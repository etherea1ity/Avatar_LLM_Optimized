"""FastAPI backend for comparing baseline vs Avatar agents.

Start with:
    uvicorn server:app --reload
"""

from __future__ import annotations

import io
import os
import random
import contextlib
from functools import lru_cache
import sys
from pathlib import Path
from typing import List, Dict, Any

import dill

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import dspy
from datasets import load_dataset
from dspy.predict.avatar import Avatar, Tool

# ---------------------------------------------------------------------------
# Environment configuration (mirrors the notebook/demo setup)
# ---------------------------------------------------------------------------
API_KEY = "sk-4c00b82305614b50a3d5d5c68b815376"
BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "openai/deepseek-v3.2-exp"

DATASETS = {
    "hotpot": {
        "title": "HotpotQA",
        "summary": "Multi-hop reasoning over Wikipedia-style passages.",
        "dataset_name": "hotpotqa/hotpot_qa",
        "dataset_config": "distractor",
        "retrieval_split": "train[:500]",
        "bundle_dir": "model",
        "bundle": "HotpotOptimized.dill",
        "questions": [
            {
                "question": "Are both Elko Regional Airport, Detroit Metro Airport and Gerald R. Ford International Airport located in Michigan?",
            },
            {
                "question": "Are both Baoan Airport, Baiyun Airport and Daxing Airport both located in Guangdong Province?",
            },
        ],
        "requires_paper_id": False,
    },
    "arxiv": {
        "title": "ArxivQA",
        "summary": "Tool-using agent grounded on arXiv paper identifiers.",
        "dataset_name": "taesiri/arxiv_qa",
        "dataset_config": None,
        "retrieval_split": "train[:200]",
        "bundle_dir": "model",
        "bundle": "optimized_arxiv_agent.dill",
        "questions": [
            {
                "question": "What problem or question is the paper addressing?",
                "paper_id": "2307.08621",
            },
            {
                "question": "Summarize the paper in one sentence.",
                "paper_id": "2401.14361",
            },
            {
                "question": "What are the keywords or key terms associated with this paper?",
                "paper_id": "2403.03485",
            },
            {
                "question": "Summarize the main method used in the paper in one paragraph.",
                "paper_id": "2111.10659",
            },
            {
                "question": "Summarize the paper in one sentence.",
                "paper_id": "2403.17726",
            },
            {
                "question": "What are the keywords or key terms associated with this paper?",
                "paper_id": "2308.09597",
            },
            {
                "question": "Write a very high-quality and detailed summary of the paper that describes the paper in a way that a human can fully understand. The summary should cover the problem, the proposed solution and highlight the main contributions of the paper.",
                "paper_id": "2402.17505",
            },
            {
                "question": "Summarize the paper in one sentence.",
                "paper_id": "1901.09005",
            },
            {
                "question": "Summarize the paper in one sentence.",
                "paper_id": "2402.18944",
            },
            {
                "question": "What is the main contribution of this paper?",
                "paper_id": "2403.09097",
            },
        ],
        "requires_paper_id": True,
    },
}

BASE_DIR = Path(__file__).resolve().parent

os_environ = {
    "OPENAI_API_KEY": API_KEY,
    "OPENAI_API_BASE": BASE,
    "OPENAI_BASE_URL": BASE,
    "MODEL_NAME": MODEL,
}

for key, value in os_environ.items():
    os.environ[key] = value

# Configure DSPy language model
if not hasattr(dspy, "TypedPredictor"):
    dspy.TypedPredictor = dspy.Predict

dspy.settings.configure(
    lm=dspy.LM(
        model=MODEL,
        api_base=BASE,
        api_key=API_KEY,
        max_tokens=16384,
        temperature=0,
    )
)

# ---------------------------------------------------------------------------
# Shared signature + helper classes
# ---------------------------------------------------------------------------

def _strip(x: str) -> str:
    return x.strip()


class HotpotQASignature(dspy.Signature):
    """You will be given a question. Your task is to answer the question."""

    question: str = dspy.InputField(prefix="Question:", desc="question to ask", format=_strip)
    answer: str = dspy.OutputField(prefix="Answer:", desc="answer to the question")


class ArxivQASignature(dspy.Signature):
    """You will be given a question and an Arxiv Paper ID. Your task is to answer the question."""
    
    question: str = dspy.InputField(
        prefix="Question:",
        desc="question to ask",
        format=lambda x: x.strip(),
    )
    paper_id: str = dspy.InputField(
        prefix="Paper ID:",
        desc="Arxiv Paper ID",
    )
    answer: str = dspy.OutputField(
        prefix="Answer:",
        desc="answer to the question",
    )

class SearchTool:
    """Tiny lexical retriever over HotpotQA passages."""

    def __init__(self, raw_ds, k: int = 5):
        self.k = k
        self.docs: List[str] = []
        for ex in raw_ds:
            ctx = ex.get("context", [])
            for page in ctx:
                sents: List[str] = []
                if isinstance(page, (list, tuple)) and len(page) >= 2 and isinstance(page[1], list):
                    sents = page[1]
                elif isinstance(page, dict):
                    sents = page.get("sentences", []) or page.get("sents", [])
                if sents:
                    snippet = " ".join(sents[:2]).strip()
                    if snippet:
                        self.docs.append(snippet)

    def run(self, query: str) -> str:
        q_tokens = query.lower().split()
        scored = []
        for doc in self.docs:
            doc_lower = doc.lower()
            hits = sum(1 for tok in q_tokens if tok in doc_lower)
            scored.append((hits, -len(doc), doc))
        scored.sort(reverse=True)
        top = [doc for _, __, doc in scored[: self.k] if doc]
        return "\n".join(top) if top else ""


class StubTool:
    """Fallback tool that returns canned strings when tool is unavailable."""

    def __init__(self, name: str):
        self.name = name

    def run(self, query: str) -> str:
        return f"[{self.name} unavailable in demo environment; query={query}]"


@lru_cache(maxsize=1)
def _build_arxiv_offline_lookup() -> Dict[str, str]:
    """Create a lightweight summary lookup from the ArxivQA dataset."""

    try:
        dataset = load_dataset("taesiri/arxiv_qa", split="train", streaming=True)
    except Exception as exc:  # pragma: no cover - dataset unavailable
        print(f"[Avatar Demo] Failed to load ArxivQA dataset: {exc}", file=sys.stderr)
        return {}

    target_ids = {
        entry.get("paper_id", "")
        for entry in DATASETS.get("arxiv", {}).get("questions", [])
        if isinstance(entry, dict) and entry.get("paper_id")
    }

    summaries: Dict[str, str] = {}
    for row in dataset:
        paper_id = str(row.get("paper_id", "")).strip()
        if not paper_id:
            continue
        if target_ids and paper_id not in target_ids:
            continue
        question = (row.get("question") or "").lower()
        answer = (row.get("answer") or "").strip()
        if not answer:
            continue
        if paper_id not in summaries:
            summaries[paper_id] = answer
        if "summary" in question:
            summaries[paper_id] = answer
        if target_ids and len(summaries) >= len(target_ids):
            break
    return summaries


class ArxivOfflineTool:
    """Offline fallback returning cached summaries from the ArxivQA dataset."""

    def run(self, paper_id: str) -> str:
        lookup = _build_arxiv_offline_lookup()
        key = (paper_id or "").strip()
        if not key:
            return "[ARXIV_SEARCH offline] Missing paper id"
        summary = lookup.get(key)
        if not summary:
            return f"[ARXIV_SEARCH offline] No cached summary for paper {key}"
        return summary


def _create_arxiv_tools() -> List[Tool]:
    """Instantiate the LangChain wrappers when possible; otherwise fall back to stubs."""

    try:
        from langchain_community.utilities import GoogleSerperAPIWrapper, ArxivAPIWrapper
    except ImportError:
        GoogleSerperAPIWrapper = None  # type: ignore[assignment]
        ArxivAPIWrapper = None  # type: ignore[assignment]

    web_impl: Any = None
    arxiv_impl: Any = None

    if GoogleSerperAPIWrapper is not None:
        try:
            web_impl = GoogleSerperAPIWrapper()
        except Exception as exc:  # pragma: no cover - configuration issues
            print(f"[Avatar Demo] WEB_SEARCH unavailable: {exc}", file=sys.stderr)
    if web_impl is None:
        web_impl = StubTool("WEB_SEARCH")

    if ArxivAPIWrapper is not None:
        try:
            arxiv_impl = ArxivAPIWrapper()
        except Exception as exc:
            print(f"[Avatar Demo] ARXIV_SEARCH unavailable: {exc}", file=sys.stderr)
    if arxiv_impl is None:
        arxiv_impl = ArxivOfflineTool()

    return [
        Tool(
            tool=web_impl,
            name="WEB_SEARCH",
            desc="If you have a question, you can use this tool to search the web for the answer.",
        ),
        Tool(
            tool=arxiv_impl,
            name="ARXIV_SEARCH",
            desc="Pass the arxiv paper id to get the paper information.",
            input_type="Arxiv Paper ID",
        ),
    ]


# ---------------------------------------------------------------------------
# Avatar bundle helpers
# ---------------------------------------------------------------------------

class AvatarBundle:
    def __init__(self, signature_cls_name: str, avatar_kwargs: Dict[str, Any], artifacts: Dict[str, Any]):
        self.signature_cls_name = signature_cls_name
        self.avatar_kwargs = avatar_kwargs
        self.artifacts = artifacts or {}

    def build(self, tools, signature_cls):
        agent = Avatar(tools=tools, signature=signature_cls, **self.avatar_kwargs)
        for key, value in self.artifacts.items():
            setattr(agent, key, value)
        return agent


@lru_cache(maxsize=1)
def load_bundle(path: Path) -> AvatarBundle:
    if not path.exists():
        raise FileNotFoundError(f"Avatar bundle missing: {path}")
    with path.open("rb") as fh:
        bundle = dill.load(fh)
    _ensure_bundle_dependencies(bundle)
    if not hasattr(bundle, "build") and not isinstance(bundle, Avatar):
        raise TypeError("Serialized object is not an Avatar bundle")
    return bundle


def _ensure_bundle_dependencies(obj) -> None:
    module_name = getattr(obj.__class__, "__module__", None)
    if not module_name:
        return
    module = sys.modules.get(module_name)
    if module is None:
        return
    if not hasattr(module, "Avatar"):
        setattr(module, "Avatar", Avatar)
    if not hasattr(module, "Tool"):
        setattr(module, "Tool", Tool)


# ---------------------------------------------------------------------------
# FastAPI app setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Avatar QA Demo", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    question: str
    dataset: str | None = None
    paper_id: str | None = None


def capture_trace(predictor, **inputs):
    buf = io.StringIO()
    predictor.verbose = True
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        output = predictor(**inputs)
    trace = buf.getvalue()
    answer = getattr(output, "answer", str(output))
    return answer, trace


def _resolve_dataset(dataset: str | None) -> str:
    if dataset is None:
        return "hotpot"
    ds = dataset.lower().strip()
    if ds not in DATASETS:
        raise HTTPException(status_code=400, detail=f"Unsupported dataset '{dataset}'. Available: {', '.join(DATASETS)}")
    return ds

def get_question_pool(dataset: str) -> List[Dict[str, Any]]:
    data = DATASETS[dataset]
    items = [dict(entry) for entry in data["questions"]]
    rng = random.Random(2024)
    rng.shuffle(items)
    return items


@lru_cache(maxsize=1)
def get_hotpot_retrieval():
    cfg = DATASETS["hotpot"]
    config = cfg["dataset_config"]
    if config:
        train_raw = load_dataset(cfg["dataset_name"], config, split=cfg["retrieval_split"])
    else:
        train_raw = load_dataset(cfg["dataset_name"], split=cfg["retrieval_split"])
    return SearchTool(train_raw, k=5)


@lru_cache(maxsize=1)
def get_hotpot_baseline():
    return dspy.TypedPredictor(HotpotQASignature)


@lru_cache(maxsize=1)
def get_hotpot_optimized():
    cfg = DATASETS["hotpot"]
    bundle_path = BASE_DIR / cfg["bundle_dir"] / cfg["bundle"]
    bundle = load_bundle(bundle_path)
    tool = get_hotpot_retrieval()
    tool_wrapper = Tool(
        tool=tool,
        name="SEARCH_TOOL",
        desc="retrieve short snippets related to the question from cached passages",
    )
    if hasattr(bundle, "build"):
        return bundle.build(tools=[tool_wrapper], signature_cls=HotpotQASignature)
    return bundle


@lru_cache(maxsize=1)
def get_arxiv_optimized():
    cfg = DATASETS["arxiv"]
    bundle_path = BASE_DIR / cfg["bundle_dir"] / cfg["bundle"]
    bundle = load_bundle(bundle_path)
    tools = _create_arxiv_tools()
    if hasattr(bundle, "build"):
        return bundle.build(tools=tools, signature_cls=ArxivQASignature)
    return bundle


@lru_cache(maxsize=1)
def get_arxiv_baseline():
    return dspy.TypedPredictor(ArxivQASignature)


@app.get("/api/questions")
def list_questions(dataset: str | None = None) -> Dict[str, Any]:
    if dataset is not None:
        key = _resolve_dataset(dataset)
        info = DATASETS[key]
        return {
            "dataset": key,
            "title": info["title"],
            "summary": info["summary"],
            "requires_paper_id": info["requires_paper_id"],
            "questions": get_question_pool(key),
        }

    return {
        "default": "hotpot",
        "datasets": {
            key: {
                "title": value["title"],
                "summary": value["summary"],
                "requires_paper_id": value["requires_paper_id"],
                "questions": get_question_pool(key),
            }
            for key, value in DATASETS.items()
        },
    }


@app.post("/api/run")
def run_agents(body: RunRequest):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    dataset_key = _resolve_dataset(body.dataset)
    cfg = DATASETS[dataset_key]

    inputs = {"question": question}
    if cfg["requires_paper_id"]:
        paper_id = (body.paper_id or "").strip()
        if not paper_id:
            raise HTTPException(status_code=400, detail="paper_id is required for the Arxiv dataset")
        inputs["paper_id"] = paper_id
    elif body.paper_id:
        inputs["paper_id"] = body.paper_id.strip()

    if dataset_key == "hotpot":
        baseline = get_hotpot_baseline()
        optimized = get_hotpot_optimized()
    elif dataset_key == "arxiv":
        baseline = get_arxiv_baseline()
        optimized = get_arxiv_optimized()
    else:
        raise HTTPException(status_code=500, detail="Dataset configuration missing")

    base_answer, base_trace = capture_trace(baseline, **inputs)
    opt_answer, opt_trace = capture_trace(optimized, **inputs)

    return {
        "question": question,
        "dataset": dataset_key,
        "paper_id": inputs.get("paper_id"),
        "baseline": {
            "answer": base_answer,
            "trace": base_trace,
        },
        "optimized": {
            "answer": opt_answer,
            "trace": opt_trace,
        },
    }
