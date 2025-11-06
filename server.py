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

DATASET_NAME = "hotpotqa/hotpot_qa"
DATASET_CONFIG = "distractor"
QUESTION_CHOICES = [
    "Are both Elko Regional Airport, Detroit Metro Airport and Gerald R. Ford International Airport located in Michigan?",
    "Are both Baoan Airport, Baiyun Airport and Daxing Airport both located in Guangdong Province?",
]
RETRIEVAL_SPLIT = "train[:500]"
BUNDLE_FILENAME = "HotpotOptimized.dill"

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


class SearchTool:
    """Very light-weight retrieval over HotpotQA passages."""

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
    if not hasattr(bundle, "build") and not isinstance(bundle, Avatar):
        raise TypeError("Serialized object is not an Avatar bundle")
    return bundle


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


def capture_trace(predictor, question: str):
    buf = io.StringIO()
    predictor.verbose = True
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        output = predictor(question=question)
    trace = buf.getvalue()
    answer = getattr(output, "answer", str(output))
    return answer, trace


@lru_cache(maxsize=1)
def get_question_pool() -> List[str]:
    random.seed(2024)
    choices = QUESTION_CHOICES.copy()
    random.shuffle(choices)
    return choices


@lru_cache(maxsize=1)
def get_retrieval_tool():
    train_raw = load_dataset(DATASET_NAME, DATASET_CONFIG, split=RETRIEVAL_SPLIT)
    return SearchTool(train_raw, k=5)


@lru_cache(maxsize=1)
def get_baseline():
    return dspy.TypedPredictor(HotpotQASignature)


@lru_cache(maxsize=1)
def get_optimized_agent():
    bundle_path = Path("model") / BUNDLE_FILENAME
    bundle = load_bundle(bundle_path)
    tool = get_retrieval_tool()
    tool_wrapper = Tool(
        tool=tool,
        name="SEARCH_TOOL",
        desc="retrieve short snippets related to the question from cached passages",
    )
    if hasattr(bundle, "build"):
        return bundle.build(tools=[tool_wrapper], signature_cls=HotpotQASignature)
    return bundle


@app.get("/api/questions")
def list_questions() -> Dict[str, Any]:
    return {"questions": get_question_pool()}


@app.post("/api/run")
def run_agents(body: RunRequest):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    baseline = get_baseline()
    optimized = get_optimized_agent()

    base_answer, base_trace = capture_trace(baseline, question)
    opt_answer, opt_trace = capture_trace(optimized, question)

    return {
        "question": question,
        "baseline": {
            "answer": base_answer,
            "trace": base_trace,
        },
        "optimized": {
            "answer": opt_answer,
            "trace": opt_trace,
        },
    }
