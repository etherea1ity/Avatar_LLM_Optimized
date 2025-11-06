# Avatar Demo

This folder contains a self-contained copy of the Avatar HotpotQA demo with a FastAPI backend, a lightweight frontend, and the optimized model bundle.

## Layout

- `server.py` – FastAPI app exposing `/api/questions` and `/api/run`.
- `frontend/` – Static site (`index.html`, `style.css`, `main.js`).
- `model/HotpotOptimized` – Serialized Avatar agent (copied from the training run).

## Prerequisites

Install the Python dependencies (Python 3.10+ recommended):

```bash
pip install \
	fastapi "uvicorn[standard]" \
	datasets dspy dill \
	langchain langchain-community \
	arxiv
```

Notes:

- The backend tries to instantiate LangChain's `GoogleSerperAPIWrapper` (web search) and `ArxivAPIWrapper` (paper metadata). If they fail to start—because the packages or credentials are missing—it falls back to an offline cache of the `taesiri/arxiv_qa` dataset. Installing the packages above enables the real tools.
- To let `GoogleSerperAPIWrapper` run, export `SERPER_API_KEY` with a valid [Serper.dev](https://serper.dev) token before starting `uvicorn`.
- The hard-coded DashScope-compatible credentials at the top of `server.py` mirror the training setup; adjust `API_KEY`, `BASE`, and `MODEL` there (or via environment variables) to match your deployment.

## Running the backend

From the `Avatar_Demo` folder:

```bash
uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

This serves the API at `http://127.0.0.1:8000`.

## Running the frontend

In a new terminal:

```bash
cd frontend
python -m http.server 5500
```

Then open `http://127.0.0.1:5500` in your browser.

Use the dropdown to pick one of the curated questions or type your own, and compare the baseline vs Avatar optimized agent outputs.
