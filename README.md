# Avatar Demo

This folder contains a self-contained copy of the Avatar HotpotQA demo with a FastAPI backend, a lightweight frontend, and the optimized model bundle.

## Layout

- `server.py` – FastAPI app exposing `/api/questions` and `/api/run`.
- `frontend/` – Static site (`index.html`, `style.css`, `main.js`).
- `model/HotpotOptimized` – Serialized Avatar agent (copied from the training run).

## Prerequisites

```bash
pip install fastapi "uvicorn[standard]" datasets dspy dill
```

Ensure your environment variables for the DashScope-compatible endpoint stay the same as in training; `server.py` sets the key/base URL automatically from the constants at the top of the file.

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
