from __future__ import annotations

import os
from pathlib import Path
from collections import defaultdict, deque
from time import monotonic

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from .config import AppConfig, load_config
from .pipeline import build_index
from .retrieval import retrieve


class QueryRequest(BaseModel):
    index_dir: str | None = Field(
        default=None,
        description="Directory containing the prebuilt GraphRAG index.",
    )
    query: str = Field(..., min_length=1, max_length=2000, description="Search query text.")
    top_k: int = Field(default=5, ge=1, le=50)
    use_graph: bool = True
    no_graph: bool = False
    expansion_hops: int = Field(default=1, ge=0)
    min_score: float = Field(default=0.2, ge=0.0)
    per_doc_cap: int = Field(default=0, ge=0)
    expand_top: int = Field(default=0, ge=0)
    expansion_cap: int = Field(default=0, ge=0)


class BuildRequest(BaseModel):
    input_dir: str = Field(..., description="Directory containing source documents.")
    output_dir: str = Field(..., description="Where to write the generated index.")
    config_path: str | None = Field(default=None, description="Optional JSON config file path.")


app = FastAPI(
    title="GraphRAG Smart Retrieval",
    version="0.1.0",
    description="Public API for querying a GraphRAG index over HTTP.",
)


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI) -> None:
        super().__init__(app)
        self.requests: dict[str, deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        if request.url.path in {"/", "/health", "/docs", "/openapi.json", "/redoc"}:
            return await call_next(request)

        limit = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
        client_id = request.client.host if request.client else "unknown"
        now = monotonic()
        window = self.requests[client_id]
        while window and now - window[0] >= 60:
            window.popleft()
        if len(window) >= limit:
            return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded."})
        window.append(now)
        return await call_next(request)


app.add_middleware(RateLimitMiddleware)


def require_api_key(api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    expected = os.getenv("API_KEY")
    if expected and api_key != expected:
        raise HTTPException(status_code=401, detail="A valid X-API-Key is required.")


def configured_path(
    requested: str | None,
    environment_name: str,
    default: str,
) -> Path:
    allowed = Path(os.getenv(environment_name, default)).resolve()
    candidate = Path(requested or allowed).resolve()
    if candidate != allowed:
        raise HTTPException(
            status_code=403,
            detail=f"Use the configured {environment_name} path.",
        )
    return allowed


def configured_file(requested: str, environment_name: str, default: str) -> Path:
    allowed_dir = Path(os.getenv(environment_name, default)).resolve()
    candidate = Path(requested).resolve()
    try:
        candidate.relative_to(allowed_dir)
    except ValueError as exc:
        raise HTTPException(
            status_code=403,
            detail=f"Use a file inside the configured {environment_name} directory.",
        ) from exc
    if not candidate.is_file():
        raise HTTPException(status_code=400, detail="Configuration file was not found.")
    return candidate


@app.get("/")
def root() -> HTMLResponse:
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>GraphRAG Smart Retrieval</title>
        <style>
            :root {
                --bg: #0b1020;
                --panel: #111827;
                --panel-2: #1f2937;
                --text: #e5e7eb;
                --muted: #9ca3af;
                --accent: #60a5fa;
                --accent-2: #34d399;
                --border: #374151;
            }
            * { box-sizing: border-box; }
            body {
                margin: 0;
                font-family: Arial, sans-serif;
                background: linear-gradient(135deg, var(--bg), #111827);
                color: var(--text);
            }
            .container {
                max-width: 960px;
                margin: 48px auto;
                padding: 20px;
            }
            .panel {
                background: rgba(17, 24, 39, 0.96);
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 24px;
                box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
            }
            h1 {
                margin-top: 0;
                font-size: 2rem;
            }
            .row {
                display: grid;
                grid-template-columns: 1fr 160px;
                gap: 12px;
                margin-bottom: 16px;
            }
            input, button {
                width: 100%;
                font: inherit;
                border-radius: 10px;
                border: 1px solid var(--border);
            }
            input {
                padding: 12px 14px;
                background: var(--panel-2);
                color: var(--text);
            }
            button {
                padding: 12px 14px;
                background: var(--accent);
                color: #08111f;
                font-weight: 700;
                cursor: pointer;
            }
            textarea {
                width: 100%;
                min-height: 130px;
                resize: vertical;
                padding: 12px 14px;
                border-radius: 10px;
                border: 1px solid var(--border);
                background: var(--panel-2);
                color: var(--text);
                font: inherit;
                margin-bottom: 16px;
            }
            .meta {
                color: var(--muted);
                margin-bottom: 18px;
                line-height: 1.5;
            }
            .result {
                background: #0f172a;
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 14px 16px;
                margin-top: 14px;
            }
            .score {
                color: var(--accent-2);
                font-weight: 700;
                margin-bottom: 8px;
            }
            .doc {
                color: var(--muted);
                font-size: 0.9rem;
                margin-top: 8px;
            }
            .empty {
                color: var(--muted);
                margin-top: 18px;
            }
            .api-key {
                margin-bottom: 16px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="panel">
                <h1>GraphRAG Smart Retrieval</h1>
                <div class="meta">
                    Query your built index and inspect the matching passages.
                    If no result is relevant, the app will return an empty set instead of weak matches.
                </div>

                <div class="row">
                    <input id="indexDir" type="text" value="./index" placeholder="Index directory" />
                    <button id="queryBtn" type="button">Search</button>
                </div>
                <input id="apiKey" class="api-key" type="password" placeholder="API key (optional)" />
                <textarea id="queryInput" placeholder="Ask a question...">coffee brewing methods</textarea>

                <div id="results"></div>
            </div>
        </div>

        <script>
            const resultsEl = document.getElementById('results');
            const queryBtn = document.getElementById('queryBtn');

            async function runQuery() {
                const indexDir = document.getElementById('indexDir').value.trim() || './index';
                const apiKey = document.getElementById('apiKey').value.trim();
                const query = document.getElementById('queryInput').value.trim();

                if (!query) {
                    resultsEl.innerHTML = '<div class="empty">Please enter a question.</div>';
                    return;
                }

                resultsEl.innerHTML = '<div class="empty">Searching...</div>';

                try {
                    const headers = { 'Content-Type': 'application/json' };
                    if (apiKey) headers['X-API-Key'] = apiKey;
                    const response = await fetch('/query', {
                        method: 'POST',
                        headers,
                        body: JSON.stringify({
                            index_dir: indexDir,
                            query,
                            top_k: 5,
                            use_graph: true,
                            no_graph: false,
                            expansion_hops: 1,
                            min_score: 0.2,
                            per_doc_cap: 0,
                            expand_top: 0,
                            expansion_cap: 0
                        })
                    });

                    const payload = await response.json();

                    if (!response.ok) {
                        throw new Error(payload.detail || 'Request failed');
                    }

                    const items = payload.results || [];
                    if (!items.length) {
                        resultsEl.innerHTML = '<div class="empty">No relevant results found for this query in the current index.</div>';
                        return;
                    }

                    resultsEl.replaceChildren(...items.map(item => {
                        const result = document.createElement('div');
                        result.className = 'result';
                        const score = document.createElement('div');
                        score.className = 'score';
                        score.textContent = `Score: ${Number(item.score).toFixed(4)}`;
                        const text = document.createElement('div');
                        text.textContent = item.text;
                        const documentInfo = document.createElement('div');
                        documentInfo.className = 'doc';
                        documentInfo.textContent = `Document: ${item.doc_id} · Chunk: ${item.chunk_id}`;
                        result.append(score, text, documentInfo);
                        return result;
                    }));
                } catch (error) {
                    resultsEl.innerHTML = `<div class="empty">Error: ${error.message}</div>`;
                }
            }

            queryBtn.addEventListener('click', runQuery);
            document.getElementById('queryInput').addEventListener('keydown', (event) => {
                if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
                    runQuery();
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "graphrag-smart-retrieval"}


@app.post("/build", dependencies=[Depends(require_api_key)])
def build_api(payload: BuildRequest) -> dict[str, str]:
    input_dir = configured_path(payload.input_dir, "GRAPHRAG_DATA_DIR", "data")
    output_dir = configured_path(payload.output_dir, "GRAPHRAG_INDEX_DIR", "index")
    config_path = configured_file(payload.config_path, "GRAPHRAG_CONFIG_DIR", "configs") if payload.config_path else None
    config = load_config(config_path) if config_path else AppConfig()
    build_index(input_dir, output_dir, config)
    return {
        "status": "ok",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
    }


@app.post("/query", dependencies=[Depends(require_api_key)])
def query_api(payload: QueryRequest) -> dict[str, object]:
    index_dir = configured_path(
        payload.index_dir,
        "GRAPHRAG_INDEX_DIR",
        os.getenv("INDEX_DIR", "index"),
    )

    use_graph = payload.use_graph and not payload.no_graph
    results = retrieve(
        index_dir=index_dir,
        query=payload.query,
        top_k=payload.top_k,
        use_graph=use_graph,
        expansion_hops=payload.expansion_hops,
        min_score=payload.min_score,
        per_doc_cap=payload.per_doc_cap,
        expand_top=payload.expand_top,
        expansion_cap=payload.expansion_cap,
    )

    filtered_results = [
        {
            "chunk_id": item.chunk_id,
            "score": float(item.score),
            "text": item.text,
            "doc_id": item.doc_id,
        }
        for item in results
        if item.score > payload.min_score
    ]

    if not filtered_results:
        return {
            "query": payload.query,
            "index_dir": str(index_dir),
            "results": [],
            "message": "No relevant results found for this query in the current index.",
        }

    return {
        "query": payload.query,
        "index_dir": str(index_dir),
        "results": filtered_results,
    }


def main() -> None:
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("graphrag_smart_retrieval.api:app", host=host, port=port)
