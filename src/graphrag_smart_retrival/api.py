from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import AppConfig, load_config
from .pipeline import build_index
from .retrieval import retrieve


class QueryRequest(BaseModel):
    index_dir: str | None = Field(
        default=None,
        description="Directory containing the prebuilt GraphRAG index.",
    )
    query: str = Field(..., min_length=1, description="Search query text.")
    top_k: int = Field(default=5, ge=1)
    use_graph: bool = True
    no_graph: bool = False
    expansion_hops: int = Field(default=1, ge=0)
    min_score: float = Field(default=0.0, ge=0.0)
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


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "GraphRAG Smart Retrieval API",
        "health": "/health",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "graphrag-smart-retrieval"}


@app.post("/build")
def build_api(payload: BuildRequest) -> dict[str, str]:
    config = load_config(payload.config_path) if payload.config_path else AppConfig()
    build_index(payload.input_dir, payload.output_dir, config)
    return {
        "status": "ok",
        "input_dir": str(Path(payload.input_dir).resolve()),
        "output_dir": str(Path(payload.output_dir).resolve()),
    }


@app.post("/query")
def query_api(payload: QueryRequest) -> dict[str, object]:
    index_dir = payload.index_dir or os.getenv("GRAPHRAG_INDEX_DIR") or os.getenv("INDEX_DIR")
    if not index_dir:
        raise HTTPException(
            status_code=400,
            detail="Provide index_dir in the request body or set the GRAPHRAG_INDEX_DIR environment variable.",
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

    return {
        "query": payload.query,
        "index_dir": str(Path(index_dir).resolve()),
        "results": [
            {
                "chunk_id": item.chunk_id,
                "score": float(item.score),
                "text": item.text,
                "doc_id": item.doc_id,
            }
            for item in results
        ],
    }


def main() -> None:
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("graphrag_smart_retrieval.api:app", host=host, port=port)
