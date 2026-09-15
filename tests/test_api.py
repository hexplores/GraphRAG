from fastapi.testclient import TestClient

from graphrag_smart_retrieval.api import app
from graphrag_smart_retrieval.pipeline import build_index
from graphrag_smart_retrieval.config import AppConfig


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_query_endpoint_returns_results(tmp_path):
    source_dir = tmp_path / "data"
    source_dir.mkdir()
    (source_dir / "doc1.txt").write_text(
        "GraphRAG is a retrieval system for finding relevant context from documents.",
        encoding="utf-8",
    )

    index_dir = tmp_path / "index"
    build_index(
        source_dir,
        index_dir,
        AppConfig(chunk_size=40, chunk_overlap=10, min_chunk_chars=20, max_keywords=4),
    )

    client = TestClient(app)
    response = client.post(
        "/query",
        json={
            "index_dir": str(index_dir),
            "query": "retrieval for documents",
            "top_k": 3,
            "use_graph": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "retrieval for documents"
    assert len(payload["results"]) > 0
    assert payload["results"][0]["text"]


def test_query_endpoint_returns_empty_on_nonmatching_query(tmp_path):
    source_dir = tmp_path / "data"
    source_dir.mkdir()
    (source_dir / "doc1.txt").write_text(
        "Coffee brewing uses drip, pour-over, french press, and espresso methods.",
        encoding="utf-8",
    )

    index_dir = tmp_path / "index"
    build_index(
        source_dir,
        index_dir,
        AppConfig(chunk_size=60, chunk_overlap=10, min_chunk_chars=20, max_keywords=4),
    )

    client = TestClient(app)
    response = client.post(
        "/query",
        json={
            "index_dir": str(index_dir),
            "query": "what is GraphRAG",
            "top_k": 5,
            "use_graph": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "what is GraphRAG"
    assert payload["results"] == []
