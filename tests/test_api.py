from fastapi.testclient import TestClient

from graphrag_smart_retrieval.api import app
from graphrag_smart_retrieval.pipeline import build_index
from graphrag_smart_retrieval.config import AppConfig
from graphrag_smart_retrieval.ingest import load_documents
from graphrag_smart_retrieval.graph import build_graph, extract_entities
from graphrag_smart_retrieval.chunking import create_chunks


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_query_endpoint_returns_results(tmp_path, monkeypatch):
    source_dir = tmp_path / "data"
    source_dir.mkdir()
    (source_dir / "doc1.txt").write_text(
        "GraphRAG is a retrieval system for finding relevant context from documents.",
        encoding="utf-8",
    )

    index_dir = tmp_path / "index"
    monkeypatch.setenv("GRAPHRAG_INDEX_DIR", str(index_dir))
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


def test_query_endpoint_returns_empty_on_nonmatching_query(tmp_path, monkeypatch):
    source_dir = tmp_path / "data"
    source_dir.mkdir()
    (source_dir / "doc1.txt").write_text(
        "Coffee brewing uses drip, pour-over, french press, and espresso methods.",
        encoding="utf-8",
    )

    index_dir = tmp_path / "index"
    monkeypatch.setenv("GRAPHRAG_INDEX_DIR", str(index_dir))
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


def test_query_rejects_index_outside_configured_path(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPHRAG_INDEX_DIR", str(tmp_path / "index"))
    client = TestClient(app)

    response = client.post(
        "/query",
        json={"index_dir": str(tmp_path / "other"), "query": "test"},
    )

    assert response.status_code == 403


def test_ingest_supports_structured_text_formats(tmp_path):
    (tmp_path / "page.html").write_text("<h1>Coffee</h1><p>Pour over</p>", encoding="utf-8")
    (tmp_path / "record.json").write_text('{"title": "Coffee", "method": "Drip"}', encoding="utf-8")
    (tmp_path / "table.csv").write_text("method,body\ndrip,clean\n", encoding="utf-8")

    documents = load_documents(tmp_path)
    contents = "\n".join(document.text for document in documents)

    assert len(documents) == 3
    assert "Coffee" in contents
    assert "Drip" in contents
    assert "clean" in contents


def test_graph_extracts_entity_nodes():
    text = "OpenAI developed ChatGPT in San Francisco."
    chunks = create_chunks("doc.txt", text, chunk_size=200, chunk_overlap=0, min_chars=1)
    graph = build_graph(["doc.txt"], chunks, max_keywords=4).graph

    assert extract_entities(text)
    assert any(data.get("node_type") == "entity" for _, data in graph.nodes(data=True))
