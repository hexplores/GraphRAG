from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from .chunking import create_chunks
from .config import AppConfig
from .embeddings import build_embeddings, save_embeddings
from .graph import build_graph, save_graph
from .ingest import load_documents


def build_index(input_dir: str | Path, output_dir: str | Path, config: AppConfig) -> None:
    documents = load_documents(input_dir)

    all_chunks = []
    chunk_records = []
    for doc in documents:
        chunks = create_chunks(
            doc.doc_id,
            doc.text,
            config.chunk_size,
            config.chunk_overlap,
            config.min_chunk_chars,
        )
        for chunk in chunks:
            all_chunks.append(chunk)
            chunk_records.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "position": chunk.position,
                    "text": chunk.text,
                }
            )

    texts = [chunk.text for chunk in all_chunks]
    embeddings = build_embeddings(texts)
    save_embeddings(embeddings, output_dir)

    graph = build_graph([doc.doc_id for doc in documents], all_chunks, config.max_keywords)
    save_graph(graph, output_dir)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "chunks.json").write_text(json.dumps(chunk_records, indent=2), encoding="utf-8")
    (output / "metadata.json").write_text(
        json.dumps({"config": asdict(config), "documents": [doc.doc_id for doc in documents]}, indent=2),
        encoding="utf-8",
    )
