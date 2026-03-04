from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

from .embeddings import EmbeddingIndex, embed_query, load_embeddings
from .graph import GraphArtifacts, load_graph


@dataclass
class RetrievedChunk:
    chunk_id: str
    score: float
    text: str
    doc_id: str


def load_chunks(output_dir: str | Path) -> list[dict]:
    output = Path(output_dir)
    return json.loads((output / "chunks.json").read_text(encoding="utf-8"))


def rank_chunks(index: EmbeddingIndex, chunk_texts: list[str], query: str) -> np.ndarray:
    query_vector = embed_query(index.vectorizer, query)
    similarities = cosine_similarity(query_vector, index.matrix)[0]
    return similarities


def expand_with_graph(
    graph: GraphArtifacts,
    chunk_nodes: list[str],
    expansion_hops: int = 1,
) -> set[str]:
    expanded: set[str] = set(chunk_nodes)
    for chunk_node in chunk_nodes:
        if not graph.graph.has_node(chunk_node):
            continue
        for neighbor in nx.single_source_shortest_path_length(
            graph.graph, chunk_node, cutoff=expansion_hops
        ):
            if neighbor.startswith("chunk::"):
                expanded.add(neighbor)
    return expanded


def retrieve(
    index_dir: str | Path,
    query: str,
    top_k: int,
    use_graph: bool = True,
    expansion_hops: int = 1,
    min_score: float = 0.0,
    per_doc_cap: int = 0,
    expand_top: int = 0,
    expansion_cap: int = 0,
) -> list[RetrievedChunk]:
    index = load_embeddings(index_dir)
    chunks = load_chunks(index_dir)
    texts = [chunk["text"] for chunk in chunks]
    scores = rank_chunks(index, texts, query)

    ranked_indices = np.argsort(scores)[::-1]
    top_indices = ranked_indices[: max(top_k, 1)]

    expansion_seed = top_indices
    if expand_top > 0:
        expansion_seed = ranked_indices[: max(expand_top, 1)]

    graph_nodes = [f"chunk::{chunks[idx]['chunk_id']}" for idx in expansion_seed]
    expanded_indices = set(top_indices.tolist())

    if use_graph:
        graph = load_graph(index_dir)
        expanded_nodes = expand_with_graph(graph, graph_nodes, expansion_hops=expansion_hops)
        chunk_index_map = {chunk["chunk_id"]: idx for idx, chunk in enumerate(chunks)}
        for node in expanded_nodes:
            chunk_id = node.replace("chunk::", "")
            idx = chunk_index_map.get(chunk_id)
            if idx is not None:
                expanded_indices.add(idx)

    results: list[RetrievedChunk] = []
    for idx in expanded_indices:
        if scores[idx] < min_score:
            continue
        chunk = chunks[idx]
        results.append(
            RetrievedChunk(
                chunk_id=chunk["chunk_id"],
                score=float(scores[idx]),
                text=chunk["text"],
                doc_id=chunk["doc_id"],
            )
        )

    results.sort(key=lambda item: item.score, reverse=True)
    if expansion_cap > 0:
        results = results[:expansion_cap]

    if per_doc_cap > 0:
        limited: list[RetrievedChunk] = []
        per_doc: dict[str, int] = {}
        for item in results:
            count = per_doc.get(item.doc_id, 0)
            if count >= per_doc_cap:
                continue
            per_doc[item.doc_id] = count + 1
            limited.append(item)
        results = limited

    return results[:top_k]
