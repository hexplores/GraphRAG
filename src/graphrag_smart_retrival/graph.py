from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path

import networkx as nx

from .chunking import Chunk


@dataclass
class GraphArtifacts:
    graph: nx.Graph


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")


def extract_keywords(text: str, max_keywords: int) -> list[str]:
    tokens = TOKEN_RE.findall(text.lower())
    if not tokens:
        return []

    frequencies: dict[str, int] = {}
    for token in tokens:
        frequencies[token] = frequencies.get(token, 0) + 1

    sorted_tokens = sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))
    return [token for token, _ in sorted_tokens[:max_keywords]]


def build_graph(doc_ids: list[str], chunks: list[Chunk], max_keywords: int) -> GraphArtifacts:
    graph = nx.Graph()

    for doc_id in doc_ids:
        graph.add_node(f"doc::{doc_id}", node_type="document", doc_id=doc_id)

    for chunk in chunks:
        chunk_node = f"chunk::{chunk.chunk_id}"
        graph.add_node(
            chunk_node,
            node_type="chunk",
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
        )
        graph.add_edge(f"doc::{chunk.doc_id}", chunk_node, edge_type="contains")

        for keyword in extract_keywords(chunk.text, max_keywords):
            keyword_node = f"kw::{keyword}"
            if not graph.has_node(keyword_node):
                graph.add_node(keyword_node, node_type="keyword", keyword=keyword)
            graph.add_edge(chunk_node, keyword_node, edge_type="mentions")

    return GraphArtifacts(graph=graph)


def save_graph(artifacts: GraphArtifacts, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = nx.readwrite.json_graph.node_link_data(artifacts.graph)
    (output / "graph.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_graph(output_dir: str | Path) -> GraphArtifacts:
    output = Path(output_dir)
    data = json.loads((output / "graph.json").read_text(encoding="utf-8"))
    graph = nx.readwrite.json_graph.node_link_graph(data)
    return GraphArtifacts(graph=graph)
