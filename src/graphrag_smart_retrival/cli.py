from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .pipeline import build_index
from .retrieval import retrieve


def build_command(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    build_index(args.input, args.output, config)
    print(f"Index built at: {Path(args.output).resolve()}")


def query_command(args: argparse.Namespace) -> None:
    query_text = args.query
    if args.query_file:
        query_text = Path(args.query_file).read_text(encoding="utf-8").strip()

    if not query_text:
        raise SystemExit("Provide --query or --query-file with non-empty content.")

    results = retrieve(
        args.index,
        query_text,
        top_k=args.top_k,
        use_graph=not args.no_graph,
        expansion_hops=args.expansion_hops,
        min_score=args.min_score,
        per_doc_cap=args.per_doc_cap,
        expand_top=args.expand_top,
        expansion_cap=args.expansion_cap,
    )
    for item in results:
        print(f"[{item.score:.4f}] {item.doc_id} :: {item.chunk_id}")
        print(item.text)
        print("---")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GraphRAG Smart Retrieval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser_cmd = subparsers.add_parser("build", help="Build retrieval index")
    build_parser_cmd.add_argument("--input", required=True, help="Input directory with documents")
    build_parser_cmd.add_argument("--output", required=True, help="Output directory for index")
    build_parser_cmd.add_argument("--config", help="Path to JSON config file")
    build_parser_cmd.set_defaults(func=build_command)

    query_parser_cmd = subparsers.add_parser("query", help="Query the index")
    query_parser_cmd.add_argument("--index", required=True, help="Index directory")
    query_parser_cmd.add_argument("--query", help="Query text")
    query_parser_cmd.add_argument("--query-file", help="Path to a file containing the query text")
    query_parser_cmd.add_argument("--top-k", type=int, default=5, help="Top K results")
    query_parser_cmd.add_argument(
        "--expand-top",
        type=int,
        default=0,
        help="Use only the top N chunks as graph expansion seeds (0 = use top-k)",
    )
    query_parser_cmd.add_argument(
        "--expansion-hops",
        type=int,
        default=1,
        help="Graph expansion hops (default 1)",
    )
    query_parser_cmd.add_argument(
        "--expansion-cap",
        type=int,
        default=0,
        help="Limit the expanded candidate pool before final top-k (0 = no cap)",
    )
    query_parser_cmd.add_argument(
        "--min-score",
        type=float,
        default=0.2,
        help="Minimum similarity score to keep a chunk (default 0)",
    )
    query_parser_cmd.add_argument(
        "--per-doc-cap",
        type=int,
        default=0,
        help="Max chunks per document in results (0 = no cap)",
    )
    query_parser_cmd.add_argument("--no-graph", action="store_true", help="Disable graph expansion")
    query_parser_cmd.set_defaults(func=query_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
