from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re
import ssl
from urllib.parse import quote
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import certifi


ATOM = "http://www.w3.org/2005/Atom"
ARXIV = "http://arxiv.org/schemas/atom"


def clean_filename(value: str) -> str:
    filename = re.sub(r"[^A-Za-z0-9._-]+", "s_", value).strip("._")
    return filename[:120] or "paper"


def text(element: ET.Element | None, tag: str) -> str:
    if element is None:
        return ""
    child = element.find(f"{{{ATOM}}}{tag}")
    return " ".join((child.text or "").split()) if child is not None else ""


def collect(query: str, output_dir: Path, max_results: int) -> int:
    search = quote(f'all:"{query}"')
    url = (
        "https://export.arxiv.org/api/query?search_query="
        f"{search}&start=0&max_results={max_results}&sortBy=relevance"
    )
    request = Request(url, headers={"User-Agent": "smart-retrieval-research-collector/1.0"})
    context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(request, timeout=30, context=context) as response:
        root = ET.fromstring(response.read())

    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for entry in root.findall(f"{{{ATOM}}}entry"):
        title = text(entry, "title")
        summary = text(entry, "summary")
        paper_url = text(entry, "id")
        authors = [
            " ".join((author.find(f"{{{ATOM}}}name").text or "").split())
            for author in entry.findall(f"{{{ATOM}}}author")
            if author.find(f"{{{ATOM}}}name") is not None
        ]
        published = text(entry, "published")
        year = datetime.fromisoformat(published.replace("Z", "+00:00")).year if published else ""
        categories = [category.attrib.get("term", "") for category in entry.findall(f"{{{ARXIV}}}primary_category")]
        filename = output_dir / f"{clean_filename(title)}.md"
        filename.write_text(
            "\n".join(
                [
                    f"# {title}",
                    "",
                    "## Authors",
                    ", ".join(authors),
                    "",
                    "## Year",
                    str(year),
                    "",
                    "## Categories",
                    ", ".join(categories),
                    "",
                    "## Abstract",
                    summary,
                    "",
                    "## Source",
                    paper_url,
                    "",
                ]
            ),
            encoding="utf-8",
        )
        count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect open ArXiv metadata and abstracts as Markdown files.")
    parser.add_argument("--query", required=True, help="Research topic, for example: explainable artificial intelligence")
    parser.add_argument("--output", default="data/papers", help="Output directory for Markdown files")
    parser.add_argument("--max-results", type=int, default=20, help="Maximum number of papers to collect")
    args = parser.parse_args()

    if not 1 <= args.max_results <= 100:
        parser.error("--max-results must be between 1 and 100")

    count = collect(args.query, Path(args.output), args.max_results)
    print(f"Collected {count} papers into {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
