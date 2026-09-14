from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SUPPORTED_EXTENSIONS = {".txt", ".md"}


@dataclass
class Document:
    doc_id: str
    path: Path
    text: str


def load_documents(input_dir: str | Path) -> list[Document]:
    base = Path(input_dir)
    documents: list[Document] = []

    for path in base.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        text = path.read_text(encoding="utf-8", errors="ignore")
        doc_id = str(path.relative_to(base))
        documents.append(Document(doc_id=doc_id, path=path, text=text))

    return documents
