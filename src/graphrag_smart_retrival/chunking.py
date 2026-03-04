from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    position: int


def chunk_text(text: str, chunk_size: int, chunk_overlap: int, min_chars: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    step = max(chunk_size - chunk_overlap, 1)
    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text_value = " ".join(chunk_words).strip()
        if len(chunk_text_value) >= min_chars:
            chunks.append(chunk_text_value)
    return chunks


def create_chunks(doc_id: str, text: str, chunk_size: int, chunk_overlap: int, min_chars: int) -> list[Chunk]:
    chunks_text = chunk_text(text, chunk_size, chunk_overlap, min_chars)
    chunks: list[Chunk] = []
    for idx, chunk in enumerate(chunks_text):
        chunk_id = f"{doc_id}::chunk-{idx}"
        chunks.append(Chunk(chunk_id=chunk_id, doc_id=doc_id, text=chunk, position=idx))
    return chunks
