from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class EmbeddingIndex:
    vectorizer: TfidfVectorizer
    matrix: sparse.csr_matrix


def build_embeddings(texts: list[str]) -> EmbeddingIndex:
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    matrix = vectorizer.fit_transform(texts)
    return EmbeddingIndex(vectorizer=vectorizer, matrix=matrix)


def save_embeddings(index: EmbeddingIndex, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    with (output / "vectorizer.pkl").open("wb") as handle:
        pickle.dump(index.vectorizer, handle)

    sparse.save_npz(output / "matrix.npz", index.matrix)


def load_embeddings(output_dir: str | Path) -> EmbeddingIndex:
    output = Path(output_dir)
    with (output / "vectorizer.pkl").open("rb") as handle:
        vectorizer = pickle.load(handle)

    matrix = sparse.load_npz(output / "matrix.npz")
    return EmbeddingIndex(vectorizer=vectorizer, matrix=matrix)


def embed_query(vectorizer: TfidfVectorizer, query: str) -> np.ndarray:
    return vectorizer.transform([query])
