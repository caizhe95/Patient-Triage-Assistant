from __future__ import annotations

from functools import lru_cache
from typing import Any, cast

import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from app.config import get_settings
from app.utils.logger import get_logger


logger = get_logger(__name__)


class TfidfEmbedder:
    def __init__(self) -> None:
        self.backend = "tfidf"
        self.model_name = "tfidf-jieba-bigram"
        self.vectorizer = TfidfVectorizer(
            tokenizer=jieba.lcut,
            lowercase=False,
            ngram_range=(1, 2),
            min_df=1,
        )
        self.fitted = False

    def fit(self, texts: list[str]) -> None:
        self.vectorizer.fit(texts)
        self.fitted = True

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        if not self.fitted:
            self.fit(texts)
        matrix = self.vectorizer.transform(texts)
        return cast(Any, matrix).astype(np.float32).toarray()

    def encode_query(self, text: str) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("tfidf embedder not fitted")
        matrix = self.vectorizer.transform([text])
        return cast(Any, matrix).astype(np.float32).toarray()


class BGEEmbedder:
    def __init__(self, model_name: str) -> None:
        self.backend = "bge"
        self.model_name = model_name
        self.model = None
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
        except Exception as exc:  # pragma: no cover
            logger.warning("failed to load bge model %s, fallback to tfidf: %s", model_name, exc)
            self.backend = "tfidf"

    def available(self) -> bool:
        return self.model is not None

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        if self.model is None:
            return tfidf_embedder().encode_documents(texts)
        vecs = self.model.encode(
            [f"passage: {t}" for t in texts],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        if self.model is None:
            return tfidf_embedder().encode_query(text)
        vec = self.model.encode(
            [f"query: {text}"],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vec, dtype=np.float32)


@lru_cache(maxsize=1)
def tfidf_embedder() -> TfidfEmbedder:
    return TfidfEmbedder()


@lru_cache(maxsize=1)
def get_embedder():
    settings = get_settings()
    if settings.embedding_backend.lower() == "bge":
        embedder = BGEEmbedder(settings.embedding_model)
        if embedder.available():
            return embedder
        return tfidf_embedder()
    return tfidf_embedder()


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
    return vectors / norms
