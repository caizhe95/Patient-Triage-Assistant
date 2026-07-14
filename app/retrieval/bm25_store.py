from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jieba
from rank_bm25 import BM25Okapi


@dataclass
class BM25IndexBundle:
    domain: str
    path: Path


class DomainBM25Store:
    def __init__(self, domain: str, index_dir: Path) -> None:
        self.domain = domain
        self.index_dir = index_dir
        self.documents: list[dict[str, Any]] = []
        self.tokenized_corpus: list[list[str]] = []
        self.bm25: BM25Okapi | None = None
        self._load()

    def _load(self) -> None:
        path = self.index_dir / f"{self.domain}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"BM25 index missing for domain={self.domain}")
        payload = pickle.loads(path.read_bytes())
        self.documents = payload["documents"]
        self.tokenized_corpus = payload["tokenized_corpus"]
        self.bm25 = payload["bm25"]

    def search(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        if not self.bm25 or not self.documents:
            return []
        tokens = jieba.lcut(query)
        scores = self.bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:top_k]
        results = []
        for idx in order:
            doc = dict(self.documents[int(idx)])
            doc["score"] = float(scores[idx])
            doc["retrieval_source"] = "bm25"
            results.append(doc)
        return results
