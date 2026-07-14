from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient

from app.config import get_settings
from app.embeddings import get_embedder, normalize
from app.retrieval.qdrant_utils import create_qdrant_client


@dataclass
class VectorIndexBundle:
    domain: str
    collection_name: str
    storage_dir: Path


class DomainVectorStore:
    def __init__(self, domain: str, index_dir: Path) -> None:
        self.domain = domain
        self.index_dir = index_dir
        self.settings = get_settings()
        self.collection_name = f"{self.settings.qdrant_collection_prefix}_{domain}"
        self.client: QdrantClient = create_qdrant_client(self.settings)

    def _query_points(self, query_vec: list[float], top_k: int) -> list[Any]:
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vec,
            limit=top_k,
            with_payload=True,
        )
        return list(response.points)

    def search(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        query_vec = normalize(get_embedder().encode_query(query).astype(np.float32))[0].tolist()
        try:
            hits = self._query_points(query_vec=query_vec, top_k=top_k)
        except Exception:
            return []

        results: list[dict[str, Any]] = []
        for hit in hits:
            payload = hit.payload or {}
            metadata = payload.get("metadata", {}) or {}
            results.append(
                {
                    "content": payload.get("content", ""),
                    "metadata": metadata,
                    "score": float(hit.score or 0.0),
                    "retrieval_source": "vector",
                    "domain": metadata.get("domain", self.domain),
                    "source": metadata.get("source", ""),
                }
            )
        return results
