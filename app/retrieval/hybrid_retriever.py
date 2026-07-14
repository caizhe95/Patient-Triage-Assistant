from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from app.config import get_settings
from app.ingestion.knowledge_layout import SHARED_DOMAIN
from app.retrieval.bm25_store import DomainBM25Store
from app.retrieval.fusion import rrf_fuse
from app.retrieval.reranker import BGEReranker, DefaultReranker
from app.retrieval.vector_store import DomainVectorStore
from app.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class RetrievalBundle:
    vector_hits: list[dict[str, Any]] = field(default_factory=list)
    bm25_hits: list[dict[str, Any]] = field(default_factory=list)
    fused_hits: list[dict[str, Any]] = field(default_factory=list)
    reranked_hits: list[dict[str, Any]] = field(default_factory=list)


def _build_reranker():
    settings = get_settings()
    if settings.enable_bge_reranker:
        reranker = BGEReranker()
        if reranker.available():
            return reranker
    return DefaultReranker()


class HybridRetriever:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.reranker = _build_reranker()
        self._vector_stores: dict[str, DomainVectorStore] = {}
        self._bm25_stores: dict[str, DomainBM25Store] = {}

    def _vector_store(self, domain: str) -> DomainVectorStore:
        if domain not in self._vector_stores:
            self._vector_stores[domain] = DomainVectorStore(
                domain=domain,
                index_dir=self.settings.qdrant_storage_dir,
            )
        return self._vector_stores[domain]

    def _bm25_store(self, domain: str) -> DomainBM25Store:
        if domain not in self._bm25_stores:
            self._bm25_stores[domain] = DomainBM25Store(
                domain=domain,
                index_dir=self.settings.bm25_index_dir,
            )
        return self._bm25_stores[domain]

    def search(self, query: str, domains: list[str], top_k: int | None = None) -> RetrievalBundle:
        top_k = top_k or self.settings.retrieval_top_k
        normalized_domains = [domain for domain in dict.fromkeys(domains or ["general"]) if domain]
        if SHARED_DOMAIN not in normalized_domains:
            normalized_domains.append(SHARED_DOMAIN)

        vector_hits: list[dict[str, Any]] = []
        bm25_hits: list[dict[str, Any]] = []
        futures = {}
        with ThreadPoolExecutor(max_workers=max(2, len(normalized_domains) * 2)) as executor:
            for domain in normalized_domains:
                futures[executor.submit(self._vector_store(domain).search, query, top_k)] = ("vector", domain)
                futures[executor.submit(self._bm25_store(domain).search, query, top_k)] = ("bm25", domain)
            for future in as_completed(futures):
                source, domain = futures[future]
                try:
                    hits = future.result()
                except Exception as exc:
                    logger.warning("%s search failed for %s: %s", source, domain, exc)
                    continue
                if source == "vector":
                    vector_hits.extend(hits)
                else:
                    bm25_hits.extend(hits)

        fuse_limit = top_k * max(2, len(normalized_domains))
        fused_hits = rrf_fuse(
            {"vector": vector_hits, "bm25": bm25_hits},
            k=self.settings.rrf_k,
            limit=fuse_limit,
        )
        rerank_top_k = self.settings.rerank_top_k or top_k
        reranked_hits = self.reranker.rerank(query=query, docs=fused_hits, top_k=rerank_top_k)
        return RetrievalBundle(
            vector_hits=vector_hits[:fuse_limit],
            bm25_hits=bm25_hits[:fuse_limit],
            fused_hits=fused_hits,
            reranked_hits=reranked_hits,
        )


@lru_cache(maxsize=1)
def get_hybrid_retriever() -> HybridRetriever:
    return HybridRetriever()
