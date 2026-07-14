from __future__ import annotations

import numpy as np
from langchain_core.documents import Document

from app.ingestion import build_indexes


class FakeClient:
    def __init__(self) -> None:
        self.upsert_sizes: list[int] = []

    def recreate_collection(self, **kwargs) -> None:
        self.collection_name = kwargs["collection_name"]

    def upsert(self, collection_name, points, wait=True):
        self.upsert_sizes.append(len(points))


class FakeSettings:
    qdrant_collection_prefix = "patient_triage"


def test_write_qdrant_collection_upserts_in_small_batches() -> None:
    client = FakeClient()
    docs = [
        Document(page_content=f"doc {index}", metadata={"domain": "general"})
        for index in range(55)
    ]
    vectors = np.ones((55, 4), dtype=np.float32)

    build_indexes._write_qdrant_collection(
        client=client,
        settings=FakeSettings(),
        domain="general",
        documents=docs,
        vectors=vectors,
        batch_size=20,
    )

    assert client.upsert_sizes == [20, 20, 15]
