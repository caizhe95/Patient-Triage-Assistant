from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., description="患者输入的自然语言症状描述")


class EvidenceItem(BaseModel):
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: str = ""
    domain: str = ""
    retrieval_source: str = ""
    score: float | None = None


class DebugInfo(BaseModel):
    route_reason: str | None = None
    route_method: str | None = None
    vector_hits: list[dict[str, Any]] = Field(default_factory=list)
    bm25_hits: list[dict[str, Any]] = Field(default_factory=list)
    fused_hits: list[dict[str, Any]] = Field(default_factory=list)
    reranked_hits: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_query: str | None = None
    planner: dict[str, Any] = Field(default_factory=dict)
    workflow_steps: list[str] = Field(default_factory=list)
    selected_domains: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    query: str
    domain: str
    candidate_domains: list[str] = Field(default_factory=list)
    route_reason: str = ""
    route_method: str = ""
    is_emergency: bool = False
    emergency_reason: str = ""
    answer: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    debug_info: DebugInfo = Field(default_factory=DebugInfo)
