  
from __future__ import annotations  
 
from typing import Any, TypedDict  
 
 
class TriageState(TypedDict, total=False):  
    query: str  
    retrieval_query: str
    is_emergency: bool  
    emergency_reason: str  
    domain: str  
    candidate_domains: list[str]  
    route_reason: str  
    route_method: str  
    retrieved_docs: list[dict[str, Any]]  
    reranked_docs: list[dict[str, Any]]  
    evidence_by_intent: dict[str, dict[str, Any] | None]
    evidence_summary: str  
    answer: str  
    debug_info: dict[str, Any]  
