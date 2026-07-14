from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.graph.workflow import get_workflow
from app.llm.langchain_deepseek import LLMConfigurationError
from app.schemas.response import ChatRequest, ChatResponse, DebugInfo, EvidenceItem


router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    workflow = get_workflow()
    try:
        state = workflow.invoke({"query": payload.query})
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    evidence = [
        EvidenceItem(
            content=str(doc.get("content", "")),
            metadata=dict(doc.get("metadata", {})),
            source=str(doc.get("metadata", {}).get("source", "")),
            domain=str(doc.get("metadata", {}).get("domain", "")),
            retrieval_source=str(doc.get("retrieval_source", "")),
            score=doc.get("fused_score", doc.get("score")),
        )
        for doc in state.get("reranked_docs", [])[:5]
    ]
    debug = state.get("debug_info", {}) or {}
    return ChatResponse(
        query=payload.query,
        domain=str(state.get("domain", "general")),
        candidate_domains=list(state.get("candidate_domains", [])),
        route_reason=str(state.get("route_reason", "")),
        route_method=str(state.get("route_method", "")),
        is_emergency=bool(state.get("is_emergency", False)),
        emergency_reason=str(state.get("emergency_reason", "")),
        answer=str(state.get("answer", "")),
        evidence=evidence,
        debug_info=DebugInfo(
            route_reason=str(state.get("route_reason", "")),
            route_method=str(state.get("route_method", "")),
            vector_hits=list(debug.get("vector_hits", [])),
            bm25_hits=list(debug.get("bm25_hits", [])),
            fused_hits=list(debug.get("fused_hits", [])),
            reranked_hits=list(debug.get("reranked_hits", [])),
            retrieval_query=debug.get("retrieval_query"),
            planner=dict(debug.get("planner", {})),
            workflow_steps=list(debug.get("workflow_steps", [])),
            selected_domains=list(debug.get("selected_domains", [])),
        ),
    )
