from __future__ import annotations

from typing import Any

from app.answering.composer import compose_answer
from app.answering.evidence import build_evidence_summary, format_evidence
from app.graph.state import TriageState
from app.retrieval.hybrid_retriever import get_hybrid_retriever
from app.safety.guard import build_safety_answer, check_safety
from app.triage.planner import create_triage_plan


def _debug_info(state: TriageState) -> dict[str, Any]:
    return dict(state.get("debug_info") or {})


def _workflow_steps(debug: dict[str, Any], step: str) -> list[str]:
    steps = debug.get("workflow_steps") or []
    return [*list(steps), step]


def safety_guard_node(state: TriageState) -> TriageState:
    query = state["query"]
    decision = check_safety(query)
    debug = _debug_info(state)
    debug["workflow_steps"] = ["safety_guard"]
    debug["safety_guard_checked"] = True
    debug["safety_guard_triggered"] = decision.triggered

    if not decision.triggered:
        return {"is_emergency": False, "emergency_reason": "", "debug_info": debug}

    return {
        "is_emergency": True,
        "emergency_reason": decision.reason,
        "domain": "safety",
        "candidate_domains": [],
        "route_reason": f"安全提示触发：{decision.reason}",
        "route_method": "safety_guard",
        "retrieval_query": query,
        "retrieved_docs": [],
        "reranked_docs": [],
        "evidence_by_intent": {},
        "evidence_summary": "",
        "answer": build_safety_answer(decision.reason),
        "debug_info": debug,
    }


def llm_planner_node(state: TriageState) -> TriageState:
    query = state["query"]
    plan = create_triage_plan(query)
    debug = _debug_info(state)
    debug["workflow_steps"] = _workflow_steps(debug, "llm_planner")
    debug.update(
        {
            "planner": plan.model_dump(),
            "retrieval_query": plan.retrieval_query,
            "selected_domains": plan.candidate_domains,
            "route_method": "llm_planner",
            "route_reason": plan.reason,
        }
    )
    return {
        "query": query,
        "retrieval_query": plan.retrieval_query,
        "domain": plan.domain,
        "candidate_domains": plan.candidate_domains,
        "route_method": "llm_planner",
        "route_reason": plan.reason,
        "debug_info": debug,
    }


def retrieve_node(state: TriageState) -> TriageState:
    query = state.get("retrieval_query") or state["query"]
    domains = state.get("candidate_domains") or [state.get("domain", "general")]
    bundle = get_hybrid_retriever().search(query=query, domains=domains)
    debug = _debug_info(state)
    debug["workflow_steps"] = _workflow_steps(debug, "retrieval_tool")
    debug["retrieval_query"] = query
    debug["candidate_domains"] = list(state.get("candidate_domains", []))
    debug["vector_hits"] = [format_evidence(doc) for doc in bundle.vector_hits]
    debug["bm25_hits"] = [format_evidence(doc) for doc in bundle.bm25_hits]
    debug["fused_hits"] = [format_evidence(doc) for doc in bundle.fused_hits]
    debug["reranked_hits"] = [format_evidence(doc) for doc in bundle.reranked_hits]

    evidence_items = [format_evidence(doc) for doc in bundle.reranked_hits]
    evidence_by_intent, evidence_summary = build_evidence_summary(
        evidence_items,
        state.get("domain", "general"),
        state.get("candidate_domains", []),
    )
    debug["evidence_summary"] = evidence_summary
    return {
        "retrieved_docs": bundle.fused_hits,
        "reranked_docs": bundle.reranked_hits,
        "evidence_by_intent": evidence_by_intent,
        "evidence_summary": evidence_summary,
        "debug_info": debug,
    }


def generate_answer_node(state: TriageState) -> TriageState:
    evidence_by_intent = state.get("evidence_by_intent") or {}
    answer = compose_answer(
        domain=state.get("domain", "general"),
        candidate_domains=state.get("candidate_domains", []),
        docs=state.get("reranked_docs", []),
        evidence_by_intent=evidence_by_intent,
    )
    debug = _debug_info(state)
    debug["workflow_steps"] = _workflow_steps(debug, "answer_composer")
    return {"answer": answer, "debug_info": debug}
