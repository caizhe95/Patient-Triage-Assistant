from __future__ import annotations

import re
from typing import Any, Callable


def format_evidence(doc: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(doc.get("metadata", {}) or {})
    return {
        "content": doc.get("content", ""),
        "metadata": metadata,
        "source": metadata.get("source", ""),
        "domain": metadata.get("domain", ""),
        "retrieval_source": doc.get("retrieval_source", ""),
        "score": doc.get("fused_score", doc.get("score")),
        "intent_type": metadata.get("intent_type", ""),
        "parent_id": metadata.get("parent_id", ""),
        "section_title": metadata.get("section_title", metadata.get("section", "")),
    }


def doc_intent(item: dict[str, Any]) -> str:
    intent = str(item.get("intent_type", "") or "")
    if intent:
        return intent
    content = str(item.get("content", "") or "")
    if any(key in content for key in ["急诊", "高危", "加重", "呼吸困难", "晕厥"]):
        return "red_flag"
    if any(key in content for key in ["准备", "病史", "用药", "检查结果", "资料"]):
        return "prep"
    if any(key in content for key in ["复诊", "观察", "记录", "时间线"]):
        return "followup"
    return "route"


def pick_first(items: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> dict[str, Any] | None:
    for item in items:
        if predicate(item):
            return item
    return None


def short_evidence_text(text: str, max_sentences: int = 2, max_chars: int = 150) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if not cleaned:
        return ""
    parts = [part.strip() for part in re.split(r"[。！？；]", cleaned) if part.strip()]
    summary = "。".join(parts[:max_sentences]).strip("。") if parts else cleaned
    if summary:
        summary += "。"
    return summary[:max_chars].strip()


def build_evidence_buckets(
    evidence_items: list[dict[str, Any]],
    primary_domain: str,
    candidate_domains: list[str],
) -> dict[str, dict[str, Any] | None]:
    backup_domains = [domain for domain in candidate_domains if domain and domain != primary_domain]
    route_doc = pick_first(
        evidence_items,
        lambda item: item.get("domain") == primary_domain and doc_intent(item) in {"route", "general"},
    )
    backup_doc = pick_first(
        evidence_items,
        lambda item: item.get("domain") in backup_domains and doc_intent(item) in {"route", "general"},
    )
    prep_doc = pick_first(evidence_items, lambda item: doc_intent(item) in {"prep", "followup", "symptom_log"})
    risk_doc = pick_first(evidence_items, lambda item: doc_intent(item) == "red_flag")
    return {"主推荐科室": route_doc, "备选科室": backup_doc, "就诊准备": prep_doc, "风险提示": risk_doc}


def build_evidence_summary(
    evidence_items: list[dict[str, Any]],
    primary_domain: str,
    candidate_domains: list[str],
) -> tuple[dict[str, dict[str, Any] | None], str]:
    buckets = build_evidence_buckets(evidence_items, primary_domain, candidate_domains)
    labels = [
        ("证据1（主推荐科室）", "主推荐科室"),
        ("证据2（备选科室）", "备选科室"),
        ("证据3（就诊准备）", "就诊准备"),
        ("证据4（风险提示）", "风险提示"),
    ]
    lines = []
    seen = set()
    for label, key in labels:
        item = buckets.get(key)
        if not item:
            continue
        content = short_evidence_text(str(item.get("content", "")))
        title = str(item.get("section_title", "") or "")
        if not content or content in seen:
            continue
        seen.add(content)
        lines.append(f"{label}（{title}）：{content}" if title else f"{label}：{content}")
    return buckets, "\n".join(lines)

