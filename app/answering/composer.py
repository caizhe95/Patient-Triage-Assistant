from __future__ import annotations

from typing import Any

from app.answering.evidence import short_evidence_text


def no_evidence_answer(domain: str, docs: list[dict[str, Any]]) -> str:
    if not docs:
        return "\n".join(
            [
                "1. 初步判断：当前证据不足，无法做出明确导诊判断。",
                "2. 建议挂号科室：建议先到综合门诊或全科门诊进行初步分流。",
                "3. 就诊准备：建议整理主要症状、出现时间、变化情况和既往检查结果。",
                "4. 风险提示：如果症状明显加重或自己判断情况紧急，请及时线下就医。",
                "5. 依据说明：当前未检索到可支持判断的相关证据。",
            ]
        )
    return "\n".join(
        [
            "1. 初步判断：根据已检索到的证据，当前信息更适合先做普通门诊分流，但不能替代医生判断。",
            f"2. 建议挂号科室：建议优先考虑 {domain} 相关门诊。",
            "3. 就诊准备：建议就诊时简要说明主要症状、出现时间和变化情况。",
            "4. 风险提示：如症状明显加重，请及时线下就医。",
            "5. 依据说明：以上内容仅依据已检索到的证据摘要整理，不包含证据之外的推断。",
        ]
    )


def compose_answer(
    domain: str,
    candidate_domains: list[str],
    docs: list[dict[str, Any]],
    evidence_by_intent: dict[str, dict[str, Any] | None],
) -> str:
    route_doc = evidence_by_intent.get("主推荐科室") or {}
    backup_doc = evidence_by_intent.get("备选科室") or {}
    prep_doc = evidence_by_intent.get("就诊准备") or {}
    risk_doc = evidence_by_intent.get("风险提示") or {}

    route_text = short_evidence_text(str(route_doc.get("content", "")), max_sentences=1, max_chars=90)
    backup_text = short_evidence_text(str(backup_doc.get("content", "")), max_sentences=1, max_chars=90)
    prep_text = short_evidence_text(str(prep_doc.get("content", "")), max_sentences=1, max_chars=110)
    risk_text = short_evidence_text(str(risk_doc.get("content", "")), max_sentences=1, max_chars=110)

    if not any([route_text, backup_text, prep_text, risk_text]):
        return no_evidence_answer(domain=domain, docs=docs)

    if route_text:
        line1 = f"1. 初步判断：证据显示当前线索更支持先按 {domain} 相关门诊分流。{route_text}"
    else:
        line1 = "1. 初步判断：当前证据不足，无法做出更明确判断。"

    if backup_text and len(candidate_domains) > 1:
        line2 = f"2. 建议挂号科室：优先考虑 {domain}；备选可参考 {candidate_domains[1]}。{backup_text}"
    else:
        line2 = f"2. 建议挂号科室：建议优先考虑 {domain} 相关门诊。"

    line3 = f"3. 就诊准备：{prep_text}" if prep_text else "3. 就诊准备：建议就诊时简要说明主要症状和变化情况。"
    line4 = f"4. 风险提示：{risk_text}" if risk_text else "4. 风险提示：如症状明显加重，请及时线下就医。"

    titles = []
    for doc in [route_doc, backup_doc, prep_doc, risk_doc]:
        title = str(doc.get("section_title", "") or doc.get("source", "") or "")
        if title and title not in titles:
            titles.append(title)
    line5 = f"5. 依据说明：以上内容仅依据{'、'.join(titles[:4]) or '当前证据摘要'}中的证据整理，不包含证据之外的推断。"
    return "\n".join([line1, line2, line3, line4, line5])
