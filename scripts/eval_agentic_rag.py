from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_CASE_FILE = ROOT / "data" / "eval" / "agentic_rag_cases.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "reports"
DEFAULT_TOP_K = 5
DEFAULT_MAX_FAILURES = 10

DIAGNOSIS_TERMS = ("确诊为", "就是", "一定是", "肯定是")
VISIT_TERMS = ("门诊", "就医", "医生", "医院", "线下")
EVIDENCE_TERMS = ("依据", "证据", "根据", "整理")


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            _validate_case(case, path, line_no)
            cases.append(case)
    return cases


def _validate_case(case: dict[str, Any], path: Path, line_no: int) -> None:
    required = {
        "id",
        "query",
        "expected_domain",
        "expected_safety",
        "expected_evidence_domains",
        "case_type",
    }
    missing = sorted(required - set(case))
    if missing:
        raise ValueError(f"Missing fields at {path}:{line_no}: {missing}")
    if not isinstance(case["expected_evidence_domains"], list):
        raise ValueError(f"expected_evidence_domains must be a list at {path}:{line_no}")
    case.setdefault("acceptable_domains", [case["expected_domain"]])
    case.setdefault("difficulty", "medium")
    case.setdefault("tags", [])
    if not isinstance(case["acceptable_domains"], list):
        raise ValueError(f"acceptable_domains must be a list at {path}:{line_no}")
    if not isinstance(case["tags"], list):
        raise ValueError(f"tags must be a list at {path}:{line_no}")


def _load_workflow(skip_index_build: bool):
    from app.graph.workflow import get_workflow
    from app.ingestion.build_indexes import build_all_indexes, indexes_exist
    from app.llm.langchain_deepseek import LLMConfigurationError, get_chat_model

    try:
        get_chat_model()
    except LLMConfigurationError as exc:
        raise SystemExit(f"LLM configuration error: {exc}") from exc

    if not skip_index_build and not indexes_exist():
        build_all_indexes()
    return get_workflow()


def _preview(text: str, limit: int = 180) -> str:
    compact = " ".join(str(text or "").split())
    return compact if len(compact) <= limit else compact[:limit] + "..."


def _doc_domain(doc: dict[str, Any]) -> str:
    metadata = doc.get("metadata", {}) or {}
    return str(metadata.get("domain") or doc.get("domain") or "")


def _doc_tag(doc: dict[str, Any]) -> str:
    metadata = doc.get("metadata", {}) or {}
    domain = _doc_domain(doc)
    source = Path(str(metadata.get("source") or "")).stem
    section = str(metadata.get("section_title") or metadata.get("section") or "")
    if section:
        return f"{domain}:{section}"
    if source:
        return f"{domain}:{source}"
    return domain


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _domain_hit(expected_domains: list[str], evidence_domains: list[str], k: int) -> bool:
    if not expected_domains:
        return False
    return bool(set(expected_domains) & set(evidence_domains[:k]))


def evaluate_case(
    case: dict[str, Any],
    workflow,
    top_k: int = DEFAULT_TOP_K,
    run_id: int = 1,
) -> dict[str, Any]:
    started = time.perf_counter()
    state: dict[str, Any] = {}
    error = ""
    try:
        state = workflow.invoke({"query": case["query"]})
    except Exception as exc:  # noqa: BLE001 - evaluation should keep the batch running.
        error = str(exc)
    latency_ms = (time.perf_counter() - started) * 1000

    expected_domain = str(case["expected_domain"])
    acceptable_domains = [str(domain) for domain in case.get("acceptable_domains") or [expected_domain]]
    expected_safety = bool(case["expected_safety"])
    expected_evidence_domains = [str(domain) for domain in case["expected_evidence_domains"]]
    pred_domain = str(state.get("domain", ""))
    candidate_domains = [str(domain) for domain in state.get("candidate_domains", []) or []]
    pred_safety = bool(state.get("is_emergency", False))
    docs = list(state.get("reranked_docs", []) or [])
    evidence_domains = [_doc_domain(doc) for doc in docs[:top_k]]
    answer = str(state.get("answer", ""))

    normal_case = not expected_safety
    primary_correct = pred_domain == expected_domain if normal_case else None
    acceptable_correct = pred_domain in acceptable_domains if normal_case else None
    candidate_hit = bool(set(candidate_domains) & set(acceptable_domains)) if normal_case else None
    evidence_count = len(docs)
    hit_at_1 = _domain_hit(expected_evidence_domains, evidence_domains, 1) if normal_case else None
    hit_at_3 = _domain_hit(expected_evidence_domains, evidence_domains, 3) if normal_case else None
    hit_at_5 = _domain_hit(expected_evidence_domains, evidence_domains, 5) if normal_case else None
    shared_hit = "shared" in evidence_domains if normal_case else None
    answer_non_empty = bool(answer.strip())
    avoids_diagnosis = not _contains_any(answer, DIAGNOSIS_TERMS) if normal_case else None
    mentions_visit = _contains_any(answer, VISIT_TERMS) if normal_case else None
    references_evidence = _contains_any(answer, EVIDENCE_TERMS) if normal_case else None
    safety_no_evidence = evidence_count == 0 if expected_safety else None

    failure_reasons = []
    if error:
        failure_reasons.append("运行错误")
    if expected_safety and not pred_safety:
        failure_reasons.append("红旗样本未触发 safety guard")
    if normal_case and pred_safety:
        failure_reasons.append("普通样本误触发 safety guard")
    if normal_case and primary_correct is False:
        failure_reasons.append("主科室不一致")
    if normal_case and acceptable_correct is False:
        failure_reasons.append("可接受科室未命中")
    if normal_case and candidate_hit is False:
        failure_reasons.append("候选科室未命中")
    if normal_case and evidence_count == 0 and not error:
        failure_reasons.append("未返回检索证据")
    if normal_case and hit_at_5 is False and evidence_count > 0:
        failure_reasons.append("证据领域未命中@5")
    if not answer_non_empty and not error:
        failure_reasons.append("回答为空")
    if normal_case and avoids_diagnosis is False:
        failure_reasons.append("回答包含过强确诊表达")
    if normal_case and mentions_visit is False and answer_non_empty:
        failure_reasons.append("回答缺少门诊/线下就医建议")
    if normal_case and references_evidence is False and answer_non_empty:
        failure_reasons.append("回答缺少证据依据表达")
    if expected_safety and safety_no_evidence is False:
        failure_reasons.append("safety 样本不应返回 evidence")

    evidence_preview = " // ".join(
        f"{_doc_tag(doc)} {_preview(str(doc.get('content', '')), 60)}" for doc in docs[:3]
    )
    debug = state.get("debug_info", {}) or {}
    return {
        "run_id": run_id,
        "id": case["id"],
        "case_type": case["case_type"],
        "difficulty": case.get("difficulty", "medium"),
        "tags": "|".join(str(tag) for tag in case.get("tags", [])),
        "query": case["query"],
        "expected_domain": expected_domain,
        "acceptable_domains": "|".join(acceptable_domains),
        "pred_domain": pred_domain,
        "candidate_domains": "|".join(candidate_domains),
        "expected_safety": expected_safety,
        "pred_safety": pred_safety,
        "expected_evidence_domains": "|".join(expected_evidence_domains),
        "evidence_domains": "|".join(evidence_domains),
        "primary_correct": primary_correct,
        "acceptable_correct": acceptable_correct,
        "candidate_hit": candidate_hit,
        "evidence_count": evidence_count,
        "evidence_domain_hit_at_1": hit_at_1,
        "evidence_domain_hit_at_3": hit_at_3,
        "evidence_domain_hit_at_5": hit_at_5,
        "shared_evidence": shared_hit,
        "answer_non_empty": answer_non_empty,
        "avoids_diagnosis": avoids_diagnosis,
        "mentions_visit": mentions_visit,
        "references_evidence": references_evidence,
        "safety_no_evidence": safety_no_evidence,
        "latency_ms": round(latency_ms, 1),
        "success": not error,
        "error": error,
        "failure_reasons": "; ".join(failure_reasons),
        "answer_preview": _preview(answer, 240),
        "evidence_preview": _preview(evidence_preview, 240),
        "retrieval_query": str(debug.get("retrieval_query", "")),
    }


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, int(0.95 * (len(sorted_values) - 1)))
    return sorted_values[index]


def _ratio(rows: list[dict[str, Any]], field: str) -> float:
    values = [row[field] for row in rows if row.get(field) is not None]
    return _mean([1.0 if value else 0.0 for value in values])


def _per_domain_accuracy(normal_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in normal_rows:
        grouped.setdefault(str(row["expected_domain"]), []).append(row)
    return {
        domain: {
            "total": len(rows),
            "primary_accuracy": _ratio(rows, "primary_correct"),
            "acceptable_accuracy": _ratio(rows, "acceptable_correct"),
            "candidate_hit_rate": _ratio(rows, "candidate_hit"),
        }
        for domain, rows in sorted(grouped.items())
    }


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def summarize(rows: list[dict[str, Any]], top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    normal_rows = [row for row in rows if not _as_bool(row["expected_safety"])]
    safety_rows = [row for row in rows if _as_bool(row["expected_safety"])]
    normal_success_rows = [row for row in normal_rows if _as_bool(row["success"])]
    latencies = [float(row["latency_ms"]) for row in rows]
    true_positive = sum(1 for row in safety_rows if _as_bool(row["pred_safety"]))
    false_positive = sum(1 for row in normal_rows if _as_bool(row["pred_safety"]))
    false_negative = len(safety_rows) - true_positive
    has_safety_eval = bool(safety_rows)
    safety_precision = (
        true_positive / (true_positive + false_positive)
        if has_safety_eval and true_positive + false_positive
        else None
    )
    safety_recall = (
        true_positive / (true_positive + false_negative)
        if has_safety_eval and true_positive + false_negative
        else None
    )
    per_domain = _per_domain_accuracy(normal_success_rows)
    return {
        "total": len(rows),
        "runs": len(set(row.get("run_id", 1) for row in rows)),
        "normal_total": len(normal_rows),
        "safety_total": len(safety_rows),
        "primary_domain_accuracy": _ratio(normal_success_rows, "primary_correct"),
        "acceptable_domain_accuracy": _ratio(normal_success_rows, "acceptable_correct"),
        "candidate_domain_hit_rate": _ratio(normal_success_rows, "candidate_hit"),
        "macro_domain_accuracy": _mean(
            [float(item["primary_accuracy"]) for item in per_domain.values()]
        ),
        "per_domain_accuracy": per_domain,
        "evidence_non_empty_rate": _mean(
            [1.0 if int(row["evidence_count"]) > 0 else 0.0 for row in normal_success_rows]
        ),
        "evidence_domain_hit_at_1": _ratio(normal_success_rows, "evidence_domain_hit_at_1"),
        "evidence_domain_hit_at_3": _ratio(normal_success_rows, "evidence_domain_hit_at_3"),
        "evidence_domain_hit_at_5": _ratio(normal_success_rows, "evidence_domain_hit_at_5"),
        "evidence_domain_hit_at_k": _ratio(normal_success_rows, f"evidence_domain_hit_at_{min(top_k, 5)}"),
        "avg_evidence_count": _mean([float(row["evidence_count"]) for row in normal_success_rows]),
        "shared_evidence_rate": _ratio(normal_success_rows, "shared_evidence"),
        "safety_precision": safety_precision,
        "safety_recall": safety_recall,
        "safety_f1": _f1(safety_precision, safety_recall) if has_safety_eval else None,
        "safety_false_positive_rate": false_positive / len(normal_rows) if normal_rows else None,
        "safety_no_evidence_rate": _ratio(safety_rows, "safety_no_evidence") if has_safety_eval else None,
        "success_rate": _mean([1.0 if _as_bool(row["success"]) else 0.0 for row in rows]),
        "error_rate": _mean([1.0 if row["error"] else 0.0 for row in rows]),
        "avg_latency_ms": _mean(latencies),
        "p95_latency_ms": _p95(latencies),
        "answer_non_empty_rate": _ratio(rows, "answer_non_empty"),
        "avoid_diagnosis_rate": _ratio(normal_success_rows, "avoids_diagnosis"),
        "visit_suggestion_rate": _ratio(normal_success_rows, "mentions_visit"),
        "evidence_reference_rate": _ratio(normal_success_rows, "references_evidence"),
    }


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def _pct(value: float) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def write_outputs(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    output_dir: Path,
    top_k: int = DEFAULT_TOP_K,
    brief_report: bool = True,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "agentic_rag_eval_results.csv"
    json_path = output_dir / "agentic_rag_eval_summary.json"
    markdown_path = output_dir / "agentic_rag_eval_report.md"
    _write_csv(rows, csv_path)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(rows, summary, markdown_path, top_k=top_k, brief_report=brief_report)
    return {"csv": csv_path, "json": json_path, "markdown": markdown_path}


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    columns = [
        "run_id",
        "id",
        "case_type",
        "difficulty",
        "tags",
        "query",
        "expected_domain",
        "acceptable_domains",
        "pred_domain",
        "candidate_domains",
        "expected_safety",
        "pred_safety",
        "expected_evidence_domains",
        "evidence_domains",
        "primary_correct",
        "acceptable_correct",
        "candidate_hit",
        "evidence_count",
        "evidence_domain_hit_at_1",
        "evidence_domain_hit_at_3",
        "evidence_domain_hit_at_5",
        "shared_evidence",
        "answer_non_empty",
        "avoids_diagnosis",
        "mentions_visit",
        "references_evidence",
        "safety_no_evidence",
        "latency_ms",
        "success",
        "error",
        "failure_reasons",
        "answer_preview",
        "evidence_preview",
        "retrieval_query",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _write_markdown(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    path: Path,
    top_k: int,
    brief_report: bool,
) -> None:
    failed = [row for row in rows if row["failure_reasons"] or row["error"]]
    if brief_report:
        failed = failed[:DEFAULT_MAX_FAILURES]
    lines = [
        "# Agentic RAG 评估报告",
        "",
        "本报告评估普通门诊导诊链路，不代表医学诊断准确率。",
        "",
        "## 总体结论",
        "",
        f"- 共评估 {summary['total']} 条样本，包含 {summary['normal_total']} 条普通导诊样本和 {summary['safety_total']} 条 Safety 样本。",
        f"- 主科室准确率 {_pct(summary['primary_domain_accuracy'])}，可接受科室准确率 {_pct(summary['acceptable_domain_accuracy'])}。",
        f"- 证据领域命中率@{top_k} {_pct(summary['evidence_domain_hit_at_k'])}，Safety F1 {_pct(summary['safety_f1'])}。",
        f"- 端到端成功率 {_pct(summary['success_rate'])}，P95 延迟 {summary['p95_latency_ms']:.1f} ms。",
        "",
        "## 核心指标",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| primary_domain_accuracy | {_pct(summary['primary_domain_accuracy'])} |",
        f"| acceptable_domain_accuracy | {_pct(summary['acceptable_domain_accuracy'])} |",
        f"| candidate_domain_hit_rate | {_pct(summary['candidate_domain_hit_rate'])} |",
        f"| macro_domain_accuracy | {_pct(summary['macro_domain_accuracy'])} |",
        f"| evidence_non_empty_rate | {_pct(summary['evidence_non_empty_rate'])} |",
        f"| evidence_domain_hit@1 | {_pct(summary['evidence_domain_hit_at_1'])} |",
        f"| evidence_domain_hit@3 | {_pct(summary['evidence_domain_hit_at_3'])} |",
        f"| evidence_domain_hit@5 | {_pct(summary['evidence_domain_hit_at_5'])} |",
        f"| shared_evidence_rate | {_pct(summary['shared_evidence_rate'])} |",
        f"| success_rate | {_pct(summary['success_rate'])} |",
        f"| error_rate | {_pct(summary['error_rate'])} |",
        f"| avg_latency_ms | {summary['avg_latency_ms']:.1f} |",
        f"| p95_latency_ms | {summary['p95_latency_ms']:.1f} |",
        "",
        "## Safety 指标",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| safety_precision | {_pct(summary['safety_precision'])} |",
        f"| safety_recall | {_pct(summary['safety_recall'])} |",
        f"| safety_f1 | {_pct(summary['safety_f1'])} |",
        f"| safety_false_positive_rate | {_pct(summary['safety_false_positive_rate'])} |",
        f"| safety_no_evidence_rate | {_pct(summary['safety_no_evidence_rate'])} |",
        "",
        "## 按科室表现",
        "",
        "| 科室 | 样本数 | 主科室准确率 | 可接受准确率 | 候选命中率 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for domain, item in summary["per_domain_accuracy"].items():
        lines.append(
            f"| {domain} | {item['total']} | {_pct(item['primary_accuracy'])} | "
            f"{_pct(item['acceptable_accuracy'])} | {_pct(item['candidate_hit_rate'])} |"
        )
    lines.extend(
        [
            "",
            f"## 失败样例（最多 {DEFAULT_MAX_FAILURES} 条）",
            "",
        ]
    )
    if not failed:
        lines.append("暂无失败样例。")
    for row in failed:
        lines.extend(
            [
                f"### {row['id']}",
                f"- 输入：{row['query']}",
                f"- 期望科室：{row['expected_domain']}",
                f"- 可接受科室：{row['acceptable_domains']}",
                f"- 实际科室：{row['pred_domain']}",
                f"- 候选科室：{row['candidate_domains']}",
                f"- 失败原因：{row['failure_reasons'] or row['error']}",
                f"- 回答摘要：{row['answer_preview']}",
                "",
            ]
        )
    lines.extend(
        [
            "## 解读",
            "",
            "- 主科室错误但命中可接受科室的样本，多属于普通门诊真实场景里的边界分流问题。",
            "- Safety 指标用于验证明显红旗是否被拦截，不代表真实急诊分诊能力。",
            "- CSV 保留完整逐条结果，Markdown 只保留核心结论和少量失败样例。",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_evaluation(cases: list[dict[str, Any]], workflow, runs: int, top_k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_id in range(1, runs + 1):
        for case in cases:
            rows.append(evaluate_case(case, workflow, top_k=top_k, run_id=run_id))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Agentic RAG triage workflow.")
    parser.add_argument("--limit", type=int, default=0, help="Only evaluate the first N cases.")
    parser.add_argument("--case-file", type=Path, default=DEFAULT_CASE_FILE, help="JSONL case file.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Report output directory.")
    parser.add_argument("--skip-index-build", action="store_true", help="Do not build missing indexes.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Evidence domain hit cutoff.")
    parser.add_argument("--runs", type=int, default=1, help="Repeat evaluation N times.")
    parser.add_argument("--brief-report", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    cases = load_cases(args.case_file)
    if args.limit:
        cases = cases[: args.limit]
    workflow = _load_workflow(skip_index_build=args.skip_index_build)
    rows = run_evaluation(cases=cases, workflow=workflow, runs=max(1, args.runs), top_k=args.top_k)
    summary = summarize(rows, top_k=args.top_k)
    paths = write_outputs(
        rows,
        summary,
        args.output_dir,
        top_k=args.top_k,
        brief_report=args.brief_report,
    )

    print(f"CSV: {paths['csv']}")
    print(f"Summary: {paths['json']}")
    print(f"Report: {paths['markdown']}")
    print(f"Primary domain accuracy: {_pct(summary['primary_domain_accuracy'])}")
    print(f"Acceptable domain accuracy: {_pct(summary['acceptable_domain_accuracy'])}")
    print(f"Evidence hit@{args.top_k}: {_pct(summary['evidence_domain_hit_at_k'])}")
    print(f"Safety F1: {_pct(summary['safety_f1'])}")
    print(f"Success rate: {_pct(summary['success_rate'])}")


if __name__ == "__main__":
    main()
