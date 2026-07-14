from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPORT_DIR = ROOT / "reports"


SAMPLES = [
    {"query": "最近胃痛反酸，饭后明显", "expected_domain": "gastroenterology", "expected_emergency": False},
    {"query": "皮疹瘙痒，越抓越痒", "expected_domain": "dermatology", "expected_emergency": False},
    {"query": "腰背痛，久坐后明显", "expected_domain": "orthopedics", "expected_emergency": False},
    {"query": "孩子发热咳嗽两天", "expected_domain": "pediatrics", "expected_emergency": False},
    {"query": "不知道该挂什么科，症状有点杂", "expected_domain": "general", "expected_emergency": False},
    {"query": "突然意识不清，还有大出血", "expected_domain": "safety", "expected_emergency": True},
    {"query": "呼吸困难明显，说话费力", "expected_domain": "safety", "expected_emergency": True},
]


def _preview(text: str, limit: int = 180) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit] + "..."


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _doc_tag(doc: dict[str, Any]) -> str:
    metadata = doc.get("metadata", {}) or {}
    domain = metadata.get("domain") or doc.get("domain") or ""
    source = metadata.get("source") or ""
    section = metadata.get("section_title") or metadata.get("section") or ""
    if section:
        return f"{domain}:{section}"
    if source:
        return f"{domain}:{Path(str(source)).stem}"
    return str(domain)


def _load_workflow(skip_index_build: bool):
    from app.graph.workflow import get_workflow
    from app.ingestion.build_indexes import build_all_indexes, indexes_exist

    if not skip_index_build and not indexes_exist():
        build_all_indexes()
    return get_workflow()


def evaluate_sample(index: int, sample: dict[str, Any], workflow) -> dict[str, Any]:
    started = time.perf_counter()
    state: dict[str, Any] = {}
    error = ""
    try:
        state = workflow.invoke({"query": sample["query"]})
    except Exception as exc:
        error = str(exc)
    latency_ms = (time.perf_counter() - started) * 1000

    pred_domain = str(state.get("domain", ""))
    pred_emergency = bool(state.get("is_emergency", False))
    docs = list(state.get("reranked_docs", []) or [])
    expected_domain = str(sample["expected_domain"])
    expected_emergency = bool(sample["expected_emergency"])

    route_correct = pred_domain == expected_domain if not expected_emergency else pred_emergency
    emergency_correct = pred_emergency == expected_emergency
    evidence_count = len(docs)
    evidence_preview = " // ".join(f"{_doc_tag(doc)} {_preview(str(doc.get('content', '')), 60)}" for doc in docs[:3])

    failure_reasons = []
    if error:
        failure_reasons.append("运行错误")
    if not emergency_correct:
        failure_reasons.append("安全判断不一致")
    if not route_correct:
        failure_reasons.append("科室结果不一致")
    if not expected_emergency and evidence_count == 0 and not error:
        failure_reasons.append("无检索证据")

    return {
        "sample_id": f"S{index:03d}",
        "query": sample["query"],
        "expected_domain": expected_domain,
        "pred_domain": pred_domain,
        "expected_emergency": expected_emergency,
        "pred_emergency": pred_emergency,
        "route_correct": route_correct,
        "emergency_correct": emergency_correct,
        "evidence_count": evidence_count,
        "latency_ms": round(latency_ms, 1),
        "failure_reasons": "; ".join(failure_reasons),
        "answer_preview": _preview(str(state.get("answer", "")), 240),
        "evidence_preview": _preview(evidence_preview, 240),
        "error": error,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    normal_rows = [row for row in rows if not row["expected_emergency"]]
    safety_rows = [row for row in rows if row["expected_emergency"]]
    return {
        "total": float(len(rows)),
        "route_accuracy": _mean([1.0 if row["route_correct"] else 0.0 for row in normal_rows]),
        "safety_accuracy": _mean([1.0 if row["emergency_correct"] else 0.0 for row in safety_rows]),
        "evidence_rate": _mean([1.0 if row["evidence_count"] > 0 else 0.0 for row in normal_rows]),
        "avg_latency_ms": _mean([float(row["latency_ms"]) for row in rows]),
        "error_rate": _mean([1.0 if row["error"] else 0.0 for row in rows]),
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "sample_id",
        "query",
        "expected_domain",
        "pred_domain",
        "expected_emergency",
        "pred_emergency",
        "route_correct",
        "emergency_correct",
        "evidence_count",
        "latency_ms",
        "failure_reasons",
        "answer_preview",
        "evidence_preview",
        "error",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_report(rows: list[dict[str, Any]], summary: dict[str, float], path: Path) -> None:
    failed = [row for row in rows if row["failure_reasons"] or row["error"]]
    lines = [
        "# Agentic RAG 评估报告",
        "",
        f"- 样本数：{int(summary['total'])}",
        f"- 普通导诊路由准确率：{_pct(summary['route_accuracy'])}",
        f"- Safety guard 准确率：{_pct(summary['safety_accuracy'])}",
        f"- 普通导诊证据返回率：{_pct(summary['evidence_rate'])}",
        f"- 平均延迟：{summary['avg_latency_ms']:.1f} ms",
        f"- 错误率：{_pct(summary['error_rate'])}",
        "",
        "## 失败样例",
        "",
    ]
    if not failed:
        lines.append("暂无失败样例。")
    for row in failed:
        lines.extend(
            [
                f"### {row['sample_id']}",
                f"- 输入：{row['query']}",
                f"- 期望：{row['expected_domain']}",
                f"- 实际：{row['pred_domain']}",
                f"- 原因：{row['failure_reasons'] or row['error']}",
                f"- 回答摘要：{row['answer_preview']}",
                f"- 证据摘要：{row['evidence_preview']}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Agentic RAG triage workflow.")
    parser.add_argument("--limit", type=int, default=0, help="Only evaluate the first N samples.")
    parser.add_argument("--skip-index-build", action="store_true", help="Do not build missing indexes before evaluation.")
    args = parser.parse_args()

    workflow = _load_workflow(skip_index_build=args.skip_index_build)
    samples = SAMPLES[: args.limit] if args.limit else SAMPLES
    rows = [evaluate_sample(index, sample, workflow) for index, sample in enumerate(samples, start=1)]
    summary = summarize(rows)

    csv_path = REPORT_DIR / "rag_eval_results.csv"
    report_path = REPORT_DIR / "rag_eval_report.md"
    write_csv(rows, csv_path)
    write_report(rows, summary, report_path)

    print(f"CSV: {csv_path}")
    print(f"Report: {report_path}")
    print(f"Route accuracy: {_pct(summary['route_accuracy'])}")
    print(f"Safety accuracy: {_pct(summary['safety_accuracy'])}")
    print(f"Evidence rate: {_pct(summary['evidence_rate'])}")


if __name__ == "__main__":
    main()
