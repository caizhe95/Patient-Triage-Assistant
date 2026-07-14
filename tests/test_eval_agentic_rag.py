from __future__ import annotations

import json
from pathlib import Path

from scripts import eval_agentic_rag as eval_rag


class FakeWorkflow:
    def invoke(self, payload):
        query = payload["query"]
        if "error" in query:
            raise RuntimeError("workflow failed")
        if "red flag" in query:
            return {
                "domain": "safety",
                "candidate_domains": [],
                "is_emergency": True,
                "answer": "请立即线下就医或前往急诊。",
                "reranked_docs": [],
                "debug_info": {"workflow_steps": ["safety_guard"]},
            }
        if "boundary" in query:
            return {
                "domain": "general",
                "candidate_domains": ["general", "neurology"],
                "is_emergency": False,
                "answer": "建议先到普通门诊就医，由医生结合症状进一步分流。依据说明：根据证据整理。",
                "reranked_docs": [
                    {
                        "content": "头晕症状可先由普通门诊分流，也可参考神经内科。",
                        "metadata": {"domain": "neurology", "source": "test.md"},
                        "retrieval_source": "vector",
                        "fused_score": 1.0,
                    }
                ],
                "debug_info": {"retrieval_query": "头晕 体位变化"},
            }
        return {
            "domain": "gastroenterology",
            "candidate_domains": ["gastroenterology"],
            "is_emergency": False,
            "answer": "建议优先消化内科门诊就医。依据说明：根据检索证据整理，不做确诊。",
            "reranked_docs": [
                {
                    "content": "胃痛反酸可优先考虑消化内科门诊。",
                    "metadata": {"domain": "gastroenterology", "source": "test.md"},
                    "retrieval_source": "bm25",
                    "fused_score": 1.0,
                },
                {
                    "content": "就诊前记录症状和持续时间。",
                    "metadata": {"domain": "shared", "source": "shared.md"},
                    "retrieval_source": "vector",
                    "fused_score": 0.5,
                },
            ],
            "debug_info": {"retrieval_query": "胃痛 反酸"},
        }


def _case(
    case_id: str,
    query: str,
    expected_domain: str,
    expected_safety: bool,
    expected_evidence_domains: list[str],
    acceptable_domains: list[str] | None = None,
    case_type: str = "normal",
) -> dict:
    return {
        "id": case_id,
        "query": query,
        "expected_domain": expected_domain,
        "acceptable_domains": acceptable_domains or [expected_domain],
        "expected_safety": expected_safety,
        "expected_evidence_domains": expected_evidence_domains,
        "case_type": case_type,
        "difficulty": "medium",
        "tags": ["unit"],
    }


def test_load_cases_from_jsonl_with_enhanced_fields(tmp_path: Path) -> None:
    case_file = tmp_path / "cases.jsonl"
    case_file.write_text(
        json.dumps(
            _case(
                "case_001",
                "胃痛反酸",
                "gastroenterology",
                False,
                ["gastroenterology"],
                acceptable_domains=["gastroenterology", "general"],
            ),
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    cases = eval_rag.load_cases(case_file)

    assert len(cases) == 1
    assert cases[0]["acceptable_domains"] == ["gastroenterology", "general"]
    assert cases[0]["difficulty"] == "medium"
    assert cases[0]["tags"] == ["unit"]


def test_evaluate_cases_and_enhanced_metrics() -> None:
    cases = [
        _case("normal_001", "胃痛反酸", "gastroenterology", False, ["gastroenterology"]),
        _case(
            "boundary_001",
            "boundary dizziness",
            "neurology",
            False,
            ["neurology"],
            acceptable_domains=["neurology", "general"],
        ),
        _case("safety_001", "red flag", "safety", True, [], case_type="safety"),
        _case("error_001", "error sample", "general", False, ["general"]),
    ]

    rows = [eval_rag.evaluate_case(case, FakeWorkflow(), top_k=3, run_id=1) for case in cases]
    summary = eval_rag.summarize(rows, top_k=3)

    assert summary["total"] == 4
    assert summary["primary_domain_accuracy"] == 0.5
    assert summary["acceptable_domain_accuracy"] == 1.0
    assert summary["candidate_domain_hit_rate"] == 1.0
    assert summary["evidence_domain_hit_at_1"] == 1.0
    assert summary["evidence_domain_hit_at_3"] == 1.0
    assert summary["shared_evidence_rate"] == 0.5
    assert summary["safety_precision"] == 1.0
    assert summary["safety_recall"] == 1.0
    assert summary["safety_f1"] == 1.0
    assert summary["success_rate"] == 0.75
    assert summary["per_domain_accuracy"]["gastroenterology"]["primary_accuracy"] == 1.0
    assert summary["per_domain_accuracy"]["neurology"]["acceptable_accuracy"] == 1.0
    assert rows[3]["error"] == "workflow failed"


def test_write_brief_report_limits_failure_details(tmp_path: Path) -> None:
    cases = [
        _case(f"case_{index:03d}", "error sample", "general", False, ["general"])
        for index in range(12)
    ]
    rows = [eval_rag.evaluate_case(case, FakeWorkflow(), top_k=3, run_id=1) for case in cases]
    summary = eval_rag.summarize(rows, top_k=3)

    paths = eval_rag.write_outputs(rows, summary, tmp_path, top_k=3, brief_report=True)
    markdown = paths["markdown"].read_text(encoding="utf-8")

    assert "Agentic RAG 评估报告" in markdown
    assert "不代表医学诊断准确率" in markdown
    assert "## 核心指标" in markdown
    assert "## 按科室表现" in markdown
    assert markdown.count("### case_") == 10
    assert "case_009" in markdown
    assert "case_010" not in markdown


def test_safety_metrics_are_none_without_safety_samples() -> None:
    rows = [
        eval_rag.evaluate_case(
            _case("normal_001", "胃痛反酸", "gastroenterology", False, ["gastroenterology"]),
            FakeWorkflow(),
            top_k=3,
            run_id=1,
        )
    ]

    summary = eval_rag.summarize(rows, top_k=3)

    assert summary["safety_precision"] is None
    assert summary["safety_recall"] is None
    assert summary["safety_f1"] is None
    assert summary["safety_no_evidence_rate"] is None
