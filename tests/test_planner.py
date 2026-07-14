from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from app.schemas.llm import TriagePlan
from app.triage import planner


def test_parse_plan_from_json_message() -> None:
    message = AIMessage(
        content='{"domain":"gastroenterology","candidate_domains":["gastroenterology"],'
        '"retrieval_query":"胃痛 反酸 饭后明显","reason":"胃部症状为主","confidence":0.9}'
    )

    plan = planner.parse_triage_plan(message)

    assert plan == TriagePlan(
        domain="gastroenterology",
        candidate_domains=["gastroenterology"],
        retrieval_query="胃痛 反酸 饭后明显",
        reason="胃部症状为主",
        confidence=0.9,
    )


def test_create_triage_plan_uses_plain_json_chain(monkeypatch) -> None:
    def fake_chat(prompt_value):
        assert "胃痛反酸" in prompt_value.to_string()
        return AIMessage(
            content='{"domain":"gastroenterology","candidate_domains":[],'
            '"retrieval_query":"胃痛 反酸","reason":"消化道症状","confidence":0.8}'
        )

    monkeypatch.setattr(planner, "get_chat_model", lambda: fake_chat)

    plan = planner.create_triage_plan("胃痛反酸")

    assert plan.domain == "gastroenterology"
    assert plan.candidate_domains == ["gastroenterology"]
    assert plan.retrieval_query == "胃痛 反酸"


def test_parse_plan_rejects_invalid_json() -> None:
    with pytest.raises(ValueError, match="valid JSON"):
        planner.parse_triage_plan(AIMessage(content="not json"))
