from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.llm import TriagePlan


@pytest.fixture(autouse=True)
def clear_workflow_cache():
    from app.graph.workflow import get_workflow

    get_workflow.cache_clear()
    yield
    get_workflow.cache_clear()


def test_health() -> None:
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_chat_uses_llm_planner_and_retrieval_tool(monkeypatch) -> None:
    from app.graph import nodes
    from app.triage import planner

    def fake_plan(query: str) -> TriagePlan:
        return TriagePlan(
            domain="gastroenterology",
            candidate_domains=["gastroenterology"],
            retrieval_query="胃痛 反酸 饭后明显",
            reason="用户描述胃痛、反酸、饭后明显",
            confidence=0.9,
        )

    class FakeRetriever:
        def search(self, query, domains, top_k=None):
            assert query == "胃痛 反酸 饭后明显"
            assert domains == ["gastroenterology"]

            class Bundle:
                vector_hits = []
                bm25_hits = []
                fused_hits = []
                reranked_hits = [
                    {
                        "content": "胃痛和反酸可优先考虑消化内科门诊。",
                        "metadata": {
                            "domain": "gastroenterology",
                            "source": "test",
                            "section_title": "消化内科导诊",
                            "intent_type": "route",
                        },
                        "retrieval_source": "test",
                        "fused_score": 1.0,
                    }
                ]

            return Bundle()

    monkeypatch.setattr(planner, "create_triage_plan", fake_plan)
    monkeypatch.setattr(nodes, "create_triage_plan", fake_plan)
    monkeypatch.setattr(nodes, "get_hybrid_retriever", lambda: FakeRetriever())

    client = TestClient(app)
    resp = client.post("/chat", json={"query": "最近胃痛反酸，饭后明显"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["domain"] == "gastroenterology"
    assert data["route_method"] == "llm_planner"
    assert data["candidate_domains"] == ["gastroenterology"]
    assert data["debug_info"]["retrieval_query"] == "胃痛 反酸 饭后明显"
    assert data["debug_info"]["workflow_steps"] == [
        "safety_guard",
        "llm_planner",
        "retrieval_tool",
        "answer_composer",
    ]
    assert data["evidence"]


def test_safety_guard_response(monkeypatch) -> None:
    from app.graph import nodes

    def fail_if_called(*args, **kwargs):
        raise AssertionError("safety guard should not call planner or retrieval")

    monkeypatch.setattr(nodes, "create_triage_plan", fail_if_called)
    monkeypatch.setattr(nodes, "get_hybrid_retriever", fail_if_called)

    client = TestClient(app)
    resp = client.post("/chat", json={"query": "突然意识不清，还有大出血"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_emergency"] is True
    assert data["domain"] == "safety"
    assert data["candidate_domains"] == []
    assert data["route_method"] == "safety_guard"
    assert data["evidence"] == []


def test_missing_llm_key_returns_clear_error(monkeypatch) -> None:
    from app.graph import nodes
    from app.llm.langchain_deepseek import LLMConfigurationError

    def raise_config_error(query: str) -> TriagePlan:
        raise LLMConfigurationError("DEEPSEEK_API_KEY is required for ordinary triage.")

    monkeypatch.setattr(nodes, "create_triage_plan", raise_config_error)

    client = TestClient(app)
    resp = client.post("/chat", json={"query": "最近胃痛反酸"})
    assert resp.status_code == 503
    assert "DEEPSEEK_API_KEY" in resp.json()["detail"]
