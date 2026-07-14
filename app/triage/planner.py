from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import BaseMessage

from app.llm.langchain_deepseek import LLMConfigurationError, get_chat_model
from app.prompts.planner_prompt import TRIAGE_PLANNER_PROMPT
from app.schemas.llm import TriagePlan


def _message_content(message: Any) -> str:
    content = message.content if isinstance(message, BaseMessage) else message
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()
    return str(content).strip()


def _extract_json_object(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM planner must return valid JSON object.")
    return cleaned[start : end + 1]


def parse_triage_plan(message: Any) -> TriagePlan:
    text = _message_content(message)
    try:
        payload = json.loads(_extract_json_object(text))
    except json.JSONDecodeError as exc:
        raise ValueError("LLM planner must return valid JSON object.") from exc
    return TriagePlan.model_validate(payload)


def create_triage_plan(query: str) -> TriagePlan:
    chat = get_chat_model()
    try:
        plan = parse_triage_plan((TRIAGE_PLANNER_PROMPT | chat).invoke({"query": query}))
    except ValueError as exc:
        raise LLMConfigurationError(f"LLM planner returned invalid JSON: {exc}") from exc

    candidate_domains = plan.candidate_domains or [plan.domain]
    if plan.domain not in candidate_domains:
        candidate_domains.insert(0, plan.domain)
    return plan.model_copy(update={"candidate_domains": candidate_domains[:2]})
