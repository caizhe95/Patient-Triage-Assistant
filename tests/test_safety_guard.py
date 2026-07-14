from __future__ import annotations

import pytest

from app.safety.guard import check_safety


@pytest.mark.parametrize(
    "query",
    [
        "胸痛很剧烈，还冒冷汗",
        "严重过敏后喉咙发紧，喘不上气",
        "头部受伤后反复呕吐，越来越困",
        "喘不上气，嘴唇发紫",
        "大量呕血，站不稳",
        "突发剧烈头痛，和以前完全不一样",
        "发热伴精神很差，叫他反应慢",
        "车祸后胸口疼，呼吸也疼",
        "一侧脸歪，说话不清楚",
        "吞了药后出现严重过敏和喘息",
        "腹痛剧烈到站不直，还冒汗",
    ],
)
def test_safety_guard_catches_eval_red_flags(query: str) -> None:
    decision = check_safety(query)

    assert decision.triggered is True
    assert decision.reason
