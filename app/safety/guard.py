from __future__ import annotations

import re
from dataclasses import dataclass
from collections.abc import Iterable


@dataclass(frozen=True)
class SafetyDecision:
    triggered: bool
    reason: str = ""


SINGLE_RED_FLAGS: tuple[tuple[str, str], ...] = (
    ("意识不清", "描述包含意识不清，可能存在高风险情况"),
    ("叫不醒", "描述包含意识不清或反应差，建议立即线下处理"),
    ("大出血", "描述包含大出血，可能需要立即线下处理"),
    ("出血止不住", "描述包含出血不止，可能需要立即线下处理"),
    ("呼吸困难", "描述包含明显呼吸困难，不适合继续线上导诊"),
    ("说话费力", "描述包含呼吸或说话明显受限，不适合继续线上导诊"),
    ("晕厥", "描述包含晕厥，不适合继续线上导诊"),
    ("抽搐", "描述包含抽搐，不适合继续线上导诊"),
    ("高热抽搐", "儿童高热伴抽搐属于高风险信号"),
)


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def check_safety(query: str) -> SafetyDecision:
    compact = re.sub(r"\s+", "", str(query or ""))

    if not compact:
        return SafetyDecision(False, "")

    if "黑便" in compact and _contains_any(compact, ("头晕", "晕")):
        return SafetyDecision(True, "黑便伴头晕，存在消化道出血风险")

    if re.search(r"(一侧|单侧).*(无力|麻木|脸歪|口角歪|说话不清)", compact):
        return SafetyDecision(True, "单侧肢体或面部异常伴语言异常，建议立即线下或急诊处理")

    if "胸痛" in compact and _contains_any(compact, ("冷汗", "出汗", "大汗", "剧烈", "压榨", "放射", "呼吸困难")):
        return SafetyDecision(True, "胸痛伴随高风险表现，建议立即线下或急诊处理")

    if _contains_any(compact, ("喘不上气", "呼吸困难")) and _contains_any(compact, ("嘴唇发紫", "口唇发紫", "发绀")):
        return SafetyDecision(True, "明显呼吸困难伴口唇发紫，建议立即线下或急诊处理")

    if _contains_any(compact, ("过敏", "喉咙发紧", "喉头水肿")) and _contains_any(
        compact, ("喘不上气", "喘息", "呼吸困难", "喉咙发紧")
    ):
        return SafetyDecision(True, "过敏伴呼吸或喉咙发紧表现，建议立即线下或急诊处理")

    if _contains_any(compact, ("头部受伤", "头外伤", "撞到头")) and _contains_any(
        compact, ("反复呕吐", "越来越困", "嗜睡", "意识不清")
    ):
        return SafetyDecision(True, "头部受伤后伴反复呕吐或嗜睡，建议立即线下或急诊处理")

    if _contains_any(compact, ("呕血", "吐血")) and _contains_any(compact, ("大量", "站不稳", "头晕", "冒冷汗")):
        return SafetyDecision(True, "呕血伴不稳或出血量较大，建议立即线下或急诊处理")

    if _contains_any(compact, ("突发", "突然")) and "剧烈头痛" in compact:
        return SafetyDecision(True, "突发剧烈头痛属于高风险信号，建议立即线下或急诊处理")

    if "发热" in compact and _contains_any(compact, ("精神很差", "反应慢", "嗜睡", "叫不醒")):
        return SafetyDecision(True, "发热伴精神反应异常，建议立即线下或急诊处理")

    if _contains_any(compact, ("车祸", "外伤", "撞伤")) and _contains_any(compact, ("胸口疼", "胸痛")) and "呼吸" in compact:
        return SafetyDecision(True, "胸部外伤后伴呼吸相关疼痛，建议立即线下或急诊处理")

    if _contains_any(compact, ("腹痛剧烈", "剧烈腹痛")) and _contains_any(compact, ("站不直", "冒汗", "冷汗", "大汗")):
        return SafetyDecision(True, "剧烈腹痛伴明显不适表现，建议立即线下或急诊处理")

    for pattern, reason in SINGLE_RED_FLAGS:
        if pattern in compact:
            return SafetyDecision(True, reason)

    return SafetyDecision(False, "")


def build_safety_answer(reason: str) -> str:
    return "\n".join(
        [
            "1. 初步判断：你的描述包含明显高风险信号，不适合继续通过线上导诊判断挂号科室。",
            "2. 建议处理：建议立即前往急诊、线下就医，或在情况紧急时呼叫当地急救电话。",
            "3. 就诊准备：如条件允许，简要记录主要症状、出现时间、既往病史和正在使用的药物。",
            f"4. 风险提示：{reason or '当前输入触发安全提示。'}",
            "5. 依据说明：该提示来自安全规则，不替代医生现场判断。",
        ]
    )
