"""Backlog builder for autonomous evolution tasks."""

from __future__ import annotations

import hashlib
from typing import Any


_TYPE_RULES: list[tuple[str, str, str]] = [
    # 转换率类规则必须排在泛化的“反思”规则之前命中，避免 too_passive/too_reactive
    # 被错误归类成“增强反思频率”这种方向相反的标题。
    ("转换率过低", "autonomy_improvement", "将近期反思转化为可验证行动"),
    ("转换率过高", "reliability_improvement", "收敛行动节奏并先固化证据目标"),
    ("工具", "tool_improvement", "提升工具调用可靠性"),
    ("外部", "tool_improvement", "增强外部依赖降级能力"),
    ("失败回复", "reliability_improvement", "降低失败回复率"),
    ("空回复", "reliability_improvement", "降低空回复率"),
    ("记忆", "memory_improvement", "增强长期记忆沉淀"),
    ("反思", "autonomy_improvement", "增强自主反思频率与质量"),
    ("主动", "autonomy_improvement", "提升主动成长行为"),
    ("安全", "safety_review", "审查安全边界与风险"),
    ("稳定", "reliability_improvement", "提升运行稳定性"),
]


def build_backlog(evaluation: dict[str, Any], metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert evaluation issues/suggestions into deterministic evolution tasks."""

    if not isinstance(evaluation, dict):
        evaluation = {}
    if not isinstance(metrics, dict):
        metrics = {}

    issues = [str(x) for x in evaluation.get("issues", []) if str(x).strip()]
    suggestions = [str(x) for x in evaluation.get("suggestions", []) if str(x).strip()]
    tasks: list[dict[str, Any]] = []

    for idx, issue in enumerate(issues):
        task_type, default_title = _classify(issue)
        suggestion = suggestions[idx] if idx < len(suggestions) else "形成最小改动并通过验证后再固化。"
        tasks.append(_task(
            task_type=task_type,
            title=f"{default_title}: {issue}",
            priority=_priority(task_type, evaluation, metrics),
            risk=_risk(task_type, evaluation),
            evidence=[issue, _metric_evidence(metrics)],
            acceptance=[suggestion, "必须写入 evolution memory，并保留可回滚记录。"],
        ))

    # Metric-driven tasks ensure the loop can generate work even when rules do
    # not produce textual issues.
    if float(metrics.get("chat_success_rate", 1.0) or 1.0) < 0.95:
        tasks.append(_task(
            "reliability_improvement",
            "提高聊天成功率并减少暂时无法回复",
            0.9,
            "medium",
            [f"chat_success_rate={metrics.get('chat_success_rate')}", f"failed_reply_count={metrics.get('failed_reply_count')}"] ,
            ["定位失败回复来源", "增加降级回复或模型重试策略", "验证公开聊天不再出现连续失败"],
        ))
    if float(metrics.get("tool_success_rate", 1.0) or 1.0) < 0.9:
        breakdown: dict = metrics.get("tool_failure_breakdown") or {}
        dominant = max(breakdown, key=breakdown.get) if breakdown else "unknown"
        tasks.append(_task(
            "tool_improvement",
            f"提高工具成功率：主要失败类型={dominant}",
            0.82,
            "medium",
            [
                f"tool_success_rate={metrics.get('tool_success_rate')}",
                f"tool_failure_count={metrics.get('tool_failure_count')}",
                f"dominant_failure={dominant}",
                f"breakdown={breakdown}",
            ],
            [f"针对 {dominant} 类失败增加重试或替代路径", "验证工具失败不会阻断主回复"],
        ))
    if int(metrics.get("reflection_count") or 0) <= 0:
        tasks.append(_task(
            "autonomy_improvement",
            "恢复或增强自主反思心跳",
            0.78,
            "low",
            ["最近窗口未检测到 reflection_start/reflection_content"],
            ["确认心跳循环运行", "下一窗口至少出现一次自主反思记录"],
        ))
    if int(metrics.get("memory_update_count") or 0) <= 0:
        tasks.append(_task(
            "memory_improvement",
            "增强关键经验的记忆沉淀",
            0.7,
            "low",
            ["最近窗口未检测到 memory_update"],
            ["关键对话或自评结果写入长期记忆", "后续 system prompt 可检索相关记忆"],
        ))

    return _dedupe_tasks(tasks)


def _classify(issue: str) -> tuple[str, str]:
    for needle, task_type, title in _TYPE_RULES:
        if needle in issue:
            return task_type, title
    return "reliability_improvement", "改进运行质量"


def _priority(task_type: str, evaluation: dict[str, Any], metrics: dict[str, Any]) -> float:
    base = {
        "safety_review": 0.95,
        "reliability_improvement": 0.8,
        "tool_improvement": 0.76,
        "memory_improvement": 0.62,
        "autonomy_improvement": 0.68,
    }.get(task_type, 0.6)
    total = float(evaluation.get("total_score", 100) or 100)
    if total < 70:
        base += 0.08
    if int(metrics.get("failed_reply_count") or 0) > 0:
        base += 0.04
    return round(min(1.0, base), 3)


def _risk(task_type: str, evaluation: dict[str, Any]) -> str:
    if task_type == "safety_review" or evaluation.get("risk_level") == "high":
        return "high"
    if task_type in {"tool_improvement", "reliability_improvement"}:
        return "medium"
    return "low"


def _task(task_type: str, title: str, priority: float, risk: str, evidence: list[str], acceptance: list[str]) -> dict[str, Any]:
    raw = f"{task_type}|{title}|{evidence}"
    digest = hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:10]
    return {
        "id": f"ev-{digest}",
        "type": task_type,
        "title": title,
        "priority": round(float(priority), 3),
        "risk": risk,
        "evidence": [x for x in evidence if x],
        "acceptance": [x for x in acceptance if x],
        "status": "open",
    }


def _metric_evidence(metrics: dict[str, Any]) -> str:
    return (
        f"chat_success_rate={metrics.get('chat_success_rate')}, "
        f"tool_success_rate={metrics.get('tool_success_rate')}, "
        f"reflection_count={metrics.get('reflection_count')}, "
        f"memory_update_count={metrics.get('memory_update_count')}"
    )


def _dedupe_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for task in sorted(tasks, key=lambda x: x.get("priority", 0), reverse=True):
        task_id = task.get("id")
        if task_id in seen:
            continue
        seen.add(task_id)
        result.append(task)
    return result[:20]
