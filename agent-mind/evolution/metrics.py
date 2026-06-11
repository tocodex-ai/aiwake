"""Runtime metric collection for AIwake's evolution loop.

The collector is intentionally conservative and standard-library only. It scans
recent persistent logs from DIARY_DIR and returns JSON-serializable counters that
can drive deterministic self-evaluation without depending on a model call.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import time
from pathlib import Path
from typing import Any

DIARY_DIR = Path(os.getenv("DIARY_DIR", "/app/data/diary"))
_FAILED_REPLY_MARKERS = (
    "[Agent 暂时无法回复",
    "暂时无法回复",
    "empty response",
    "failed to answer",
)
_MOJIBAKE_MARKERS = ("�", "\ufffd", "â", "â€™", "â€œ", "â€", "æ", "ä", "å", "ç", "è", "é")
_EXTERNAL_TOOL_NAMES = ("web_search", "web_fetch", "url_fetch", "arxiv_search", "weather")


def collect_metrics(hours: int = 24, limit: int = 1000) -> dict[str, Any]:
    """Collect recent runtime metrics from activity/public-chat logs.

    Failures never propagate to callers. When an unexpected error occurs, the
    returned payload is marked as degraded and includes the error text.
    """

    generated_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
    try:
        window_hours = max(1, min(int(hours), 168))
    except Exception:
        window_hours = 24
    try:
        max_items = max(1, min(int(limit), 5000))
    except Exception:
        max_items = 1000

    base: dict[str, Any] = {
        "generated_at": generated_at,
        "window_hours": window_hours,
        "limit": max_items,
        "source_dir": str(DIARY_DIR),
        "degraded": False,
        "errors": [],
        "latest_tick": None,
        "user_message_count": 0,
        "agent_reply_count": 0,
        "failed_reply_count": 0,
        "tool_call_count": 0,
        "tool_result_count": 0,
        "tool_failure_count": 0,
        "tool_success_count": 0,
        "external_tool_failure_count": 0,
        "tool_failure_breakdown": {},
        "reflection_count": 0,
        "proactive_count": 0,
        "memory_update_count": 0,
        "memory_compress_count": 0,
        "chat_success_rate": 1.0,
        "tool_success_rate": 1.0,
        "reflection_to_action_ratio": 0.0,
        "reflection_to_action_status": "unknown",
        "last_events": [],
        "experiment_task_count": 0,
        "experiment_artifact_count": 0,
        "experiment_tool_call_count": 0,
        "experiment_reflection_count": 0,
    }

    try:
        cutoff = time.time() - window_hours * 3600
        if not DIARY_DIR.exists():
            base["degraded"] = True
            base["errors"].append(f"日志目录不存在: {DIARY_DIR}")
            return base

        _read_activity_logs(base, cutoff, max_items)
        _read_public_chat_logs(base, cutoff, max_items)
        _read_experiment_logs(base, cutoff, max_items)
        _finalize_rates(base)
        return base
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        base["degraded"] = True
        base["errors"].append(str(exc))
        _finalize_rates(base)
        return base


def _read_activity_logs(metrics: dict[str, Any], cutoff: float, limit: int) -> None:
    today = _dt.date.today()
    days = max(1, min(8, int(metrics["window_hours"] / 24) + 2))
    collected = 0
    for offset in range(days):
        day = today - _dt.timedelta(days=offset)
        path = DIARY_DIR / f"activity_{day.isoformat()}.log"
        if not path.exists():
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception as exc:
            metrics["degraded"] = True
            metrics["errors"].append(f"读取 {path.name} 失败: {exc}")
            continue
        # 日志文件按时间追加；统计窗口应优先读取最新记录，避免当天日志超过 limit
        # 时只统计到最早的工具调用，导致最新成功调用没有进入 /experiment/status。
        for line in reversed(lines):
            parsed = _parse_activity_line(day, line)
            if not parsed or parsed["ts"] < cutoff:
                continue
            collected += 1
            _count_activity(metrics, parsed)
            if len(metrics["last_events"]) < limit:
                metrics["last_events"].append({
                    "ts": parsed["ts"],
                    "event": parsed["event"],
                    "content": parsed["content"][:240],
                })
            if collected >= limit:
                return


def _parse_activity_line(day: _dt.date, line: str) -> dict[str, Any] | None:
    line = line.lstrip("\ufeff")
    match = re.match(r"^\[(?P<hms>\d{2}:\d{2}:\d{2})\] \[(?P<event>[^\]]+)\] (?P<content>.*)$", line)
    if not match:
        return None
    try:
        ts = _dt.datetime.strptime(f"{day.isoformat()} {match.group('hms')}", "%Y-%m-%d %H:%M:%S").timestamp()
    except Exception:
        ts = 0.0
    return {"ts": ts, "event": match.group("event"), "content": match.group("content")}


def _count_activity(metrics: dict[str, Any], item: dict[str, Any]) -> None:
    event = item["event"]
    content = item["content"]
    if event == "user_message":
        metrics["user_message_count"] += 1
    elif event == "llm_response":
        metrics["agent_reply_count"] += 1
        if _is_failed_reply(content):
            metrics["failed_reply_count"] += 1
    elif event == "tool_call":
        metrics["tool_call_count"] += 1
    elif event == "tool_result":
        metrics["tool_result_count"] += 1
        if _is_tool_failure(content):
            metrics["tool_failure_count"] += 1
            category = _classify_tool_failure(content)
            breakdown = metrics.setdefault("tool_failure_breakdown", {})
            breakdown[category] = int(breakdown.get(category, 0)) + 1
            if any(name in content for name in _EXTERNAL_TOOL_NAMES):
                metrics["external_tool_failure_count"] += 1
    elif event in {"reflection_start", "reflection_content"}:
        metrics["reflection_count"] += 1
    elif event == "proactive_speak":
        metrics["proactive_count"] += 1
    elif event == "memory_update":
        metrics["memory_update_count"] += 1
    elif event == "memory_compress":
        metrics["memory_compress_count"] += 1

    tick_match = re.search(r"tick\s*#(?P<tick>\d+)", content, flags=re.I)
    if tick_match:
        try:
            tick = int(tick_match.group("tick"))
            metrics["latest_tick"] = max(metrics["latest_tick"] or 0, tick)
        except Exception:
            pass


def _read_experiment_logs(metrics: dict[str, Any], cutoff: float, limit: int) -> None:
    exp_dir = Path(os.getenv("EXPERIMENT_DIR", "/app/data/experiments"))
    files = {
        "tasks.jsonl": "experiment_task_count",
        "artifacts.jsonl": "experiment_artifact_count",
        "tool_calls.jsonl": "experiment_tool_call_count",
        "reflections.jsonl": "experiment_reflection_count",
    }
    if not exp_dir.exists():
        return
    for filename, counter in files.items():
        path = exp_dir / filename
        if not path.exists():
            continue
        seen = 0
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception as exc:
            metrics["degraded"] = True
            metrics["errors"].append(f"读取 {path.name} 失败: {exc}")
            continue
        for line in lines:
            try:
                item = json.loads(line)
                ts = float(item.get("ts") or 0)
            except Exception:
                continue
            if ts < cutoff:
                continue
            metrics[counter] += 1
            if counter == "experiment_tool_call_count":
                metrics["tool_call_count"] += 1
                metrics["tool_result_count"] += 1
            elif counter == "experiment_reflection_count":
                metrics["reflection_count"] += 1
            elif filename == "tasks.jsonl" and item.get("type") == "self_task_update":
                metrics["memory_update_count"] += 1 if item.get("status") == "done" else 0
            seen += 1
            if seen >= limit:
                break


def _read_public_chat_logs(metrics: dict[str, Any], cutoff: float, limit: int) -> None:
    today = _dt.date.today()
    days = max(1, min(8, int(metrics["window_hours"] / 24) + 2))
    seen = 0
    for offset in range(days):
        day = today - _dt.timedelta(days=offset)
        path = DIARY_DIR / f"public_chat_{day.isoformat()}.jsonl"
        if not path.exists():
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception as exc:
            metrics["degraded"] = True
            metrics["errors"].append(f"读取 {path.name} 失败: {exc}")
            continue
        for line in lines:
            try:
                item = json.loads(line)
                ts = float(item.get("ts") or 0)
            except Exception:
                continue
            if ts < cutoff:
                continue
            role = item.get("role")
            content = str(item.get("content") or "")
            if role == "user":
                metrics["user_message_count"] += 1
            elif role == "agent":
                metrics["agent_reply_count"] += 1
                if _is_failed_reply(content):
                    metrics["failed_reply_count"] += 1
            seen += 1
            if seen >= limit:
                return


def _is_failed_reply(content: str) -> bool:
    text = content or ""
    return any(marker in text for marker in _FAILED_REPLY_MARKERS) or not text.strip() or _looks_like_mojibake_reply(text)


def _looks_like_mojibake_reply(content: str) -> bool:
    """Conservatively count visibly corrupted agent replies as failed replies."""
    text = (content or "").strip()
    if len(text) < 8:
        return False
    visible_count = sum(1 for ch in text if not ch.isspace())
    if visible_count <= 0:
        return False
    question_count = text.count("?")
    marker_count = sum(text.count(marker) for marker in _MOJIBAKE_MARKERS)
    c1_control_count = sum(1 for ch in text if "\u0080" <= ch <= "\u009f")
    cjk_count = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    damaged_count = question_count + marker_count + c1_control_count
    damaged_ratio = damaged_count / visible_count
    if text.startswith("[Agent") and (question_count >= 4 or marker_count >= 1):
        return True
    if cjk_count == 0 and question_count >= 6 and damaged_ratio >= 0.22:
        return True
    if cjk_count == 0 and damaged_count >= 10 and damaged_ratio >= 0.12:
        return True
    if cjk_count == 0 and question_count >= 4 and (marker_count + c1_control_count) >= 2:
        return True
    return cjk_count == 0 and marker_count >= 4 and damaged_ratio >= 0.08


def _is_tool_failure(content: str) -> bool:
    text = (content or "").lower()
    return any(marker in text for marker in ("error", "失败", "不可用", "timeout", "403", "429", "521"))


# 失败类型分类规则：按优先级匹配，第一个命中的类别即为该失败的归类。
# 仅对已判定为失败的内容调用，用于把聚合的 tool_failure_count 拆分成可区分的
# 失败画像，使进化 backlog 能生成差异化提案，避免 NO_PATCH 空循环。
_FAILURE_CATEGORY_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("rate_limited", ("429", "rate limit", "too many requests", "频率", "限流")),
    ("forbidden", ("403", "forbidden", "unauthorized", "401", "无权", "拒绝访问")),
    ("upstream_error", ("521", "522", "523", "502", "503", "504", "bad gateway", "gateway")),
    ("timeout", ("timeout", "timed out", "超时")),
    ("unavailable", ("不可用", "unavailable", "connection", "连接", "网络", "network")),
]


def _classify_tool_failure(content: str) -> str:
    """Map a failed tool_result into a coarse failure category.

    Only called for entries already classified as failures by
    :func:`_is_tool_failure`. Returns a stable category key so the backlog
    builder can produce differentiated, evidence-rich proposals instead of
    repeating an identical generic proposal every cycle.
    """
    text = (content or "").lower()
    for category, markers in _FAILURE_CATEGORY_RULES:
        if any(marker in text for marker in markers):
            return category
    return "other_error"


def _finalize_rates(metrics: dict[str, Any]) -> None:
    replies = int(metrics.get("agent_reply_count") or 0)
    failed = int(metrics.get("failed_reply_count") or 0)
    metrics["chat_success_rate"] = round((replies - failed) / replies, 4) if replies > 0 else 1.0

    tool_results = int(metrics.get("tool_result_count") or 0)
    tool_failed = int(metrics.get("tool_failure_count") or 0)
    tool_success = max(0, tool_results - tool_failed)
    metrics["tool_success_count"] = tool_success
    metrics["tool_success_rate"] = round(tool_success / tool_results, 4) if tool_results > 0 else 1.0

    reflections = int(metrics.get("reflection_count") or 0)
    actions = int(metrics.get("tool_call_count") or 0) + int(metrics.get("proactive_count") or 0)
    ratio = round(actions / reflections, 4) if reflections > 0 else 0.0
    metrics["reflection_to_action_ratio"] = ratio
    if reflections <= 0:
        metrics["reflection_to_action_status"] = "no_reflection"
    elif ratio < 0.10:
        metrics["reflection_to_action_status"] = "too_passive"
    elif ratio > 0.70:
        # 只有 action/reflection 超过 0.70 才算 too_reactive；
        # 0.30~0.70 区间是反思中积极使用工具的正常表现。
        metrics["reflection_to_action_status"] = "too_reactive"
    else:
        metrics["reflection_to_action_status"] = "balanced"

    metrics["last_events"] = sorted(metrics.get("last_events", []), key=lambda x: x.get("ts", 0))[-50:]
