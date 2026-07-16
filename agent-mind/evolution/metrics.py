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
    # empty_result 是“工具可用但无语义内容”，不计入基础设施失败。
    if "classification=empty_result" in text or "failure_type=empty_result" in text:
        return False
    if "classification=failure" in text:
        return True
    if any(marker in text for marker in ("status=error", "status': 'error'", '"status": "error"')):
        return True
    if re.search(r"returncode\s*[=:]\s*(?!0\b)\d+", text):
        return True
    return any(
        marker in text
        for marker in (
            "失败",
            "不可用",
            "timeout",
            "403",
            "429",
            "521",
            "command_missing",
            "missing_binary",
            ": not found",
            "no such file",
        )
    )


# 失败类型分类规则：按优先级匹配，第一个命中的类别即为该失败的归类。
# 仅对已判定为失败的内容调用，用于把聚合的 tool_failure_count 拆分成可区分的
# 失败画像，使进化 backlog 能生成差异化提案，避免 NO_PATCH 空循环。
_FAILURE_CATEGORY_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("rate_limited", ("429", "rate limit", "too many requests", "频率", "限流")),
    ("command_missing", (
        "command_missing", "missing_binary", "returncode=127", "returncode': 127",
        '"returncode": 127', ": not found", "/bin/sh: 1:", "command not found",
    )),
    # config_blocked / not_found 必须排在 forbidden 之前：这两类是“确定性失败”，
    # 重试同样的工具名/路径永远不会成功，应给出“不要重试、改名/换工具/确认存在”的
    # 止损指引，避免被笼统计入 other_error 后又按“原样重试一次”空转，长期卡住
    # tool_failure_count 不收敛（线上证据：Glob/Read/TodoWrite 未被实验配置允许、
    # file_read 文件不存在，反复出现却始终归到 other_error）。
    ("config_blocked", (
        "未被当前实验配置允许", "未被当前实验", "未被允许", "not allowed",
        "not permitted", "disabled", "已禁用", "未配置", "未注册",
    )),
    ("not_found", (
        "不存在", "not found", "no such file", "找不到", "未找到",
        "does not exist", "404",
    )),
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


# 工具失败恢复决策表：把"失败类型 → 首选恢复动作 → 何时停止"固化成稳定规则，
# 而不是让反思层每轮重新文字推演同一套策略（线上证据显示 reflection_count 高、
# 但 tool_failure_count 长期卡住不收敛，根因正是"高反思、低规则化"）。
# 每条规则只描述读-only 的决策建议，真正执行仍由调用方/反思层按需采纳。
_FAILURE_RECOVERY_PLAYBOOK: dict[str, dict[str, str]] = {
    "rate_limited": {
        "first_action": "退避后单次重试（指数退避，起步 1-2s）",
        "fallback": "切换到等价的低频工具或缩小请求范围",
        "stop_when": "连续 2 次仍 429 即停止重试，转为延后执行并记录证据",
    },
    "command_missing": {
        "first_action": "先用 command -v/which 预检命令是否存在；若不存在，立即换用已安装的等价命令",
        "fallback": "用 grep/find/python 等基础命令替代 rg/fd 等缺失二进制，并把缺失命令写入证据",
        "stop_when": "确认二进制缺失即停止重复调用同一命令（重试无意义），改走等价工具或脚本",
    },
    "forbidden": {
        "first_action": "校验参数/权限范围，确认是否误用需要鉴权的入口",
        "fallback": "改用只读等价工具或公开数据源",
        "stop_when": "确认无权限即停止，不要反复重试（重试无意义）",
    },
    "upstream_error": {
        "first_action": "单次重试（上游 5xx/网关错误通常是瞬时抖动）",
        "fallback": "降级为替代信息源或延后到下一轮执行",
        "stop_when": "重试 1 次仍失败即停止，记录上游不可用证据",
    },
    "timeout": {
        "first_action": "缩小查询范围/降低数据量后单次重试",
        "fallback": "切换到更轻量的等价工具",
        "stop_when": "2 次超时即停止，标记该路径为高延迟并延后",
    },
    "unavailable": {
        "first_action": "确认网络/连接是否瞬时不可用，单次重试",
        "fallback": "切换到本地缓存或替代来源",
        "stop_when": "连续 2 次不可用即停止，转为延后执行",
    },
    "config_blocked": {
        "first_action": "确认工具名是否被当前实验/白名单允许；若是幻觉命名先归一化到真实工具名",
        "fallback": "改用语义等价的已允许工具（如 Glob→file_list、Read→file_read、TodoWrite→self_task_create）",
        "stop_when": "确认工具未被配置允许即停止，绝不重复调用同一未授权工具名（重试无意义）",
    },
    "not_found": {
        "first_action": "先用 file_list 确认目标路径/资源是否真实存在，再决定是否调用",
        "fallback": "改用存在的正确路径，或换公开数据源/等价工具获取所需信息",
        "stop_when": "确认目标不存在即停止，不要对同一不存在的路径反复重试（重试无意义）",
    },
    "other_error": {
        "first_action": "先检查参数/输入完整性，再原样重试一次",
        "fallback": "改用等价工具或缩小任务范围分步执行",
        "stop_when": "同类失败第二次仍走同样弯路即停止，标记为需要规则修正",
    },
}


def build_failure_recovery_playbook(breakdown: dict[str, Any] | None) -> dict[str, Any]:
    """根据实际失败画像生成一份可执行的恢复决策指引。

    输入是 ``tool_failure_breakdown``（category -> count）。只为真实出现过的
    失败类别生成规则条目，并按出现次数降序排列，让反思层优先处理高频失败。
    返回结构是只读建议，不触发任何执行，供 metrics / 反思 prompt 直接引用，
    把"高反思、低规则化"补上一层稳定的"下次默认动作"。
    """
    breakdown = breakdown or {}
    rules: list[dict[str, Any]] = []
    for category, count in sorted(
        breakdown.items(), key=lambda kv: int(kv[1] or 0), reverse=True
    ):
        if int(count or 0) <= 0:
            continue
        rule = _FAILURE_RECOVERY_PLAYBOOK.get(category, _FAILURE_RECOVERY_PLAYBOOK["other_error"])
        rules.append({
            "category": category,
            "count": int(count or 0),
            "first_action": rule["first_action"],
            "fallback": rule["fallback"],
            "stop_when": rule["stop_when"],
        })
    return {
        "rules": rules,
        "principle": (
            "每次工具失败都补三元证据：失败类型 / 恢复动作 / 恢复结果；"
            "若同类失败第二次仍走同样弯路，视为闭环失效信号，优先修正规则而非增加泛化反思。"
        ),
        "evidence_fields": ["failure_category", "recovery_action", "recovery_result"],
    }


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

    # 把"失败类型 → 恢复动作 → 何时停止"决策表按实际失败画像挂进 metrics，
    # 让反思层/backlog 能直接引用稳定恢复规则，而不是每轮重新文字推演。
    metrics["failure_recovery_playbook"] = build_failure_recovery_playbook(
        metrics.get("tool_failure_breakdown")
    )
    # 暴露 failure_triads 计数，便于观察“证据库是否真在增长”
    try:
        from .failure_triads import triad_stats

        metrics["failure_triad_stats"] = triad_stats(limit=500)
    except Exception:
        metrics["failure_triad_stats"] = {"total": 0, "real_count": 0, "by_category": {}}

    metrics["last_events"] = sorted(metrics.get("last_events", []), key=lambda x: x.get("ts", 0))[-50:]
