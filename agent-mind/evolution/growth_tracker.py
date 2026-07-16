"""Growth milestone tracking for AIwake's self-evolution.

Records growth events when AIwake acquires new capabilities through
self-learning and uses tools to modify itself. Each milestone carries
a reward score; the cumulative curve is exposed via a public API for
the homepage growth chart.

All records are append-only JSONL under the evolution data directory.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


# ── Reward points by event type ──
REWARD_TABLE: dict[str, int] = {
    "self_upgrade_applied": 50,
    "proposal_generated": 10,
    "proposal_approved": 15,
    "external_learning": 20,
    "experiment_completed": 25,
    "tool_self_modify": 40,
    "code_patch_applied": 50,
    "deploy_success": 60,
    "test_passed": 30,
    "config_change": 15,
    "new_capability": 70,
    "reflection_insight": 10,
    # ── 突破性事件 (breakthroughs) ──
    "self_code_analysis": 35,        # AIwake 主动分析自身源码
    "upgrade_plan_generated": 45,    # AIwake 生成自我修改方案
    "autonomous_debugging": 30,      # AIwake 自主排查问题
    "first_tool_breakthrough": 25,   # 首次使用某种重要工具
    # ── 目标闭环（改→度量→收敛）──
    "goal_closed": 65,               # 一个可度量目标被验证达成并闭合（重大成长）
    "goal_abandoned": 5,             # 目标超限/恶化被自动放弃（如实记录，不刷分）
    "capability_registered": 55,     # 闭环达成沉淀为一项可复用能力，写入工具库
}


# ── Achievement (成果) event types ──
# These are the concrete "AIwake autonomously used tools to improve its own
# code" outcomes. Unlike routine score accrual, each of these is a real,
# auditable event we want to surface as a distinct milestone POINT on the
# growth curve (with file / edit / archive evidence), not just a number.
ACHIEVEMENT_META: dict[str, dict[str, str]] = {
    "self_upgrade_applied": {"icon": "🔧", "label": "自主应用代码升级"},
    "code_patch_applied": {"icon": "🩹", "label": "应用代码补丁"},
    "tool_self_modify": {"icon": "🛠️", "label": "用工具改造自身代码"},
    "deploy_success": {"icon": "🚀", "label": "自我部署成功"},
    "new_capability": {"icon": "✨", "label": "获得新能力"},
    "capability_registered": {"icon": "📦", "label": "沉淀可复用能力"},
    "goal_closed": {"icon": "🎯", "label": "目标闭环达成"},
    "self_code_analysis": {"icon": "🔍", "label": "自主分析源码"},
    "upgrade_plan_generated": {"icon": "📐", "label": "生成自我修改方案"},
    "feature_module_completed": {"icon": "🏆", "label": "完成完整功能模块"},
}
ACHIEVEMENT_EVENT_TYPES: frozenset[str] = frozenset(ACHIEVEMENT_META.keys())


def _build_achievements(
    window_milestones: list[dict[str, Any]],
    date_to_score: dict[str, int],
) -> list[dict[str, Any]]:
    """Turn code-improvement milestones into concrete achievement POINTS.

    Each returned item carries an ``x`` (date, matching a chart label) and a
    ``y`` (the cumulative score the curve shows on that date) so the frontend
    can plot it directly ON the growth curve, plus the concrete evidence
    (files changed, edit count, durable-archive status, source proposal) so a
    viewer sees *what* was achieved, not just a number.
    """
    achievements: list[dict[str, Any]] = []
    for item in window_milestones:
        if not isinstance(item, dict):
            continue
        event_type = str(item.get("event_type", ""))
        if event_type not in ACHIEVEMENT_EVENT_TYPES:
            continue

        date_key = str(item.get("date", ""))[:10] or str(item.get("timestamp", ""))[:10]
        try:
            own_cumulative = int(item.get("cumulative_score", 0) or 0)
        except (ValueError, TypeError):
            own_cumulative = 0
        # Snap the point onto the rendered cumulative line for the day.
        y_value = date_to_score.get(date_key, own_cumulative)

        meta = ACHIEVEMENT_META.get(event_type, {"icon": "⭐", "label": event_type})
        extra = item.get("extra") or {}
        if not isinstance(extra, dict):
            extra = {}

        achievements.append({
            "id": str(item.get("id", "")),
            "timestamp": str(item.get("timestamp", "")),
            "date": date_key,
            "event_type": event_type,
            "icon": meta["icon"],
            "title": meta["label"],
            "description": str(item.get("description", ""))[:500],
            "reward_points": item.get("reward_points", 0),
            "cumulative_score": own_cumulative,
            "x": date_key,
            "y": y_value,
            "files": str(extra.get("files", "")),
            "edits_applied": str(extra.get("edits_applied", "")),
            "patch_method": str(extra.get("patch_method", "")),
            "archived": str(extra.get("archived", "")),
            "risk_level": str(extra.get("risk_level", "")),
            "source_proposal_id": str(item.get("source_proposal_id", "")),
        })

    # Keep the payload light: most recent 60 achievement points.
    return achievements[-60:]


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _milestone_id(event_type: str, description: str) -> str:
    now = _utc_now()
    digest = hashlib.sha1(f"{now}|{event_type}|{description}".encode("utf-8", errors="ignore")).hexdigest()[:10]
    stamp = now.replace(":", "").replace("-", "").replace(".", "")[:15]
    return f"gm_{stamp}_{digest}"


def _data_dir() -> Path:
    base = os.getenv("EVOLUTION_DIR") or os.getenv("AIWAKE_EVOLUTION_DIR") or "/app/data/evolution"
    return Path(base)


def _milestones_file() -> Path:
    return _data_dir() / "growth_milestones.jsonl"


def _read_cumulative_score() -> int:
    """Read the latest cumulative score from existing milestones."""
    file_path = _milestones_file()
    if not file_path.exists():
        return 0
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                return int(item.get("cumulative_score", 0))
            except (json.JSONDecodeError, ValueError):
                continue
    except OSError:
        pass
    return 0


def record_growth_milestone(
    event_type: str,
    description: str,
    *,
    reward_points: int | None = None,
    source_proposal_id: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Append a growth milestone record. Returns the milestone dict on success."""
    file_path = _milestones_file()
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if reward_points is None:
        reward_points = REWARD_TABLE.get(event_type, 10)

    cumulative = _read_cumulative_score() + reward_points
    now = _utc_now()

    milestone = {
        "id": _milestone_id(event_type, description),
        "timestamp": now,
        "date": now[:10],
        "event_type": event_type,
        "description": description[:500],
        "reward_points": reward_points,
        "cumulative_score": cumulative,
        "source_proposal_id": source_proposal_id,
    }
    if extra:
        milestone["extra"] = {k: str(v)[:200] for k, v in extra.items()}

    try:
        with file_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(milestone, ensure_ascii=False, default=str))
            f.write("\n")
        return milestone
    except OSError:
        return None


def read_milestones(days: int = 30, limit: int = 500) -> list[dict[str, Any]]:
    """Read growth milestones within the specified day range."""
    file_path = _milestones_file()
    if not file_path.exists():
        return []

    cutoff = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=max(1, days))).isoformat()
    items: list[dict[str, Any]] = []

    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    for line in lines:
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and item.get("timestamp", "") >= cutoff:
            items.append(item)

    return items[-max(1, min(limit, 1000)):]


def growth_chart_data(days: int = 30) -> dict[str, Any]:
    """Aggregate milestones into daily chart data for the frontend.

    Scans the full milestone log (append-only JSONL) and aggregates by day so
    that high-frequency milestone activity never causes earlier days to be
    dropped. The cumulative line uses each record's stored ``cumulative_score``
    (monotonic) rather than re-summing reward points, so the total stays
    accurate even when the window holds thousands of records.

    Returns:
        {
            "labels": ["2026-06-01", "2026-06-02", ...],
            "scores": [0, 50, 110, ...],          # cumulative score per day
            "daily_points": [0, 50, 60, ...],      # points earned each day
            "milestones": [{...}, ...],             # individual events (capped)
            "total_score": 110,
            "total_milestones": 5,
        }
    """
    # Build date range for the requested window
    today = _dt.date.today()
    start = today - _dt.timedelta(days=max(1, days) - 1)
    start_iso = start.isoformat()
    date_range: list[str] = []
    d = start
    while d <= today:
        date_range.append(d.isoformat())
        d += _dt.timedelta(days=1)

    daily_points: dict[str, int] = defaultdict(int)
    daily_last_cumulative: dict[str, int] = {}
    window_milestones: list[dict[str, Any]] = []
    total_in_window = 0
    base_score = 0  # cumulative score of the last record before the window

    file_path = _milestones_file()
    if file_path.exists():
        try:
            lines = file_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []

        for line in lines:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict):
                continue

            date_key = str(item.get("date", ""))[:10] or str(item.get("timestamp", ""))[:10]
            if not date_key:
                continue

            if date_key < start_iso:
                # Before the window: keep tracking the latest cumulative baseline.
                try:
                    base_score = int(item.get("cumulative_score", base_score))
                except (ValueError, TypeError):
                    pass
                continue

            # Within the window: aggregate per day from the full log (no cap).
            try:
                daily_points[date_key] += int(item.get("reward_points", 0) or 0)
            except (ValueError, TypeError):
                pass
            try:
                daily_last_cumulative[date_key] = int(item.get("cumulative_score", 0))
            except (ValueError, TypeError):
                pass
            total_in_window += 1
            window_milestones.append(item)

    # Build per-day cumulative line from the real cumulative_score, carrying
    # the previous value forward on days without any milestone.
    scores: list[int] = []
    daily_pts_list: list[int] = []
    date_to_score: dict[str, int] = {}
    running = base_score
    for date_str in date_range:
        if date_str in daily_last_cumulative:
            running = daily_last_cumulative[date_str]
        scores.append(running)
        daily_pts_list.append(daily_points.get(date_str, 0))
        date_to_score[date_str] = running

    total_score = scores[-1] if scores else base_score

    # Concrete achievement points: real "AIwake improved its own code" events,
    # snapped onto the cumulative curve so the chart shows *what* happened, not
    # just a rising number.
    achievements = _build_achievements(window_milestones, date_to_score)

    return {
        "labels": date_range,
        "scores": scores,
        "daily_points": daily_pts_list,
        "milestones": window_milestones[-100:],
        "achievements": achievements,
        "total_score": total_score,
        "total_milestones": total_in_window,
        "total_achievements": len(achievements),
        "days": days,
    }
