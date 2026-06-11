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

    Returns:
        {
            "labels": ["2026-06-01", "2026-06-02", ...],
            "scores": [0, 50, 110, ...],          # cumulative score per day
            "daily_points": [0, 50, 60, ...],      # points earned each day
            "milestones": [{...}, ...],             # individual events
            "total_score": 110,
            "total_milestones": 5,
        }
    """
    milestones = read_milestones(days=days)

    # Build date range
    today = _dt.date.today()
    start = today - _dt.timedelta(days=max(1, days) - 1)
    date_range = []
    d = start
    while d <= today:
        date_range.append(d.isoformat())
        d += _dt.timedelta(days=1)

    # Aggregate daily points
    daily_points: dict[str, int] = defaultdict(int)
    for m in milestones:
        date_key = str(m.get("date", ""))[:10]
        if date_key:
            daily_points[date_key] += int(m.get("reward_points", 0))

    # Find the cumulative score just before the window
    base_score = 0
    file_path = _milestones_file()
    if file_path.exists():
        cutoff = start.isoformat()
        try:
            lines = file_path.read_text(encoding="utf-8").splitlines()
            for line in lines:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    if item.get("date", "")[:10] < cutoff:
                        base_score = int(item.get("cumulative_score", 0))
                except (json.JSONDecodeError, ValueError):
                    continue
        except OSError:
            pass

    # Build cumulative scores array
    cumulative = base_score
    scores = []
    daily_pts_list = []
    for date_str in date_range:
        pts = daily_points.get(date_str, 0)
        cumulative += pts
        scores.append(cumulative)
        daily_pts_list.append(pts)

    return {
        "labels": date_range,
        "scores": scores,
        "daily_points": daily_pts_list,
        "milestones": milestones[-100:],
        "total_score": cumulative,
        "total_milestones": len(milestones),
        "days": days,
    }
