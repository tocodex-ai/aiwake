"""Goal tracker: measurable improvement targets with convergence judgment.

AIwake can register a goal (e.g. "raise tool_success_rate from 0.97 to 0.99"),
and the evolution engine will re-measure the metric each cycle, check whether
the target is met, and close/abandon the goal automatically. This closes the
"改 → 度量 → 收敛" loop that was previously missing.

All records are append-only JSONL under the evolution data directory.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .growth_tracker import record_growth_milestone

# ── Metric metadata ──
# Each entry: (direction, description)
# direction: "up" = higher is better, "down" = lower is better
METRIC_REGISTRY: dict[str, tuple[str, str]] = {
    "tool_success_rate": ("up", "工具调用成功率"),
    "chat_success_rate": ("up", "聊天回复成功率"),
    "tool_failure_count": ("down", "工具调用失败次数"),
    "external_tool_failure_count": ("down", "外部工具失败次数"),
    "reflection_to_action_ratio": ("target", "反思→行动转换率"),
    "proactive_count": ("up", "主动行动次数"),
    "memory_update_count": ("up", "记忆更新次数"),
    "reflection_count": ("down", "反思次数（过高可能空转）"),
    "failed_reply_count": ("down", "失败回复次数"),
}

DEFAULT_MAX_CYCLES = 12
ABANDON_THRESHOLD = 0.20  # if current is >20% worse than baseline, auto-abandon


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _data_dir() -> Path:
    base = os.getenv("EVOLUTION_DIR") or os.getenv("AIWAKE_EVOLUTION_DIR") or "/app/data/evolution"
    return Path(base)


def _goals_file() -> Path:
    return _data_dir() / "goals.jsonl"


def _capabilities_file() -> Path:
    """Capability library: durable record of goal-driven capabilities AIwake gained."""
    return _data_dir() / "capability_library.jsonl"


def _record_capability(goal: dict[str, Any], current: float) -> dict[str, Any] | None:
    """Append a capability-library entry when a goal closes successfully.

    The entry captures the *what / how / evidence*: which metric improved, from
    what baseline to what value, in how many cycles. Future reflections can read
    this to recall reusable improvements rather than re-deriving them.
    """
    file_path = _capabilities_file()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "id": f"cap_{goal['id']}",
        "registered_at": _utc_now(),
        "goal_id": goal["id"],
        "metric": goal["metric"],
        "direction": goal["direction"],
        "baseline": goal.get("baseline"),
        "target": goal["target"],
        "achieved": current,
        "cycles": int(goal.get("cycle_count", 0)),
        "description": goal.get("description", "")[:500],
        "source": goal.get("source", ""),
    }
    try:
        with file_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str))
            f.write("\n")
        return entry
    except OSError:
        return None


def read_capabilities(limit: int = 50) -> list[dict[str, Any]]:
    """Read the capability library (latest first)."""
    file_path = _capabilities_file()
    if not file_path.exists():
        return []
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    items = []
    for line in lines:
        if not line.strip():
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    items.reverse()
    return items[: max(1, min(limit, 1000))]


def _goal_id(description: str) -> str:
    now = _utc_now()
    digest = hashlib.sha1(f"{now}|{description}".encode("utf-8", errors="ignore")).hexdigest()[:10]
    stamp = now.replace(":", "").replace("-", "").replace(".", "")[:15]
    return f"goal_{stamp}_{digest}"


def _validate_metric(metric: str) -> str:
    """Validate metric name, return normalized form or raise."""
    if metric not in METRIC_REGISTRY:
        valid = ", ".join(sorted(METRIC_REGISTRY.keys()))
        raise ValueError(f"未知指标 '{metric}'，可用指标: {valid}")
    return metric


def _read_all_goals(limit: int = 200) -> list[dict[str, Any]]:
    """Read all goal records from JSONL, returning the latest line per goal_id."""
    file_path = _goals_file()
    if not file_path.exists():
        return []
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    latest: dict[str, dict[str, Any]] = {}
    for line in lines:
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and item.get("id"):
            latest[item["id"]] = item
    goals = list(latest.values())
    goals.sort(key=lambda g: g.get("created_at", ""))
    return goals[-max(1, min(limit, 1000)):]


def _append_goal(goal: dict[str, Any]) -> Path | None:
    """Append one goal record to the JSONL file."""
    file_path = _goals_file()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with file_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(goal, ensure_ascii=False, default=str))
            f.write("\n")
        return file_path
    except OSError:
        return None


def _extract_metric_value(metrics: dict[str, Any], metric: str) -> float | None:
    """Extract a numeric value for *metric* from the metrics dict."""
    raw = metrics.get(metric)
    if raw is None:
        return None
    try:
        return float(raw)
    except (ValueError, TypeError):
        return None


def _convergence_judgment(
    current: float,
    baseline: float,
    target: float,
    direction: str,
    tolerance: float = 0.001,
) -> tuple[bool, str]:
    """Check if *current* meets the target.

    Returns (is_closed, reason).
    """
    if direction == "up":
        if current >= target - tolerance:
            return True, f"当前值 {current:.4f} 已达到目标 {target}（方向：上升）"
        if current > baseline:
            return False, f"当前值 {current:.4f} 较基线 {baseline} 有改善，但尚未达到目标 {target}"
        return False, f"当前值 {current:.4f} 未达目标 {target}，且未优于基线 {baseline}"
    elif direction == "down":
        if current <= target + tolerance:
            return True, f"当前值 {current:.4f} 已达到目标 {target}（方向：下降）"
        if current < baseline:
            return False, f"当前值 {current:.4f} 较基线 {baseline} 有改善，但尚未达到目标 {target}"
        return False, f"当前值 {current:.4f} 未达目标 {target}，且未优于基线 {baseline}"
    else:  # target
        diff = abs(current - target)
        if diff <= tolerance:
            return True, f"当前值 {current:.4f} 在目标 {target} 的容差 {tolerance} 范围内"
        return False, f"当前值 {current:.4f} 距目标 {target} 偏差 {diff:.4f}，超出容差 {tolerance}"
    return False, "未知方向"


def _should_abandon(current: float, baseline: float, direction: str) -> tuple[bool, str]:
    """Check if the current value is significantly worse than baseline."""
    if baseline == 0:
        return False, ""
    if direction == "up":
        if current < baseline * (1 - ABANDON_THRESHOLD):
            return True, f"当前值 {current:.4f} 较基线 {baseline} 下降超过 {ABANDON_THRESHOLD*100:.0f}%，自动放弃"
    elif direction == "down":
        if current > baseline * (1 + ABANDON_THRESHOLD):
            return True, f"当前值 {current:.4f} 较基线 {baseline} 上升超过 {ABANDON_THRESHOLD*100:.0f}%，自动放弃"
    return False, ""


# ── Public API ──


def register_goal(
    metric: str,
    direction: str,
    target: float,
    description: str,
    *,
    source: str = "manual",
    baseline: float | None = None,
    baseline_at: str | None = None,
    max_cycles: int = DEFAULT_MAX_CYCLES,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Register a new measurable improvement goal.

    Args:
        metric: Metric name from METRIC_REGISTRY (e.g. "tool_success_rate").
        direction: "up" (higher is better), "down" (lower is better), or "target".
        target: Target value for the metric.
        description: Human-readable description of the goal.
        source: How the goal was created ("reflection", "manual", "auto").
        baseline: Baseline measurement. If None, will be set when first evaluated.
        baseline_at: Timestamp of baseline measurement.
        max_cycles: Max evolution cycles before auto-abandon.
        extra: Optional extra metadata.

    Returns:
        The goal dict, or raises on validation failure.
    """
    _validate_metric(metric)
    direction = direction.strip().lower()
    if direction not in ("up", "down", "target"):
        raise ValueError(f"方向必须是 'up'、'down' 或 'target'，收到 '{direction}'")

    goal = {
        "id": _goal_id(description),
        "created_at": _utc_now(),
        "metric": metric,
        "direction": direction,
        "target": float(target),
        "baseline": float(baseline) if baseline is not None else None,
        "baseline_at": baseline_at or "",
        "status": "open",
        "description": str(description)[:500],
        "source": str(source)[:50],
        "max_cycles": int(max_cycles),
        "cycle_count": 0,
        "last_measurement": None,
        "last_measured_at": None,
        "closed_at": None,
        "closed_reason": None,
    }
    if extra:
        goal["extra"] = {k: str(v)[:200] for k, v in extra.items()}

    _append_goal(goal)
    return goal


def read_open_goals() -> list[dict[str, Any]]:
    """Read all currently open goals."""
    return [g for g in _read_all_goals() if g.get("status") == "open"]


def read_all_goals(limit: int = 50) -> list[dict[str, Any]]:
    """Read all goals (open + closed + abandoned), latest first."""
    goals = _read_all_goals(limit=limit)
    goals.reverse()
    return goals


def evaluate_open_goals(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Evaluate all open goals against current metrics.

    For each open goal:
    - Extract the metric value from *metrics*.
    - If no baseline yet, set it now.
    - Apply convergence judgment.
    - If closed → record growth milestone, append closure line.
    - If abandoned → append abandonment line.
    - Otherwise → update cycle_count and last_measurement.

    Returns a list of result dicts (one per goal evaluated).
    """
    open_goals = read_open_goals()
    if not open_goals:
        return []

    results: list[dict[str, Any]] = []
    for goal in open_goals:
        result = _evaluate_one(goal, metrics)
        results.append(result)
    return results


def _evaluate_one(goal: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a single open goal."""
    gid = goal["id"]
    metric = goal["metric"]
    direction = goal["direction"]
    target = float(goal["target"])
    baseline = goal.get("baseline")
    max_cycles = int(goal.get("max_cycles", DEFAULT_MAX_CYCLES))
    cycle_count = int(goal.get("cycle_count", 0))

    current = _extract_metric_value(metrics, metric)
    if current is None:
        return {
            "goal_id": gid,
            "status": "skipped",
            "reason": f"指标 '{metric}' 在当前 metrics 中不可用",
        }

    now = _utc_now()

    # ── First evaluation: set baseline ──
    if baseline is None:
        baseline = current
        goal["baseline"] = baseline
        goal["baseline_at"] = now
        # Persist baseline update
        goal["cycle_count"] = cycle_count + 1
        goal["last_measurement"] = current
        goal["last_measured_at"] = now
        _append_goal(goal)
        return {
            "goal_id": gid,
            "status": "baseline_set",
            "metric": metric,
            "baseline": baseline,
            "current": current,
            "reason": f"基线已设定: {metric}={baseline:.4f}，目标={target}",
        }

    # ── Subsequent evaluations ──
    cycle_count += 1
    goal["cycle_count"] = cycle_count
    goal["last_measurement"] = current
    goal["last_measured_at"] = now

    # Check auto-abandon (significantly worse than baseline)
    abandon, abandon_reason = _should_abandon(current, baseline, direction)
    if abandon:
        goal["status"] = "abandoned"
        goal["closed_at"] = now
        goal["closed_reason"] = abandon_reason
        _append_goal(goal)
        # 如实记录放弃事件（不刷分），便于后续从重大成长日志中追溯
        try:
            record_growth_milestone(
                event_type="goal_abandoned",
                description=f"目标自动放弃（恶化）: {goal.get('description', '')[:200]}",
                extra={
                    "goal_id": gid,
                    "metric": metric,
                    "baseline": str(baseline),
                    "current": str(current),
                    "target": str(target),
                    "cycles": str(cycle_count),
                    "reason": "regression",
                },
            )
        except Exception:
            pass
        return {
            "goal_id": gid,
            "status": "abandoned",
            "metric": metric,
            "baseline": baseline,
            "current": current,
            "target": target,
            "cycle_count": cycle_count,
            "reason": abandon_reason,
        }

    # Check max cycles exceeded
    if cycle_count >= max_cycles:
        goal["status"] = "abandoned"
        goal["closed_at"] = now
        goal["closed_reason"] = f"超过最大轮次 {max_cycles} 仍未达成目标"
        _append_goal(goal)
        try:
            record_growth_milestone(
                event_type="goal_abandoned",
                description=f"目标自动放弃（超限）: {goal.get('description', '')[:200]}",
                extra={
                    "goal_id": gid,
                    "metric": metric,
                    "baseline": str(baseline),
                    "current": str(current),
                    "target": str(target),
                    "cycles": str(cycle_count),
                    "reason": "max_cycles_exceeded",
                },
            )
        except Exception:
            pass
        return {
            "goal_id": gid,
            "status": "abandoned",
            "metric": metric,
            "baseline": baseline,
            "current": current,
            "target": target,
            "cycle_count": cycle_count,
            "reason": f"超过最大轮次 {max_cycles} 仍未达成目标（当前 {current:.4f}，目标 {target}）",
        }

    # Convergence judgment
    closed, reason = _convergence_judgment(current, baseline, target, direction)
    if closed:
        goal["status"] = "closed"
        goal["closed_at"] = now
        goal["closed_reason"] = reason
        _append_goal(goal)
        # 重大成长：目标闭合
        try:
            record_growth_milestone(
                event_type="goal_closed",
                description=f"目标达成: {goal.get('description', '')[:200]}",
                extra={
                    "goal_id": gid,
                    "metric": metric,
                    "baseline": str(baseline),
                    "current": str(current),
                    "target": str(target),
                    "cycles": str(cycle_count),
                },
            )
        except Exception:
            pass
        # 沉淀为可复用能力，写入工具/能力库 + 记一次重大成长
        cap_entry: dict[str, Any] | None = None
        try:
            cap_entry = _record_capability(goal, current)
            if cap_entry is not None:
                record_growth_milestone(
                    event_type="capability_registered",
                    description=f"能力沉淀入库: {goal.get('description', '')[:200]}",
                    extra={
                        "capability_id": cap_entry["id"],
                        "goal_id": gid,
                        "metric": metric,
                        "achieved": str(current),
                        "cycles": str(cycle_count),
                    },
                )
        except Exception:
            cap_entry = None
        return {
            "goal_id": gid,
            "status": "closed",
            "metric": metric,
            "baseline": baseline,
            "current": current,
            "target": target,
            "cycle_count": cycle_count,
            "reason": reason,
            "capability_id": (cap_entry or {}).get("id"),
        }

    # Still iterating — persist updated cycle_count + last_measurement
    _append_goal(goal)
    return {
        "goal_id": gid,
        "status": "iterating",
        "metric": metric,
        "baseline": baseline,
        "current": current,
        "target": target,
        "cycle_count": cycle_count,
        "max_cycles": max_cycles,
        "reason": reason,
    }
