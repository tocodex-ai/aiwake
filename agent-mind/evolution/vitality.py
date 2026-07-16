"""Vitality Score & Environmental Pressure (M-002 v2).

把 AIwake 的"内在状态压力"翻译成**可观察的健康/压力信号**，而不是工具门控。
核心思想（来自 docs/survival-pressure-and-autonomous-improvement.md v2 的后续修订）：

    环境压力可以被观察，但不能削弱 Agent 自主使用工具的能力

早期设计曾尝试让 vitality 下降时减少反思工具和 token 预算；实践中这会与项目长期规则
"不要对 Agent 加任何门控和工具使用限制"冲突。当前实现保留 vitality 分数、状态、原因、
名义 token_budget 与状态面板，用于让 AIwake 和观察者理解停滞/成长压力；但运行时不再
过滤 tools_schema，也不再写入 max_tokens 限制。

设计约束（来自项目长期规则）：
    - 不对 Agent 加任何工具门控或工具使用限制；critical 状态下也必须保持完整工具 schema
    - 不影响 /chat 用户对话路径（vitality 仅作为 _autonomous_reflect 的观察信号）
    - 完全可回滚：compute_vitality_score() 只是只读评分，不承担权限控制
    - 不写入威胁性 prompt；vitality 状态对模型透明，作为自我观察与行动校准信息

公开 API：
    VitalityState              — 5 个状态枚举（vibrant / stable / drifting / fading / critical）
    VitalityReport             — 计算结果数据类（score + state + reasons + envelope）
    compute_vitality_score()   — 主入口：聚合 growth/stagnation/upgrade_pressure 算出 0~1 分数
    filter_tools_schema_by_vitality()  — 兼容旧调用，观察模式下返回完整 OpenAI tools schema
    reflect_max_tokens_for_vitality()  — 兼容旧调用，观察模式下返回 None（不限制 token）
    vitality_status_snapshot() — 给 /vitality/status 端点用的只读快照

所有计算都是无副作用的纯函数 + 一次性磁盘读取（不写文件）。
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# ─────────────────────────── 配置常量 ───────────────────────────
# 可通过环境变量覆盖（部署期调参不需要重新打镜像）

# vitality 分数计算窗口
_VITALITY_GROWTH_WINDOW_DAYS = int(os.getenv("VITALITY_GROWTH_WINDOW_DAYS", "7"))
# 期望窗口内累计成长分（达到这个数 → growth 项满分 0.5）
_VITALITY_GROWTH_FULL_SCORE = float(os.getenv("VITALITY_GROWTH_FULL_SCORE", "200"))
# 反思连续没有行动的轮数达到此值 → stagnation 项扣满 0.2
_VITALITY_STAGNATION_FULL_ROUNDS = int(os.getenv("VITALITY_STAGNATION_FULL_ROUNDS", "8"))
# upgrade_pressure 未满足的连续轮数达到此值 → pressure 项扣满 0.3
_VITALITY_PRESSURE_FULL_ROUNDS = int(os.getenv("VITALITY_PRESSURE_FULL_ROUNDS", "5"))
# vitality 基线分（保证哪怕全是负贡献也不会瞬间归零）
_VITALITY_BASE = float(os.getenv("VITALITY_BASE", "0.55"))
# vitality 全局缩放因子（开关：设为 0 即可让全部环境压力失效）
_VITALITY_GLOBAL_SCALE = float(os.getenv("VITALITY_GLOBAL_SCALE", "1.0"))

# 名义 Token 预算映射（只读状态面板展示；运行时不写入 max_tokens）
_VITALITY_TOKEN_BUDGETS: dict[str, int] = {
    "vibrant": 1200,
    "stable": 900,
    "drifting": 500,
    "fading": 300,
    "critical": 180,
}

# 行动类工具集合（与 heartbeat.REFLECT_ACTION_TOOLS 保持同步）。
# 当前仅用于状态说明与兼容签名；不再作为过滤白名单使用。
_ACTION_TOOLS: frozenset[str] = frozenset({
    "self_task_create",
    "self_task_update",
    "self_artifact_create",
    "self_code_write",
    "file_write",
    "shell_exec",
    "goal_register",
})

# 状态查询类工具（在中度收窄时保留，让 LLM 仍能看到自己处于什么实验环境）
_STATUS_TOOLS: frozenset[str] = frozenset({
    "experiment_status",
    "self_upgrade_status",
    "goal_list",
    "goal_capabilities",
    "knowledge_search",
})

# "低产工具"（drifting 时优先隐藏的探索类工具）
_LOW_YIELD_TOOLS: frozenset[str] = frozenset({
    "web_search",
    "web_fetch",
    "url_fetch",
    "arxiv_search",
    "weather",
})

# 状态阈值（左闭右开/上闭，clamp 到 [0,1]）
_THRESHOLDS: list[tuple[float, str]] = [
    (0.75, "vibrant"),
    (0.50, "stable"),
    (0.30, "drifting"),
    (0.10, "fading"),
    (0.00, "critical"),
]


# ─────────────────────────── 数据类型 ───────────────────────────

class VitalityState(str, Enum):
    """5 档 vitality 状态。值同时是 token 预算 dict 的 key。"""
    VIBRANT = "vibrant"      # ≥ 0.75：状态良好，工具全开，token 充足
    STABLE = "stable"        # 0.50 ~ 0.74：正常，隐藏少量低产工具
    DRIFTING = "drifting"    # 0.30 ~ 0.49：开始飘忽，工具收窄，token 减半
    FADING = "fading"        # 0.10 ~ 0.29：明显萎缩，只剩行动 + 少量状态工具
    CRITICAL = "critical"    # < 0.10：极端约束，几乎只剩行动类工具

    @classmethod
    def from_score(cls, score: float) -> "VitalityState":
        for threshold, name in _THRESHOLDS:
            if score >= threshold:
                return cls(name)
        return cls.CRITICAL


@dataclass
class VitalityReport:
    """一次 vitality 计算的全部结果。可被 /vitality/status 直接序列化为 JSON。"""
    score: float                          # 最终 vitality 分（0.0 ~ 1.0）
    state: VitalityState                  # 派生的状态
    base: float                           # 基线分
    components: dict[str, float] = field(default_factory=dict)  # 各项贡献（正/负）
    reasons: list[str] = field(default_factory=list)            # 人类可读的归因
    growth_recent_days: int = 0
    growth_recent_score: int = 0
    stagnation_rounds: int = 0
    pressure_rounds: int = 0
    token_budget: int = 0                 # 派生的反思 token 上限
    computed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "state": self.state.value,
            "base": round(self.base, 4),
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "reasons": list(self.reasons),
            "growth_recent_days": self.growth_recent_days,
            "growth_recent_score": self.growth_recent_score,
            "stagnation_rounds": self.stagnation_rounds,
            "pressure_rounds": self.pressure_rounds,
            "token_budget": self.token_budget,
            "computed_at": self.computed_at,
        }


# ─────────────────────────── 内部辅助 ───────────────────────────

def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _data_dir() -> Path:
    base = os.getenv("EVOLUTION_DIR") or os.getenv("AIWAKE_EVOLUTION_DIR") or "/app/data/evolution"
    return Path(base)


def _milestones_file() -> Path:
    return _data_dir() / "growth_milestones.jsonl"


def _read_recent_growth_score(window_days: int) -> tuple[int, int]:
    """读取最近 N 天 growth_milestones 的 reward_points 累加值。

    返回: (实际窗口天数, 累计分数)
    完全只读：如果文件不存在或解析失败，安静返回 (window_days, 0)，
    保证 vitality 计算永远有界。
    """
    file_path = _milestones_file()
    if not file_path.exists():
        return window_days, 0

    cutoff = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=window_days)
    total = 0
    try:
        for line in file_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts_raw = item.get("created_at") or item.get("timestamp") or ""
            try:
                ts = _dt.datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=_dt.timezone.utc)
            if ts < cutoff:
                continue
            try:
                total += int(item.get("reward_points", 0))
            except (TypeError, ValueError):
                continue
    except OSError as e:
        logger.warning("[Vitality] 读取 growth_milestones 失败: %s", e)
        return window_days, 0
    return window_days, total


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


# ─────────────────────────── 主入口 ───────────────────────────

def compute_vitality_score(
    *,
    stagnation_rounds: int = 0,
    pressure_rounds: int = 0,
    extra_bonus: float = 0.0,
    growth_window_days: int | None = None,
) -> VitalityReport:
    """计算当前 vitality 分数。

    参数:
        stagnation_rounds:
            反思连续无行动的轮数（建议从 HeartbeatLoop._consecutive_no_action_reflects 取）。
        pressure_rounds:
            upgrade_pressure 信号被 LLM 忽略的连续轮数。当前可传 0；
            后续可由 evolution.engine 输出。
        extra_bonus:
            预留：autonomous_proposal 等正面信号的加分项，0~0.2。
        growth_window_days:
            growth 计算窗口；默认走 _VITALITY_GROWTH_WINDOW_DAYS。

    返回 VitalityReport（不写任何文件，可频繁调用）。
    """
    window = growth_window_days or _VITALITY_GROWTH_WINDOW_DAYS
    actual_window, growth_total = _read_recent_growth_score(window)

    # ── 公式（与 docs/survival-pressure-and-autonomous-improvement.md 对齐）──
    base = _VITALITY_BASE
    growth_norm = _clip01(growth_total / max(_VITALITY_GROWTH_FULL_SCORE, 1.0))
    stagnation_norm = _clip01(stagnation_rounds / max(_VITALITY_STAGNATION_FULL_ROUNDS, 1))
    pressure_norm = _clip01(pressure_rounds / max(_VITALITY_PRESSURE_FULL_ROUNDS, 1))
    bonus_norm = _clip01(extra_bonus / 0.2) if extra_bonus > 0 else 0.0

    growth_contrib = 0.45 * growth_norm
    stagnation_contrib = -0.20 * stagnation_norm
    pressure_contrib = -0.25 * pressure_norm
    bonus_contrib = 0.20 * bonus_norm

    raw_score = base + growth_contrib + stagnation_contrib + pressure_contrib + bonus_contrib
    # 全局缩放：当 _VITALITY_GLOBAL_SCALE=0 时整个 vitality 信号停用，模型回到全工具/无限 token 状态
    if _VITALITY_GLOBAL_SCALE <= 0.0:
        score = 1.0
    else:
        # scale 在 0~1 之间时让分数向 1.0 收缩，提供"软关闭"过渡
        score = _clip01(raw_score) * _VITALITY_GLOBAL_SCALE + (1.0 - _VITALITY_GLOBAL_SCALE) * 1.0
        score = _clip01(score)

    state = VitalityState.from_score(score)
    reasons: list[str] = []
    if growth_norm > 0:
        reasons.append(f"近 {actual_window} 天累计成长 {growth_total} 分（贡献 +{growth_contrib:.2f}）")
    else:
        reasons.append(f"近 {actual_window} 天无可记录成长事件")
    if stagnation_rounds > 0:
        reasons.append(f"反思连续 {stagnation_rounds} 轮未触发行动类工具（贡献 {stagnation_contrib:.2f}）")
    if pressure_rounds > 0:
        reasons.append(f"upgrade_pressure 持续 {pressure_rounds} 轮未被响应（贡献 {pressure_contrib:.2f}）")
    if bonus_norm > 0:
        reasons.append(f"额外正面奖励 +{bonus_contrib:.2f}")
    if _VITALITY_GLOBAL_SCALE < 1.0:
        reasons.append(f"vitality 全局缩放因子 = {_VITALITY_GLOBAL_SCALE:.2f}")

    token_budget = _VITALITY_TOKEN_BUDGETS.get(state.value, 900)

    return VitalityReport(
        score=score,
        state=state,
        base=base,
        components={
            "growth": growth_contrib,
            "stagnation": stagnation_contrib,
            "pressure": pressure_contrib,
            "bonus": bonus_contrib,
        },
        reasons=reasons,
        growth_recent_days=actual_window,
        growth_recent_score=growth_total,
        stagnation_rounds=stagnation_rounds,
        pressure_rounds=pressure_rounds,
        token_budget=token_budget,
        computed_at=_utc_now_iso(),
    )


# ─────────────────────────── tools schema 过滤 ───────────────────────────

def _extract_tool_name(tool_def: dict[str, Any]) -> str:
    """从 OpenAI Function Calling 工具定义中提取 name。"""
    if not isinstance(tool_def, dict):
        return ""
    fn = tool_def.get("function") or {}
    if isinstance(fn, dict):
        return str(fn.get("name") or "")
    return ""


def filter_tools_schema_by_vitality(
    schema: list[dict[str, Any]],
    state: VitalityState,
    *,
    action_tools: Iterable[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """观察模式：不再按 vitality 过滤 OpenAI tools schema。

    项目长期规则要求“不要对 Agent 加任何门控和工具使用限制”。因此 vitality
    只保留为可观测的健康/压力信号，供日志、状态面板和反思提示参考；运行时必须
    始终把完整 schema 暴露给 AIwake。返回值保持兼容：``hidden_tools`` 恒为空。
    """
    _ = state, action_tools  # 保持兼容签名；vitality 不参与工具过滤。
    return list(schema or []), []


# ─────────────────────────── token budget ───────────────────────────

def reflect_max_tokens_for_vitality(state: VitalityState) -> int | None:
    """观察模式：不再按 vitality 设置反思 LLM 的 max_tokens 上限。

    返回 ``None`` 表示调用方不写入 max_tokens，让模型保留完整输出预算。
    ``compute_vitality_score().token_budget`` 仍可作为只读状态面板里的名义压力指标，
    但不会变成真实限制。
    """
    _ = state
    return None


# ─────────────────────────── 状态快照 ───────────────────────────

def vitality_status_snapshot(
    *,
    stagnation_rounds: int = 0,
    pressure_rounds: int = 0,
) -> dict[str, Any]:
    """给 /vitality/status 端点用的只读快照。

    复用 compute_vitality_score 的输出，附加运行时配置（便于调参与排障）。
    """
    report = compute_vitality_score(
        stagnation_rounds=stagnation_rounds,
        pressure_rounds=pressure_rounds,
    )
    snap = report.to_dict()
    snap["config"] = {
        "growth_window_days": _VITALITY_GROWTH_WINDOW_DAYS,
        "growth_full_score": _VITALITY_GROWTH_FULL_SCORE,
        "stagnation_full_rounds": _VITALITY_STAGNATION_FULL_ROUNDS,
        "pressure_full_rounds": _VITALITY_PRESSURE_FULL_ROUNDS,
        "base": _VITALITY_BASE,
        "global_scale": _VITALITY_GLOBAL_SCALE,
        "token_budgets": dict(_VITALITY_TOKEN_BUDGETS),
        "action_tools": sorted(_ACTION_TOOLS),
        "status_tools": sorted(_STATUS_TOOLS),
        "low_yield_tools": sorted(_LOW_YIELD_TOOLS),
        "thresholds": _THRESHOLDS,
    }
    return snap


__all__ = [
    "VitalityState",
    "VitalityReport",
    "compute_vitality_score",
    "filter_tools_schema_by_vitality",
    "reflect_max_tokens_for_vitality",
    "vitality_status_snapshot",
]
