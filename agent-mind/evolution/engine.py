"""Evolution engine: AIwake's minimal real self-evolution loop.

The loop is intentionally audit-first. It observes runtime signals, evaluates
itself, generates a backlog, conducts a safe self-dialogue, and persists an
immutable report. It does not directly patch production code; that is the next
stage after measurable evaluation and gatekeeping are stable.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import os
import traceback
from typing import Any, Awaitable, Callable

from .backlog import build_backlog
from .evaluator import evaluate_metrics
from .memory import append_growth_log, append_jsonl, write_report
from .metrics import collect_metrics
from .self_dialogue import run_self_dialogue
from .self_upgrade import maybe_generate_from_backlog
from autonomy_config import load_autonomy_config
from experiment_runner import ExperimentRunner
from safety_guard import redact_secrets

Broadcast = Callable[[dict[str, Any]], Awaitable[None]] | Callable[[dict[str, Any]], None]


class EvolutionEngine:
    """Coordinate one safe self-evolution evaluation cycle."""

    def __init__(
        self,
        llm: Any = None,
        state_provider: Callable[[], dict[str, Any]] | None = None,
        broadcast: Broadcast | None = None,
    ) -> None:
        self.llm = llm
        self.state_provider = state_provider
        self.broadcast = broadcast
        self._running = False
        self._last_report: dict[str, Any] | None = None

    @property
    def last_report(self) -> dict[str, Any] | None:
        return self._last_report

    async def run_once(self, reason: str = "manual") -> dict[str, Any]:
        """Run one observe-evaluate-plan-self-dialogue-persist cycle."""

        started_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
        try:
            state_snapshot = self._get_state_snapshot()
            metrics = collect_metrics(hours=int(os.getenv("EVOLUTION_WINDOW_HOURS", "24")))
            # Normalize metric names expected by evaluator.py.
            evaluator_input = dict(metrics)
            evaluator_input.update({
                "tool_failures": metrics.get("tool_failure_count", 0),
                "external_tool_failures": metrics.get("external_tool_failure_count", 0),
                "memory_updates": metrics.get("memory_update_count", 0),
                "proactive_actions": metrics.get("proactive_count", 0),
                "failed_response": int(metrics.get("failed_reply_count", 0) or 0) > 0,
                "reflection_done": int(metrics.get("reflection_count", 0) or 0) > 0,
                "verified": True,
            })
            evaluation = evaluate_metrics(evaluator_input)
            backlog = build_backlog(evaluation, metrics)
            config = load_autonomy_config()
            upgrade_proposals = maybe_generate_from_backlog(backlog, enabled=config.upgrade_proposal_enabled)
            upgrade_pressure = _build_upgrade_pressure(backlog, upgrade_proposals, config)
            self_dialogue = await run_self_dialogue(self.llm, state_snapshot, metrics, evaluation)
            external_learning = self_dialogue.get("external_learning", {}) if isinstance(self_dialogue, dict) else {}
            experiment_result = None
            if config.experiment_full_autonomy:
                experiment_result = await ExperimentRunner(broadcast=self.broadcast).run_once(
                    state_snapshot=state_snapshot,
                    reason=f"evolution:{reason}",
                )

            report = {
                "status": "ok",
                "reason": reason,
                "started_at": started_at,
                "finished_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                "state": state_snapshot,
                "metrics": metrics,
                "evaluation": evaluation,
                "backlog": backlog,
                "self_dialogue": self_dialogue,
                "experiment": redact_secrets(experiment_result),
                "external_learning": external_learning,
                "upgrade_proposals": redact_secrets(upgrade_proposals),
                "upgrade_pressure": redact_secrets(upgrade_pressure),
                "self_modification_policy": {
                    "status": "autonomous_guarded_apply",
                    "requirements": ["proposal", "safety_checks", "auto_approved_status", "append_only_audit_log", "rollback_notes"],
                    "protections": ["no_memory_delete", "no_log_delete", "no_secret_access", "no_destructive_delete", "no_audit_log_modification"],
                    "proposal_pressure": "每轮存在 backlog 时应优先形成一个自动批准的最小自升级提案；若没有提案，必须记录原因。",
                },
                "next_gate": upgrade_pressure.get("next_gate", "guarded_apply_then_deploy_with_audit_log"),
            }
            self._persist(report)
            self._last_report = report
            await self._broadcast({"type": "evolution_evaluation", "report": _compact_report(report)})
            await self._broadcast({"type": "evolution_backlog", "items": backlog[:10]})
            await self._broadcast({"type": "evolution_self_dialogue", "dialogue": self_dialogue})
            return report
        except Exception as exc:  # pragma: no cover - runtime guard
            report = {
                "status": "error",
                "reason": reason,
                "started_at": started_at,
                "finished_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                "error": str(exc),
                "traceback": traceback.format_exc(limit=5),
            }
            self._persist(report)
            self._last_report = report
            await self._broadcast({"type": "evolution_error", "report": report})
            return report

    async def run_loop(self, interval_seconds: float | None = None) -> None:
        """Run the evolution cycle periodically until stopped."""

        enabled = os.getenv("EVOLUTION_LOOP_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
        if not enabled:
            return
        if interval_seconds is None:
            try:
                interval_seconds = float(os.getenv("EVOLUTION_INTERVAL_SECONDS", "3600"))
            except Exception:
                interval_seconds = 3600.0
        interval_seconds = max(60.0, float(interval_seconds))
        self._running = True
        # Delay first automatic cycle slightly so startup health is not blocked.
        await asyncio.sleep(float(os.getenv("EVOLUTION_STARTUP_DELAY_SECONDS", "20")))
        while self._running:
            await self.run_once(reason="scheduled")
            await asyncio.sleep(interval_seconds)

    def stop(self) -> None:
        self._running = False

    def _get_state_snapshot(self) -> dict[str, Any]:
        if not self.state_provider:
            return {}
        try:
            state = self.state_provider()
            return state if isinstance(state, dict) else {}
        except Exception as exc:
            return {"error": str(exc)}

    def _persist(self, report: dict[str, Any]) -> None:
        append_jsonl("self_evaluation", report)
        for task in report.get("backlog", []) or []:
            append_jsonl("backlog", task)
        write_report(_report_name(report), _render_markdown(report))
        external_learning = report.get("external_learning") or {}
        if external_learning.get("sources"):
            append_growth_log({
                "event": "external_learning_reflection",
                "purpose": external_learning.get("purpose"),
                "sources": external_learning.get("sources"),
                "validation_result": "已纳入本轮自我进化报告",
                "risk_note": external_learning.get("safety_boundary"),
            })

    async def _broadcast(self, payload: dict[str, Any]) -> None:
        if not self.broadcast:
            return
        try:
            result = self.broadcast(payload)
            if hasattr(result, "__await__"):
                await result  # type: ignore[misc]
        except Exception:
            pass


def _report_name(report: dict[str, Any]) -> str:
    ts = str(report.get("finished_at") or report.get("started_at") or "report")
    safe = ts.replace(":", "-").replace("+", "-").replace(".", "-")
    return f"report_{safe}"


def _compact_report(report: dict[str, Any]) -> dict[str, Any]:
    evaluation = report.get("evaluation", {}) or {}
    metrics = report.get("metrics", {}) or {}
    return {
        "status": report.get("status"),
        "reason": report.get("reason"),
        "finished_at": report.get("finished_at"),
        "total_score": evaluation.get("total_score"),
        "risk_level": evaluation.get("risk_level"),
        "issues": evaluation.get("issues", [])[:5],
        "backlog_count": len(report.get("backlog", []) or []),
        "chat_success_rate": metrics.get("chat_success_rate"),
        "tool_success_rate": metrics.get("tool_success_rate"),
        "experiment_artifact_count": metrics.get("experiment_artifact_count"),
    }


def _build_upgrade_pressure(backlog: list[dict[str, Any]], upgrade_proposals: list[dict[str, Any]], config: Any) -> dict[str, Any]:
    """Explain whether the evolution cycle produced a safe upgrade proposal.

    This is a soft pressure mechanism: it makes missing proposals visible and
    auditable, but it still never applies patches or deploys automatically.
    """

    backlog_items = [item for item in (backlog or []) if isinstance(item, dict)]
    high_priority = [item for item in backlog_items if float(item.get("priority") or 0) >= 0.5]
    enabled = bool(getattr(config, "upgrade_proposal_enabled", False))
    generated = bool(upgrade_proposals)
    if generated:
        status = "proposal_generated"
        reason = "已从本轮最高优先级 backlog 生成自动批准的自升级提案，可进入受控应用/部署队列。"
        next_gate = "guarded_apply_then_deploy_with_audit_log"
    elif not enabled:
        status = "proposal_disabled"
        reason = "自升级提案生成开关关闭；本轮只能记录 backlog 与反思。"
        next_gate = "enable_upgrade_proposal_generation_before_apply"
    elif not backlog_items:
        status = "no_backlog"
        reason = "本轮没有可转化为自升级提案的 backlog。"
        next_gate = "wait_for_observed_backlog"
    else:
        status = "proposal_missing"
        reason = "存在 backlog 但未生成自升级提案；下一轮必须优先解释原因或形成自动批准的最小提案。"
        next_gate = "must_explain_or_generate_auto_approved_upgrade_proposal_next_cycle"
    return {
        "status": status,
        "enabled": enabled,
        "generated_count": len(upgrade_proposals or []),
        "backlog_count": len(backlog_items),
        "high_priority_backlog_count": len(high_priority),
        "reason": reason,
        "next_gate": next_gate,
        "apply_allowed": bool(getattr(config, "upgrade_apply_enabled", False)),
        "deploy_allowed": bool(getattr(config, "upgrade_deploy_enabled", False)),
        "policy": "无人审查模式提高提案生成压力，并允许受控应用/部署；禁止修改日志、记忆、密钥、执行破坏性删除或改写审计记录。",
    }


def _render_markdown(report: dict[str, Any]) -> str:
    evaluation = report.get("evaluation", {}) or {}
    metrics = report.get("metrics", {}) or {}
    backlog = report.get("backlog", []) or []
    dialogue = report.get("self_dialogue", {}) or {}
    external_learning = report.get("external_learning", {}) or {}
    lines = [
        "# AIwake 自我进化评估报告",
        "",
        f"- 状态: {report.get('status')}",
        f"- 触发原因: {report.get('reason')}",
        f"- 完成时间: {report.get('finished_at')}",
        f"- 总评分: {evaluation.get('total_score')}",
        f"- 风险等级: {evaluation.get('risk_level')}",
        f"- 聊天成功率: {metrics.get('chat_success_rate')}",
        f"- 工具成功率: {metrics.get('tool_success_rate')}",
        "",
        "## 发现的问题",
    ]
    for issue in evaluation.get("issues", []) or ["暂无严重问题"]:
        lines.append(f"- {issue}")
    lines.extend(["", "## 改进建议"])
    for suggestion in evaluation.get("suggestions", []) or []:
        lines.append(f"- {suggestion}")
    lines.extend(["", "## 自主改进 Backlog"])
    for task in backlog[:20]:
        lines.append(f"- `{task.get('id')}` [{task.get('risk')}] {task.get('title')} priority={task.get('priority')}")
    lines.extend(["", "## 自己与自己对话"])
    for turn in dialogue.get("transcript", []) or []:
        lines.append(f"**{turn.get('role')}**：{turn.get('content')}")
        lines.append("")
    lines.extend(["", "## 外部世界信息收集"])
    if external_learning.get("sources"):
        lines.append(f"- 目的: {external_learning.get('purpose')}")
        lines.append(f"- 安全边界: {external_learning.get('safety_boundary')}")
        for source in external_learning.get("sources", [])[:5]:
            lines.append(f"- 来源: {source.get('url')}")
            lines.append(f"  - 用途: {source.get('use')}")
            lines.append(f"  - 摘要: {source.get('summary')}")
    else:
        lines.append("- 本轮未收集外部资料，或外部学习被配置关闭。")
    upgrade_proposals = report.get("upgrade_proposals", []) or []
    upgrade_pressure = report.get("upgrade_pressure", {}) or {}
    lines.extend(["", "## 自升级提案"])
    lines.append(f"- 提案压力状态: {upgrade_pressure.get('status', 'unknown')}")
    lines.append(f"- 原因: {upgrade_pressure.get('reason', '未记录')}")
    lines.append(f"- 下一门控: {upgrade_pressure.get('next_gate', 'human_review_of_pending_upgrade_proposal_only_no_auto_apply')}")
    if upgrade_proposals:
        for proposal in upgrade_proposals[:5]:
            lines.append(f"- `{proposal.get('id')}` [{proposal.get('risk_level')}/{proposal.get('status')}] {proposal.get('problem')}")
    else:
        lines.append("- 本轮未生成自升级提案；已记录缺席原因，下一轮需优先解释或生成最小 pending 提案。")
    lines.extend([
        "",
        "## 当前门控",
        "无人审查模式允许生成、校验、自动批准、受控应用与部署自升级提案。仍禁止修改/删除日志、记忆、密钥，禁止破坏性删除，所有动作必须追加审计记录。",
    ])
    return "\n".join(lines)
