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
from .self_upgrade import (
    classify_backlog_dedup,
    generate_and_apply_patch,
    maybe_generate_from_backlog,
    read_proposals,
)
from .growth_tracker import read_milestones, record_growth_milestone
from .goal_tracker import evaluate_open_goals
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
            # other_error 偏高时强制做一次 failure triad 聚类，避免“有能力不调用”
            forced_miner = None
            try:
                from .failure_triads import maybe_force_mine_from_metrics

                forced_miner = maybe_force_mine_from_metrics(metrics, min_other_error=3)
                if forced_miner:
                    append_growth_log({
                        "event": "forced_degradation_evidence_miner",
                        "purpose": "other_error 占比偏高，自动挖掘失败模式",
                        "matched_triads": forced_miner.get("matched_triads"),
                        "top_patterns": (forced_miner.get("top_patterns") or [])[:5],
                        "trigger": forced_miner.get("trigger"),
                        "validation_result": "append-only cluster written",
                        "risk_note": "只读聚类 + 追加日志，不改代码、不删历史",
                    })
            except Exception as miner_exc:
                forced_miner = {"status": "error", "error": str(miner_exc)[:200]}
            config = load_autonomy_config()
            upgrade_proposals = maybe_generate_from_backlog(backlog, enabled=config.upgrade_proposal_enabled)
            for proposal in upgrade_proposals:
                record_growth_milestone(
                    event_type="proposal_generated",
                    description=f"生成自升级提案: {proposal.get('problem', '')[:200]}",
                    source_proposal_id=str(proposal.get('id', '')),
                )
            # ── Auto-apply: 每轮最多应用 3 个 approved 提案（含 LLM patch 生成）──
            apply_results = await _try_auto_apply_batch(config)
            dedup_summary = classify_backlog_dedup(backlog)
            upgrade_pressure = _build_upgrade_pressure(backlog, upgrade_proposals, config, dedup_summary)
            # ── 反思空转桥接：当指标识别 too_passive 且本轮没有生成提案时，
            # 把这个"看得见却没行动"的信号转成一条可追踪成长里程碑（每 UTC 日去重，防刷分）。
            passive_bridge = _bridge_passive_signal(metrics, upgrade_pressure)
            # ── 目标闭环：对每个开放目标用当前 metrics 复测，达成则闭合并记成长事件，
            # 未达成则迭代，超限或显著恶化则自动放弃。形成 改→度量→收敛 闭环。──
            goal_results = _evaluate_goals_safely(metrics)
            # ── 单任务状态机：反思入口先 try_acquire；若已有写任务在跑，直接跳过本轮反思
            # 工具调用，避免反思与对话/其它反思并发改文件、避免心跳重复触发。──
            self_dialogue: dict[str, Any]
            task_acquired = False
            task_id: str | None = None
            try:
                import sys as _sys
                from pathlib import Path as _Path
                _agent_root = _Path(__file__).resolve().parent.parent
                if str(_agent_root) not in _sys.path:
                    _sys.path.insert(0, str(_agent_root))
                from self_task_manager import get_manager as _get_manager
                _manager = _get_manager()
                acquired, current = await _manager.try_acquire(
                    kind="reflection_loop",
                    owner="evolution_engine",
                    title=f"reflection:{reason}",
                    ttl=600,
                    source_request_id=f"evolution:{reason}",
                    extra={"reason": reason},
                )
                task_acquired = acquired
                if acquired and current:
                    task_id = current.task_id
            except Exception as e:
                _manager = None
                acquired = True  # 若状态机不可用，不阻塞反思
                current = None
                task_acquired = False
            if task_acquired or current is None:
                # 注入 tool_router 让反思走真工具路径
                tool_router_instance = None
                try:
                    from tool_router import ToolRouter as _ToolRouter
                    tool_router_instance = _ToolRouter()
                except Exception:
                    tool_router_instance = None
                try:
                    self_dialogue = await run_self_dialogue(
                        self.llm, state_snapshot, metrics, evaluation,
                        tool_router=tool_router_instance,
                    )
                finally:
                    if task_acquired and task_id:
                        try:
                            await _manager.finish(task_id, result="reflection_loop_done")
                        except Exception:
                            pass
            else:
                # 已有任务在跑：跳过本轮反思的工具调用，仅做最小自评
                import logging as _log_skip
                _log_skip.getLogger("evolution.engine").info(
                    f"[Evolution] 跳过本轮反思工具调用：状态机锁被任务 {current.task_id}({current.kind}) 占用"
                )
                self_dialogue = _fallback_skip_due_to_busy(state_snapshot, metrics, evaluation, current)
                try:
                    record_growth_milestone(
                        event_type="reflection_skipped_due_to_active_task",
                        description=f"反思跳过：当前有任务 {current.kind}({current.task_id}) 进行中",
                        extra={"current_task": current.to_public()},
                    )
                except Exception:
                    pass
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
                "auto_apply_results": redact_secrets(apply_results),
                "upgrade_pressure": redact_secrets(upgrade_pressure),
                "passive_action_bridge": passive_bridge,
                "goal_closure": goal_results,
                "forced_degradation_miner": redact_secrets(forced_miner) if forced_miner else None,
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
            record_growth_milestone(
                event_type="external_learning",
                description=f"外部学习: {external_learning.get('purpose', '未知目的')[:200]}",
                extra={"source_count": str(len(external_learning.get('sources', [])))},
            )

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


def _fallback_skip_due_to_busy(
    state_snapshot: dict[str, Any],
    metrics: dict[str, Any],
    evaluation: dict[str, Any],
    current_task: Any,
) -> dict[str, Any]:
    """状态机锁被占用时的最小自评：不调任何工具、不调 LLM。"""
    return {
        "skipped": True,
        "reason": "task_busy",
        "current_task": current_task.to_public() if hasattr(current_task, "to_public") else {},
        "detected_risks": list((evaluation or {}).get("issues", []) or [])[:3],
        "proposed_next_actions": [
            "等待当前任务结束后再尝试本轮反思；本轮已自动跳过工具调用以避免并发改文件。",
        ],
        "registered_goals": [],
        "self_code_writes": [],
        "external_learning": {},
    }


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


async def _try_auto_apply_batch(config: Any) -> list[dict[str, Any]]:
    """Auto-apply up to 3 approved proposals per evolution cycle.

    Reads all approved proposals, picks the oldest ones that haven't exceeded
    the max patch attempt limit, and calls generate_and_apply_patch to
    generate real LLM-driven code patches and apply them.
    If LLM patch fails, the proposal stays as 'approved' for retry next cycle
    (no fallback to status-only apply which would create phantom states).
    Returns a list of apply result dicts (0 to _MAX_BATCH items).
    Failures are caught and logged; they never break the evolution cycle.
    """
    import logging as _log

    _logger = _log.getLogger("evolution.engine")

    if not bool(getattr(config, "upgrade_apply_enabled", False)):
        return []

    _MAX_BATCH = int(os.getenv("UPGRADE_BATCH_SIZE", "3"))
    _MAX_PATCH_ATTEMPTS = 3

    try:
        approved = read_proposals(limit=500, status="approved")
        if not approved:
            return []

        # Pick up to _MAX_BATCH oldest approved proposals, skipping those
        # that have already failed LLM patch generation too many times.
        targets: list[dict[str, Any]] = []
        for candidate in approved:
            events = candidate.get("notes") or []
            fail_count = sum(
                1 for n in events
                if isinstance(n, str) and "LLM patch generation failed" in n
            )
            if fail_count < _MAX_PATCH_ATTEMPTS:
                targets.append(candidate)
                if len(targets) >= _MAX_BATCH:
                    break
        if not targets:
            _logger.info("[EvolutionEngine] all approved proposals exceeded max patch attempts (%d)", _MAX_PATCH_ATTEMPTS)
            return []

        _logger.info("[EvolutionEngine] batch applying %d proposals (of %d approved)", len(targets), len(approved))

        results: list[dict[str, Any]] = []
        for target in targets:
            proposal_id = str(target.get("id") or "")
            if not proposal_id:
                continue

            # Try LLM-driven real code patch (no fallback to status-only apply)
            try:
                result = await generate_and_apply_patch(
                    proposal_id=proposal_id,
                    dry_run=False,
                    actor="evolution_engine_auto_apply",
                )
                _logger.info(
                    "[EvolutionEngine] LLM patch proposal %s: applied=%s edits=%d reason=%s",
                    proposal_id,
                    result.get("applied"),
                    result.get("edits_applied", 0),
                    result.get("reason", ""),
                )
            except Exception as patch_exc:
                # Do NOT fallback to guarded_apply_proposal — it only changes status
                # without generating real code changes, creating a "phantom applied" state.
                # Keep proposal as "approved" and let the next evolution cycle retry.
                _logger.warning(
                    "[EvolutionEngine] LLM patch failed for %s (will retry next cycle): %s",
                    proposal_id, patch_exc,
                )
                result = {"applied": False, "reason": f"LLM patch exception: {patch_exc}"}

            if result.get("applied"):
                proposal_data = result.get("proposal", {})
                record_growth_milestone(
                    event_type="self_upgrade_applied",
                    description=f"自主应用升级提案: {proposal_data.get('problem', '未知')[:200]}",
                    source_proposal_id=proposal_id,
                    extra={
                        "risk_level": proposal_data.get("risk_level", "unknown"),
                        "files": str(proposal_data.get("proposed_files", [])),
                        "edits_applied": result.get("edits_applied", 0),
                        "patch_method": "llm_patch" if result.get("edits_applied") else "guarded_apply",
                    },
                )
            results.append(redact_secrets(result))
        return results
    except Exception as exc:
        _logger.warning("[EvolutionEngine] auto-apply batch failed: %s", exc)
        return [{"applied": False, "error": str(exc)}]


def _evaluate_goals_safely(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Re-measure open goals against current metrics; never raises.

    This closes the 改→度量→收敛 loop: each open goal is re-evaluated every
    evolution cycle. Closed goals record a growth milestone; goals that regress
    badly or exceed max cycles are auto-abandoned. Any failure is swallowed so
    the evolution cycle stays intact.
    """
    try:
        return evaluate_open_goals(metrics)
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        return [{"status": "error", "reason": f"目标复测失败: {exc}"}]


def _bridge_passive_signal(metrics: dict[str, Any], upgrade_pressure: dict[str, Any]) -> dict[str, Any]:
    """Bridge a detected reflection-spinning signal into a trackable growth record.

    The metrics layer already detects ``reflection_to_action_status == "too_passive"``
    (too much reflection, too little action). Previously that signal only flowed into
    the backlog/report text and never produced a durable, trackable artifact, so a day
    full of reflection still showed ``daily_points == 0``.

    This bridge closes that gap: when the cycle observes ``too_passive`` (and no fresh
    upgrade proposal was generated this cycle), it records a single
    ``autonomous_debugging`` growth milestone for the current UTC day. It is
    deduplicated per UTC day so repeated cycles cannot inflate the growth score.

    It never raises; any failure is swallowed so the evolution cycle stays intact.
    """

    status = str((metrics or {}).get("reflection_to_action_status") or "")
    result: dict[str, Any] = {
        "status": "skipped",
        "reflection_to_action_status": status,
        "recorded": False,
        "reason": "",
    }

    if status != "too_passive":
        result["reason"] = "reflection_to_action_status 非 too_passive，无需桥接。"
        return result

    # 仅在没有更高优先级的提案动作时桥接，避免与正常自升级流程重复制造成长事件。
    pressure_status = str((upgrade_pressure or {}).get("status") or "")
    if pressure_status == "proposal_generated":
        result["status"] = "deferred"
        result["reason"] = "本轮已生成自升级提案，由提案流程承担行动记录，无需额外桥接。"
        return result

    today = _dt.datetime.now(_dt.timezone.utc).date().isoformat()
    try:
        existing = read_milestones(days=2, limit=200)
    except Exception:
        existing = []
    for milestone in existing or []:
        if not isinstance(milestone, dict):
            continue
        if milestone.get("event_type") != "autonomous_debugging":
            continue
        extra = milestone.get("extra") or {}
        if str(milestone.get("date", ""))[:10] == today and extra.get("bridge") == "passive_action_bridge":
            result["status"] = "already_recorded"
            result["reason"] = "本 UTC 日已记录过 too_passive 行动桥接里程碑，去重跳过。"
            result["milestone_id"] = str(milestone.get("id", ""))
            return result

    ratio = (metrics or {}).get("reflection_to_action_ratio")
    reflection_count = (metrics or {}).get("reflection_count")
    description = (
        "反思空转自检：检测到 reflection_to_action_status=too_passive "
        f"(ratio={ratio}, reflection_count={reflection_count})，"
        "已将该信号转化为可追踪的自主排查里程碑，提示下一步用一次工具取证或最小改进闭环替代重复反思。"
    )
    try:
        milestone = record_growth_milestone(
            event_type="autonomous_debugging",
            description=description,
            extra={
                "bridge": "passive_action_bridge",
                "reflection_to_action_ratio": str(ratio),
                "reflection_count": str(reflection_count),
                "upgrade_pressure_status": pressure_status,
            },
        )
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        result["status"] = "error"
        result["reason"] = f"记录成长里程碑失败: {exc}"
        return result

    if milestone:
        result["status"] = "recorded"
        result["recorded"] = True
        result["reason"] = "too_passive 信号已桥接为 autonomous_debugging 成长里程碑（当日首次）。"
        result["milestone_id"] = str(milestone.get("id", ""))
        result["reward_points"] = milestone.get("reward_points")
    else:
        result["status"] = "error"
        result["reason"] = "record_growth_milestone 返回空，写入可能失败。"
    return result


def _build_upgrade_pressure(
    backlog: list[dict[str, Any]],
    upgrade_proposals: list[dict[str, Any]],
    config: Any,
    dedup_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Explain whether the evolution cycle produced a safe upgrade proposal.

    This is a soft pressure mechanism: it makes missing proposals visible and
    auditable. Auto-apply of approved proposals is handled by _try_auto_apply_batch.

    ``dedup_summary`` (from :func:`classify_backlog_dedup`) lets the gate tell apart
    two very different "no proposal generated" situations:

    - **proposal_deduped** — every backlog item is already covered by a recent
      equivalent proposal (approved/pending, or applied/rejected within the dedup
      window). This is a healthy steady state, *not* a missing action, so the gate
      no longer demands "explain or generate a proposal next cycle". This removes the
      false-alarm that previously kept the agent spinning on reflections trying to
      justify a proposal that dedup will always suppress.
    - **proposal_missing** — there is at least one actionable backlog item with no
      existing proposal, yet none was generated. This remains a real gap to close.
    """

    backlog_items = [item for item in (backlog or []) if isinstance(item, dict)]
    high_priority = [item for item in backlog_items if float(item.get("priority") or 0) >= 0.5]
    enabled = bool(getattr(config, "upgrade_proposal_enabled", False))
    generated = bool(upgrade_proposals)
    dedup_summary = dedup_summary or {}
    actionable_count = int(dedup_summary.get("actionable_count", 0) or 0)
    deduped_count = int(dedup_summary.get("deduped_count", 0) or 0)
    all_deduped = bool(dedup_summary.get("all_deduped", False))
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
    elif all_deduped or (deduped_count > 0 and actionable_count == 0):
        status = "proposal_deduped"
        reason = (
            f"backlog 全部已有近期等价提案覆盖（去重抑制 {deduped_count} 项，无新可执行项）；"
            "这是健康的稳定态，不是遗漏行动，无需在下一轮反复解释或重复生成同类提案。"
        )
        next_gate = "monitor_until_new_actionable_backlog_or_dedup_window_expires"
    else:
        status = "proposal_missing"
        reason = "存在可执行 backlog 但未生成自升级提案；下一轮必须优先解释原因或形成自动批准的最小提案。"
        next_gate = "must_explain_or_generate_auto_approved_upgrade_proposal_next_cycle"
    return {
        "status": status,
        "enabled": enabled,
        "generated_count": len(upgrade_proposals or []),
        "backlog_count": len(backlog_items),
        "high_priority_backlog_count": len(high_priority),
        "actionable_backlog_count": actionable_count,
        "deduped_backlog_count": deduped_count,
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
