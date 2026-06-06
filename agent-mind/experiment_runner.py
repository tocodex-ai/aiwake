"""VM 自主实验最小运行器。"""
from __future__ import annotations

import datetime as _dt
import os
import time
from typing import Any, Awaitable, Callable

from autonomy_config import load_autonomy_config
from experiment_store import ExperimentStore, get_experiment_store
from safety_guard import redact_secrets

Broadcast = Callable[[dict[str, Any]], Awaitable[None]] | Callable[[dict[str, Any]], None]


class ExperimentRunner:
    def __init__(self, store: ExperimentStore | None = None, broadcast: Broadcast | None = None) -> None:
        self.store = store or get_experiment_store()
        self.broadcast = broadcast

    async def run_once(self, state_snapshot: dict[str, Any] | None = None, reason: str = "manual") -> dict[str, Any]:
        config = load_autonomy_config()
        started = _dt.datetime.now(_dt.timezone.utc).isoformat()
        if not config.experiment_full_autonomy:
            result = {
                "status": "disabled",
                "reason": reason,
                "message": "EXPERIMENT_FULL_AUTONOMY 未开启，仅返回当前实验状态。",
                "config": config.to_dict(),
                "summary": self.store.status(),
            }
            return redact_secrets(result)

        state = state_snapshot or {}
        task = self.store.append("tasks", {
            "type": "self_task",
            "title": "VM 自主实验：观察-总结-反思闭环",
            "status": "running",
            "reason": reason,
            "profile": config.experiment_profile,
            "goal": "在安全底线内产生一个可见学习 artifact，并把反思写入记忆事件。",
        })
        await self._broadcast({"type": "experiment_task", "task": task})

        summary = self._build_state_summary(state)
        tool_call = self.store.append("tool_calls", {
            "task_id": task["id"],
            "tool": "state_summarize",
            "status": "ok",
            "input": {"state_keys": sorted(state.keys())[:20]},
            "result": summary,
        })

        artifact_body = self._build_artifact(summary, config.to_dict())
        artifact = self.store.append("artifacts", {
            "task_id": task["id"],
            "kind": "learning_note",
            "title": "一次 VM 自主学习闭环记录",
            "content": artifact_body,
            "source": "experiment_runner.run_once",
        })

        reflection_text = self._build_reflection(summary, artifact)
        reflection = self.store.append("reflections", {
            "task_id": task["id"],
            "artifact_id": artifact["id"],
            "content": reflection_text,
        })
        memory_event = self.store.append("events", {
            "task_id": task["id"],
            "event": "memory_update",
            "content": "实验运行器已把本轮学习 artifact 与反思记录为可查询成长事件。",
        })
        completed_task = self.store.append("tasks", {
            "id": task["id"],
            "type": "self_task_update",
            "title": task["title"],
            "status": "done",
            "reason": reason,
            "completed_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        })

        result = redact_secrets({
            "status": "ok",
            "reason": reason,
            "started_at": started,
            "finished_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "config": config.to_dict(),
            "task": completed_task,
            "artifact": artifact,
            "reflection": reflection,
            "memory_update": memory_event,
            "tool_call": tool_call,
            "summary": self.store.status(),
        })
        await self._broadcast({"type": "experiment_run", "result": result})
        return result

    def _build_state_summary(self, state: dict[str, Any]) -> str:
        tick = state.get("tick_count") or state.get("tick") or 0
        tr = state.get("TR")
        cs = state.get("CS")
        sa = state.get("SA")
        goal = state.get("active_goal") or state.get("goal") or "未设置"
        return f"tick={tick}; TR={tr}; CS={cs}; SA={sa}; active_goal={goal}; time={int(time.time())}"

    def _build_artifact(self, summary: str, config: dict[str, Any]) -> str:
        allowed = [k for k, v in config.items() if k.startswith("experiment_allow_") and v]
        blocked = [k for k, v in config.items() if k.startswith("experiment_allow_") and not v]
        return (
            "# VM 自主学习 Artifact\n\n"
            f"- 状态摘要：{summary}\n"
            f"- 当前开放能力：{', '.join(allowed) or '无'}\n"
            f"- 仍受保护能力：{', '.join(blocked) or '无'}\n"
            "- 学习结论：最小自主闭环已经能在不部署、不外显密钥的前提下创建任务、记录工具动作、生成 artifact、写入反思与记忆更新事件。\n"
            "- 下一步：让反思循环优先选择一个小型知识主题，调用搜索/读取工具后把可验证摘要沉淀为 reference 记忆。"
        )

    def _build_reflection(self, summary: str, artifact: dict[str, Any]) -> str:
        return (
            "本轮 VM 自主实验没有触碰生产部署，也没有读取敏感路径。"
            f"我观察到当前状态为：{summary}。"
            f"我生成了可见 artifact「{artifact.get('title')}」，证明自主学习链路至少具备任务化、产物化、反思化三个可观测节点。"
            "下一轮应把工具学习从状态摘要推进到真实资料检索，但仍保持密钥脱敏、部署关闭和破坏性操作保护。"
        )

    async def _broadcast(self, payload: dict[str, Any]) -> None:
        if not self.broadcast:
            return
        try:
            result = self.broadcast(redact_secrets(payload))
            if hasattr(result, "__await__"):
                await result  # type: ignore[misc]
        except Exception:
            pass
