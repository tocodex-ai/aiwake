"""VM 自主实验 JSONL 存储。"""
from __future__ import annotations

import datetime as _dt
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from safety_guard import redact_secrets

EXPERIMENT_DIR = Path(os.getenv("EXPERIMENT_DIR", "/app/data/experiments"))
_COLLECTIONS = ("tasks", "artifacts", "tool_calls", "reflections", "events")


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


class ExperimentStore:
    def __init__(self, base_dir: Path | str | None = None) -> None:
        self.base_dir = Path(base_dir) if base_dir else EXPERIMENT_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, collection: str) -> Path:
        if collection not in _COLLECTIONS:
            raise ValueError(f"unknown experiment collection: {collection}")
        return self.base_dir / f"{collection}.jsonl"

    def append(self, collection: str, item: dict[str, Any]) -> dict[str, Any]:
        clean = redact_secrets(dict(item))
        clean.setdefault("id", f"{collection[:3]}_{int(time.time())}_{uuid.uuid4().hex[:8]}")
        clean.setdefault("ts", time.time())
        clean.setdefault("created_at", _now_iso())
        path = self._path(collection)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(clean, ensure_ascii=False, default=str) + "\n")
        return clean

    def read(self, collection: str, limit: int = 100) -> list[dict[str, Any]]:
        path = self._path(collection)
        if not path.exists():
            return []
        max_items = max(1, min(int(limit or 100), 1000))
        items: list[dict[str, Any]] = []
        for line in self._read_tail_lines(path, max_items=max_items):
            try:
                item = json.loads(line)
            except Exception:
                continue
            items.append(redact_secrets(item))
        return items[-max_items:]

    def _read_tail_lines(self, path: Path, *, max_items: int, max_bytes: int = 1024 * 1024) -> list[str]:
        """Read the end of an append-only JSONL file without loading old history."""
        try:
            size = path.stat().st_size
            if size <= max_bytes:
                return path.read_text(encoding="utf-8", errors="ignore").splitlines()
            with open(path, "rb") as f:
                f.seek(max(0, size - max_bytes))
                f.readline()
                data = f.read()
            lines = data.decode("utf-8", errors="ignore").splitlines()
            return lines[-max_items * 3 :]
        except Exception:
            return []

    def _effective_tasks(self, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """按任务 id 折叠 append-only 任务事件，识别 self_task_update 的最新状态。"""
        by_id: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for item in tasks:
            task_id = str(item.get("id") or "").strip()
            if not task_id:
                continue
            if task_id not in by_id:
                by_id[task_id] = dict(item)
                order.append(task_id)
                continue
            merged = dict(by_id[task_id])
            if item.get("type") == "self_task_update":
                merged["status"] = item.get("status", merged.get("status"))
                merged["last_update"] = item
                if item.get("note"):
                    merged["note"] = item.get("note")
            else:
                merged.update(item)
            by_id[task_id] = merged
        return [by_id[task_id] for task_id in order]

    def status(self) -> dict[str, Any]:
        tasks = self.read("tasks", limit=1000)
        effective_tasks = self._effective_tasks(tasks)
        artifacts = self.read("artifacts", limit=1000)
        tool_calls = self.read("tool_calls", limit=1000)
        reflections = self.read("reflections", limit=1000)
        events = self.read("events", limit=1000)
        closed_statuses = {"done", "closed", "cancelled", "failed"}
        open_tasks = [t for t in effective_tasks if t.get("status") not in closed_statuses]
        # 为每个 open task 生成精简摘要，暴露 id/title/status/goal 供 agent 自主管理
        open_tasks_summary = [
            {
                "id": t.get("id", ""),
                "title": t.get("title", ""),
                "status": t.get("status", "open"),
                "goal": (t.get("goal") or "")[:200],
                "created_at": t.get("created_at", ""),
            }
            for t in open_tasks
        ]
        return redact_secrets({
            "task_count": len(effective_tasks),
            "task_event_count": len(tasks),
            "open_task_count": len(open_tasks),
            "open_tasks": open_tasks_summary,
            "artifact_count": len(artifacts),
            "tool_call_count": len(tool_calls),
            "reflection_count": len(reflections),
            "event_count": len(events),
            "latest_task": effective_tasks[-1] if effective_tasks else None,
            "latest_artifact": artifacts[-1] if artifacts else None,
            "latest_reflection": reflections[-1] if reflections else None,
            "next_plan": self._next_plan(effective_tasks, artifacts, reflections),
        })

    def _next_plan(self, tasks: list[dict[str, Any]], artifacts: list[dict[str, Any]], reflections: list[dict[str, Any]]) -> str:
        if not tasks:
            return "创建第一条自我学习任务，建立可观察闭环。"

        latest_task = tasks[-1]
        closed_statuses = {"done", "closed", "cancelled", "failed"}
        task_is_open = latest_task.get("status") not in closed_statuses
        latest_artifact = artifacts[-1] if artifacts else None
        artifact_matches_task = bool(latest_artifact and latest_artifact.get("task_id") == latest_task.get("id"))

        # 优先处理仍 open 的最新自任务，避免自主反思在任务已创建后又退回泛化轮询。
        # 这是公开 /experiment/status 的下一步提示，必须可由 open_task_count/latest_task 反查验证。
        if task_is_open and not artifact_matches_task:
            return (
                "最新自任务仍 open 且尚无匹配 artifact；先围绕 latest_task 做最小闭环："
                "读取任务 goal/done_when 和 daily_note，若证据仍零增量则记录早退验证或生成学习 artifact，"
                "不要直接发起新的外部搜索。"
            )
        if task_is_open and artifact_matches_task:
            return "复盘最新 artifact 是否满足任务 done_when；若满足，记录可观察闭环已达成但任务仍 open，并提出关闭或补充任务卡的下一步。"

        if latest_task.get("status") in {"done", "closed"}:
            return "最新任务已关闭；继续轮询状态，选择一个新的小问题进行搜索/总结/反思，累积可见学习成果。"
        if not reflections:
            return "围绕最新 artifact 写入反思并沉淀记忆。"
        return "继续轮询状态，选择一个小问题进行搜索/总结/反思，累积可见学习成果。"


_default_store: ExperimentStore | None = None


def get_experiment_store() -> ExperimentStore:
    global _default_store
    if _default_store is None:
        _default_store = ExperimentStore()
    return _default_store
