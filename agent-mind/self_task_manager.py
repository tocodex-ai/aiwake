"""SelfTaskManager: AIwake 单任务状态机（进程内全局互斥）。

设计目标：
- 反思 / /chat / self_upgrade_apply / experiment_loop 同一时间只能有一个 *写* 任务在跑。
- /chat 拿锁失败时不被拒绝：调用方应回退到只读模式（不调用写工具）。
- 心跳层在反思入口检查锁；锁被占用时直接跳过本轮反思，避免重复触发与 LLM 浪费。
- 任务带 ttl，超时自动释放，不存在死锁。
- 持久化到 append-only JSONL，进程重启时回放最近 24h 推断当前状态。

不依赖任何第三方库；线程安全靠 asyncio.Lock 串行化，调用方约定 await 完整过程。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# 持久化路径，与 evolution/* 数据保持同目录
_DATA_DIR = Path("/app/data") if Path("/app/data").exists() else Path(__file__).parent / "data"
_STATE_FILE = _DATA_DIR / "evolution" / "self_task_state.jsonl"

# 任务种类与默认 ttl（秒）
_DEFAULT_TTL: dict[str, int] = {
    "chat_session": 180,
    "reflection_loop": 600,
    "self_upgrade_apply": 600,
    "experiment_loop": 900,
    "admin_op": 300,
}

# 全局 ttl 上限，避免误传巨大值
_MAX_TTL = 3600


@dataclass
class SelfTask:
    task_id: str
    kind: str
    owner: str
    title: str
    started_at: float
    expires_at: float
    heartbeat_at: float
    status: str  # running|finished|aborted|timeout
    tool_calls_count: int = 0
    last_tool_name: str = ""
    source_request_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_public(self) -> dict[str, Any]:
        d = asdict(self)
        d["seconds_remaining"] = max(0, int(self.expires_at - time.time())) if self.status == "running" else 0
        return d

    def is_expired(self, now: float | None = None) -> bool:
        return (now or time.time()) > self.expires_at


class SelfTaskManager:
    """进程内全局单任务状态机。"""

    _instance: Optional["SelfTaskManager"] = None
    _instance_lock: Optional[asyncio.Lock] = None

    def __init__(self, state_file: Path | None = None):
        self._state_file: Path = state_file or _STATE_FILE
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._current: Optional[SelfTask] = None
        self._restore_from_disk()

    # ── 单例 ──
    @classmethod
    def get(cls) -> "SelfTaskManager":
        if cls._instance is None:
            cls._instance = SelfTaskManager()
        return cls._instance

    # ── 持久化 ──
    def _append(self, task: SelfTask, *, transition: str) -> None:
        try:
            with self._state_file.open("a", encoding="utf-8") as f:
                row = task.to_public()
                row["transition"] = transition
                row["recorded_at"] = time.time()
                f.write(json.dumps(row, ensure_ascii=False, default=str))
                f.write("\n")
        except OSError as e:
            logger.warning(f"[SelfTaskManager] 持久化失败: {e}")

    def _restore_from_disk(self) -> None:
        if not self._state_file.exists():
            return
        try:
            lines = self._state_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        # 找最近一条状态记录
        last_running: Optional[SelfTask] = None
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            status = row.get("status")
            if status not in ("running", "finished", "aborted", "timeout"):
                continue
            # 拿到最新一条的话直接构建 SelfTask；如果是 running 但已超时，回放为 timeout
            try:
                task = SelfTask(
                    task_id=row["task_id"],
                    kind=row["kind"],
                    owner=row.get("owner", ""),
                    title=row.get("title", ""),
                    started_at=float(row.get("started_at", 0)),
                    expires_at=float(row.get("expires_at", 0)),
                    heartbeat_at=float(row.get("heartbeat_at", 0)),
                    status=status,
                    tool_calls_count=int(row.get("tool_calls_count", 0)),
                    last_tool_name=row.get("last_tool_name", ""),
                    source_request_id=row.get("source_request_id", ""),
                    extra=row.get("extra", {}) or {},
                )
            except (KeyError, ValueError, TypeError):
                continue
            if status == "running":
                if task.is_expired():
                    task.status = "timeout"
                    self._append(task, transition="timeout_on_restore")
                else:
                    last_running = task
            break
        self._current = last_running
        if last_running:
            logger.info(f"[SelfTaskManager] 恢复进行中任务: {last_running.task_id} kind={last_running.kind}")

    # ── 公共 API ──
    async def try_acquire(
        self,
        *,
        kind: str,
        owner: str,
        title: str,
        ttl: int | None = None,
        source_request_id: str = "",
        extra: dict[str, Any] | None = None,
    ) -> tuple[bool, Optional[SelfTask]]:
        """尝试获取全局锁。

        返回：
        - (True, task)：获取成功，返回新建的 task；
        - (False, current)：失败，返回当前正在跑的任务（供调用方决定回退策略）。
        """
        async with self._lock:
            self._gc_locked()
            if self._current and self._current.status == "running":
                return False, self._current
            now = time.time()
            ttl_seconds = max(10, min(_MAX_TTL, ttl if ttl is not None else _DEFAULT_TTL.get(kind, 300)))
            task = SelfTask(
                task_id=f"task_{int(now*1000)}_{uuid.uuid4().hex[:8]}",
                kind=kind,
                owner=owner,
                title=title or kind,
                started_at=now,
                expires_at=now + ttl_seconds,
                heartbeat_at=now,
                status="running",
                source_request_id=source_request_id,
                extra=extra or {},
            )
            self._current = task
            self._append(task, transition="acquired")
            return True, task

    async def heartbeat(self, task_id: str, *, extend_ttl: int | None = None, last_tool_name: str = "") -> bool:
        """续期当前任务。"""
        async with self._lock:
            cur = self._current
            if not cur or cur.task_id != task_id or cur.status != "running":
                return False
            now = time.time()
            cur.heartbeat_at = now
            if extend_ttl:
                cur.expires_at = max(cur.expires_at, now + min(_MAX_TTL, max(10, extend_ttl)))
            if last_tool_name:
                cur.last_tool_name = last_tool_name
                cur.tool_calls_count += 1
            self._append(cur, transition="heartbeat")
            return True

    async def finish(self, task_id: str, *, result: str = "ok") -> bool:
        async with self._lock:
            cur = self._current
            if not cur or cur.task_id != task_id:
                return False
            cur.status = "finished"
            cur.extra = {**cur.extra, "result": result[:500]}
            self._append(cur, transition="finished")
            self._current = None
            return True

    async def abort(self, task_id: str, *, reason: str = "aborted") -> bool:
        async with self._lock:
            cur = self._current
            if not cur or cur.task_id != task_id:
                return False
            cur.status = "aborted"
            cur.extra = {**cur.extra, "reason": reason[:500]}
            self._append(cur, transition="aborted")
            self._current = None
            return True

    def _gc_locked(self) -> None:
        """无锁内调用：超时回收。"""
        cur = self._current
        if cur and cur.status == "running" and cur.is_expired():
            cur.status = "timeout"
            self._append(cur, transition="timeout")
            self._current = None

    async def current(self) -> Optional[dict[str, Any]]:
        async with self._lock:
            self._gc_locked()
            return self._current.to_public() if self._current else None

    async def is_busy(self) -> bool:
        async with self._lock:
            self._gc_locked()
            return self._current is not None and self._current.status == "running"


def get_manager() -> SelfTaskManager:
    """方便 sync 调用方拿到单例。"""
    return SelfTaskManager.get()
