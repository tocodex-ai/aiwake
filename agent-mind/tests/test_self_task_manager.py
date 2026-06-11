"""SelfTaskManager 状态机核心：start/finish/abort/timeout/restore 自检。

不依赖 pytest，单文件可独立运行：

    python src/agent-mind/tests/test_self_task_manager.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path


def _find_agent_dir() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if parent.name == "agent-mind":
            return parent
        candidate = parent / "src" / "agent-mind"
        if candidate.exists():
            return candidate
    raise RuntimeError("Cannot locate agent-mind directory")


AGENT_DIR = _find_agent_dir()
sys.path.insert(0, str(AGENT_DIR))

from self_task_manager import SelfTaskManager  # noqa: E402


def _new_manager(tmp: Path) -> SelfTaskManager:
    state_file = tmp / "evolution" / "self_task_state.jsonl"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    return SelfTaskManager(state_file=state_file)


def _read_transitions(state_file: Path) -> list[dict]:
    if not state_file.exists():
        return []
    out = []
    for line in state_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


async def _case_acquire_then_finish() -> None:
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        m = _new_manager(tmp)
        ok, t = await m.try_acquire(
            kind="reflection_loop", owner="test", title="case1", ttl=60
        )
        assert ok is True and t is not None
        assert t.status == "running"
        # 互斥：第二次必须失败，且返回的是当前任务
        ok2, t2 = await m.try_acquire(kind="chat_session", owner="test", title="case1b")
        assert ok2 is False
        assert t2 is not None and t2.task_id == t.task_id
        # finish
        finished = await m.finish(t.task_id, result="done")
        assert finished is True
        assert (await m.is_busy()) is False
        # 现在应能再拿
        ok3, t3 = await m.try_acquire(kind="chat_session", owner="test", title="case1c")
        assert ok3 is True and t3 is not None and t3.task_id != t.task_id
        assert (await m.current()) is not None


async def _case_abort() -> None:
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        m = _new_manager(tmp)
        ok, t = await m.try_acquire(kind="admin_op", owner="test", title="abort_case", ttl=60)
        assert ok and t is not None
        aborted = await m.abort(t.task_id, reason="user_cancel")
        assert aborted is True
        # 错误 task_id abort 应当返回 False
        ok2, t2 = await m.try_acquire(kind="admin_op", owner="test", title="abort_case2")
        assert ok2 and t2 is not None
        assert (await m.abort("not-exist")) is False
        await m.finish(t2.task_id)


async def _case_timeout() -> None:
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        m = _new_manager(tmp)
        # ttl 强制为最小值 10 秒；为了不等待，这里直接手动篡改 expires_at。
        ok, t = await m.try_acquire(
            kind="reflection_loop", owner="test", title="timeout", ttl=10
        )
        assert ok and t is not None
        # 把过期时间往回拨 100 秒
        m._current.expires_at = time.time() - 100  # type: ignore[attr-defined]
        # is_busy 应触发 _gc_locked → 释放
        busy = await m.is_busy()
        assert busy is False
        cur = await m.current()
        assert cur is None
        # 之后能再拿到锁
        ok2, t2 = await m.try_acquire(kind="reflection_loop", owner="test", title="after_timeout")
        assert ok2 is True


async def _case_heartbeat_extends_ttl() -> None:
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        m = _new_manager(tmp)
        ok, t = await m.try_acquire(kind="reflection_loop", owner="test", title="hb", ttl=20)
        assert ok and t is not None
        original_expire = t.expires_at
        # 续期到 +600 秒；heartbeat 应抬升 expires_at
        ok_hb = await m.heartbeat(t.task_id, extend_ttl=600, last_tool_name="self_code_write")
        assert ok_hb is True
        assert m._current is not None  # type: ignore[attr-defined]
        assert m._current.expires_at >= original_expire + 500  # type: ignore[attr-defined]
        assert m._current.last_tool_name == "self_code_write"  # type: ignore[attr-defined]
        assert m._current.tool_calls_count == 1  # type: ignore[attr-defined]
        # 不正确的 task_id 心跳应失败
        bad = await m.heartbeat("nope", extend_ttl=10)
        assert bad is False


async def _case_restore_from_disk() -> None:
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        # 第一段：写入一条 acquired，但不 finish
        m1 = _new_manager(tmp)
        ok, t = await m1.try_acquire(kind="reflection_loop", owner="test", title="restore", ttl=600)
        assert ok and t is not None
        # 模拟进程重启：构造新实例
        state_file = tmp / "evolution" / "self_task_state.jsonl"
        m2 = SelfTaskManager(state_file=state_file)
        # 重启后应认为自己仍处于 busy 状态（任务未 finish 也未过期）
        assert (await m2.is_busy()) is True
        # 重启后 try_acquire 应该失败（互斥）
        ok2, t2 = await m2.try_acquire(kind="chat_session", owner="test", title="afterrestore")
        assert ok2 is False and t2 is not None and t2.task_id == t.task_id

        # 验证转换链：至少有 acquired 一条
        rows = _read_transitions(state_file)
        assert any(r.get("transition") == "acquired" for r in rows)


async def _case_restore_timeout() -> None:
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        state_file = tmp / "evolution" / "self_task_state.jsonl"
        state_file.parent.mkdir(parents=True, exist_ok=True)
        # 直接写一条已过期的 running 记录
        now = time.time()
        row = {
            "task_id": "expired_xx",
            "kind": "reflection_loop",
            "owner": "test",
            "title": "x",
            "started_at": now - 1000,
            "expires_at": now - 500,
            "heartbeat_at": now - 800,
            "status": "running",
            "tool_calls_count": 0,
            "last_tool_name": "",
            "source_request_id": "",
            "extra": {},
            "transition": "acquired",
            "recorded_at": now - 1000,
        }
        state_file.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        m = SelfTaskManager(state_file=state_file)
        # 过期记录应在 restore 阶段被标记为 timeout，进程不应认为自己 busy
        assert (await m.is_busy()) is False
        rows = _read_transitions(state_file)
        assert any(r.get("transition") == "timeout_on_restore" for r in rows)


async def _main() -> None:
    await _case_acquire_then_finish()
    await _case_abort()
    await _case_timeout()
    await _case_heartbeat_extends_ttl()
    await _case_restore_from_disk()
    await _case_restore_timeout()
    print("self_task_manager_selfcheck: ok")


if __name__ == "__main__":
    asyncio.run(_main())
