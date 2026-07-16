"""Self-checks for ToolRouter tool-name normalization and unknown-tool guidance.

Background: 线上反思层多次调用不存在的工具名（如 Read / Glob / TodoWrite），
落入旧的 "未被当前实验配置允许" 硬失败分支，被计入 tool_failure_count，
污染工具可靠性指标。本测试覆盖：

1. 别名归一化：Read → file_read、Glob → file_list、Bash → shell_exec 等。
2. 真实工具名（KNOWN_TOOLS）不被改名。
3. 真正未知/幻觉工具名返回 status=degraded 的结构化指引，而非 error，
   因此不会被 metrics._is_tool_failure 计为失败。

Run from repository root:
    python src/agent-mind/tests/test_tool_router_name_normalization.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch


def _find_agent_dir() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "heartbeat.py").exists() and (parent / "tool_router.py").exists():
            return parent
        if parent.name == "agent-mind":
            return parent
        candidate = parent / "src" / "agent-mind"
        if candidate.exists():
            return candidate
        candidate = parent / "agent-mind"
        if candidate.exists():
            return candidate
    raise RuntimeError("Cannot locate agent-mind directory")


AGENT_DIR = _find_agent_dir()
sys.path.insert(0, str(AGENT_DIR))

from tool_router import ToolRouter, KNOWN_TOOLS  # noqa: E402
from evolution.metrics import _is_tool_failure  # noqa: E402


def main() -> None:
    router = ToolRouter()

    # 1) 别名归一化（大小写不敏感）
    assert router._normalize_tool_name("Read") == "file_read"
    assert router._normalize_tool_name("read_file") == "file_read"
    assert router._normalize_tool_name("Glob") == "file_list"
    assert router._normalize_tool_name("LS") == "file_list"
    assert router._normalize_tool_name("Bash") == "shell_exec"
    assert router._normalize_tool_name("WebSearch") == "web_search"
    assert router._normalize_tool_name("recall") == "knowledge_search"

    # 2) 真实工具名不被改写
    for real in ("file_read", "web_fetch", "goal_register", "self_task_create"):
        assert router._normalize_tool_name(real) == real, real

    # 3) 真正未知工具名 → 结构化指引（status=degraded，非 error），不计为失败
    async def _run_unknown() -> dict:
        return await router.call("TodoWrite", {"todos": []})

    with patch.object(router.store, "append") as store_append:
        guidance = asyncio.run(_run_unknown())
    assert guidance["status"] == "degraded", guidance
    assert guidance["diagnosis"] == "unknown_tool_name"
    assert "error" not in guidance, "未知工具名不应返回 error 字段，避免污染 tool_failure_count"
    assert "available_tools" in guidance and "file_read" in guidance["available_tools"]
    store_append.assert_called()  # 记录了 tool_name_unknown 审计事件

    # metrics 不应把这条指引文本判为工具失败
    assert _is_tool_failure(guidance["result"]) is False, (
        "未知工具指引文本不应触发 tool_failure 计数"
    )

    # 4) 归一化后的真实工具能正常路由（Read → file_read，参数缺失返回受控结果而非崩溃）
    async def _run_alias() -> dict:
        return await router.call("Read", {"path": ""})

    alias_result = asyncio.run(_run_alias())
    assert alias_result.get("tool") == "file_read", alias_result

    # 一致性：别名表里的目标都应是真实工具
    from tool_router import _TOOL_NAME_ALIASES
    for alias, target in _TOOL_NAME_ALIASES.items():
        assert target in KNOWN_TOOLS, f"别名 {alias} 指向了非真实工具 {target}"

    print("tool_router_name_normalization_selfcheck: ok")


if __name__ == "__main__":
    main()
