# -*- coding: utf-8 -*-
"""Self-checks for deterministic shell missing-binary probe in /chat.

背景：task-60 训练提示要求 AIwake 调用一个无害缺失命令验证 shell_exec
主命令预检；线上 work 模型空输出时，旧逻辑只返回 empty_fallback，未执行工具，
导致可验证闭环无法形成。本测试覆盖确定性 shell_probe 分支：

1. 只接受明确 shell_exec + missing_binary/缺失命令语境下的 definitely_missing_aiwake_probe* 命令；
2. 触发后先执行 shell_exec，随后为指定 task 生成 self_artifact_create 证据；
3. 回复中复述 status/failure_type/reason 等字段，便于 activity_logs 验证。
"""
from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path


_MAIN_PATH = Path(__file__).resolve().parents[1] / "main.py"


def _load_main():
    agent_dir = str(_MAIN_PATH.parent)
    if agent_dir not in sys.path:
        sys.path.insert(0, agent_dir)
    spec = importlib.util.spec_from_file_location("aiwake_main_shell_probe_test", _MAIN_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("aiwake_main_shell_probe_test", module)
    spec.loader.exec_module(module)
    return module


def test_extract_explicit_shell_probe_command_accepts_safe_probe() -> None:
    main_mod = _load_main()
    msg = (
        "请只做一件事：调用 shell_exec 执行 "
        "definitely_missing_aiwake_probe_20260614_postcheck --version，"
        "用来验证 shell_exec 主命令预检是否拦截 missing_binary。"
    )
    assert (
        main_mod._extract_explicit_shell_probe_command(msg)
        == "definitely_missing_aiwake_probe_20260614_postcheck --version"
    )


def test_extract_explicit_shell_probe_command_rejects_general_shell() -> None:
    main_mod = _load_main()
    assert main_mod._extract_explicit_shell_probe_command("调用 shell_exec 执行 ls -la") is None
    assert (
        main_mod._extract_explicit_shell_probe_command(
            "调用 shell_exec 执行 definitely_missing_aiwake_probe_20260614_postcheck; rm -rf /"
        )
        is None
    )


def test_deterministic_shell_probe_records_artifact_and_fields() -> None:
    main_mod = _load_main()
    calls: list[tuple[str, dict]] = []

    async def fake_tool_executor(tool_name: str, params: dict) -> dict:
        calls.append((tool_name, params))
        if tool_name == "shell_exec":
            return {
                "status": "error",
                "tool": "shell_exec",
                "command": params["command"],
                "failure_type": "unavailable",
                "reason": "missing_binary",
                "error": "命令 'definitely_missing_aiwake_probe_20260614_postcheck' 在当前容器内不可用（未找到对应二进制）",
            }
        if tool_name == "self_artifact_create":
            return {
                "status": "ok",
                "tool": "self_artifact_create",
                "result": {
                    "id": "art_shell_probe",
                    "task_id": params["task_id"],
                    "kind": params["kind"],
                    "status": "draft",
                },
            }
        raise AssertionError(f"unexpected tool call: {tool_name}")

    msg = (
        "短训练：优先调用 self_artifact_create，latest_task=tas_1781429735_b9778e93。"
        "请调用 shell_exec 执行 definitely_missing_aiwake_probe_20260614_postcheck --version，"
        "用来验证 shell_exec 主命令预检是否拦截 missing_binary。"
    )
    reply, tier = asyncio.run(main_mod._run_deterministic_self_learning_tool(msg, fake_tool_executor))

    assert tier == "deterministic:shell_probe"
    assert calls[0] == (
        "shell_exec",
        {"command": "definitely_missing_aiwake_probe_20260614_postcheck --version"},
    )
    assert calls[1][0] == "self_artifact_create"
    assert calls[1][1]["task_id"] == "tas_1781429735_b9778e93"
    assert "missing_binary" in calls[1][1]["content"]
    assert "status" in reply
    assert "failure_type" in reply
    assert "missing_binary" in reply


if __name__ == "__main__":
    test_extract_explicit_shell_probe_command_accepts_safe_probe()
    test_extract_explicit_shell_probe_command_rejects_general_shell()
    test_deterministic_shell_probe_records_artifact_and_fields()
    print("chat_shell_probe_deterministic_selfcheck: ok")
