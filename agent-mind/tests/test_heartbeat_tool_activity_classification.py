"""Self-checks for heartbeat tool-result activity classification.

Run from repository root:
    python src/agent-mind/tests/test_heartbeat_tool_activity_classification.py
"""
from __future__ import annotations

import sys
from pathlib import Path


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

from heartbeat import _classify_tool_result_for_activity  # noqa: E402


def main() -> None:
    missing_binary = _classify_tool_result_for_activity(
        "shell_exec",
        {
            "status": "error",
            "tool": "shell_exec",
            "result": "[stderr]: /bin/sh: 1: rg: not found",
            "failure_type": "command_missing",
            "returncode": 127,
        },
        "[stderr]: /bin/sh: 1: rg: not found",
    )
    assert missing_binary["success"] is False
    assert missing_binary["classification"] == "failure"
    assert "status=error" in missing_binary["diagnostic_suffix"]
    assert "failure_type=command_missing" in missing_binary["diagnostic_suffix"]
    assert "returncode=127" in missing_binary["diagnostic_suffix"]

    ok_shell = _classify_tool_result_for_activity(
        "shell_exec",
        {"status": "ok", "tool": "shell_exec", "result": "hello", "returncode": 0},
        "hello",
    )
    assert ok_shell["success"] is True
    assert ok_shell["classification"] == "success"

    empty_note = _classify_tool_result_for_activity(
        "daily_note_read",
        {"status": "ok", "tool": "daily_note_read", "result": "", "failure_type": "empty_result"},
        "",
    )
    assert empty_note["success"] is True
    assert empty_note["classification"] == "empty_result"

    search_empty = _classify_tool_result_for_activity(
        "shell_exec",
        {
            "status": "error",
            "tool": "shell_exec",
            "command": "grep -rn definitely_absent_pattern /app/tool_router.py",
            "result": "",
            "returncode": 1,
        },
        "",
    )
    assert search_empty["success"] is True
    assert search_empty["classification"] == "empty_result"
    assert search_empty["failure_type"] == "empty_result"
    assert "returncode=1" in search_empty["diagnostic_suffix"]

    # ghost_tool / dead_path：能力库自创工具返回 status=ok，但 result 体里
    # 嵌套 error（实现/产出缺失）。旧逻辑只看顶层 error 会误记为 success。
    ghost_tool = _classify_tool_result_for_activity(
        "stale_task_detector",
        {
            "status": "ok",
            "tool": "stale_task_detector",
            "result": {"error": "status file not found: /app/data/experiment/status.json", "stale": []},
            "source": "capability_library",
        },
        "{'error': 'status file not found: /app/data/experiment/status.json', 'stale': []}",
    )
    assert ghost_tool["success"] is False
    assert ghost_tool["classification"] == "failure"
    assert ghost_tool["failure_type"] == "ghost_tool/dead_path"
    assert "failure_type=ghost_tool/dead_path" in ghost_tool["diagnostic_suffix"]

    # status=ok 且 result 为正常 dict（无 error）不应被误判为失败。
    ok_dict = _classify_tool_result_for_activity(
        "goal_list",
        {"status": "ok", "tool": "goal_list", "result": {"goals": [], "count": 0}},
        "{'goals': [], 'count': 0}",
    )
    assert ok_dict["success"] is True
    assert ok_dict["classification"] == "success"

    print("heartbeat_tool_activity_classification_selfcheck: ok")


if __name__ == "__main__":
    main()
