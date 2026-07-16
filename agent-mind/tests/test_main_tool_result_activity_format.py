"""Self-checks for /chat tool-result activity formatting.

Run from repository root:
    python src/agent-mind/tests/test_main_tool_result_activity_format.py
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

from main import _format_tool_result_activity, _safe_tool_result_summary  # noqa: E402


def main() -> None:
    missing_binary = {
        "status": "error",
        "tool": "shell_exec",
        "result": "[stderr]: /bin/sh: 1: rg: not found",
        "failure_type": "command_missing",
        "returncode": 127,
        "command": "rg token src",
    }
    content, extra = _format_tool_result_activity("shell_exec", missing_binary)
    assert "shell_exec 失败 (error)" in content
    assert "classification=failure" in content
    assert "failure_type=command_missing" in content
    assert "returncode=127" in content
    assert extra["success"] is False
    assert extra["classification"] == "failure"
    assert extra["failure_type"] == "command_missing"
    assert extra["returncode"] == 127

    empty_note = {
        "status": "ok",
        "tool": "daily_note_read",
        "result": "",
        "failure_type": "empty_result",
    }
    content, extra = _format_tool_result_activity("daily_note_read", empty_note)
    assert "[空结果]" in content
    assert "classification=empty_result" in content
    assert extra["success"] is True
    assert extra["classification"] == "empty_result"
    assert extra["failure_type"] == "empty_result"

    summary = _safe_tool_result_summary(
        {
            "status": "error",
            "tool": "web_fetch",
            "failure_type": "external_access_restricted",
            "http_status": 403,
            "result": {"id": "art_1", "task_id": "tas_1", "kind": "learning_artifact"},
        }
    )
    assert "failure_type" in summary
    assert "http_status" in summary
    assert "tas_1" in summary

    synthetic_token = "sk-" + "abcdefghijklmnop"
    redacted_content, _ = _format_tool_result_activity(
        "web_fetch",
        {
            "status": "error",
            "tool": "web_fetch",
            "error": f"Authorization: Bearer {synthetic_token} is invalid",
            "failure_type": "external_access_restricted",
        },
    )
    assert synthetic_token not in redacted_content
    assert "[REDACTED_SECRET]" in redacted_content

    print("main_tool_result_activity_format_selfcheck: ok")


if __name__ == "__main__":
    main()
