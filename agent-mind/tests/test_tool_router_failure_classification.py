"""Self-checks for ToolRouter failure classification and fallback hints.

Run from repository root:
    python src/agent-mind/tests/test_tool_router_failure_classification.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch


def _find_agent_dir() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
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

from tool_router import ToolRouter  # noqa: E402


def main() -> None:
    router = ToolRouter()

    api_misuse = router._classify_failure(
        "file_read",
        {
            "status": "error",
            "tool": "file_read",
            "path": "/experiment/status",
            "error": "文件不存在: /experiment/status",
            "diagnosis": "file_not_found",
        },
    )
    assert api_misuse["failure_type"] == "execution_context_mismatch/tool_path_boundary_misuse"
    assert "web_fetch" in api_misuse["fallback_hint"] or "url_fetch" in api_misuse["fallback_hint"]

    private_network = router._classify_failure(
        "web_fetch",
        {
            "status": "error",
            "tool": "web_fetch",
            "url": "http://localhost:8000/experiment/status",
            "error": "安全限制：目标解析到非公网地址 127.0.0.1，已拒绝抓取",
            "diagnosis": "private_network_safety_restriction",
        },
    )
    assert private_network["failure_type"] == "private_network_safety_restriction"
    assert "公开 HTTPS" in private_network["fallback_hint"]

    not_found = router._classify_failure(
        "web_fetch",
        {
            "status": "error",
            "tool": "web_fetch",
            "url": "https://example.com/missing",
            "http_status": 404,
            "error": "目标站点返回 HTTP 错误。 HTTP 404 Not Found",
        },
    )
    assert not_found["failure_type"] == "external_content_not_found/content_stale"
    assert "HTTP 状态" in not_found["fallback_hint"]

    empty = router._classify_failure(
        "daily_note_read",
        {
            "status": "ok",
            "tool": "daily_note_read",
            "date": "today",
            "result": "[today 暂无日记]",
        },
    )
    assert empty["status"] == "ok"
    assert empty["failure_type"] == "empty_result"
    assert "明确日期" in empty["fallback_hint"]

    local_endpoint = router._classify_failure(
        "shell_exec",
        {
            "status": "error",
            "tool": "shell_exec",
            "command": "curl http://localhost:8000/experiment/status",
            "result": "STATUS_UNAVAILABLE",
            "returncode": 7,
        },
    )
    assert local_endpoint["failure_type"] == "local_endpoint_unavailable/environment_boundary"
    assert "experiment_status" in local_endpoint["fallback_hint"]

    with patch("tool_router.append_growth_log") as append_growth_log:
        audit_result = router._finalize_tool_result(
            "web_fetch",
            {
                "status": "error",
                "tool": "web_fetch",
                "url": "https://example.com/flaky",
                "http_status": 503,
                "error": "目标站点返回 HTTP 503 Service Unavailable",
            },
        )
    assert audit_result["failure_type"] == "external_rate_limited_or_unavailable"
    append_growth_log.assert_called_once()
    audit_item = append_growth_log.call_args.args[0]
    assert audit_item["event"] == "tool_failure_audit"
    assert audit_item["tool"] == "web_fetch"
    assert audit_item["failure_type"] == "external_rate_limited_or_unavailable"
    assert "退避" in audit_item["fallback_hint"]

    with patch("tool_router.append_growth_log") as append_growth_log:
        router._finalize_tool_result(
            "daily_note_read",
            {
                "status": "ok",
                "tool": "daily_note_read",
                "result": "[today 暂无日记]",
            },
        )
    append_growth_log.assert_not_called()

    print("tool_router_failure_classification_selfcheck: ok")


if __name__ == "__main__":
    main()
