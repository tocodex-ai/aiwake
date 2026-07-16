"""Self-checks for evolution.metrics tool-failure classification.

Run from repository root:
    python src/agent-mind/tests/test_metrics_failure_category.py

Covers the deterministic-failure split added after the 2026-06-11 online
evidence: Glob/Read/TodoWrite「未被当前实验配置允许」与 file_read「文件不存在」
长期被笼统计入 other_error，导致 playbook 建议「原样重试一次」而空转，
tool_failure_count 卡住不收敛。新增 config_blocked / not_found 两个确定性类别，
给出「不要重试」的止损指引。
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

from evolution.metrics import (  # noqa: E402
    _classify_tool_failure,
    _is_tool_failure,
    build_failure_recovery_playbook,
)


def test_config_blocked_from_real_online_evidence() -> None:
    # 线上真实失败文本（2026-06-11 17:12/17:23）
    for content in (
        "[反思] 工具 Glob 返回: {'error': \"工具 'Glob' 未被当前实验配置允许\", 'tool': 'Glob', 'status': 'error'}",
        "[反思] 工具 Read 返回: {'error': \"工具 'Read' 未被当前实验配置允许\"}",
        "[反思] 工具 TodoWrite 返回: {'error': \"工具 'TodoWrite' 未被当前实验配置允许\"}",
        "tool 'X' not allowed in current experiment config",
        "该工具已禁用",
    ):
        assert _classify_tool_failure(content) == "config_blocked", content


def test_not_found_from_real_online_evidence() -> None:
    for content in (
        "file_read 失败 (error) — 文件不存在: src/agent-mind/evolution_engine.py",
        "no such file or directory",
        "目标资源 not found",
        "HTTP 404 Not Found",
        "找不到该路径",
    ):
        assert _classify_tool_failure(content) == "not_found", content


def test_command_missing_from_shell_exec_activity_log() -> None:
    # 线上真实形态：旧 heartbeat 曾把 `rg: not found` 误写成 classification=success。
    content = "[反思] 工具 shell_exec 返回: [stderr]: /bin/sh: 1: rg: not found | classification=success"
    assert _is_tool_failure(content)
    assert _classify_tool_failure(content) == "command_missing"

    new_content = "[反思] 工具 shell_exec 返回: [stderr]: /bin/sh: 1: rg: not found | classification=failure | status=error failure_type=command_missing returncode=127"
    assert _is_tool_failure(new_content)
    assert _classify_tool_failure(new_content) == "command_missing"


def test_existing_categories_not_regressed() -> None:
    assert _classify_tool_failure("HTTP 429 too many requests") == "rate_limited"
    assert _classify_tool_failure("HTTP 403 forbidden") == "forbidden"
    assert _classify_tool_failure("502 bad gateway") == "upstream_error"
    assert _classify_tool_failure("operation timed out 超时") == "timeout"
    assert _classify_tool_failure("connection refused 网络不可用") == "unavailable"
    # 真正无法归类的仍落到 other_error
    assert _classify_tool_failure("error: 未知的内部错误 12345") == "other_error"


def test_playbook_stop_when_no_retry_for_deterministic_failures() -> None:
    playbook = build_failure_recovery_playbook(
        {"config_blocked": 5, "command_missing": 3, "not_found": 2, "other_error": 1}
    )
    rules = {r["category"]: r for r in playbook["rules"]}
    assert "config_blocked" in rules and "not_found" in rules and "command_missing" in rules
    # 确定性失败的止损规则必须强调“重试无意义/不要重复”
    assert "重试无意义" in rules["config_blocked"]["stop_when"]
    assert "重试无意义" in rules["command_missing"]["stop_when"]
    assert "重试无意义" in rules["not_found"]["stop_when"]
    # 降序排列：config_blocked(5) > command_missing(3) > not_found(2) > other_error(1)
    assert [r["category"] for r in playbook["rules"]] == [
        "config_blocked",
        "command_missing",
        "not_found",
        "other_error",
    ]


def main() -> None:
    test_config_blocked_from_real_online_evidence()
    test_not_found_from_real_online_evidence()
    test_command_missing_from_shell_exec_activity_log()
    test_existing_categories_not_regressed()
    test_playbook_stop_when_no_retry_for_deterministic_failures()
    print("OK: metrics failure category tests passed (5 cases)")


if __name__ == "__main__":
    main()
