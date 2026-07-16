# -*- coding: utf-8 -*-
"""Regression tests for self_code_write project-test path recognition."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch


AGENT_DIR = Path(__file__).resolve().parents[1]
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from safety_guard import (  # noqa: E402
    is_memory_path,
    is_self_project_code_path,
    is_sensitive_path,
)
from tool_router import ToolRouter  # noqa: E402


def test_self_project_test_paths_are_allowed(monkeypatch) -> None:
    monkeypatch.setenv("APP_ROOT", "/app")

    assert is_self_project_code_path("tests/test_new_regression.py")
    assert is_self_project_code_path("src/agent-mind/tests/test_new_regression.py")
    assert is_self_project_code_path("/app/tests/test_new_regression.py")
    assert is_self_project_code_path(AGENT_DIR / "tests" / "test_new_regression.py")


def test_external_test_like_paths_are_rejected(monkeypatch) -> None:
    monkeypatch.setenv("APP_ROOT", "/app")

    assert not is_self_project_code_path("/tmp/tests/test_external.py")
    assert not is_self_project_code_path("/tmp/agent-mind/tests/test_external.py")
    assert not is_self_project_code_path("other/tests/test_external.py")


def test_sensitive_and_memory_guards_remain_active() -> None:
    assert is_sensitive_path("tests/test_token_secret.py")
    assert is_memory_path("data/users/memory.json")


def test_self_code_write_accepts_project_test_path_without_real_write(monkeypatch) -> None:
    monkeypatch.setenv("APP_ROOT", "/app")
    router = ToolRouter()
    audit = {
        "purpose": "验证 AIwake 可为自身改进新增回归测试。",
        "change_summary": "仅验证 tests 路径识别，不执行真实文件写入。",
        "files": ["tests/test_new_regression.py"],
        "validation_result": "路径守卫回归测试。",
        "risk_note": "通过 mock 隔离写入；不触碰记忆、日志或密钥。",
    }

    with (
        patch.object(
            router,
            "_write_file_with_snapshot",
            return_value={
                "status": "ok",
                "tool": "self_code_write",
                "path": "tests/test_new_regression.py",
                "result": "mock write",
            },
        ) as write_file,
        patch("tool_router.append_growth_log"),
        patch("tool_router.record_growth_milestone"),
        patch.object(router.store, "append"),
    ):
        result = router._self_code_write(
            "tests/test_new_regression.py",
            "def test_placeholder():\n    assert True\n",
            audit,
        )

    assert result["status"] == "ok", result
    write_file.assert_called_once_with(
        "tests/test_new_regression.py",
        "def test_placeholder():\n    assert True\n",
        tool="self_code_write",
    )
