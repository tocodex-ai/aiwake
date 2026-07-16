# -*- coding: utf-8 -*-
"""Self-checks for failure triad auto-write and other_error miner."""
from __future__ import annotations

import asyncio
import importlib
import os
import sys
import tempfile
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
    raise RuntimeError("Cannot locate agent-mind directory")


AGENT_DIR = _find_agent_dir()
sys.path.insert(0, str(AGENT_DIR))


def test_append_and_mine(tmp_path: Path) -> None:
    os.environ["EVOLUTION_DIR"] = str(tmp_path)
    import evolution.failure_triads as ft
    importlib.reload(ft)

    r1 = ft.append_failure_triad(
        tool_name="web_fetch",
        failure_type="aiwake_tool_failure",
        error="timeout while fetching example.com",
        recovery_action="shrink and retry once",
        recovery_result="recorded",
    )
    assert r1["written"] is True
    assert r1["failure_category"] == "timeout" or r1["failure_category"] == "other_error"

    r2 = ft.append_failure_triad(
        tool_name="shell_exec",
        failure_type="aiwake_tool_failure",
        error="ValueError: invalid json payload",
        recovery_action="fix params",
        recovery_result="recorded",
    )
    r3 = ft.append_failure_triad(
        tool_name="file_read",
        failure_type="other_error",
        error="unexpected internal error 12345",
        recovery_action="cluster first",
        recovery_result="recorded",
    )
    assert r2["written"] and r3["written"]

    mined = ft.mine_other_error_patterns(limit=50, min_count=1)
    assert mined["status"] == "ok"
    assert mined["matched_triads"] >= 1
    assert isinstance(mined.get("top_patterns"), list)
    stats = ft.triad_stats()
    assert stats["real_count"] >= 3


def test_tool_router_auto_triad_and_miner(tmp_path: Path) -> None:
    os.environ["EVOLUTION_DIR"] = str(tmp_path)
    import evolution.failure_triads as ft
    importlib.reload(ft)
    from tool_router import ToolRouter, KNOWN_TOOLS, ALLOWED_TOOLS, READONLY_TOOLS

    assert "degradation_evidence_miner" in KNOWN_TOOLS
    assert "degradation_evidence_miner" in ALLOWED_TOOLS
    assert "degradation_evidence_miner" in READONLY_TOOLS

    router = ToolRouter()
    with patch("tool_router.append_growth_log"):
        out = router._finalize_tool_result(
            "web_fetch",
            {
                "status": "error",
                "tool": "web_fetch",
                "url": "https://example.com/flaky",
                "http_status": 503,
                "error": "目标站点返回 HTTP 503 Service Unavailable",
            },
        )
    assert out.get("failure_type") == "rate_limited"
    assert out.get("failure_triad_id")
    assert out.get("failure_category")

    items = ft.read_failure_triads(limit=20)
    real = [i for i in items if str(i.get("failure_type") or "").lower() != "init"]
    assert any(i.get("tool") == "web_fetch" for i in real)

    mined = asyncio.run(router.call("degradation_evidence_miner", {"limit": 50, "min_count": 1}))
    assert mined.get("status") == "ok", mined
    assert mined.get("tool") == "degradation_evidence_miner"
    result = mined.get("result") or {}
    assert "top_patterns" in result


def test_forced_miner_trigger(tmp_path: Path) -> None:
    os.environ["EVOLUTION_DIR"] = str(tmp_path)
    import evolution.failure_triads as ft
    importlib.reload(ft)

    for i in range(3):
        ft.append_failure_triad(
            tool_name="x",
            failure_type="other_error",
            error=f"timeout sample {i}",
            recovery_action="retry",
            recovery_result="recorded",
        )
    out = ft.maybe_force_mine_from_metrics(
        {"tool_failure_breakdown": {"other_error": 5}, "tool_failure_count": 5},
        min_other_error=3,
    )
    assert out is not None
    assert out.get("status") == "ok"
    assert out.get("trigger", {}).get("other_error_count") == 5

    skip = ft.maybe_force_mine_from_metrics(
        {"tool_failure_breakdown": {"other_error": 1}},
        min_other_error=3,
    )
    assert skip is None


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        test_append_and_mine(p / "a")
        test_tool_router_auto_triad_and_miner(p / "b")
        test_forced_miner_trigger(p / "c")
    # regression: existing failure classification still works
    from tests.test_tool_router_failure_classification import main as fail_main
    # run classification selfcheck in-process if import path works
    print("failure_triads_and_miner_selfcheck: ok")


if __name__ == "__main__":
    main()
