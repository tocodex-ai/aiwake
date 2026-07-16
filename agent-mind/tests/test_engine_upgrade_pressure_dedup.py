"""Self-checks for dedup-aware upgrade-pressure gating.

Run from repository root:
    python src/agent-mind/tests/test_engine_upgrade_pressure_dedup.py

Covers:
- evolution.self_upgrade.classify_backlog_dedup:
  * counts actionable vs deduped backlog items using existing proposals
  * marks all_deduped when every backlog item is covered
- evolution.engine._build_upgrade_pressure:
  * reports proposal_deduped (not proposal_missing) when the only reason no
    proposal was generated is that backlog is already covered by recent proposals
  * still reports proposal_missing when an actionable backlog item exists
  * reports proposal_generated when a proposal was produced this cycle

This protects against the regression where a fully-deduped backlog kept the
upgrade-pressure gate stuck at proposal_missing, driving repeated reflection
spinning to "explain" a proposal that dedup will always suppress.
"""
from __future__ import annotations

import datetime as _dt
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

from evolution import engine as engine_mod  # noqa: E402
from evolution import self_upgrade as su_mod  # noqa: E402


class _FakeConfig:
    upgrade_proposal_enabled = True
    upgrade_apply_enabled = True
    upgrade_deploy_enabled = True


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def main() -> None:
    orig_read = su_mod.read_proposals
    try:
        # ── classify_backlog_dedup: all backlog items already covered ──
        # An applied proposal created "just now" (within the 6h dedup window)
        # for the same rule category as the only backlog item.
        def fake_read_all_covered(limit=50, status=None, path=None):
            return [
                {
                    "status": "applied",
                    "problem": "提升工具调用可靠性: 检测到工具调用失败",
                    "created_at": _now_iso(),
                }
            ]

        su_mod.read_proposals = fake_read_all_covered  # type: ignore[assignment]
        backlog = [
            {
                "id": "ev-1",
                "title": "提升工具调用可靠性: 检测到工具调用失败",
                "priority": 0.76,
                "risk": "medium",
            }
        ]
        summary = su_mod.classify_backlog_dedup(backlog)
        assert summary["backlog_count"] == 1, summary
        assert summary["actionable_count"] == 0, summary
        assert summary["deduped_count"] == 1, summary
        assert summary["all_deduped"] is True, summary

        # Gate must report proposal_deduped, NOT proposal_missing.
        pressure = engine_mod._build_upgrade_pressure(backlog, [], _FakeConfig(), summary)
        assert pressure["status"] == "proposal_deduped", pressure
        assert pressure["next_gate"] != "must_explain_or_generate_auto_approved_upgrade_proposal_next_cycle", pressure
        assert pressure["actionable_backlog_count"] == 0, pressure
        assert pressure["deduped_backlog_count"] == 1, pressure

        # ── classify_backlog_dedup: an actionable (uncovered) item exists ──
        def fake_read_none(limit=50, status=None, path=None):
            return []

        su_mod.read_proposals = fake_read_none  # type: ignore[assignment]
        summary2 = su_mod.classify_backlog_dedup(backlog)
        assert summary2["actionable_count"] == 1, summary2
        assert summary2["deduped_count"] == 0, summary2
        assert summary2["all_deduped"] is False, summary2

        # Gate must report proposal_missing — a real gap remains.
        pressure2 = engine_mod._build_upgrade_pressure(backlog, [], _FakeConfig(), summary2)
        assert pressure2["status"] == "proposal_missing", pressure2
        assert pressure2["actionable_backlog_count"] == 1, pressure2

        # ── proposal generated this cycle wins regardless of dedup ──
        pressure3 = engine_mod._build_upgrade_pressure(
            backlog, [{"id": "upg-x"}], _FakeConfig(), summary
        )
        assert pressure3["status"] == "proposal_generated", pressure3

        # ── expired applied proposal no longer suppresses (outside dedup window) ──
        def fake_read_expired(limit=50, status=None, path=None):
            old = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(hours=48)).isoformat()
            return [
                {
                    "status": "applied",
                    "problem": "提升工具调用可靠性: 检测到工具调用失败",
                    "created_at": old,
                }
            ]

        su_mod.read_proposals = fake_read_expired  # type: ignore[assignment]
        summary4 = su_mod.classify_backlog_dedup(backlog)
        assert summary4["actionable_count"] == 1, summary4
        assert summary4["all_deduped"] is False, summary4

        print("engine_upgrade_pressure_dedup_selfcheck: ok")
    finally:
        su_mod.read_proposals = orig_read


if __name__ == "__main__":
    main()
