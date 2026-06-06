"""Minimal self-checks for the self-upgrade proposal MVP.

Run from repository root:
    python src/agent-mind/tests/test_self_upgrade_mvp.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

def _find_agent_dir() -> Path:
    """Locate agent-mind in both the source workspace and exported repo layout."""
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


def _load_main_module():
    path = AGENT_DIR / "main.py"
    spec = importlib.util.spec_from_file_location("aiwake_main_for_self_upgrade_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        proposal_file = Path(tmp) / "self_upgrade_proposals.jsonl"
        os.environ["SELF_UPGRADE_PROPOSALS_FILE"] = str(proposal_file)
        os.environ.pop("AIWAKE_ADMIN_TOKEN", None)
        os.environ.pop("ADMIN_TOKEN", None)

        from evolution.self_upgrade import (
            append_proposal,
            check_proposal_safety,
            create_candidate_proposal,
            proposal_status_summary,
            read_proposals,
            record_approval_status,
        )

        proposal = create_candidate_proposal(
            source="test",
            problem="提高公开聊天失败降级的可观测性",
            evidence=["failed_reply_count=1"],
            proposed_files=["src/agent-mind/main.py"],
            patch_summary="只增加只读状态摘要，允许受控应用并追加审计记录。",
        )
        assert proposal.status == "approved", proposal
        assert proposal.human_approval_required is False
        append_proposal(proposal)
        assert proposal_file.exists(), "proposal JSONL should be created append-only"
        items = read_proposals(limit=10)
        assert len(items) == 1 and items[0]["id"] == proposal.id

        checks, risk, blocked = check_proposal_safety(
            ["src/agent-mind/data/users/memory.json"],
            "rm -rf /app/data/users",
            [],
        )
        assert blocked is True
        assert risk == "high"
        assert any(check["status"] == "blocked" for check in checks)

        updated = record_approval_status(proposal.id, "approved", actor="test_admin", notes="smoke approval only")
        assert updated["status"] == "approved"
        assert updated["approval_event"]["allow_apply"] is True
        assert updated["approval_event"]["allow_deploy"] is True
        summary = proposal_status_summary()
        assert summary["counts"]["approved"] == 1
        assert summary["auto_apply_enabled"] is True
        assert summary["auto_deploy_enabled"] is True
        assert summary["human_approval_required"] is False

        main_module = _load_main_module()
        route_paths = {route.path for route in main_module.app.routes}
        assert "/self-upgrade/status" in route_paths
        assert "/self-upgrade/proposals" in route_paths
        assert "/self-upgrade/proposals/{proposal_id}/approval" in route_paths

    print("self_upgrade_mvp_selfcheck: ok")


if __name__ == "__main__":
    main()
