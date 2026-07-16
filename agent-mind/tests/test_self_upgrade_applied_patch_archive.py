"""Regression test for the self-upgrade applied-patch durable archive.

Evidence (online audit, 2026-06-16): the self-upgrade pipeline reported 90/91
proposals as 'applied' with notes like "LLM 生成并应用了 N 个编辑块，修改文件:
tool_router.py", yet every edit landed on the image layer (/app/tool_router.py),
which is EPHEMERAL — a restart or re-deploy rebuilds the image from source and
wipes those edits. Net effect: the loop "applied" patches that silently vanished
(real work, zero durable effect = the "空转闭环").

This test asserts the durable-archive behaviour added to close that loop:
- after a successful apply, a copy of each modified file is written under the
  persistent volume dir (_data_dir()/applied_patches/<proposal_id>/),
- a manifest.jsonl records what was archived,
- the archived copy survives independently of the in-container source file
  (simulating a restart/redeploy that resets the image layer).

Run from repository root:
    python src/agent-mind/tests/test_self_upgrade_applied_patch_archive.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
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


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # Simulated container image layer (ephemeral) and persistent volume.
        app_root = tmp_path / "app"
        app_root.mkdir(parents=True, exist_ok=True)
        data_dir = tmp_path / "app" / "data" / "evolution"
        data_dir.mkdir(parents=True, exist_ok=True)

        os.environ["APP_ROOT"] = str(app_root)
        os.environ["EVOLUTION_DIR"] = str(data_dir)

        import importlib

        import evolution.self_upgrade as su
        importlib.reload(su)  # pick up APP_ROOT / EVOLUTION_DIR from env

        # Create a fake editable source file on the (ephemeral) image layer.
        target_rel = "tool_router.py"
        target_abs = app_root / target_rel
        target_abs.write_text("# patched content v2\nprint('hello')\n", encoding="utf-8")

        proposal_id = "upg_test_archive_0001"
        apply_result = {
            "applied_count": 1,
            "files_modified": [target_rel, target_rel],  # intentional duplicate
            "errors": [],
        }

        result = su.archive_applied_patch(proposal_id, apply_result, actor="test_actor")
        assert result.get("archived") is True, result
        assert target_rel in (result.get("archived_files") or []), result

        archive_dir = Path(result["archive_dir"])
        assert archive_dir.exists(), "archive dir must exist on the persistent volume"
        # Must live under the persistent data dir, not the image layer.
        assert str(archive_dir).startswith(str(data_dir)), archive_dir

        # The flattened copy must hold the post-patch content.
        safe_name = target_rel.replace("/", "__")
        copy_path = archive_dir / safe_name
        assert copy_path.exists(), f"archived copy missing: {copy_path}"
        assert copy_path.read_text(encoding="utf-8") == target_abs.read_text(encoding="utf-8")

        # Manifest recorded the archive event.
        manifest = archive_dir / "manifest.jsonl"
        assert manifest.exists(), "manifest.jsonl must be written"
        lines = [l for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 1, lines
        entry = json.loads(lines[0])
        assert entry["proposal_id"] == proposal_id
        assert entry["files"] == [target_rel]

        # Simulate a restart / re-deploy: the image layer is rebuilt from source,
        # wiping the in-container edit. The durable archive must still survive.
        target_abs.unlink()
        assert not target_abs.exists()
        assert copy_path.exists(), "durable archive must survive image-layer reset"
        assert copy_path.read_text(encoding="utf-8") == "# patched content v2\nprint('hello')\n"

        # No files modified -> no archive (defensive path).
        empty = su.archive_applied_patch("upg_empty", {"files_modified": []}, actor="test_actor")
        assert empty.get("archived") is False, empty

    print("self_upgrade_applied_patch_archive: ok")


if __name__ == "__main__":
    main()
