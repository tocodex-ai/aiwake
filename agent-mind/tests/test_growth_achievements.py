"""Regression test for the growth-curve achievement points.

Goal (task request, 2026-06-16): surface AIwake's *concrete* "autonomously used
tools to improve its own code" outcomes as achievement POINTS on the growth
curve — not just a rising score. The backend ``growth_chart_data`` must emit an
``achievements`` array where each item:
  - is derived only from real code-improvement event types (self_upgrade_applied,
    code_patch_applied, tool_self_modify, deploy_success, ...),
  - carries x (date label) + y (cumulative score on that day) so the frontend can
    plot it directly ON the cumulative line,
  - exposes concrete evidence (files / edits_applied / archived / proposal id),
  - and excludes routine non-code milestones (e.g. reflection_insight).

Run from repository root:
    python src/agent-mind/tests/test_growth_achievements.py
"""
from __future__ import annotations

import datetime as _dt
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
        data_dir = Path(tmp) / "evolution"
        data_dir.mkdir(parents=True, exist_ok=True)
        os.environ["EVOLUTION_DIR"] = str(data_dir)

        import importlib

        import evolution.growth_tracker as gt
        importlib.reload(gt)  # pick up EVOLUTION_DIR

        today = _dt.date.today()
        d1 = (today - _dt.timedelta(days=2)).isoformat()
        d2 = (today - _dt.timedelta(days=1)).isoformat()
        d3 = today.isoformat()

        # Hand-craft an append-only milestone log mixing real code-improvement
        # events with routine ones, with monotonic cumulative_score.
        records = [
            {
                "id": "gm_1", "timestamp": f"{d1}T01:00:00+00:00", "date": d1,
                "event_type": "self_upgrade_applied",
                "description": "自主应用升级提案: 修复 tool_router 路径回退",
                "reward_points": 50, "cumulative_score": 50,
                "source_proposal_id": "upg_aaa",
                "extra": {"files": "['tool_router.py']", "edits_applied": "3",
                          "patch_method": "llm_patch", "archived": "True", "risk_level": "low"},
            },
            {
                "id": "gm_2", "timestamp": f"{d2}T02:00:00+00:00", "date": d2,
                "event_type": "reflection_insight",  # routine, must be excluded
                "description": "一次反思洞察",
                "reward_points": 10, "cumulative_score": 60,
            },
            {
                "id": "gm_3", "timestamp": f"{d3}T03:00:00+00:00", "date": d3,
                "event_type": "deploy_success",
                "description": "自我部署成功: 上线持久化归档",
                "reward_points": 60, "cumulative_score": 120,
                "source_proposal_id": "",
                "extra": {"archived": "False"},
            },
        ]
        log = data_dir / "growth_milestones.jsonl"
        with log.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False))
                f.write("\n")

        data = gt.growth_chart_data(days=7)

        # Achievements present and correctly filtered.
        ach = data.get("achievements")
        assert isinstance(ach, list), data
        assert data.get("total_achievements") == 2, data
        types = {a["event_type"] for a in ach}
        assert types == {"self_upgrade_applied", "deploy_success"}, types
        assert all(a["event_type"] != "reflection_insight" for a in ach), ach

        # Each achievement carries x/y aligned to the cumulative curve.
        labels = data["labels"]
        scores = data["scores"]
        score_by_date = dict(zip(labels, scores))
        for a in ach:
            assert a["x"] in labels, (a["x"], labels)
            assert a["y"] == score_by_date[a["x"]], (a, score_by_date)
            assert "title" in a and a["title"], a
            assert "icon" in a and a["icon"], a

        # Concrete evidence surfaced for the self_upgrade_applied event.
        upg = next(a for a in ach if a["event_type"] == "self_upgrade_applied")
        assert "tool_router.py" in upg["files"], upg
        assert upg["edits_applied"] == "3", upg
        assert upg["archived"] == "True", upg
        assert upg["source_proposal_id"] == "upg_aaa", upg
        assert upg["reward_points"] == 50, upg

        # The y of the day-1 achievement equals that day's cumulative (50).
        assert upg["y"] == 50, upg

        # Total milestones still counts everything (incl. routine).
        assert data.get("total_milestones") == 3, data

    print("growth_achievements: ok")


if __name__ == "__main__":
    main()
