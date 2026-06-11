"""Self-checks for the reflection-spinning (too_passive) -> growth bridge.

Run from repository root:
    python src/agent-mind/tests/test_engine_passive_action_bridge.py

Covers evolution.engine._bridge_passive_signal:
- records a milestone once when status is too_passive and no proposal generated
- deduplicates within the same UTC day (no score inflation)
- defers when a fresh upgrade proposal was generated this cycle
- skips when status is not too_passive
"""
from __future__ import annotations

import sys
from pathlib import Path


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

from evolution import engine as engine_mod  # noqa: E402


class _FakeStore:
    """In-memory stand-in for the append-only growth milestone store."""

    def __init__(self) -> None:
        self.milestones: list[dict] = []
        self._counter = 0

    def record(self, event_type, description, *, reward_points=None, source_proposal_id="", extra=None):
        self._counter += 1
        import datetime as _dt

        now = _dt.datetime.now(_dt.timezone.utc).isoformat()
        milestone = {
            "id": f"gm_test_{self._counter}",
            "timestamp": now,
            "date": now[:10],
            "event_type": event_type,
            "description": description,
            "reward_points": reward_points if reward_points is not None else 30,
            "extra": {k: str(v) for k, v in (extra or {}).items()},
        }
        self.milestones.append(milestone)
        return milestone

    def read(self, days=30, limit=500):
        return list(self.milestones)


def _install(monkeypatch_store: _FakeStore) -> None:
    engine_mod.record_growth_milestone = monkeypatch_store.record  # type: ignore[assignment]
    engine_mod.read_milestones = monkeypatch_store.read  # type: ignore[assignment]


def main() -> None:
    # Preserve originals so the import side effects on other tests stay clean.
    orig_record = engine_mod.record_growth_milestone
    orig_read = engine_mod.read_milestones
    try:
        # Case 1: too_passive + no proposal -> records exactly one milestone.
        store = _FakeStore()
        _install(store)
        metrics = {
            "reflection_to_action_status": "too_passive",
            "reflection_to_action_ratio": 0.02,
            "reflection_count": 1300,
        }
        pressure = {"status": "proposal_missing"}
        r1 = engine_mod._bridge_passive_signal(metrics, pressure)
        assert r1["status"] == "recorded", r1
        assert r1["recorded"] is True
        assert len(store.milestones) == 1
        assert store.milestones[0]["event_type"] == "autonomous_debugging"
        assert store.milestones[0]["extra"]["bridge"] == "passive_action_bridge"

        # Case 2: second call same UTC day -> deduped, no new milestone.
        r2 = engine_mod._bridge_passive_signal(metrics, pressure)
        assert r2["status"] == "already_recorded", r2
        assert r2["recorded"] is False
        assert len(store.milestones) == 1, "must not inflate growth score within the same day"

        # Case 3: proposal generated this cycle -> deferred, nothing recorded.
        store3 = _FakeStore()
        _install(store3)
        r3 = engine_mod._bridge_passive_signal(metrics, {"status": "proposal_generated"})
        assert r3["status"] == "deferred", r3
        assert len(store3.milestones) == 0

        # Case 4: balanced status -> skipped, nothing recorded.
        store4 = _FakeStore()
        _install(store4)
        r4 = engine_mod._bridge_passive_signal(
            {"reflection_to_action_status": "balanced"}, {"status": "proposal_missing"}
        )
        assert r4["status"] == "skipped", r4
        assert len(store4.milestones) == 0

        print("engine_passive_action_bridge_selfcheck: ok")
    finally:
        engine_mod.record_growth_milestone = orig_record
        engine_mod.read_milestones = orig_read


if __name__ == "__main__":
    main()
