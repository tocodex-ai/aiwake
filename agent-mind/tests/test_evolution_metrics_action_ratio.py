"""Self-checks for evolution reflection-to-action metrics.

Run from repository root:
    python src/agent-mind/tests/test_evolution_metrics_action_ratio.py
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

from evolution.evaluator import evaluate_metrics  # noqa: E402
from evolution.metrics import _finalize_rates  # noqa: E402


def main() -> None:
    metrics = {
        "agent_reply_count": 10,
        "failed_reply_count": 0,
        "tool_call_count": 5,
        "tool_result_count": 5,
        "tool_failure_count": 1,
        "reflection_count": 50,
        "proactive_count": 0,
        "last_events": [],
    }
    _finalize_rates(metrics)
    assert metrics["tool_success_count"] == 4
    assert metrics["tool_success_rate"] == 0.8
    assert metrics["reflection_to_action_ratio"] == 0.1
    assert metrics["reflection_to_action_status"] == "balanced"

    passive = dict(metrics, tool_call_count=1, tool_result_count=1, tool_failure_count=0, reflection_count=50)
    _finalize_rates(passive)
    assert passive["reflection_to_action_ratio"] == 0.02
    assert passive["reflection_to_action_status"] == "too_passive"
    passive_eval = evaluate_metrics(passive)
    assert "反思→行动转换率过低" in passive_eval["issues"]

    # too_reactive 阈值为 action/reflection > 0.70（0.30~0.70 视为 balanced），
    # 故用 40/50=0.8 触发；20/50=0.4 现在属于 balanced。
    reactive = dict(metrics, tool_call_count=40, tool_result_count=40, tool_failure_count=0, reflection_count=50)
    _finalize_rates(reactive)
    assert reactive["reflection_to_action_ratio"] == 0.8
    assert reactive["reflection_to_action_status"] == "too_reactive"
    reactive_eval = evaluate_metrics(reactive)
    assert "反思→行动转换率过高" in reactive_eval["issues"]

    print("evolution_metrics_action_ratio_selfcheck: ok")


if __name__ == "__main__":
    main()
