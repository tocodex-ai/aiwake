"""Self-checks for the failure recovery playbook (metrics + reflection prompt).

Run from repository root:
    python src/agent-mind/tests/test_failure_recovery_playbook.py

Covers:
  1. 已知失败类别生成对应规则；
  2. 按出现次数降序排列，高频失败优先；
  3. count<=0 与空 breakdown 被忽略；
  4. 未知类别回退到 other_error 规则；
  5. metrics.collect 之后挂载 failure_recovery_playbook 字段；
  6. self_dialogue._build_prompt 注入恢复决策表文本。
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

from evolution.metrics import build_failure_recovery_playbook  # noqa: E402
from evolution import self_dialogue  # noqa: E402


def test_known_categories_and_ordering() -> None:
    playbook = build_failure_recovery_playbook(
        {"other_error": 9, "upstream_error": 2, "rate_limited": 5}
    )
    rules = playbook["rules"]
    # 降序：other_error(9) > rate_limited(5) > upstream_error(2)
    assert [r["category"] for r in rules] == ["other_error", "rate_limited", "upstream_error"]
    for r in rules:
        assert r["first_action"] and r["fallback"] and r["stop_when"]
    assert playbook["evidence_fields"] == ["failure_category", "recovery_action", "recovery_result"]
    assert "三元证据" in playbook["principle"]


def test_zero_and_empty_ignored() -> None:
    assert build_failure_recovery_playbook({})["rules"] == []
    assert build_failure_recovery_playbook(None)["rules"] == []
    assert build_failure_recovery_playbook({"other_error": 0, "timeout": 0})["rules"] == []


def test_unknown_category_falls_back_to_other_error() -> None:
    playbook = build_failure_recovery_playbook({"some_new_kind": 3})
    rule = playbook["rules"][0]
    assert rule["category"] == "some_new_kind"
    # 回退使用 other_error 的规则文本
    assert "参数" in rule["first_action"] or "输入" in rule["first_action"]


def test_metrics_attaches_playbook() -> None:
    from evolution.metrics import _finalize_rates

    metrics = {
        "agent_reply_count": 10,
        "failed_reply_count": 0,
        "tool_result_count": 20,
        "tool_failure_count": 3,
        "tool_failure_breakdown": {"other_error": 2, "upstream_error": 1},
        "reflection_count": 5,
        "tool_call_count": 2,
        "proactive_count": 0,
        "last_events": [],
    }
    _finalize_rates(metrics)
    assert "failure_recovery_playbook" in metrics
    cats = [r["category"] for r in metrics["failure_recovery_playbook"]["rules"]]
    assert "other_error" in cats and "upstream_error" in cats


def test_prompt_injects_recovery_table() -> None:
    metrics = {
        "tool_failure_count": 11,
        "tool_failure_breakdown": {"other_error": 9, "upstream_error": 2},
        "failure_recovery_playbook": build_failure_recovery_playbook(
            {"other_error": 9, "upstream_error": 2}
        ),
    }
    prompt = self_dialogue._build_prompt(
        state_snapshot={"TR": 0.9},
        metrics=metrics,
        evaluation={"issues": ["检测到工具调用失败"], "suggestions": []},
        external_learning={},
    )
    assert "工具失败恢复决策表" in prompt
    assert "other_error(x9)" in prompt
    assert "upstream_error(x2)" in prompt
    assert "三元证据" in prompt


def main() -> None:
    test_known_categories_and_ordering()
    test_zero_and_empty_ignored()
    test_unknown_category_falls_back_to_other_error()
    test_metrics_attaches_playbook()
    test_prompt_injects_recovery_table()
    print("OK: failure recovery playbook tests passed (5 cases)")


if __name__ == "__main__":
    main()
