"""Self-checks for evolution backlog classification of reflection-to-action issues.

Run from repository root:
    python src/agent-mind/tests/test_backlog_action_ratio_classification.py

回归目标：too_passive（反思→行动转换率过低）必须被归类为“将近期反思转化为
可验证行动”，而不是方向相反的“增强自主反思频率与质量”；too_reactive 同理
被归类为收敛行动节奏，而不是继续加码行动。
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

from evolution.backlog import build_backlog  # noqa: E402


def _titles_for(issue: str) -> list[str]:
    evaluation = {
        "issues": [issue],
        "suggestions": ["将至少一次近期反思转化为可验证行动。"],
        "total_score": 90,
        "risk_level": "low",
    }
    # 指标全部健康，确保只有 issue 文本驱动 backlog 分类，避免指标兜底任务干扰。
    metrics = {
        "chat_success_rate": 1.0,
        "tool_success_rate": 1.0,
        "reflection_count": 100,
        "memory_update_count": 10,
        "failed_reply_count": 0,
    }
    tasks = build_backlog(evaluation, metrics)
    return [t["title"] for t in tasks]


def main() -> None:
    # too_passive：方向必须是“把反思转成行动”，绝不能是“增强反思频率”。
    passive_titles = _titles_for("反思→行动转换率过低")
    assert any("将近期反思转化为可验证行动" in t for t in passive_titles), passive_titles
    assert not any("增强自主反思频率与质量" in t for t in passive_titles), passive_titles

    # too_reactive：方向必须是收敛行动节奏。
    reactive_titles = _titles_for("反思→行动转换率过高")
    assert any("收敛行动节奏" in t for t in reactive_titles), reactive_titles

    # 真正的“反思不足/缺失”仍应归类为增强反思频率，确保未误伤原有规则。
    missing_titles = _titles_for("未检测到反思/复盘信号")
    assert any("增强自主反思频率与质量" in t for t in missing_titles), missing_titles

    print("backlog_action_ratio_classification_selfcheck: ok")


if __name__ == "__main__":
    main()
