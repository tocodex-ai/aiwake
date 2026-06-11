"""Self-check for the self_dialogue [GOAL_REGISTER] protocol parser.

Verifies that the reflection layer can parse [GOAL_REGISTER]{...} blocks emitted
by the LLM and turn them into real goal_tracker registrations.

Run:
    python src/agent-mind/tests/test_self_dialogue_goal_protocol.py
"""
from __future__ import annotations

import os
import sys
import tempfile
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


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["EVOLUTION_DIR"] = tmp

        for name in list(sys.modules.keys()):
            if name.startswith("evolution.goal_tracker") or name.startswith("evolution.self_dialogue"):
                del sys.modules[name]

        from evolution.self_dialogue import _parse_and_register_goals  # type: ignore
        from evolution.goal_tracker import read_open_goals  # type: ignore

        # Case 1: 单条合法协议块 → 登记成功
        text1 = (
            "Critic: 工具调用偶发失败，影响每日成长。\n"
            "Builder: 把 tool_success_rate 提到 0.99。\n"
            'detected_risks: ["工具偶发失败"]\n'
            '[GOAL_REGISTER] {"metric": "tool_success_rate", "direction": "up", '
            '"target": 0.99, "description": "把 tool_success_rate 从约 0.97 提升到 0.99"}\n'
        )
        r1 = _parse_and_register_goals(text1)
        assert len(r1) == 1 and r1[0]["status"] == "ok", f"Case 1 应登记成功: {r1}"
        opens = read_open_goals()
        assert any(g["metric"] == "tool_success_rate" for g in opens), "应能在 open goals 中读到刚登记的目标"

        # Case 2: 同 metric 重复登记应被反思层防重，跳过
        r2 = _parse_and_register_goals(text1)
        assert r2 and r2[0]["status"] == "skipped", f"Case 2 应被防重跳过: {r2}"

        # Case 3: 非法 direction → 跳过，不抛
        text3 = '[GOAL_REGISTER] {"metric": "tool_failure_count", "direction": "sideways", "target": 0, "description": "..."}'
        r3 = _parse_and_register_goals(text3)
        assert r3 and r3[0]["status"] == "skipped" and "direction" in r3[0]["reason"], f"Case 3 应非法 direction: {r3}"

        # Case 4: 未知 metric → 跳过
        text4 = '[GOAL_REGISTER] {"metric": "unknown_xyz", "direction": "up", "target": 1.0, "description": "..."}'
        r4 = _parse_and_register_goals(text4)
        assert r4 and r4[0]["status"] == "skipped" and "未知 metric" in r4[0]["reason"], f"Case 4 应未知 metric: {r4}"

        # Case 5: JSON 损坏 → 跳过，不抛
        text5 = '[GOAL_REGISTER] {"metric": "tool_failure_count", "direction": "down", target=0, "description":"bad json"}'
        r5 = _parse_and_register_goals(text5)
        assert r5 and r5[0]["status"] == "skipped" and "JSON" in r5[0]["reason"], f"Case 5 应 JSON 解析失败: {r5}"

        # Case 6: 文本中没有协议块 → 空列表
        r6 = _parse_and_register_goals("纯反思文本，没有协议块。")
        assert r6 == [], f"Case 6 应返回空列表: {r6}"

        # Case 7: 多条协议块 + 不同 metric → 都登记
        text7 = (
            '[GOAL_REGISTER] {"metric": "tool_failure_count", "direction": "down", "target": 0, "description": "降到 0"}\n'
            '[GOAL_REGISTER] {"metric": "chat_success_rate", "direction": "up", "target": 0.99, "description": "聊天成功率到 0.99"}\n'
        )
        r7 = _parse_and_register_goals(text7)
        ok_count = sum(1 for x in r7 if x["status"] == "ok")
        assert ok_count == 2, f"Case 7 应两条都登记成功: {r7}"

        print("OK self_dialogue [GOAL_REGISTER] protocol parser self-check passed.")


if __name__ == "__main__":
    main()
