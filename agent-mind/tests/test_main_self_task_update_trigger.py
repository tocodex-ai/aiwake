"""Self-checks for deterministic self-task update trigger parsing.

Run from repository root:
    python src/agent-mind/tests/test_main_self_task_update_trigger.py
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

from main import _is_explicit_self_task_update_request  # noqa: E402


def main() -> None:
    conditional_prompt = (
        "若能提取至少3个可验证高位TR模式规律，请调用 self_artifact_create 绑定 task_id=tas_123，"
        "随后如 done_when 已满足，可调用 self_task_update 关闭任务并记录 done_when_met；"
        "若证据不足，也创建补充 artifact。"
    )
    assert not _is_explicit_self_task_update_request(conditional_prompt)

    decide_later_prompt = (
        "对已有 matching artifact 的 open task，下一步应检查 done_when，"
        "再决定是否 self_task_update。"
    )
    assert not _is_explicit_self_task_update_request(decide_later_prompt)

    imperative_prompt = (
        "task-30 关闭当前自学习闭环：调用 self_task_update，"
        "task_id=tas_123，status=closed。done_when_met: artifact 已匹配。"
    )
    assert _is_explicit_self_task_update_request(imperative_prompt)

    must_prompt = "必须调用 self_task_update，task_id=tas_123，status=closed。"
    assert _is_explicit_self_task_update_request(must_prompt)

    print("main_self_task_update_trigger_selfcheck: ok")


if __name__ == "__main__":
    main()
