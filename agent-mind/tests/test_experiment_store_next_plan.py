"""Self-checks for ExperimentStore next_plan sequencing.

Run from repository root:
    python src/agent-mind/tests/test_experiment_store_next_plan.py
"""
from __future__ import annotations

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

from experiment_store import ExperimentStore


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = ExperimentStore(base_dir=tmp)
        assert store.status()["next_plan"] == "创建第一条自我学习任务，建立可观察闭环。"

        task = store.append("tasks", {
            "type": "self_learning_task_card",
            "title": "测试任务",
            "status": "open",
            "done_when": "生成 artifact 并可查询。",
        })
        status_after_task = store.status()
        assert status_after_task["open_task_count"] == 1
        assert status_after_task["next_plan"] == (
            "最新自任务仍 open 且尚无匹配 artifact；先围绕 latest_task 做最小闭环："
            "读取任务 goal/done_when 和 daily_note，若证据仍零增量则记录早退验证或生成学习 artifact，"
            "不要直接发起新的外部搜索。"
        )

        store.append("artifacts", {
            "task_id": task["id"],
            "kind": "learning_artifact_draft",
            "title": "测试 artifact",
            "status": "draft",
        })
        status_after_artifact = store.status()
        assert status_after_artifact["artifact_count"] == 1
        assert status_after_artifact["task_count"] == 1
        assert status_after_artifact["task_event_count"] == 1
        assert status_after_artifact["next_plan"] == (
            "复盘最新 artifact 是否满足任务 done_when；若满足，记录可观察闭环已达成但任务仍 open，"
            "并提出关闭或补充任务卡的下一步。"
        )

        store.append("tasks", {
            "id": task["id"],
            "type": "self_task_update",
            "status": "closed",
            "note": "done_when 已由 artifact_count>=1 与 latest_artifact 指向目标产物证明满足。",
        })
        status_after_close = store.status()
        assert status_after_close["task_count"] == 1
        assert status_after_close["task_event_count"] == 2
        assert status_after_close["open_task_count"] == 0
        assert status_after_close["latest_task"]["status"] == "closed"
        assert status_after_close["latest_task"]["last_update"]["type"] == "self_task_update"
        assert status_after_close["next_plan"] == (
            "最新任务已关闭；继续轮询状态，选择一个新的小问题进行搜索/总结/反思，累积可见学习成果。"
        )

    print("experiment_store_next_plan_selfcheck: ok")


if __name__ == "__main__":
    main()
