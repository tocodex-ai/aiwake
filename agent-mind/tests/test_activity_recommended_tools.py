"""Self-check: file_not_found 等失败结果若带 recommended_tools，
活动日志 content 文本与 extra 都应持久化该自愈提示；无该字段时保持原样。

覆盖 /chat 路径的 _format_tool_result_activity（main.py）。心跳路径 heartbeat.py
使用相同的拼接约定（recommend_suffix），逻辑等价，此处以纯函数路径做断言。

Run from repository root:
    python src/agent-mind/tests/test_activity_recommended_tools.py
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

from main import _format_tool_result_activity  # noqa: E402


def main() -> None:
    # 1. file_not_found 带 recommended_tools → content 与 extra 都应包含
    res = {
        "error": "文件不存在: /tmp/self_upgrade_status.md",
        "tool": "file_read",
        "path": "/tmp/self_upgrade_status.md",
        "diagnosis": "file_not_found",
        "recommended_tools": ["self_upgrade_status"],
    }
    content, extra = _format_tool_result_activity("file_read", res)
    assert "recommended_tools=['self_upgrade_status']" in content, f"content 缺少推荐: {content}"
    assert extra.get("recommended_tools") == ["self_upgrade_status"], f"extra 缺少推荐: {extra}"
    assert "classification=" in content, f"content 应保留 classification: {content}"

    # 2. experiment_status 同理
    res2 = {
        "error": "文件不存在: data/experiment_status.json",
        "tool": "file_read",
        "diagnosis": "file_not_found",
        "recommended_tools": ["experiment_status"],
    }
    content2, extra2 = _format_tool_result_activity("file_read", res2)
    assert "recommended_tools=['experiment_status']" in content2, f"content2 缺少推荐: {content2}"
    assert extra2.get("recommended_tools") == ["experiment_status"]

    # 3. 普通失败（无 recommended_tools）→ content 不应出现该后缀，extra 不应有该键
    res3 = {
        "error": "文件不存在: data/does_not_exist.py",
        "tool": "file_read",
        "diagnosis": "file_not_found",
    }
    content3, extra3 = _format_tool_result_activity("file_read", res3)
    assert "recommended_tools" not in content3, f"普通失败不应带推荐后缀: {content3}"
    assert "recommended_tools" not in extra3, f"普通失败 extra 不应带推荐键: {extra3}"

    # 4. 成功结果不应受影响
    res4 = {"status": "ok", "tool": "self_upgrade_status", "result": "状态正常"}
    content4, extra4 = _format_tool_result_activity("self_upgrade_status", res4)
    assert "recommended_tools" not in content4
    assert "recommended_tools" not in extra4

    # 5. 空列表视为无推荐
    res5 = {"error": "x", "tool": "file_read", "recommended_tools": []}
    content5, extra5 = _format_tool_result_activity("file_read", res5)
    assert "recommended_tools" not in content5
    assert "recommended_tools" not in extra5

    print("activity_recommended_tools_selfcheck: ok")


if __name__ == "__main__":
    main()
