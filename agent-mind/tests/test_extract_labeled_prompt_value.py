# -*- coding: utf-8 -*-
"""Self-checks for labeled prompt value extraction used by deterministic tools."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_MAIN_PATH = Path(__file__).resolve().parents[1] / "main.py"


def _load_main():
    agent_dir = str(_MAIN_PATH.parent)
    if agent_dir not in sys.path:
        sys.path.insert(0, agent_dir)
    spec = importlib.util.spec_from_file_location("aiwake_main_label_extract_test", _MAIN_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("aiwake_main_label_extract_test", module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_extract_title_from_chinese_quoted_suggestion() -> None:
    main_mod = _load_main()
    msg = (
        "A. 用 self_task_create 新建唯一 open task（标题建议）：\n"
        "   「激活 failure_triads + 挖掘 other_error 模式」\n"
        "   goal：对最近 other_error 失败做结构化证据记录与聚类，产出可验证学习产物。"
    )
    title = main_mod._extract_labeled_prompt_value(
        msg,
        "title",
        default="短消息下避免空输出降级并保留工具证据",
    )
    assert title == "激活 failure_triads + 挖掘 other_error 模式", title


def test_extract_goal_from_chinese_colon() -> None:
    main_mod = _load_main()
    msg = "goal：对最近 other_error 失败做结构化证据记录与聚类，产出可验证学习产物。"
    goal = main_mod._extract_labeled_prompt_value(msg, "goal", default="fallback")
    assert "other_error" in goal
    assert "结构化证据" in goal


def main() -> None:
    test_extract_title_from_chinese_quoted_suggestion()
    test_extract_goal_from_chinese_colon()
    print("test_extract_labeled_prompt_value: ok")


if __name__ == "__main__":
    main()
