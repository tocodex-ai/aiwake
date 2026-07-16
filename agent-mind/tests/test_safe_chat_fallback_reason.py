# -*- coding: utf-8 -*-
"""empty_fallback 降级回复必须如实归因，不得把"上游空输出"误标为"编码损坏"。

背景（task-60 第 36 轮线上证据）：work API 达到本小时调用上限导致 LLM 空输出时，
/chat 走 empty_fallback 路径却复用了乱码安全降级文案（"检测到……编码损坏"），
把配额限制误报为编码事故，污染公开聊天与训练证据链。
"""
import importlib.util
import sys
from pathlib import Path

import pytest

_MAIN_PATH = Path(__file__).resolve().parents[1] / "main.py"


def _load_main():
    spec = importlib.util.spec_from_file_location("aiwake_main_for_test", _MAIN_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("aiwake_main_for_test", module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def main_mod():
    return _load_main()


def test_empty_reason_mentions_quota_not_mojibake(main_mod):
    reply = main_mod._safe_chat_fallback_reply("训练导师消息", reason="empty")
    assert "配额" in reply or "速率限制" in reply
    assert "编码损坏" not in reply
    assert "乱码" not in reply


def test_default_reason_keeps_mojibake_wording(main_mod):
    reply = main_mod._safe_chat_fallback_reply("训练导师消息")
    assert "编码损坏" in reply


def test_empty_reason_preserves_topic_echo(main_mod):
    reply = main_mod._safe_chat_fallback_reply("task-60 第 36 轮训练", reason="empty")
    assert "task-60" in reply


def test_empty_reason_without_topic_is_safe(main_mod):
    reply = main_mod._safe_chat_fallback_reply("", reason="empty")
    assert reply.strip()
    assert "编码损坏" not in reply
