"""Vitality 必须保持观察模式，不得限制 Agent 的工具或反思输出预算。"""
from __future__ import annotations

import sys
from pathlib import Path


AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENT_DIR))

from evolution.vitality import (  # noqa: E402
    VitalityState,
    filter_tools_schema_by_vitality,
    reflect_max_tokens_for_vitality,
)


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_every_vitality_state_keeps_full_tool_schema() -> None:
    schema = [
        _tool("web_search"),
        _tool("web_fetch"),
        _tool("weather"),
        _tool("experiment_status"),
        _tool("self_code_write"),
        _tool("shell_exec"),
    ]

    for state in VitalityState:
        filtered, hidden = filter_tools_schema_by_vitality(
            schema,
            state,
            action_tools={"self_code_write", "shell_exec"},
        )
        assert filtered == schema, state
        assert filtered is not schema, "应返回列表副本，避免调用方意外修改原 schema"
        assert hidden == [], state


def test_every_vitality_state_leaves_reflection_tokens_unlimited() -> None:
    for state in VitalityState:
        assert reflect_max_tokens_for_vitality(state) is None, state
