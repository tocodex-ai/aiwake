# -*- coding: utf-8 -*-
"""Prompt Cache 路由、usage 可观测性与稳定前缀回归测试。"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace


AGENT_DIR = Path(__file__).resolve().parents[1]
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

import heartbeat  # noqa: E402
import llm_gate  # noqa: E402


class FakeResponse:
    def __init__(self, data: dict, *, status_code: int = 200, text: str = ""):
        self._data = data
        self.status_code = status_code
        self.text = text

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}: {self.text}")

    def json(self) -> dict:
        return self._data


def _configured_gate(monkeypatch) -> llm_gate.LLMGate:
    monkeypatch.setenv("PROMPT_CACHE_ENABLED", "true")
    monkeypatch.setenv("PROMPT_CACHE_KEY_PREFIX", "aiwake-test")
    gate = llm_gate.LLMGate()
    profile = gate.profiles["work"]
    profile.api_url = "https://example.invalid/v1"
    profile.api_key = "test-only-key"
    profile.model = "test-model"
    return gate


def test_prompt_cache_usage_variants_are_parsed() -> None:
    openai_usage = llm_gate.LLMGate._extract_prompt_cache_usage({
        "usage": {
            "prompt_tokens": 2048,
            "prompt_tokens_details": {
                "cached_tokens": 1536,
                "cache_write_tokens": 256,
            },
        },
    })
    responses_usage = llm_gate.LLMGate._extract_prompt_cache_usage({
        "usage": {
            "input_tokens": 1800,
            "input_tokens_details": {"cached_tokens": 1024},
        },
    })
    compatible_usage = llm_gate.LLMGate._extract_prompt_cache_usage({
        "usage": {
            "prompt_tokens": 1600,
            "prompt_cache_hit_tokens": 896,
            "cache_creation_input_tokens": 128,
        },
    })

    assert openai_usage == {
        "reported": True,
        "prompt_tokens": 2048,
        "cached_tokens": 1536,
        "cache_write_tokens": 256,
    }
    assert responses_usage["cached_tokens"] == 1024
    assert compatible_usage["cached_tokens"] == 896
    assert compatible_usage["cache_write_tokens"] == 128


def test_call_cloud_injects_stable_key_and_records_cache_hit(monkeypatch) -> None:
    payloads: list[dict] = []
    responses = [
        FakeResponse({
            "choices": [{"message": {"content": "first"}}],
            "usage": {
                "prompt_tokens": 2048,
                "prompt_tokens_details": {
                    "cached_tokens": 0,
                    "cache_write_tokens": 1536,
                },
            },
        }),
        FakeResponse({
            "choices": [{"message": {"content": "second"}}],
            "usage": {
                "prompt_tokens": 2050,
                "prompt_tokens_details": {
                    "cached_tokens": 1536,
                    "cache_write_tokens": 0,
                },
            },
        }),
    ]

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):  # noqa: ANN001
            return False

        async def post(self, url, *, headers, json):  # noqa: ANN001
            payloads.append(dict(json))
            return responses.pop(0)

    monkeypatch.setattr(llm_gate.httpx, "AsyncClient", FakeAsyncClient)
    gate = _configured_gate(monkeypatch)

    first = asyncio.run(gate.call_cloud("stable-system", "dynamic-one"))
    second = asyncio.run(gate.call_cloud("stable-system", "dynamic-two"))

    assert (first, second) == ("first", "second")
    assert payloads[0]["prompt_cache_key"] == payloads[1]["prompt_cache_key"]
    assert payloads[0]["prompt_cache_key"] == "aiwake-test:work:test-model"
    assert payloads[0]["messages"][0] == payloads[1]["messages"][0]
    status = gate.prompt_cache_status()["work"]
    assert status == {
        "responses": 2,
        "usage_reported": 2,
        "cache_hits": 1,
        "prompt_tokens": 4098,
        "cached_tokens": 1536,
        "cache_write_tokens": 1536,
    }


def test_unsupported_prompt_cache_key_retries_without_field(monkeypatch) -> None:
    payloads: list[dict] = []
    responses = [
        FakeResponse(
            {"error": {"message": "Unknown parameter prompt_cache_key"}},
            status_code=400,
            text="Unknown parameter: prompt_cache_key",
        ),
        FakeResponse({
            "choices": [{"message": {"content": "fallback-ok"}}],
            "usage": {"prompt_tokens": 1200},
        }),
    ]

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):  # noqa: ANN001
            return False

        async def post(self, url, *, headers, json):  # noqa: ANN001
            payloads.append(dict(json))
            return responses.pop(0)

    monkeypatch.setattr(llm_gate.httpx, "AsyncClient", FakeAsyncClient)
    gate = _configured_gate(monkeypatch)

    result = asyncio.run(gate.call_cloud("stable-system", "user"))

    assert result == "fallback-ok"
    assert "prompt_cache_key" in payloads[0]
    assert "prompt_cache_key" not in payloads[1]
    assert gate._prompt_cache_key_supported["work"] is False


def test_dynamic_time_is_after_long_static_system_prefix() -> None:
    loop = heartbeat.HeartbeatLoop.__new__(heartbeat.HeartbeatLoop)
    loop.personality_prompt = "STATIC-PERSONALITY-MARKER\n"
    loop.state = SimpleNamespace(
        TR=0.5,
        CS=0.5,
        SA=0.4,
        energy_level="平稳",
        mood_level="平稳",
        patience_level="平稳",
        active_goal="",
    )
    loop._consecutive_no_tool_reflects = 0
    loop._consecutive_no_action_reflects = 0

    prompt = loop._build_system_prompt(["专注"])

    identity_index = prompt.index("【身份锚定 - 最高优先级指令】")
    personality_index = prompt.index("STATIC-PERSONALITY-MARKER")
    core_index = prompt.index("【总纲：意识的涌现模型】")
    time_index = prompt.index("【当前真实时间】")
    state_index = prompt.index("【当前内在状态仪表盘】")
    assert identity_index < personality_index < core_index < time_index < state_index
    # 仅业务 system prompt 的稳定区已超过 2000 个字符；LLMGate 还会在其前面
    # 注入更长的 agent_card 与 runtime rules，整体稳定前缀高于 1024-token 门槛。
    assert time_index > 2000
