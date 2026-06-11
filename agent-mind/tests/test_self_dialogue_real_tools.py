"""self_dialogue 走真工具路径自检：tool_router 注入后，反思 LLM 走 call_with_tools。

重点验证：
- 当 tool_router 提供 get_openai_tools_schema/call 时，反思路径会把它们注入 call_reflect_messages；
- 文本协议 [GOAL_REGISTER]、[SELF_CODE_WRITE] 仍然作为兜底正确解析；
- 当 tool_router 缺失/异常时，自动回退到旧的纯文本路径，不抛错。

运行：
    python src/agent-mind/tests/test_self_dialogue_real_tools.py
"""
from __future__ import annotations

import asyncio
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
    raise RuntimeError("Cannot locate agent-mind directory")


AGENT_DIR = _find_agent_dir()
sys.path.insert(0, str(AGENT_DIR))


class _RecorderLLM:
    """模拟 LLMGate.call_reflect_messages：记录是否注入了 tools_schema/tool_executor。"""

    def __init__(self, response_text: str):
        self.response_text = response_text
        self.last_kwargs: dict = {}
        self.calls: int = 0

    async def call_reflect_messages(self, messages, **kwargs):  # noqa: D401
        self.calls += 1
        self.last_kwargs = kwargs
        return self.response_text


class _StubToolRouter:
    """模拟 ToolRouter，提供反思路径所需的最小接口。"""

    def __init__(self):
        self.last_call: tuple[str, dict] | None = None

    def get_openai_tools_schema(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "goal_list",
                    "description": "stub",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

    async def call(self, tool_name: str, params: dict) -> dict:
        self.last_call = (tool_name, params)
        return {"status": "ok", "tool": tool_name, "result": []}


async def _case_real_tool_path_injection() -> None:
    from evolution import self_dialogue

    state = {"name": "AIwake"}
    metrics = {"tool_failure_count": 0}
    evaluation = {"summary": "ok"}

    llm = _RecorderLLM(
        response_text="""Critic: ...\nBuilder: ...\nNext: 登记 tool_failure_count\n[GOAL_REGISTER]{"metric":"tool_failure_count","direction":"down","target":0,"description":"reflection-test"}\n"""
    )
    router = _StubToolRouter()

    out = await self_dialogue.run_self_dialogue(
        llm, state, metrics, evaluation, tool_router=router
    )
    assert llm.calls == 1
    # tools_schema/tool_executor/max_rounds 必须被注入到 call_reflect_messages
    assert "tools_schema" in llm.last_kwargs, "tools_schema must be injected"
    assert "tool_executor" in llm.last_kwargs, "tool_executor must be injected"
    assert llm.last_kwargs.get("max_rounds") == 8
    assert isinstance(out, dict)
    # 文本协议同时也应被解析（兜底，不破坏）
    registered = out.get("registered_goals") or []
    assert isinstance(registered, list), "registered_goals should be a list"
    # 至少有一条对 tool_failure_count 的记录（实际 success 取决于 goal_tracker，但调用必须发生）
    assert any(g.get("metric") == "tool_failure_count" for g in registered), (
        f"text protocol [GOAL_REGISTER] should be parsed, got {registered}"
    )


async def _case_no_router_falls_back_to_text_only() -> None:
    from evolution import self_dialogue

    state = {"name": "AIwake"}
    metrics = {"tool_failure_count": 0}
    evaluation = {"summary": "ok"}

    llm = _RecorderLLM(response_text="Critic: ...\nBuilder: ...\n")

    out = await self_dialogue.run_self_dialogue(llm, state, metrics, evaluation, tool_router=None)
    assert llm.calls == 1
    # 没有 router 时，不应注入 tools_schema/tool_executor
    assert "tools_schema" not in llm.last_kwargs
    assert "tool_executor" not in llm.last_kwargs
    assert isinstance(out, dict)


async def _case_broken_router_does_not_crash() -> None:
    from evolution import self_dialogue

    class _BrokenRouter:
        def get_openai_tools_schema(self):
            raise RuntimeError("schema build broken")

        async def call(self, tool_name, params):
            return {"error": "should not be called"}

    llm = _RecorderLLM(response_text="Critic: ok\nBuilder: ok\n")
    out = await self_dialogue.run_self_dialogue(
        llm, {}, {"tool_failure_count": 0}, {"summary": "ok"}, tool_router=_BrokenRouter()
    )
    # 抛错 → 应自动降级为纯文本路径，仍然返回 dict
    assert isinstance(out, dict)
    assert "tools_schema" not in llm.last_kwargs
    assert "tool_executor" not in llm.last_kwargs


async def _main() -> None:
    await _case_real_tool_path_injection()
    await _case_no_router_falls_back_to_text_only()
    await _case_broken_router_does_not_crash()
    print("self_dialogue_real_tools_selfcheck: ok")


if __name__ == "__main__":
    asyncio.run(_main())
