"""能力库 → 工具库 桥接自检：capability_library.jsonl + tool_spec → ToolRouter 可调用。

不依赖 pytest，单文件可独立运行：

    python src/agent-mind/tests/test_capability_to_tool_bridge.py
"""
from __future__ import annotations

import asyncio
import json
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


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _case_validate_and_load() -> None:
    import capability_tools  # noqa: WPS433

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        cap_file = tmp / "evolution" / "capability_library.jsonl"
        # 1) 合法 entry：调用 evolution.goal_tracker.read_capabilities（参数 limit:int）
        # 2) 非法 module_path：os.system 这种应被拒
        # 3) 缺字段：tool_spec 缺 description 应被拒
        rows = [
            {
                "id": "cap_ok_1",
                "tool_spec": {
                    "name": "read_caps_test",
                    "module_path": "evolution.goal_tracker",
                    "callable": "read_capabilities",
                    "description": "只读：读取能力库（测试用）",
                    "params_schema": {
                        "type": "object",
                        "properties": {
                            "limit": {"type": "integer", "minimum": 1, "maximum": 100}
                        },
                    },
                    "risk": 0.1,
                    "readonly": True,
                },
            },
            {
                "id": "cap_bad_module",
                "tool_spec": {
                    "name": "evil_call",
                    "module_path": "os",
                    "callable": "system",
                    "description": "应被拒",
                    "params_schema": {"type": "object", "properties": {}},
                },
            },
            {
                "id": "cap_missing_desc",
                "tool_spec": {
                    "name": "no_desc",
                    "module_path": "evolution.goal_tracker",
                    "callable": "read_capabilities",
                    "description": "",
                    "params_schema": {"type": "object", "properties": {}},
                },
            },
            {
                "id": "cap_no_spec_ok_skip",
                # 没有 tool_spec 字段：应直接 skip，不视为错误
                "metric": "passive_action_score",
            },
        ]
        _write_jsonl(cap_file, rows)
        # 用 monkeypatch 方式临时改 capability_tools 模块的常量
        original = capability_tools._CAPABILITIES_FILE
        capability_tools._CAPABILITIES_FILE = cap_file
        try:
            registered = capability_tools.load_capability_tools()
        finally:
            capability_tools._CAPABILITIES_FILE = original

        assert "read_caps_test" in registered, f"valid spec should register, got {list(registered.keys())}"
        assert "evil_call" not in registered, "non-evolution.* module_path must be rejected"
        assert "no_desc" not in registered, "empty description must be rejected"

        rec = registered["read_caps_test"]
        assert callable(rec["callable"]), "callable should be imported"
        assert rec["risk"] <= 0.9
        assert rec["readonly"] is True
        schema = rec["schema"]
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "read_caps_test"
        assert "limit" in schema["function"]["parameters"]["properties"]


def _case_call_capability_tool() -> None:
    import capability_tools  # noqa: WPS433

    spec = {
        "name": "read_caps_test",
        "module_path": "evolution.goal_tracker",
        "callable": "read_capabilities",
        "description": "只读：读取能力库（测试用）",
        "params_schema": {
            "type": "object",
            "properties": {"limit": {"type": "integer"}},
        },
        "risk": 0.1,
        "readonly": True,
    }
    # 直接构造 record，绕过 jsonl
    from evolution.goal_tracker import read_capabilities  # noqa: WPS433
    record = {
        "spec": spec,
        "callable": read_capabilities,
        "schema": {},
        "risk": 0.1,
        "readonly": True,
        "source_capability_id": "manual",
    }
    out = capability_tools.call_capability_tool(record, {"limit": 5, "extra_unknown": "must_be_filtered"})
    assert out.get("status") == "ok", f"call should succeed, got {out}"
    assert out.get("tool") == "read_caps_test"
    assert isinstance(out.get("result"), list), "read_capabilities returns list"
    assert out.get("source") == "capability_library"


async def _case_tool_router_dispatch() -> None:
    """ToolRouter 启动时应通过 _reload_capability_tools 把能力库工具暴露给 schema/dispatch。"""
    import capability_tools  # noqa: WPS433
    import tool_router  # noqa: WPS433

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        cap_file = tmp / "evolution" / "capability_library.jsonl"
        _write_jsonl(cap_file, [
            {
                "id": "cap_router_1",
                "tool_spec": {
                    "name": "router_read_caps",
                    "module_path": "evolution.goal_tracker",
                    "callable": "read_capabilities",
                    "description": "只读：路由器测试调用 read_capabilities",
                    "params_schema": {
                        "type": "object",
                        "properties": {"limit": {"type": "integer"}},
                    },
                    "risk": 0.1,
                    "readonly": True,
                },
            }
        ])
        # 临时换掉 jsonl 路径并清空已注册的能力工具
        original = capability_tools._CAPABILITIES_FILE
        capability_tools._CAPABILITIES_FILE = cap_file
        # 清空类级能力工具注册表，避免被旧实例污染
        tool_router.ToolRouter._capability_tools = {}
        try:
            r = tool_router.ToolRouter()
            schema = r.get_openai_tools_schema()
            names = {item["function"]["name"] for item in schema if isinstance(item, dict) and isinstance(item.get("function"), dict)}
            assert "router_read_caps" in names, f"capability tool should appear in schema, got {sorted(names)}"
            # ALLOWED_TOOLS 也应被打入
            assert "router_read_caps" in tool_router.ALLOWED_TOOLS

            # dispatch 调用：能力库工具应能被 call() 真正执行并返回 status=ok
            res = await r.call("router_read_caps", {"limit": 3})
            assert res.get("status") == "ok", f"capability dispatch failed: {res}"
            assert res.get("source") == "capability_library"
            assert isinstance(res.get("result"), list)
        finally:
            capability_tools._CAPABILITIES_FILE = original
            # 清理：移除测试注册到 ALLOWED_TOOLS 的工具，避免污染其它测试
            tool_router.ALLOWED_TOOLS.pop("router_read_caps", None)
            tool_router.ToolRouter._capability_tools = {}


def main() -> None:
    _case_validate_and_load()
    _case_call_capability_tool()
    asyncio.run(_case_tool_router_dispatch())
    print("capability_to_tool_bridge_selfcheck: ok")


if __name__ == "__main__":
    main()
