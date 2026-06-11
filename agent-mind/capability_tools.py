"""能力库 → 工具库 桥接：从 capability_library.jsonl 自动注册新工具。

设计要点：
- 能力库条目里的 tool_spec 字段是可选的——只有 spec 完整且通过安全检查的才会被注入。
- 注册的工具仅限 evolution.* 子模块下的 callable，且模块路径必须通过
  is_self_project_code_path 校验，避免任意 import。
- 注册后，工具同时出现在 ALLOWED_TOOLS / get_openai_tools_schema / dispatcher，
  对反思层（call_with_tools）/ /chat / 容器内三条路径同时可见。
- 启动失败时静默跳过单条记录，整体加载不应让进程崩溃。
"""
from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import Any, Callable

from safety_guard import is_self_project_code_path

logger = logging.getLogger(__name__)

# 能力库文件位置（与 evolution.goal_tracker._capabilities_file 保持一致）
_DATA_DIR = Path("/app/data") if Path("/app/data").exists() else Path(__file__).parent / "data"
_CAPABILITIES_FILE = _DATA_DIR / "evolution" / "capability_library.jsonl"

# 仅允许这些前缀作为 module_path，避免任意 import
_ALLOWED_MODULE_PREFIXES = ("evolution.",)


def _capability_tool_module_path_to_relpath(module_path: str) -> str:
    """把 'evolution.self_check_helper' 翻译成 'evolution/self_check_helper.py' 用于安全路径检查。"""
    return module_path.replace(".", "/") + ".py"


def _validate_tool_spec(spec: dict[str, Any]) -> tuple[bool, str]:
    """校验 tool_spec 是否合法且安全。"""
    if not isinstance(spec, dict):
        return False, "tool_spec 不是 dict"
    name = spec.get("name")
    module_path = spec.get("module_path")
    callable_name = spec.get("callable")
    description = spec.get("description")
    params_schema = spec.get("params_schema")
    if not isinstance(name, str) or not name.isidentifier():
        return False, f"name 必须是合法 Python 标识符，得到 {name!r}"
    if not isinstance(module_path, str) or not any(module_path.startswith(p) for p in _ALLOWED_MODULE_PREFIXES):
        return False, f"module_path 必须以 {_ALLOWED_MODULE_PREFIXES} 之一开头，得到 {module_path!r}"
    if not isinstance(callable_name, str) or not callable_name.isidentifier():
        return False, f"callable 必须是合法 Python 标识符，得到 {callable_name!r}"
    if not isinstance(description, str) or not description.strip():
        return False, "description 不能为空"
    if not isinstance(params_schema, dict) or params_schema.get("type") != "object":
        return False, "params_schema 必须是 type=object 的 JSON Schema"
    rel = _capability_tool_module_path_to_relpath(module_path)
    if not is_self_project_code_path(rel):
        return False, f"module_path 不在 AIwake 项目代码路径范围: {rel}"
    return True, ""


def _read_capability_entries() -> list[dict[str, Any]]:
    if not _CAPABILITIES_FILE.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        for line in _CAPABILITIES_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError as e:
        logger.warning(f"[capability_tools] 读取能力库失败: {e}")
    return out


def load_capability_tools() -> dict[str, dict[str, Any]]:
    """扫描 capability_library.jsonl，返回 {name: registration_record}。

    registration_record 包含：
      - "spec": 原始 tool_spec 字典
      - "callable": 已 import 的函数对象
      - "schema": OpenAI tools schema 条目
      - "risk": 工具风险评分
      - "readonly": 是否并发安全只读
      - "source_capability_id": 能力库条目 id
    """
    out: dict[str, dict[str, Any]] = {}
    seen_capability_id_to_name: dict[str, str] = {}
    for entry in _read_capability_entries():
        spec = entry.get("tool_spec")
        if not spec:
            continue
        ok, reason = _validate_tool_spec(spec)
        if not ok:
            logger.warning(f"[capability_tools] 跳过非法 tool_spec: {reason}")
            continue
        name = spec["name"]
        # 同名后写覆盖前写：能力库本就是 append-only，新一行视为更新
        cap_id = entry.get("id") or ""
        seen_capability_id_to_name[cap_id] = name
        try:
            module = importlib.import_module(spec["module_path"])
            callable_obj: Callable[..., Any] = getattr(module, spec["callable"])
        except Exception as e:
            logger.warning(f"[capability_tools] import 失败 {spec.get('module_path')}.{spec.get('callable')}: {e}")
            continue
        if not callable(callable_obj):
            logger.warning(f"[capability_tools] {spec.get('callable')!r} 不是 callable")
            continue
        risk = float(spec.get("risk", 0.2))
        risk = max(0.0, min(0.9, risk))  # 自动注册的工具上限 0.9，避免破坏性
        readonly = bool(spec.get("readonly", False))
        schema_entry = {
            "type": "function",
            "function": {
                "name": name,
                "description": spec["description"][:500],
                "parameters": spec["params_schema"],
            },
        }
        out[name] = {
            "spec": spec,
            "callable": callable_obj,
            "schema": schema_entry,
            "risk": risk,
            "readonly": readonly,
            "source_capability_id": cap_id,
        }
    if out:
        logger.info(f"[capability_tools] 已注册 {len(out)} 个能力库工具: {list(out.keys())}")
    return out


def call_capability_tool(record: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """安全调用一个能力库工具，统一返回 {status, tool, result|error}."""
    spec = record.get("spec", {})
    name = spec.get("name", "<unknown>")
    callable_obj = record.get("callable")
    if not callable(callable_obj):
        return {"status": "error", "tool": name, "error": "callable 缺失"}
    if not isinstance(params, dict):
        params = {}
    try:
        # 仅按 params_schema 顶层 properties 中声明的键传参，避免被注入未知关键词
        props = (spec.get("params_schema") or {}).get("properties", {}) or {}
        kwargs = {k: params[k] for k in props.keys() if k in params}
        value = callable_obj(**kwargs)
        return {"status": "ok", "tool": name, "result": value, "source": "capability_library"}
    except TypeError as e:
        return {"status": "error", "tool": name, "error": f"参数不匹配: {e}"}
    except Exception as e:
        return {"status": "error", "tool": name, "error": f"执行异常: {e}"}
