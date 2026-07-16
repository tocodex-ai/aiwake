"""
DSML 工具调用解析器
===================

解析 LLM 文本输出中 ｜DSML｜ 格式（以及标准 XML、<function>、[SELF_CODE_WRITE] 等）
的工具调用标签，将其转换为统一的工具调用字典列表。

线上 AIwake 的 LLM 在反思中经常输出类似以下的伪 XML 工具调用格式::

    <｜｜DSML｜｜tool_calls>
      <｜｜DSML｜｜invoke name="shell_exec">
        <｜｜DSML｜｜parameter name="command" string="true">sed -n '213,400p' /app/tool_router.py</｜｜DSML｜｜parameter>
      </｜｜DSML｜｜invoke>
    </｜｜DSML｜｜tool_calls>

这些标签此前不被代码解析，只是被当作纯文本保存，导致 LLM "想调用工具但没调用"的空转。
本模块提供 ``parse_dsml_tool_calls()`` 将其解析为标准工具调用结构::

    [{"name": "shell_exec", "arguments": {"command": "sed -n ..."}}]

支持的格式:
  1. ｜DSML｜ 伪 XML（全角 ｜ U+FF5C / 半角 | U+007C 竖线均兼容）
  2. 标准 XML: ``<tool_calls><invoke name="..."><parameter name="...">...</parameter></invoke></tool_calls>``
  3. ``<function>tool_name(args)</function>``
  4. ``[SELF_CODE_WRITE]{...json...}``

无敏感信息，可安全开源。
"""

from __future__ import annotations

import json
import re

# ---------------------------------------------------------------------------
# 正则预编译
# ---------------------------------------------------------------------------

# ｜DSML｜ 标记中的竖线：兼容全角 ｜(U+FF5C) 与半角 |(U+007C)
_BAR = r'[\uFF5C\u007C]'
# 完整的 DSML 标记，如 ｜｜DSML｜｜ 或 ||DSML||（竖线数量 1~2 均兼容）
_DSML_MARKER = _BAR + r'{1,2}DSML' + _BAR + r'{1,2}'

# invoke 开始标签：<｜｜DSML｜｜invoke name="xxx"> 或 <invoke name="xxx">
_INVOKE_OPEN = re.compile(
    r'<(?:' + _DSML_MARKER + r')?invoke\s+name\s*=\s*["\']([^"\']*)["\']\s*>',
    re.DOTALL,
)

# invoke 结束标签
_INVOKE_CLOSE = re.compile(
    r'</(?:' + _DSML_MARKER + r')?invoke\s*>',
    re.DOTALL,
)

# parameter 标签：<...parameter name="xxx" string="true/false">value</...parameter>
_PARAM = re.compile(
    r'<(?:' + _DSML_MARKER + r')?parameter\s+name\s*=\s*["\']([^"\']*)["\']'
    r'(?:\s+string\s*=\s*["\']([^"\']*)["\'])?\s*>'
    r'(.*?)'
    r'</(?:' + _DSML_MARKER + r')?parameter\s*>',
    re.DOTALL,
)

# <function>tool_name(args)</function>
_FUNCTION = re.compile(
    r'<function>\s*(\w+)\s*\((.*?)\)\s*</function>',
    re.DOTALL,
)

# [SELF_CODE_WRITE] 标记
_SELF_CODE_WRITE_MARK = re.compile(r'\[SELF_CODE_WRITE\]')


# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------

def _coerce_value(raw: str, string_attr: str | None):
    """根据 string 属性尝试将参数值转为 int/float，否则保持字符串。

    string="false" 表示该参数不是字符串，尝试转为数字；
    string="true" 或缺省时保持原始字符串。
    """
    raw = raw.strip()
    if string_attr and string_attr.strip().lower() == 'false':
        try:
            if '.' in raw or 'e' in raw.lower():
                return float(raw)
            return int(raw)
        except (ValueError, TypeError):
            # 转换失败则保留原始字符串
            return raw
    return raw


def _extract_json_object(text: str, start: int) -> str | None:
    """从 ``start`` 位置开始提取一个花括号平衡的 JSON 对象字符串。

    正确处理嵌套花括号和字符串内的转义字符。
    """
    if start >= len(text) or text[start] != '{':
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == '\\':
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _parse_invoke_format(text: str) -> list[dict]:
    """解析 ｜DSML｜ / 标准 XML 的 invoke + parameter 结构。"""
    results: list[dict] = []
    for m in _INVOKE_OPEN.finditer(text):
        tool_name = m.group(1).strip()
        if not tool_name:
            continue
        # 从当前 invoke 开始标签之后查找对应的结束标签
        start = m.end()
        close_m = _INVOKE_CLOSE.search(text, start)
        block = text[start:close_m.start()] if close_m else text[start:]

        arguments: dict = {}
        for pm in _PARAM.finditer(block):
            pname = pm.group(1).strip()
            string_attr = pm.group(2)  # 可能为 None
            pvalue = pm.group(3)
            if pname:
                arguments[pname] = _coerce_value(pvalue, string_attr)

        results.append({"name": tool_name, "arguments": arguments})
    return results


def _parse_args_string(args_str: str) -> dict:
    """将参数字符串解析为字典，兼容 JSON 与 key=value 形式。"""
    if not args_str:
        return {}
    # 先尝试整体 JSON
    try:
        parsed = json.loads(args_str)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    # 尝试 key=value 形式: key="value" / key='value' / key=number / key=bare
    arguments: dict = {}
    kv_pat = re.compile(
        r'(\w+)\s*=\s*'
        r'(?:"([^"]*)"'
        r"|'([^']*)'"
        r'|([^,\s]+))',
    )
    for km in kv_pat.finditer(args_str):
        key = km.group(1)
        if km.group(2) is not None:
            val: object = km.group(2)
        elif km.group(3) is not None:
            val = km.group(3)
        else:
            raw = km.group(4)
            # 尝试数字转换
            try:
                if '.' in raw or 'e' in raw.lower():
                    val = float(raw)
                else:
                    val = int(raw)
            except (ValueError, TypeError):
                val = raw
        arguments[key] = val
    return arguments


def _parse_function_format(text: str) -> list[dict]:
    """解析 ``<function>tool_name(args)</function>`` 格式。"""
    results: list[dict] = []
    for m in _FUNCTION.finditer(text):
        tool_name = m.group(1).strip()
        args_str = m.group(2).strip()
        if not tool_name:
            continue
        results.append({"name": tool_name, "arguments": _parse_args_string(args_str)})
    return results


def _parse_self_code_write_format(text: str) -> list[dict]:
    """解析 ``[SELF_CODE_WRITE]{...json...}`` 格式。"""
    results: list[dict] = []
    for m in _SELF_CODE_WRITE_MARK.finditer(text):
        pos = m.end()
        # 跳过空白
        while pos < len(text) and text[pos] in ' \t\r\n':
            pos += 1
        json_str = _extract_json_object(text, pos)
        if not json_str:
            continue
        try:
            parsed = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(parsed, dict):
            continue
        name = parsed.get('name') or parsed.get('tool') or parsed.get('function')
        args = parsed.get('arguments') or parsed.get('args') or parsed.get('parameters') or {}
        if name:
            if not isinstance(args, dict):
                args = {}
            results.append({"name": str(name), "arguments": args})
    return results


# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------

def parse_dsml_tool_calls(text: str) -> list[dict]:
    """解析 LLM 文本输出中的工具调用标签。

    依次尝试以下格式，返回首个成功解析的结果:

      1. ｜DSML｜ / 标准 XML 的 invoke + parameter 结构
      2. ``<function>tool_name(args)</function>``
      3. ``[SELF_CODE_WRITE]{...json...}``

    Args:
        text: LLM 输出的原始文本。

    Returns:
        工具调用字典列表，格式为 ``[{"name": "...", "arguments": {...}}, ...]``。
        解析失败或无工具调用时返回空列表 ``[]``。
    """
    if not text or not isinstance(text, str):
        return []

    # 1. 优先解析 invoke 格式（最结构化）
    results = _parse_invoke_format(text)
    if results:
        return results

    # 2. <function>...</function> 格式
    results = _parse_function_format(text)
    if results:
        return results

    # 3. [SELF_CODE_WRITE]{...} 格式
    results = _parse_self_code_write_format(text)
    if results:
        return results

    return []


# ---------------------------------------------------------------------------
# 自检测试
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    tests = [
        # 1. DSML 全角竖线格式
        (
            '<｜｜DSML｜｜tool_calls> <｜｜DSML｜｜invoke name="shell_exec"> '
            '<｜｜DSML｜｜parameter name="command" string="true">'
            "sed -n '213,400p' /app/tool_router.py"
            '</｜｜DSML｜｜parameter> '
            '</｜｜DSML｜｜invoke> </｜｜DSML｜｜tool_calls>',
            [{'name': 'shell_exec', 'arguments': {'command': "sed -n '213,400p' /app/tool_router.py"}}],
        ),
        # 2. 标准 XML 格式
        (
            '<tool_calls><invoke name="read_file">'
            '<parameter name="path">/app/main.py</parameter>'
            '</invoke></tool_calls>',
            [{'name': 'read_file', 'arguments': {'path': '/app/main.py'}}],
        ),
        # 3. string="false" 数字转换
        (
            '<invoke name="search">'
            '<parameter name="query" string="true">hello</parameter>'
            '<parameter name="limit" string="false">10</parameter>'
            '<parameter name="ratio" string="false">0.5</parameter>'
            '</invoke>',
            [{'name': 'search', 'arguments': {'query': 'hello', 'limit': 10, 'ratio': 0.5}}],
        ),
        # 4. <function> 格式
        (
            '<function>shell_exec(command="ls -la")</function>',
            [{'name': 'shell_exec', 'arguments': {'command': 'ls -la'}}],
        ),
        # 5. [SELF_CODE_WRITE] 格式（含嵌套 JSON）
        (
            '[SELF_CODE_WRITE]{"name": "write_file", '
            '"arguments": {"path": "/app/x.py", "content": "print(1)"}}',
            [{'name': 'write_file', 'arguments': {'path': '/app/x.py', 'content': 'print(1)'}}],
        ),
        # 6. 无工具调用
        (
            '这是一段普通文本，没有工具调用。',
            [],
        ),
        # 7. 多个 invoke
        (
            '<invoke name="a"><parameter name="x">1</parameter></invoke>'
            '<invoke name="b"><parameter name="y">2</parameter></invoke>',
            [
                {'name': 'a', 'arguments': {'x': '1'}},
                {'name': 'b', 'arguments': {'y': '2'}},
            ],
        ),
        # 8. 半角竖线 ||DSML|| 格式
        (
            '<||DSML||invoke name="web_fetch">'
            '<||DSML||parameter name="url" string="true">https://example.com</||DSML||parameter>'
            '</||DSML||invoke>',
            [{'name': 'web_fetch', 'arguments': {'url': 'https://example.com'}}],
        ),
    ]

    passed = 0
    for i, (inp, expected) in enumerate(tests, 1):
        got = parse_dsml_tool_calls(inp)
        ok = got == expected
        status = 'PASS' if ok else 'FAIL'
        print(f'[{status}] 测试 {i}: {got}')
        if not ok:
            print(f'         期望: {expected}')
        else:
            passed += 1

    print(f'\n{passed}/{len(tests)} 通过')
