"""Self-check: _quote_aware_split 不应误拆引号内的 |、;、&&、||。

针对 open task：
- tas_1781734097_7b35ec3b 窄验证：引号内管道符不被误拆
- tas_1781744716_2c0a8802 修复命令预检对引号内分隔符的误切分

结论：行为已经正确（按 quote-aware 实现），本 selfcheck 用来固化期望，避免回归。
"""
from __future__ import annotations

import sys
from pathlib import Path


def _find_agent_dir() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "command_precheck.py").exists():
            return parent
    raise RuntimeError("command_precheck.py not found from test path")


AGENT_DIR = _find_agent_dir()
sys.path.insert(0, str(AGENT_DIR))

from command_precheck import _quote_aware_split, precheck_command  # noqa: E402


CASES = [
    ("echo a | grep b", ["echo a ", "|", "grep b"]),
    ("echo 'a|b' | grep c", ["echo 'a|b' ", "|", "grep c"]),
    ('echo "a|b" | grep c', ['echo "a|b" ', "|", "grep c"]),
    ("grep 'a&&b' /tmp/x && ls", ["grep 'a&&b' /tmp/x ", "&&", "ls"]),
    ('grep -E "a;b" /tmp/x; echo done', ['grep -E "a;b" /tmp/x', ";", "echo done"]),
    ('cmd1 "a||b" || cmd2', ['cmd1 "a||b" ', "||", "cmd2"]),
    ('printf %s "hello;world"; echo ok', ['printf %s "hello;world"', ";", "echo ok"]),
    ('rg -n "^def |^class " .', ['rg -n "^def |^class " .']),
]


def main() -> int:
    failures: list[str] = []
    for raw, expected in CASES:
        got = _quote_aware_split(raw)
        if got != expected:
            failures.append(f"input={raw!r} expected={expected!r} got={got!r}")
    if failures:
        print("FAIL command_precheck_quote_split_selfcheck")
        for line in failures:
            print(line)
        return 1

    # 顺带保证引号内含管道符的 rg 命令仍可被预检识别为单个子命令
    res = precheck_command('rg -n "^def |^class " .')
    if not isinstance(res.rewritten_command, str) or not res.rewritten_command:
        print("FAIL precheck returned empty rewritten_command")
        return 1

    print(f"command_precheck_quote_split_selfcheck: ok (cases={len(CASES)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
