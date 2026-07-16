"""Self-check: precheck_command 把 rg 降级为 grep 时必须保留 shell 元字符引号语义。

历史 bug：_transform_args 用 `" ".join(shlex.quote(t) if " " in t or "*" in t or "?" in t else t)`
只对空格/通配符加引号，让 `rg "foo|bar"` 降级后变成 `grep ... foo|bar`，
shell 会把 `|` 当管道符切走，语义被破坏。

修复后改为统一 shlex.quote，本 selfcheck 固化期望，防止回归。
"""
from __future__ import annotations

import shlex
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

from command_precheck import precheck_command  # noqa: E402


def _split_subcommands(text: str) -> list[str]:
    """按 shell 顶层 |/;/&&/|| 切分，引号内不切分，复制粘贴 precheck 的简化版语义。"""
    parts: list[str] = []
    current = ""
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in ('"', "'"):
            quote = ch
            current += ch
            i += 1
            while i < n and text[i] != quote:
                current += text[i]
                i += 1
            if i < n:
                current += text[i]
                i += 1
            continue
        if ch == "|" or ch == ";":
            parts.append(current.strip())
            current = ""
            i += 1
            continue
        if ch == "&" and i + 1 < n and text[i + 1] == "&":
            parts.append(current.strip())
            current = ""
            i += 2
            continue
        current += ch
        i += 1
    if current.strip():
        parts.append(current.strip())
    return parts


def main() -> int:
    failures: list[str] = []

    # 场景 1：rg "foo|bar" 降级后 | 不能被外层 shell 当成管道符。
    res = precheck_command('rg "foo|bar" /tmp/x')
    rewritten = res.rewritten_command
    subs = _split_subcommands(rewritten)
    if len(subs) != 1:
        failures.append(
            f"case1 expected single subcommand after split, got {len(subs)} -> {subs!r} (rewritten={rewritten!r})"
        )
    # 还需确认管道字符确实被引号包住（否则即便只有 1 段，重写出来的字符串里也不应裸出 |）
    tokens = shlex.split(rewritten)
    if not any("foo|bar" in tok for tok in tokens):
        failures.append(
            f"case1 expected literal 'foo|bar' token after shlex.split, got tokens={tokens!r}"
        )

    # 场景 2：rg "a;b" 降级后分号同样不能被识别为命令分隔符。
    res2 = precheck_command('rg "a;b" .')
    subs2 = _split_subcommands(res2.rewritten_command)
    if len(subs2) != 1:
        failures.append(
            f"case2 expected single subcommand, got {len(subs2)} -> {subs2!r} (rewritten={res2.rewritten_command!r})"
        )
    tokens2 = shlex.split(res2.rewritten_command)
    if not any("a;b" in tok for tok in tokens2):
        failures.append(
            f"case2 expected literal 'a;b' token, got tokens={tokens2!r}"
        )

    # 场景 3：复合命令里的引号内分隔符也不被破坏。
    res3 = precheck_command('rg "a&&b" /tmp && echo done')
    subs3 = _split_subcommands(res3.rewritten_command)
    # 引号内 && 不算分隔符，加上外层 && echo done 才算第二段
    if len(subs3) != 2:
        failures.append(
            f"case3 expected two subcommands, got {len(subs3)} -> {subs3!r} (rewritten={res3.rewritten_command!r})"
        )
    tokens3 = shlex.split(res3.rewritten_command)
    if not any("a&&b" in tok for tok in tokens3):
        failures.append(
            f"case3 expected literal 'a&&b' token, got tokens={tokens3!r}"
        )

    if failures:
        print("FAIL command_precheck_quote_preservation_selfcheck")
        for line in failures:
            print(line)
        return 1

    print("command_precheck_quote_preservation_selfcheck: ok (cases=3)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
