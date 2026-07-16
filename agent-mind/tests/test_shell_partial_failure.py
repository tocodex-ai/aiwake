"""Self-check: shell_exec 复合命令整体 returncode=0 但 stderr 报路径不存在时，
应被识别为 shell 软失败（classification=failure / failure_type=shell_partial_failure），
而不是被错标为 success。

复现 AIwake 线上真实证据（2026-06-16 活动日志）：
  - `cat: /app/data/diary/2026-06-16.md: No such file or directory` | returncode=0
  - `find: '/home/aiwake': No such file or directory` | returncode=0
这两条命令历史上都被标成 classification=success，掩盖了"路径假设错误"这一失败信号。

Run from repository root:
    python src/agent-mind/tests/test_shell_partial_failure.py
"""
from __future__ import annotations

import sys
from pathlib import Path


def _find_agent_dir() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "heartbeat.py").exists() and (parent / "tool_router.py").exists():
            return parent
        if parent.name == "agent-mind":
            return parent
        candidate = parent / "src" / "agent-mind"
        if candidate.exists():
            return candidate
        candidate = parent / "agent-mind"
        if candidate.exists():
            return candidate
    raise RuntimeError("Cannot locate agent-mind directory")


AGENT_DIR = _find_agent_dir()
sys.path.insert(0, str(AGENT_DIR))

from heartbeat import (  # noqa: E402
    _classify_tool_result_for_activity,
    _looks_like_shell_stderr_failure,
)
from tool_router import ToolRouter  # noqa: E402


def _activity(tool_name: str, data: dict) -> dict:
    """模拟活动日志分类入口：result_text 取 data['result']。"""
    return _classify_tool_result_for_activity(tool_name, data, str(data.get("result") or ""))


def test_heartbeat_classification() -> None:
    # 1. cat 路径不存在但管道 returncode=0 → 应判为 failure / shell_partial_failure
    res = _activity("shell_exec", {
        "status": "ok",
        "tool": "shell_exec",
        "returncode": 0,
        "result": "0\n[stderr]: cat: /app/data/diary/2026-06-16.md: No such file or directory",
    })
    assert res["classification"] == "failure", f"应判为 failure: {res}"
    assert res["failure_type"] == "shell_partial_failure", f"failure_type 应为 shell_partial_failure: {res}"
    assert res["success"] is False

    # 2. find 路径不存在 → 同样
    res2 = _activity("shell_exec", {
        "status": "ok",
        "tool": "shell_exec",
        "returncode": 0,
        "result": "[stderr]: find: \u2018/home/aiwake\u2019: No such file or directory",
    })
    assert res2["classification"] == "failure", f"应判为 failure: {res2}"
    assert res2["failure_type"] == "shell_partial_failure"

    # 3. 正常成功命令（无 stderr 失败标记）仍是 success，不被误伤
    res3 = _activity("shell_exec", {
        "status": "ok",
        "tool": "shell_exec",
        "returncode": 0,
        "result": "42: classify_command_risk, 183:class ToolRouter:",
    })
    assert res3["classification"] == "success", f"正常命令应为 success: {res3}"
    assert res3["success"] is True

    # 4. stdout 里恰好出现 'no such file' 文本但无命令报错前缀结构 → 不误判
    res4 = _activity("shell_exec", {
        "status": "ok",
        "tool": "shell_exec",
        "returncode": 0,
        "result": "grep match in docs: the error message 'No such file or directory' is logged here",
    })
    assert res4["classification"] == "success", f"stdout 内容不应被误判: {res4}"

    # 4b. 真实线上证据：命令用 2>&1 把 stderr 合并进 stdout（无 [stderr]: 段），
    #     returncode=0，但 `ls: cannot access ...: No such file or directory` 应被识别为软失败。
    res4b = _activity("shell_exec", {
        "status": "ok",
        "tool": "shell_exec",
        "returncode": 0,
        "result": "ls: cannot access '/app/data/diary/this_path_does_not_exist_xyz': No such file or directory\nEXIT_CODE=2\ndone",
    })
    assert res4b["classification"] == "failure", f"2>&1 合并的 ls 报错应判 failure: {res4b}"
    assert res4b["failure_type"] == "shell_partial_failure"

    # 4c. cat 报错（2>&1 合并）同样
    res4c = _activity("shell_exec", {
        "status": "ok",
        "tool": "shell_exec",
        "returncode": 0,
        "result": "cat: /app/data/diary/2026-06-16.md: No such file or directory\ndone",
    })
    assert res4c["classification"] == "failure", f"2>&1 合并的 cat 报错应判 failure: {res4c}"

    # 4d. 回归：用 shell_exec 打印自身源码做诊断时，源码里恰好出现的错误短语字面量
    #     （纯数字行号 + 错误短语，如 `400: "no such file or directory",`）不应被误判为
    #     shell_partial_failure。这是 2026-06-25 线上 AIwake 反思空转的根因：
    #     returncode=0 的成功诊断命令被错标为失败，导致它反复诊断却无法落地修复。
    res4d = _activity("shell_exec", {
        "status": "ok",
        "tool": "shell_exec",
        "returncode": 0,
        "result": (
            '--- 398-410 ---\n'
            '398:     # shell 复合命令整体 returncode=0，但 stderr/stdout 里藏着明确失败信号时识别这些高置信度短语。\n'
            '399:     _SHELL_STDERR_FAILURE_MARKERS = (\n'
            '400:         "no such file or directory",\n'
            '401:         "not a directory",\n'
            '402:         "command not found",\n'
            '403:         "permission denied",\n'
            '404:         "cannot access",\n'
            '405:         "cannot open",\n'
            '406:         "cannot stat",\n'
            '407:         "cannot remove",\n'
            '408:         "operation not permitted",\n'
            '409:     )\n'
        ),
    })
    assert res4d["classification"] == "success", (
        f"源码行号+错误短语字面量不应被误判为 shell_partial_failure: {res4d}"
    )
    assert res4d["success"] is True

    # 4e. 回归：单行纯数字行号 + 错误短语（最简复现）也不应误判
    res4e = _activity("shell_exec", {
        "status": "ok",
        "tool": "shell_exec",
        "returncode": 0,
        "result": '400:         "no such file or directory",',
    })
    assert res4e["classification"] == "success", (
        f"纯数字行号+错误短语字面量不应被误判: {res4e}"
    )

    # 5. returncode!=0 走原有 nonzero 分支，不被新分支抢占（仍 failure）
    res5 = _activity("shell_exec", {
        "status": "error",
        "tool": "shell_exec",
        "returncode": 1,
        "result": "[stderr]: cat: x: No such file or directory",
    })
    assert res5["classification"] == "failure", f"非零退出仍应 failure: {res5}"

    print("  [ok] heartbeat classification: 5/5")


def test_helper_marker() -> None:
    router = ToolRouter()

    cases = [
        # 带真实的行首 [stderr]: 段。
        ("0\n[stderr]: cat: x: No such file or directory", True),
        ("[stderr]: bash: foo: command not found", True),
        # 2>&1 合并 stderr（无 [stderr]: 段）但有命令报错前缀结构。
        ("ls: cannot access 'x': No such file or directory", True),
        ("cat: /app/x.md: No such file or directory\ndone", True),
        ("bash: foo: command not found", True),
        # 裸短语无命令名前缀结构 → 不误判。
        ("No such file or directory", False),
        ("the file is missing, no such file or directory mentioned in prose", False),
        # stdout 打印源码中的错误短语，而独立 stderr 只有 warning → 成功。
        (
            '400:         "no such file or directory",\n'
            "[stderr]: pip: WARNING: cache path cannot open on this platform",
            False,
        ),
        # stdout 中的 `[stderr]:` 只是源码字面量，不能充当真实分隔符。
        (
            '463: stderr_marker = "[stderr]:"\n'
            '464: sample = "cat: x: No such file or directory"',
            False,
        ),
        # 2>&1 合并输出里的多行 warning 即使引用错误短语也不是失败。
        (
            "download completed\n"
            "pip: WARNING: ignored cache entry: No such file or directory\n"
            "done",
            False,
        ),
        # warning 行后仍有真实错误行时必须判为失败。
        (
            "[stderr]: pip: WARNING: cache directory cannot open\n"
            "cat: /app/missing.py: No such file or directory",
            True,
        ),
        # 空文本。
        ("", False),
        (None, False),
    ]

    for text, expected in cases:
        heartbeat_result = _looks_like_shell_stderr_failure(text)
        router_result = router._looks_like_shell_stderr_failure(text)
        assert heartbeat_result is expected, (text, heartbeat_result, expected)
        assert router_result is expected, (text, router_result, expected)
        assert heartbeat_result is router_result, (
            f"heartbeat/tool_router 分类不一致: {text!r} -> "
            f"{heartbeat_result}/{router_result}"
        )

    print(f"  [ok] helper marker parity: {len(cases)}/{len(cases)}")


def test_tool_router_classify_failure() -> None:
    router = ToolRouter()
    # returncode=0 但 stderr 报路径不存在 → tool_router 应补 failure_type + fallback_hint
    enriched = router._classify_failure("shell_exec", {
        "status": "ok",
        "tool": "shell_exec",
        "returncode": 0,
        "result": "0\n[stderr]: cat: /app/data/diary/2026-06-16.md: No such file or directory",
    })
    assert enriched.get("failure_type") == "shell_partial_failure", f"应补 shell_partial_failure: {enriched}"
    assert "fallback_hint" in enriched and enriched["fallback_hint"], "应附 fallback_hint"
    assert "file_list" in enriched["fallback_hint"], "fallback_hint 应引导 file_list 锚定真实路径"

    # 正常命令不被改写
    normal = router._classify_failure("shell_exec", {
        "status": "ok",
        "tool": "shell_exec",
        "returncode": 0,
        "result": "183:class ToolRouter:",
    })
    assert normal.get("failure_type") is None, f"正常命令不应被分类为失败: {normal}"

    # stdout 打印源码错误短语 + stderr 仅 warning 时仍应保持成功。
    warning_only = router._classify_failure("shell_exec", {
        "status": "ok",
        "tool": "shell_exec",
        "returncode": 0,
        "result": (
            '400:         "no such file or directory",\n'
            "[stderr]: pip: WARNING: cache path cannot open on this platform"
        ),
    })
    assert warning_only.get("failure_type") is None, (
        f"源码文本 + warning 不应被分类为失败: {warning_only}"
    )

    print("  [ok] tool_router classify_failure: 3/3")


def main() -> None:
    test_helper_marker()
    test_heartbeat_classification()
    test_tool_router_classify_failure()
    print("shell_partial_failure_selfcheck: ok")


if __name__ == "__main__":
    main()
