# -*- coding: utf-8 -*-
"""Reflection action gate: detect spinning patterns in self-reflection.

来源：task-60 闭环训练（2026-06-19）。线上活动日志显示反思链路在
tick #912 ~ #968 持续重复 ``experiment_status -> file_read evolution/self_dialogue.py
-> shell_exec rg(empty_result)`` 链路，但既没有 ``self_code_write`` 调用、也没有
``[SELF_CODE_WRITE]`` 协议落盘。本模块只做一件事：对最近 N 轮反思的工具调用
链路做指纹去重；若连续 ``window_size`` 轮（默认 3）链路指纹完全相同，
返回 ``stop_signal='spinning_pattern'``，提示上层在反思末尾改用
``[SELF_CODE_WRITE]`` 协议或调用 ``self_code_write`` 真正动笔。

设计原则：
- 仅使用 Python 标准库；
- 纯函数式：不读写任何记忆/日志/配置文件，由调用方自行决定如何记录；
- 不抛异常：任何不合法输入都返回 ``no_signal``，便于在 heartbeat 中以软提醒方式接入；
- 安全边界：不访问密钥、不修改记忆/活动日志/聊天记录，不发起网络请求。

audit:
  - purpose: 让反思阶段在反复同一只读探查链路时主动停下并改走落地补丁路径
  - change_summary: 新增 ``evolution/reflection_action_gate.py``，提供 ``detect_spinning_pattern``
  - files: ['src/agent-mind/evolution/reflection_action_gate.py']
  - safety: 纯函数；不访问密钥/日志/记忆/网络；输入异常时返回 no_signal
  - rollback: 直接删除本文件即可，不影响既有运行（无任何模块强依赖）
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable


def _normalize_step(step: Any) -> str:
    """把单步工具调用归一化成稳定字符串，用于指纹哈希。

    对 file_read 类工具：只保留 path 参数（忽略 offset/limit），这样对同一文件
    的多次分页读取会被视为等价操作，避免反复读同一文件却逃过空转检测。
    对 shell_exec 类只读命令（cat/head/grep/python -c print）：归一化为
    "shell_exec|readonly_probe"，避免参数微调逃过检测。
    """
    if isinstance(step, dict):
        tool = str(step.get("tool") or step.get("name") or "").strip().lower()
        args = step.get("args") or step.get("arguments") or {}
        if isinstance(args, dict):
            # file_read: 只保留 path，忽略 offset/limit 等分页参数
            if tool == "file_read":
                path_val = str(args.get("path") or "").strip()
                return f"{tool}|path={path_val}"
            # shell_exec: 检测是否为只读探查命令
            if tool == "shell_exec":
                cmd = str(args.get("command") or "")
                if _is_readonly_probe_command(cmd):
                    return "shell_exec|readonly_probe"
            keys = ",".join(sorted(str(k) for k in args.keys()))
        else:
            keys = str(args)[:64]
        return f"{tool}|{keys}"
    return str(step).strip().lower()


# 只读探查命令的判定模式
_READONLY_PROBE_PATTERNS = (
    "cat ", "head ", "tail ", "grep ", "rg ", "find ",
    "python -c", "python - <<", "python -m py_compile",
    "sed -n", "awk ", "wc ",
)


def _is_readonly_probe_command(cmd: str) -> bool:
    """判断 shell_exec 命令是否为只读探查（不产生副作用）。"""
    cmd_stripped = cmd.strip().lstrip("cd /app && ").lstrip("cd /app;").strip()
    for pattern in _READONLY_PROBE_PATTERNS:
        if cmd_stripped.startswith(pattern):
            return True
    # python heredoc 读文件也是只读探查
    if "Path(" in cmd and ".read_text()" in cmd and ".write_text(" not in cmd:
        return True
    return False


def _fingerprint(chain: Iterable[Any]) -> str:
    """给一轮反思的工具链路计算稳定指纹。"""
    parts = [_normalize_step(s) for s in chain if s is not None]
    raw = ">".join(parts)
    return hashlib.sha1(raw.encode("utf-8", errors="replace")).hexdigest()


def detect_spinning_pattern(
    recent_chains: list[list[Any]],
    *,
    window_size: int = 3,
) -> dict[str, Any]:
    """检测最近若干轮反思是否在同一工具链路上空转。

    Args:
        recent_chains: 最近几轮反思的工具链路列表，最新的在末尾。每一轮是
            该轮反思按顺序调用的工具步骤列表（dict 含 ``tool``/``args`` 或字符串）。
        window_size: 视为空转的最少连续相同轮数，默认 3。

    Returns:
        dict，固定字段：
            - ``stop_signal``: ``"spinning_pattern"`` | ``"no_signal"``
            - ``window_size``: 实际比较窗口
            - ``matched_rounds``: 命中的连续轮数（无信号时为 0）
            - ``fingerprint``: 命中的链路指纹（无信号时为空串）
            - ``hint``: 给上层的下一步提示（中文，软建议）
    """
    if not isinstance(recent_chains, list) or window_size < 2:
        return {"stop_signal": "no_signal", "window_size": window_size,
                "matched_rounds": 0, "fingerprint": "", "hint": ""}
    valid = [c for c in recent_chains if isinstance(c, list) and c]
    if len(valid) < window_size:
        return {"stop_signal": "no_signal", "window_size": window_size,
                "matched_rounds": 0, "fingerprint": "", "hint": ""}
    tail = valid[-window_size:]
    fps = [_fingerprint(chain) for chain in tail]
    if len(set(fps)) == 1 and fps[0]:
        return {
            "stop_signal": "spinning_pattern",
            "window_size": window_size,
            "matched_rounds": window_size,
            "fingerprint": fps[0],
            "hint": (
                "最近 {n} 轮反思工具链路完全一致，已构成空转。"
                "下一步请改用 [SELF_CODE_WRITE] 协议在反思末尾落地一处最小补丁，"
                "或直接调用 self_code_write 工具，避免继续只读探查。"
            ).format(n=window_size),
        }
    return {"stop_signal": "no_signal", "window_size": window_size,
            "matched_rounds": 0, "fingerprint": "", "hint": ""}


__all__ = ["detect_spinning_pattern"]
