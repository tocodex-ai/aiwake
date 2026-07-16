"""
Command Precheck - Shell 命令预检与别名降级模块

功能：
1. 解析复合 shell 命令（含 &&, ||, |, ; 等连接符）
2. 对每个子命令的主二进制做存在性检查
3. 若不存在但有已知降级映射，自动替换为可用命令
4. 对降级命令执行参数转换（移除/替换目标命令不认识的参数）
5. 返回重写后的命令字符串和预检报告

这解决了 AIwake 在容器内使用 rg/fd/bat 等命令时频繁 command_missing 的问题，
即使命令出现在 cd /app && rg xxx 这类复合命令中也能正确降级。
增强版还处理参数不兼容的场景（如 rg --files → find .）。
"""

import logging
import re
import shlex
import shutil
from typing import NamedTuple

logger = logging.getLogger(__name__)

# ── 命令降级映射表 ──
# key: 可能缺失的命令名
# value: 替换为的等价命令（只替换命令名部分，保留原始参数）
COMMAND_FALLBACK_MAP: dict[str, str] = {
    "rg": "grep -rn",
    "ripgrep": "grep -rn",
    "fd": "find . -name",
    "bat": "cat",
    "exa": "ls",
    "eza": "ls",
    "delta": "diff",
    "sd": "sed",
    "dust": "du -sh",
    "procs": "ps aux",
    "tokei": "wc -l",
    "hyperfine": "time",
    "xh": "curl",
    "httpie": "curl",
    "http": "curl",
}

# ── 参数转换规则 ──
# 当命令从 source → target 降级时，某些参数需要转换或移除
# 格式: {source_binary: ArgTransformSpec}
# ArgTransformSpec 包含:
#   - "mode_flags": 当检测到这些 flag 时，使用完全不同的降级命令
#   - "drop_flags": 需要移除的参数（不影响语义，只是目标不支持）
#   - "translate_flags": 需要转换的参数 {原参数: 新参数 或 None(移除)}
#   - "drop_options": 需要移除的带值选项（如 --type-add xxx）

ARGUMENT_TRANSFORM_RULES: dict[str, dict] = {
    "rg": {
        # rg --files 模式 → 完全不同的语义：列出文件而非搜索
        "mode_flags": {
            "--files": {
                "replacement_cmd": "find .",
                "arg_map": {
                    "-g": "-name",          # rg -g 'pattern' → find -name 'pattern'
                    "--glob": "-name",
                },
                "drop_flags": ["--hidden", "--no-ignore", "--follow", "-L",
                               "--sort-files", "--sort", "--sortr", "-j",
                               "--threads", "--no-heading", "--heading"],
                "drop_options": ["--type", "--type-add", "--type-not",
                                 "-t", "-T", "--max-depth", "--maxdepth",
                                 "--ignore-file"],
            },
            "--files-with-matches": {
                "replacement_cmd": "grep -rln",
                "arg_map": {},
                "drop_flags": ["--no-heading", "--heading", "--color",
                               "--no-color", "--smart-case", "-S"],
                "drop_options": ["--type", "-t", "-T", "--type-add",
                                 "--type-not"],
            },
            "-l": {
                "replacement_cmd": "grep -rln",
                "arg_map": {},
                "drop_flags": ["--no-heading", "--heading", "--color",
                               "--no-color", "--smart-case", "-S"],
                "drop_options": ["--type", "-t", "-T", "--type-add",
                                 "--type-not"],
            },
        },
        # 普通搜索模式下不兼容的参数
        "drop_flags": ["--no-heading", "--heading", "--smart-case", "-S",
                       "--pcre2", "-P", "--multiline", "-U",
                       "--sort-files", "--sort", "--sortr",
                       "--json", "--vimgrep", "--column",
                       "--hidden", "--no-ignore", "--follow", "-L",
                       "--count-matches", "--null", "-0"],
        "translate_flags": {
            "--count": "-c",
            "-c": "-c",
            "--ignore-case": "-i",
            "-i": "-i",
            "--word-regexp": "-w",
            "-w": "-w",
            "--fixed-strings": "-F",
            "-F": "-F",
            "--invert-match": "-v",
            "-v": "-v",
            "--files-without-match": "-L",
        },
        "drop_options": ["--type", "-t", "-T", "--type-add", "--type-not",
                         "--max-count", "-m",
                         "--max-depth", "--maxdepth",
                         "-j", "--threads",
                         "--ignore-file", "--glob", "-g",
                         "--after-context", "-A",
                         "--before-context", "-B",
                         "--context", "-C",
                         "--replace", "-r",
                         "--color", "--colors"],
        # grep 支持的等效选项（保留对应值）
        "translate_options": {
            "--after-context": "-A",
            "-A": "-A",
            "--before-context": "-B",
            "-B": "-B",
            "--context": "-C",
            "-C": "-C",
            "--max-count": "-m",
            "-m": "-m",
        },
    },
    "fd": {
        "drop_flags": ["--hidden", "--no-ignore", "--follow", "-L",
                       "--absolute-path", "-a", "--glob", "-g"],
        "translate_flags": {},
        "drop_options": ["--type", "-t", "--extension", "-e",
                         "--exclude", "-E", "--max-depth", "-d",
                         "--min-depth", "--size", "-S",
                         "--changed-within", "--changed-before",
                         "--color", "-j", "--threads"],
    },
}

# Shell 连接符正则 —— 匹配 &&, ||, |, ; 以及命令替换 $() 内部不处理
_SHELL_SEPARATORS = re.compile(r"\s*(&&|\|\||\||;)\s*")


def _quote_aware_split(command: str) -> list[str]:
    """
    引号感知的 Shell 命令分割器。
    
    按顶层的 &&, ||, |, ; 分割命令，但不拆分引号内或命令替换 $(...) / `...` 内的分隔符。
    
    返回格式和 re.split 一致：[subcmd, sep, subcmd, sep, ...]
    """
    parts: list[str] = []
    current = ""
    i = 0
    n = len(command)
    
    while i < n:
        ch = command[i]
        
        # 处理引号（单引号、双引号）
        if ch in ('"', "'"):
            quote_char = ch
            current += ch
            i += 1
            while i < n and command[i] != quote_char:
                if command[i] == '\\' and quote_char == '"' and i + 1 < n:
                    current += command[i:i+2]
                    i += 2
                else:
                    current += command[i]
                    i += 1
            if i < n:
                current += command[i]  # 关闭引号
                i += 1
            continue
        
        # 处理命令替换 $(...)
        if ch == '$' and i + 1 < n and command[i+1] == '(':
            depth = 1
            current += command[i:i+2]
            i += 2
            while i < n and depth > 0:
                if command[i] == '(':
                    depth += 1
                elif command[i] == ')':
                    depth -= 1
                current += command[i]
                i += 1
            continue
        
        # 处理反引号命令替换 `...`
        if ch == '`':
            current += ch
            i += 1
            while i < n and command[i] != '`':
                current += command[i]
                i += 1
            if i < n:
                current += command[i]
                i += 1
            continue
        
        # 检测分隔符
        # &&
        if ch == '&' and i + 1 < n and command[i+1] == '&':
            parts.append(current)
            parts.append("&&")
            current = ""
            i += 2
            # 跳过后续空白
            while i < n and command[i] == ' ':
                i += 1
            continue
        
        # ||
        if ch == '|' and i + 1 < n and command[i+1] == '|':
            parts.append(current)
            parts.append("||")
            current = ""
            i += 2
            while i < n and command[i] == ' ':
                i += 1
            continue
        
        # | (单独管道，不是 ||)
        if ch == '|' and (i + 1 >= n or command[i+1] != '|'):
            parts.append(current)
            parts.append("|")
            current = ""
            i += 1
            while i < n and command[i] == ' ':
                i += 1
            continue
        
        # ;
        if ch == ';':
            parts.append(current)
            parts.append(";")
            current = ""
            i += 1
            while i < n and command[i] == ' ':
                i += 1
            continue
        
        # 处理转义字符
        if ch == '\\' and i + 1 < n:
            current += command[i:i+2]
            i += 2
            continue
        
        current += ch
        i += 1
    
    # 最后一段
    parts.append(current)
    return parts


class PrecheckResult(NamedTuple):
    """预检结果"""
    rewritten_command: str       # 重写后的完整命令
    substitutions: list[dict]    # 所有替换记录 [{original, replacement, reason}]
    missing_commands: list[str]  # 无法降级的缺失命令列表


def _extract_primary_binary(subcmd: str) -> tuple[str, str, str]:
    """
    从子命令中提取主二进制名。
    返回 (prefix, binary, tail)
    prefix: cd /app && 后的前缀空白或环境变量赋值
    binary: 主命令名
    tail:   命令参数部分
    """
    stripped = subcmd.strip()
    if not stripped:
        return ("", "", "")
    
    # 跳过前导的环境变量赋值 (如 FOO=bar cmd args)
    prefix_parts = []
    remaining = stripped
    while remaining:
        match = re.match(r'^([A-Za-z_][A-Za-z0-9_]*=\S*)\s+', remaining)
        if match:
            prefix_parts.append(match.group(1))
            remaining = remaining[match.end():]
        else:
            break
    
    prefix = " ".join(prefix_parts)
    if prefix:
        prefix += " "
    
    # 跳过 cd xxx && 这种已经在外层被拆分的情况，这里只处理单个子命令
    # 提取首个 token 作为命令名
    parts = remaining.split(None, 1)
    if not parts:
        return (prefix, "", "")
    
    binary = parts[0]
    tail = parts[1] if len(parts) > 1 else ""
    
    return (prefix, binary, tail)


def _is_checkable_binary(binary: str) -> bool:
    """判断是否应该对该二进制做存在性检查"""
    if not binary:
        return False
    # 带路径的命令（/usr/bin/xxx 或 ./script.sh）不做检查
    if "/" in binary or binary.startswith("."):
        return False
    # shell 内建命令不检查
    builtins = frozenset({
        "cd", "echo", "export", "source", ".", "alias", "unalias",
        "set", "unset", "eval", "exec", "exit", "return", "shift",
        "test", "[", "[[", "true", "false", "read", "printf",
        "type", "hash", "command", "builtin", "declare", "local",
        "readonly", "trap", "wait", "jobs", "fg", "bg", "kill",
        "umask", "ulimit", "pwd", "dirs", "pushd", "popd",
        "break", "continue", "for", "while", "until", "if",
        "then", "else", "elif", "fi", "case", "esac", "do", "done",
        "function", "in", "select", "time", "coproc",
    })
    if binary in builtins:
        return False
    return True


def _transform_args(binary: str, tail: str) -> tuple[str, str]:
    """
    对降级命令的参数进行转换。
    
    检查是否触发了"模式 flag"（如 rg --files 表示列出文件而非搜索），
    如果触发，使用完全不同的降级命令+参数转换。
    否则，移除目标命令不认识的参数，转换可映射的参数。
    
    Returns:
        (replacement_cmd, transformed_tail)
        - replacement_cmd: 可能被 mode_flag 改变的降级命令
        - transformed_tail: 转换后的参数字符串
    """
    rules = ARGUMENT_TRANSFORM_RULES.get(binary)
    if not rules:
        return ("", tail)  # 没有转换规则，保持原样
    
    # 解析参数（安全 shlex，失败则回退到简单 split）
    try:
        tokens = shlex.split(tail)
    except ValueError:
        tokens = tail.split()
    
    # 检查 mode_flags（优先级最高）
    mode_flags = rules.get("mode_flags", {})
    triggered_mode = None
    for flag, mode_spec in mode_flags.items():
        if flag in tokens:
            triggered_mode = (flag, mode_spec)
            break
    
    if triggered_mode:
        flag, mode_spec = triggered_mode
        replacement_cmd = mode_spec["replacement_cmd"]
        arg_map = mode_spec.get("arg_map", {})
        mode_drop_flags = set(mode_spec.get("drop_flags", []))
        mode_drop_options = set(mode_spec.get("drop_options", []))
        
        # 移除 mode flag 本身，然后转换剩余参数
        new_tokens = []
        skip_next = False
        for idx, tok in enumerate(tokens):
            if skip_next:
                skip_next = False
                continue
            if tok == flag:
                continue  # 移除 mode flag
            if tok in mode_drop_flags:
                continue
            if tok in mode_drop_options:
                skip_next = True  # 移除该选项及其值
                continue
            # 检查 --opt=val 格式
            opt_name = tok.split("=", 1)[0] if "=" in tok else tok
            if opt_name in mode_drop_options:
                if "=" not in tok:
                    skip_next = True
                continue
            if opt_name in mode_drop_flags:
                continue
            # 转换参数
            if tok in arg_map:
                new_tokens.append(arg_map[tok])
                continue
            if opt_name in arg_map and "=" in tok:
                val = tok.split("=", 1)[1]
                new_tokens.append(f"{arg_map[opt_name]}={val}")
                continue
            # 带值选项转换 (如 -g 'pattern' → -name 'pattern')
            if tok in arg_map and idx + 1 < len(tokens):
                new_tokens.append(arg_map[tok])
                continue
            new_tokens.append(tok)
        
        # 用 shlex.quote 统一处理所有 token，正确转义 shell 元字符（|、&、;、<、>、$、`、引号等），
        # 避免 rg "foo|bar" 降级 grep 时丢引号让 | 被 shell 当成管道符（先前只检查空格/通配符不够安全）。
        transformed_tail = " ".join(shlex.quote(t) for t in new_tokens)
        return (replacement_cmd, transformed_tail)
    
    # 非 mode flag 场景：普通降级的参数过滤
    drop_flags = set(rules.get("drop_flags", []))
    translate_flags = rules.get("translate_flags", {})
    drop_options = set(rules.get("drop_options", []))
    translate_options = rules.get("translate_options", {})
    
    new_tokens = []
    skip_next = False
    for idx, tok in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        
        # --opt=val 格式
        if "=" in tok and tok.startswith("-"):
            opt_name = tok.split("=", 1)[0]
            opt_val = tok.split("=", 1)[1]
            if opt_name in drop_options or opt_name in drop_flags:
                continue
            if opt_name in translate_options:
                new_tokens.append(f"{translate_options[opt_name]}={opt_val}")
                continue
            if opt_name in translate_flags:
                new_tokens.append(translate_flags[opt_name])
                continue
            new_tokens.append(tok)
            continue
        
        # 需要丢弃的 flag
        if tok in drop_flags:
            continue
        
        # 需要丢弃的带值选项
        if tok in drop_options:
            # 下一个 token 是值，跳过
            skip_next = True
            continue
        
        # 需要转换的 flag
        if tok in translate_flags:
            mapped = translate_flags[tok]
            if mapped:  # None 表示移除
                new_tokens.append(mapped)
            continue
        
        # 需要转换的带值选项
        if tok in translate_options:
            new_tokens.append(translate_options[tok])
            # 保留下一个 token（值）
            continue
        
        new_tokens.append(tok)
    
    # 同上：统一用 shlex.quote 保证 shell 元字符被正确引用，避免管道/重定向被误解析。
    transformed_tail = " ".join(shlex.quote(t) for t in new_tokens)
    return ("", transformed_tail)


def precheck_command(command: str, extra_aliases: dict[str, str] | None = None) -> PrecheckResult:
    """
    对 shell 命令进行预检和降级替换（含参数转换）。
    
    Args:
        command: 原始 shell 命令字符串
        extra_aliases: 额外的别名映射（会覆盖默认映射）
    
    Returns:
        PrecheckResult 包含重写后的命令、替换记录和无法降级的命令列表
    """
    aliases = dict(COMMAND_FALLBACK_MAP)
    if extra_aliases:
        aliases.update(extra_aliases)
    
    substitutions: list[dict] = []
    missing_commands: list[str] = []
    
    # 拆分复合命令（引号感知，不会拆分引号内或命令替换内的分隔符）
    parts = _quote_aware_split(command)
    # parts 是交替的 [subcmd, separator, subcmd, separator, ...]
    
    rewritten_parts: list[str] = []
    
    for i, part in enumerate(parts):
        # 偶数索引是子命令，奇数索引是分隔符
        if i % 2 == 1:
            # 这是分隔符，原样保留
            rewritten_parts.append(f" {part} ")
            continue
        
        subcmd = part.strip()
        if not subcmd:
            rewritten_parts.append(part)
            continue
        
        prefix, binary, tail = _extract_primary_binary(subcmd)
        
        if not _is_checkable_binary(binary):
            rewritten_parts.append(part)
            continue
        
        # 检查命令是否存在
        if shutil.which(binary) is not None:
            # 命令存在，不需要替换
            rewritten_parts.append(part)
            continue
        
        # 命令不存在，尝试降级
        if binary in aliases:
            # 先做参数转换
            mode_replacement, transformed_tail = _transform_args(binary, tail)
            
            # 如果 mode_flag 触发了不同的命令，使用它
            replacement = mode_replacement if mode_replacement else aliases[binary]
            
            # 构建新子命令
            new_subcmd = f"{prefix}{replacement}"
            if transformed_tail:
                new_subcmd += f" {transformed_tail}"
            rewritten_parts.append(new_subcmd)
            
            detail = ""
            if mode_replacement:
                detail = f" (mode 降级: 参数触发替代命令)"
            elif transformed_tail != tail:
                detail = f" (参数已转换)"
            
            substitutions.append({
                "original": binary,
                "original_tail": tail,
                "replacement": replacement,
                "transformed_tail": transformed_tail,
                "reason": f"'{binary}' 不存在，降级为 '{replacement}'{detail}",
            })
            logger.info(f"[CommandPrecheck] 命令降级: {binary} → {replacement}{detail}")
        else:
            # 无法降级，记录缺失
            missing_commands.append(binary)
            rewritten_parts.append(part)
            logger.warning(f"[CommandPrecheck] 命令 '{binary}' 不存在且无已知降级路径")
    
    rewritten_command = "".join(rewritten_parts)
    
    return PrecheckResult(
        rewritten_command=rewritten_command,
        substitutions=substitutions,
        missing_commands=missing_commands,
    )
