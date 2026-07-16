"""Unit tests for heartbeat._is_action_like_shell_command.

This classifier decides whether a shell_exec command should be counted as an
"action" for the reflection stagnation counter. Read-only exploration commands
(rg/grep/sed/cat/find/ls/wc/python -c 'print(...)') must NOT be classified as
action; state-changing commands (write redirects, git commit, pip install,
docker, mkdir, rm, mv, self_code_write surrogates) MUST be classified as
action.

Run from repository root:
    python src/agent-mind/tests/test_heartbeat_shell_action_classifier.py
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
    _build_no_action_circuit_breaker_prompt,
    _is_action_like_shell_command,
)


# ── 1. 只读命令：均应判为 False（非行动） ──────────────────
READONLY_COMMANDS = [
    # 基础查看
    "cat src/agent-mind/heartbeat.py",
    "head -n 50 main.py",
    "tail -f log/app.log",
    "less README.md",
    # 列举/查找
    "ls -la src/",
    "find . -name '*.py'",
    "which python",
    # 文本搜索/统计（即使用了管道）
    "grep -rn 'def ' src/",
    "rg 'TODO' --type py",
    "rg --files src/agent-mind",
    "awk '{print $1}' log.txt",
    "sed -n '10,20p' file.txt",
    "wc -l src/agent-mind/heartbeat.py",
    # 管道组合（每段都是只读）
    "cat file.txt | grep 'error' | head -n 20",
    "rg 'reflect' src/ | wc -l",
    # 系统状态
    "ps aux",
    "uptime",
    "env",
    # Python 只读探针
    'python -c "print(1+1)"',
    'python3 -c "import sys; print(sys.version)"',
    # 网络只读
    "curl -s https://example.com/health",
    # 包管理只读子命令
    "git status",
    "git log --oneline -n 5",
    "git diff HEAD~1",
    "pip list",
    "pip show fastapi",
    "npm ls",
    # 复合命令但全部为只读
    "cd src && ls -la",
    "cd src/agent-mind && grep -rn 'TODO' .",
    # echo / 内置
    "echo hello world",
    "true",
]


# ── 2. 行动命令：均应判为 True（行动） ──────────────────
ACTION_COMMANDS = [
    # 文件状态变更
    "rm tmp.txt",
    "mv old.py new.py",
    "cp src.py dst.py",
    "mkdir -p build/dist",
    "touch placeholder.txt",
    "chmod +x deploy.sh",
    # 重定向（写文件）
    "echo 'hello' > greeting.txt",
    "echo 'append' >> log.txt",
    "cat template.txt > result.txt",
    "python -c \"print('hi')\" > out.txt",
    # git 写子命令
    "git add .",
    "git commit -m 'fix'",
    "git push origin main",
    "git pull --rebase",
    "git checkout -b new-branch",
    "git reset --hard HEAD~1",
    # 包管理写子命令
    "pip install fastapi",
    "pip3 uninstall requests",
    "npm install lodash",
    "npm i",
    "yarn add react",
    "go install github.com/foo/bar",
    "cargo build",
    # 部署/容器/系统服务
    "docker build -t app:latest .",
    "docker run -d app:latest",
    "kubectl apply -f deploy.yaml",
    "flyctl deploy",
    "fly deploy --remote-only",
    "systemctl restart nginx",
    "ssh user@host 'whoami'",
    "rsync -av src/ dst/",
    # python 写操作（-c 内嵌）
    "python -c \"open('out.txt','w').write('hi')\"",
    "python -c \"import os; os.remove('tmp.txt')\"",
    "python -c \"import subprocess; subprocess.run(['ls'])\"",
    "python -c \"from pathlib import Path; Path('a.txt').write_text('x')\"",
    "python -c \"import shutil; shutil.rmtree('build')\"",
    # 复合命令中混入一个写命令
    "cd src && pip install -e .",
    "git pull && pip install -r requirements.txt",
    "mkdir -p out && cat foo.txt > out/bar.txt",
    # 未知二进制：保守视为行动
    "my_custom_deploy_script.sh",
    "./build_and_publish.py",
    # bash/sh 直接执行脚本
    "bash deploy.sh",
    "sh release.sh production",
    # 打包/归档
    "tar -czf release.tgz src/",
    "zip -r out.zip src/",
]


# ── 3. 边界情况 ──────────────────
EDGE_CASES = [
    # 空命令
    ("", False),
    ("   ", False),
    (None, False),
    # 仅 cd（不算落地）
    ("cd src/agent-mind", False),
    # 带环境变量前缀
    ("PYTHONPATH=src python -c \"print(1)\"", False),
    ("DEBUG=1 git add foo.py", True),
    # 含引号但里面没有真正的写关键字
    ("grep 'open(\"w\"' src/", False),
    # 子 shell 包裹
    ("(rg TODO src/)", False),
    ("(rm tmp.txt)", True),
    # 管道里仅最后一段是 tee（写）
    ("cat foo.txt | tee bar.txt", True),
]


def _label(cmd: str | None) -> str:
    if cmd is None:
        return "<None>"
    if not cmd.strip():
        return f"<empty:{cmd!r}>"
    return cmd if len(cmd) < 70 else (cmd[:67] + "...")


def _test_no_action_circuit_breaker_prompt() -> None:
    assert _build_no_action_circuit_breaker_prompt(4, {}) == ""

    open_prompt = _build_no_action_circuit_breaker_prompt(5, {
        "open_tasks": [{
            "id": "task-open-1",
            "title": "提高工具成功率",
            "status": "open",
            "goal": "将 tool_success_rate 提升到 0.90",
        }],
        "latest_task": {
            "id": "task-closed-old",
            "title": "旧任务不应复活",
            "status": "closed",
        },
    })
    assert "task-open-1" in open_prompt, open_prompt
    assert "提高工具成功率" in open_prompt, open_prompt
    assert "task-closed-old" not in open_prompt, open_prompt
    assert "/app/tool_router.py" not in open_prompt, open_prompt
    assert "自主选择相关文件" in open_prompt, open_prompt

    closed_only_prompt = _build_no_action_circuit_breaker_prompt(8, {
        "open_tasks": [],
        "latest_task": {
            "id": "task-closed-old",
            "title": "修复 _looks_like_shell_stderr_failure 误判",
            "status": "closed",
        },
    })
    assert "task-closed-old" not in closed_only_prompt, closed_only_prompt
    assert "_looks_like_shell_stderr_failure" not in closed_only_prompt, closed_only_prompt
    assert "不得把已关闭任务重新当作当前目标" in closed_only_prompt, closed_only_prompt
    assert "self_task_create" in closed_only_prompt, closed_only_prompt
    assert "P1–P5" in closed_only_prompt, closed_only_prompt

    stale_open_list_prompt = _build_no_action_circuit_breaker_prompt(6, {
        "open_tasks": [{
            "id": "task-closed-in-open-list",
            "title": "状态快照里的终态任务",
            "status": "done",
        }],
    })
    assert "task-closed-in-open-list" not in stale_open_list_prompt, stale_open_list_prompt
    assert "self_task_create" in stale_open_list_prompt, stale_open_list_prompt


def main() -> None:
    failures: list[str] = []

    for cmd in READONLY_COMMANDS:
        got = _is_action_like_shell_command(cmd)
        if got is not False:
            failures.append(
                f"  [readonly→action误判] {_label(cmd)} -> got={got}, expected=False"
            )

    for cmd in ACTION_COMMANDS:
        got = _is_action_like_shell_command(cmd)
        if got is not True:
            failures.append(
                f"  [action→readonly漏判] {_label(cmd)} -> got={got}, expected=True"
            )

    for cmd, expected in EDGE_CASES:
        got = _is_action_like_shell_command(cmd)
        if got != expected:
            failures.append(
                f"  [edge case] {_label(cmd)} -> got={got}, expected={expected}"
            )

    if failures:
        print("heartbeat_shell_action_classifier_selfcheck: FAILED")
        for line in failures:
            print(line)
        sys.exit(1)

    _test_no_action_circuit_breaker_prompt()

    total = len(READONLY_COMMANDS) + len(ACTION_COMMANDS) + len(EDGE_CASES)
    print(
        f"heartbeat_shell_action_classifier_selfcheck: ok "
        f"({total} cases: {len(READONLY_COMMANDS)} readonly + "
        f"{len(ACTION_COMMANDS)} action + {len(EDGE_CASES)} edge)"
    )


if __name__ == "__main__":
    main()
