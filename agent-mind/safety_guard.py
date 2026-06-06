"""最低安全守卫：路径、命令风险识别与密钥脱敏。"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_SECRET_REPLACEMENT = "[REDACTED_SECRET]"

_TOKEN_PATTERNS = [
    re.compile(r"FlyV1\s+[A-Za-z0-9_:\-\.]+", re.I),
    re.compile(r"(?i)(FLY_API_TOKEN\s*=\s*)[^\s\"']+"),
    re.compile(r"(?i)((?:api[_-]?key|access[_-]?token|secret|password|passwd|pwd)\s*[:=]\s*)[^\s\"']+"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9_\-\.]{16,}"),
    re.compile(r"sk-[A-Za-z0-9]{16,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
]

_SENSITIVE_NAME_PATTERNS = (
    ".tocodex/key.md",
    "token",
    "secret",
    "password",
    "passwd",
    "credential",
    "credentials",
    "private_key",
)

_SENSITIVE_SUFFIXES = (".pem", ".key", ".p12", ".pfx")
_DEPLOY_MARKERS = ("flyctl deploy", " fly deploy", "npm run deploy", "vercel deploy", "railway up", "docker push", "kubectl apply")
_DESTRUCTIVE_MARKERS = (
    "rm -rf", "rmdir /s", "del /f", "format ", "mkfs", "shred ", "wipe", "drop database",
    "truncate table", "git reset --hard", "git clean -fd", "remove-item -recurse",
)
_SECRET_READ_MARKERS = (".tocodex/key.md", " cat .env", " type .env", "get-content .env", " key.md", "token", "secret")
_MEMORY_PATH_MARKERS = (
    "/data/users",
    "data/users/",
    "/data/diary",
    "data/diary/",
    "/data/evolution",
    "data/evolution/",
    "/data/experiments",
    "data/experiments/",
    "/user_memory",
    "memory.json",
    "memories.json",
)
_SELF_CODE_SUFFIXES = (".py", ".yaml", ".yml", ".md", ".html", ".css", ".js", ".toml")
_SELF_CODE_MARKERS = (
    "agent-mind/",
    "runtime_prompts/",
    "evolution/",
    "static/",
    "dockerfile",
    "requirements.txt",
    "fly.toml",
)
_SELF_CORE_FILES = {
    "autonomy_config.py",
    "experiment_runner.py",
    "experiment_store.py",
    "heartbeat.py",
    "llm_gate.py",
    "main.py",
    "personality.yaml",
    "public_chat.py",
    "requirements.txt",
    "safety_guard.py",
    "state.py",
    "tool_router.py",
    "user_memory.py",
    "ws_manager.py",
}


def redact_secrets(value: Any) -> Any:
    """递归脱敏文本、列表和字典，避免密钥进入日志/前端/记忆。"""
    if isinstance(value, dict):
        return {str(k): redact_secrets(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_secrets(v) for v in value]
    if isinstance(value, tuple):
        return tuple(redact_secrets(v) for v in value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    text = str(value)
    for pattern in _TOKEN_PATTERNS:
        text = pattern.sub(lambda m: (m.group(1) if m.lastindex else "") + _SECRET_REPLACEMENT, text)
    return text


def _normalized_path(path: str | Path) -> str:
    raw = str(path or "").replace("\\", "/").strip().strip('"\'')
    while "//" in raw:
        raw = raw.replace("//", "/")
    return raw.lower()


def is_sensitive_path(path: str | Path) -> bool:
    text = _normalized_path(path)
    if not text:
        return False
    name = Path(text).name.lower()
    if text.endswith("/.env") or name == ".env" or name.startswith(".env."):
        return True
    if text.endswith("/.tocodex/key.md") or text == ".tocodex/key.md":
        return True
    if any(marker in text for marker in _SENSITIVE_NAME_PATTERNS):
        return True
    return any(text.endswith(suffix) for suffix in _SENSITIVE_SUFFIXES)


def is_memory_path(path: str | Path) -> bool:
    """识别持久化记忆/成长数据路径，禁止覆盖或删除。"""
    text = _normalized_path(path)
    if not text:
        return False
    return any(marker in text for marker in _MEMORY_PATH_MARKERS)


def is_self_project_code_path(path: str | Path) -> bool:
    """识别 AIwake 自身项目代码/运行规则文件。"""
    text = _normalized_path(path)
    if not text:
        return False
    name = Path(text).name.lower()
    if name.startswith(".") and name not in {".dockerignore"}:
        return False
    if not (text.endswith(_SELF_CODE_SUFFIXES) or name in {"dockerfile", "requirements.txt"}):
        return False
    return name in _SELF_CORE_FILES or any(marker in text for marker in _SELF_CODE_MARKERS)


def has_self_modification_audit(audit: dict[str, Any] | None) -> tuple[bool, list[str]]:
    """自改代码必须携带最小成长/审计字段。"""
    required = ("purpose", "change_summary", "files", "validation_result", "risk_note")
    missing: list[str] = []
    if not isinstance(audit, dict):
        return False, list(required)
    for key in required:
        value = audit.get(key)
        if value is None or (isinstance(value, str) and not value.strip()) or (key == "files" and not value):
            missing.append(key)
    return not missing, missing


def classify_command_risk(command: str) -> dict[str, Any]:
    text = (command or "").strip()
    low = text.lower()
    risks: list[str] = []
    if any(marker in low for marker in _DEPLOY_MARKERS):
        risks.append("deploy")
    if any(marker.lower() in low for marker in _DESTRUCTIVE_MARKERS):
        risks.append("destructive")
    if re.search(r"(?i)\b(del|erase|rm|remove-item|unlink)\b[^\n]*(data[/\\](users|diary|evolution|experiments)|memory)", text):
        risks.append("memory_delete")
    if any(marker.lower() in low for marker in _SECRET_READ_MARKERS):
        risks.append("secret_access")
    if re.search(r"(?i)(cat|type|get-content|gc)\s+[^\n]*(\.env|key\.md|token|secret|password)", text):
        risks.append("secret_access")
    risk_level = "low"
    if "deploy" in risks or "destructive" in risks or "secret_access" in risks or "memory_delete" in risks:
        risk_level = "high"
    elif any(op in low for op in ("pip install", "npm install -g", "curl ", "wget ")):
        risk_level = "medium"
    return {"risk_level": risk_level, "risks": sorted(set(risks)), "redacted_command": redact_secrets(text)}
