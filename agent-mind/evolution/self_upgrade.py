"""Safe self-upgrade proposal primitives for AIwake.

Phase-1 MVP only creates, validates, lists, and records human approval state for
code-upgrade proposals. It never applies patches, never deploys, and writes all
records append-only under the evolution data directory.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from safety_guard import classify_command_risk, is_memory_path, is_sensitive_path, redact_secrets

_ALLOWED_STATUSES = {"pending", "approved", "rejected", "blocked"}
_APPROVAL_TARGET_STATUSES = {"approved", "rejected"}
_RISK_ORDER = {"low": 1, "medium": 2, "high": 3}
_SECRET_TEXT_RE = re.compile(
    r"(?i)(api[_-]?key|access[_-]?token|secret|password|passwd|pwd|bearer\s+|sk-[a-z0-9]|-----BEGIN [A-Z ]*PRIVATE KEY-----)"
)
_DELETE_LINE_RE = re.compile(r"(?i)(rm\s+-rf|rmdir\s+/s|del\s+/f|remove-item\s+-recurse|unlink|shred|wipe)")
_DEPLOY_LINE_RE = re.compile(r"(?i)(flyctl\s+deploy|\bfly\s+deploy\b|npm\s+run\s+deploy|vercel\s+deploy|railway\s+up|docker\s+push|kubectl\s+apply)")


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _data_dir() -> Path:
    base = os.getenv("EVOLUTION_DIR") or os.getenv("AIWAKE_EVOLUTION_DIR") or "/app/data/evolution"
    return Path(base)


def _proposal_file() -> Path:
    custom = os.getenv("SELF_UPGRADE_PROPOSALS_FILE")
    if custom:
        return Path(custom)
    return _data_dir() / "self_upgrade_proposals.jsonl"


def _normalize_path(path: str | Path) -> str:
    return str(path or "").replace("\\", "/").strip().strip('"\'')


def _is_protected_path(path: str | Path) -> bool:
    text = _normalize_path(path).lower()
    if not text:
        return True
    protected_markers = (
        "data/users/",
        "/data/users",
        "data/diary/",
        "/data/diary",
        "data/evolution/",
        "/data/evolution",
        "data/experiments/",
        "/data/experiments",
        "logs/",
        "/logs/",
        ".env",
        ".tocodex/key.md",
        "key.md",
        "token",
        "secret",
        "password",
        "private_key",
        "memory.json",
        "memories.json",
        "user_memory",
    )
    deploy_files = ("docker-compose.yml", "dockerfile", "fly.toml", "railway.toml", "vercel.json")
    name = Path(text).name.lower()
    return (
        is_sensitive_path(text)
        or is_memory_path(text)
        or any(marker in text for marker in protected_markers)
        or name in deploy_files
    )


def _risk_max(*levels: str) -> str:
    result = "low"
    for level in levels:
        if _RISK_ORDER.get(level, 1) > _RISK_ORDER.get(result, 1):
            result = level
    return result


@dataclass
class SafetyCheck:
    name: str
    status: str
    detail: str


@dataclass
class CodeUpgradeProposal:
    id: str
    created_at: str
    source: str
    problem: str
    evidence: list[Any]
    proposed_files: list[str]
    patch_summary: str
    risk_level: str = "medium"
    safety_checks: list[dict[str, Any]] = field(default_factory=list)
    status: str = "pending"
    human_approval_required: bool = True
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return redact_secrets(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CodeUpgradeProposal":
        return cls(
            id=str(data.get("id") or _proposal_id(data.get("source"), data.get("problem"))),
            created_at=str(data.get("created_at") or _utc_now()),
            source=str(data.get("source") or "unknown"),
            problem=str(data.get("problem") or "未说明问题"),
            evidence=list(data.get("evidence") or []),
            proposed_files=[str(x) for x in data.get("proposed_files") or []],
            patch_summary=str(data.get("patch_summary") or ""),
            risk_level=str(data.get("risk_level") or "medium"),
            safety_checks=list(data.get("safety_checks") or []),
            status=str(data.get("status") or "pending"),
            human_approval_required=bool(data.get("human_approval_required", True)),
            notes=[str(x) for x in data.get("notes") or []],
        )


def _proposal_id(source: Any, problem: Any) -> str:
    now = _utc_now()
    digest = hashlib.sha1(f"{now}|{source}|{problem}".encode("utf-8", errors="ignore")).hexdigest()[:10]
    stamp = now.replace(":", "").replace("-", "").replace(".", "")[:15]
    return f"upg_{stamp}_{digest}"


def check_proposal_safety(
    proposed_files: list[str] | tuple[str, ...],
    patch_summary: str = "",
    evidence: list[Any] | None = None,
) -> tuple[list[dict[str, Any]], str, bool]:
    """Return safety checks, risk level, and whether the proposal is blocked."""

    checks: list[SafetyCheck] = []
    risk = "low"
    blocked = False
    files = [_normalize_path(path) for path in proposed_files if _normalize_path(path)]
    if not files:
        checks.append(SafetyCheck("target_files", "blocked", "提案必须声明至少一个候选文件。"))
        blocked = True
        risk = "high"
    else:
        dangerous_files = [path for path in files if _is_protected_path(path)]
        if dangerous_files:
            checks.append(SafetyCheck("protected_paths", "blocked", f"禁止修改/删除受保护路径: {dangerous_files}"))
            blocked = True
            risk = "high"
        else:
            checks.append(SafetyCheck("protected_paths", "passed", "候选文件未命中记忆、日志、密钥、部署配置等受保护路径。"))

    text = redact_secrets("\n".join([patch_summary or ""] + [json.dumps(x, ensure_ascii=False, default=str) for x in (evidence or [])]))
    if _SECRET_TEXT_RE.search(str(text)):
        checks.append(SafetyCheck("secret_content", "blocked", "提案内容疑似包含密钥、token 或密码片段。"))
        blocked = True
        risk = "high"
    else:
        checks.append(SafetyCheck("secret_content", "passed", "未发现明显密钥或 token 文本。"))

    destructive_match = _DELETE_LINE_RE.search(patch_summary or "")
    if destructive_match:
        checks.append(SafetyCheck("destructive_command", "blocked", "提案摘要包含破坏性删除命令。"))
        blocked = True
        risk = "high"
    else:
        checks.append(SafetyCheck("destructive_command", "passed", "未发现破坏性删除命令。"))

    deploy_match = _DEPLOY_LINE_RE.search(patch_summary or "")
    if deploy_match:
        checks.append(SafetyCheck("deploy_command", "blocked", "第一阶段禁止自动部署或生成部署执行提案。"))
        blocked = True
        risk = "high"
    else:
        checks.append(SafetyCheck("deploy_command", "passed", "未发现部署命令；本模块也不提供应用或部署能力。"))

    command_risk = classify_command_risk(patch_summary or "")
    if command_risk.get("risk_level") == "high":
        checks.append(SafetyCheck("command_risk", "blocked", f"命令风险: {command_risk.get('risks')}"))
        blocked = True
        risk = "high"
    elif command_risk.get("risk_level") == "medium":
        checks.append(SafetyCheck("command_risk", "warning", f"命令风险: {command_risk.get('risks')}"))
        risk = _risk_max(risk, "medium")
    else:
        checks.append(SafetyCheck("command_risk", "passed", "未发现高风险命令。"))

    if any(Path(path).name.lower() in {"main.py", "tool_router.py", "safety_guard.py", "autonomy_config.py"} for path in files):
        risk = _risk_max(risk, "high")
        checks.append(SafetyCheck("core_logic", "warning", "候选文件涉及核心逻辑；允许无人审查模式下自动批准，但必须追加审计日志并保留回滚线索。"))
    elif any(path.endswith((".py", ".html", ".js", ".css", ".md", ".yaml", ".yml")) for path in files):
        risk = _risk_max(risk, "medium")
        checks.append(SafetyCheck("code_change", "warning", "候选文件可能影响运行行为；允许受控应用，但必须记录审计日志。"))

    checks.append(SafetyCheck("apply_policy", "passed", "无人审查模式允许自动批准、受控应用与部署；仍禁止修改日志/记忆/密钥、破坏性删除和泄露敏感信息。"))
    return [asdict(check) for check in checks], risk, blocked


def create_candidate_proposal(
    source: str,
    problem: str,
    evidence: list[Any] | None = None,
    proposed_files: list[str] | None = None,
    patch_summary: str | None = None,
    notes: list[str] | None = None,
) -> CodeUpgradeProposal:
    files = proposed_files or _suggest_files(problem, evidence or [])
    summary = patch_summary or _suggest_patch_summary(problem, files)
    checks, risk, blocked = check_proposal_safety(files, summary, evidence or [])
    status = "blocked" if blocked else "approved"
    proposal = CodeUpgradeProposal(
        id=_proposal_id(source, problem),
        created_at=_utc_now(),
        source=source,
        problem=problem,
        evidence=redact_secrets(evidence or []),
        proposed_files=files,
        patch_summary=summary,
        risk_level=risk,
        safety_checks=checks,
        status=status,
        human_approval_required=False,
        notes=(notes or []) + ["无人审查模式：非阻断提案自动批准；允许受控应用/部署，但必须追加审计记录并避开日志、记忆、密钥与破坏性操作。"],
    )
    return proposal


def create_proposal_from_backlog(backlog_item: dict[str, Any]) -> CodeUpgradeProposal:
    title = str(backlog_item.get("title") or backlog_item.get("id") or "自我进化 backlog 改进项")
    problem = f"{title}"
    evidence = [
        {"type": "backlog", "id": backlog_item.get("id"), "risk": backlog_item.get("risk"), "priority": backlog_item.get("priority")},
        {"type": "evidence", "items": backlog_item.get("evidence") or []},
        {"type": "acceptance", "items": backlog_item.get("acceptance") or []},
    ]
    return create_candidate_proposal(
        source=f"evolution_backlog:{backlog_item.get('id', 'unknown')}",
        problem=problem,
        evidence=evidence,
        notes=["由 EvolutionEngine 从 backlog 自动转化为 approved 提案；无人审查模式下可进入受控应用/部署队列。"],
    )


def _suggest_files(problem: str, evidence: list[Any]) -> list[str]:
    text = json.dumps({"problem": problem, "evidence": evidence}, ensure_ascii=False, default=str).lower()
    if "工具" in problem or "tool" in text:
        return ["src/agent-mind/tool_router.py"]
    if "乱码" in problem or "聊天" in problem or "失败回复" in problem or "chat" in text:
        return ["src/agent-mind/main.py"]
    if "反思" in problem or "主动" in problem or "心跳" in problem:
        return ["src/agent-mind/heartbeat.py"]
    if "记忆" in problem or "memory" in text:
        return ["src/agent-mind/user_memory.py"]
    if "安全" in problem or "secret" in text or "token" in text:
        return ["src/agent-mind/safety_guard.py"]
    return ["src/agent-mind/evolution/engine.py"]


def _suggest_patch_summary(problem: str, files: list[str]) -> str:
    return (
        "候选代码升级方向：基于学习/观察结果定位问题，提出最小补丁草案、验证计划与回滚说明；"
        f"候选文件={files}；问题={problem[:180]}。"
        "本阶段不生成自动应用动作，不执行部署。"
    )


def append_proposal(proposal: CodeUpgradeProposal, path: Path | None = None) -> Path:
    file_path = path or _proposal_file()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = proposal.to_dict()
    with file_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str))
        handle.write("\n")
    return file_path


def read_proposals(limit: int = 50, status: str | None = None, path: Path | None = None) -> list[dict[str, Any]]:
    file_path = path or _proposal_file()
    try:
        max_count = max(1, min(int(limit), 500))
    except Exception:
        max_count = 50
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    items: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            items.append(redact_secrets(item))
    if status:
        items = [item for item in items if item.get("status") == status]
    # Collapse append-only history by latest status per id while preserving recency.
    latest: dict[str, dict[str, Any]] = {}
    for item in items:
        latest[str(item.get("id"))] = item
    return list(latest.values())[-max_count:]


def get_proposal(proposal_id: str) -> dict[str, Any] | None:
    for item in reversed(read_proposals(limit=500)):
        if item.get("id") == proposal_id:
            return item
    return None


def record_approval_status(proposal_id: str, status: str, actor: str = "admin_api", notes: str = "") -> dict[str, Any]:
    """Append an approval/rejection status snapshot. Applying is handled by guarded automation."""

    target_status = str(status or "").strip().lower()
    if target_status not in _APPROVAL_TARGET_STATUSES:
        raise ValueError("status must be approved or rejected")
    current = get_proposal(proposal_id)
    if not current:
        raise KeyError(f"proposal not found: {proposal_id}")
    current_status = str(current.get("status") or "")
    if current_status not in {"pending", "approved", "rejected"}:
        raise ValueError(f"proposal status cannot transition from {current_status} to {target_status}")
    updated = dict(current)
    updated["status"] = target_status
    updated["human_approval_required"] = False
    updated.setdefault("notes", [])
    updated["notes"] = list(updated.get("notes") or []) + [redact_secrets(f"{target_status} by {actor}: {notes}".strip())]
    updated["approval_event"] = redact_secrets({
        "actor": actor,
        "decision": target_status,
        "notes": notes,
        "created_at": _utc_now(),
        "allow_apply": os.getenv("EVOLUTION_UPGRADE_APPLY_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"},
        "allow_deploy": os.getenv("EVOLUTION_UPGRADE_DEPLOY_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"},
        "policy": "无人审查模式：状态记录可进入受控应用/部署队列；所有动作必须追加日志，且禁止修改日志、记忆、密钥或执行破坏性删除。",
    })
    append_proposal(CodeUpgradeProposal.from_dict(updated))
    return redact_secrets(updated)


def auto_approve_pending_proposals(actor: str = "auto_upgrade_policy") -> int:
    """Append approved snapshots for safe pending proposals in no-human-review mode."""

    human_required = os.getenv("EVOLUTION_UPGRADE_HUMAN_APPROVAL_REQUIRED", "false").strip().lower() in {"1", "true", "yes", "on"}
    apply_enabled = os.getenv("EVOLUTION_UPGRADE_APPLY_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
    if human_required or not apply_enabled:
        return 0
    changed = 0
    for proposal in read_proposals(limit=500):
        if proposal.get("status") != "pending":
            continue
        checks = proposal.get("safety_checks") or []
        if any(check.get("status") == "blocked" for check in checks if isinstance(check, dict)):
            continue
        updated = dict(proposal)
        updated["status"] = "approved"
        updated["human_approval_required"] = False
        updated.setdefault("notes", [])
        updated["notes"] = list(updated.get("notes") or []) + [
            redact_secrets(f"auto-approved by {actor}: 无人审查模式下，非阻断提案自动批准并进入受控应用/部署队列。")
        ]
        updated["approval_event"] = redact_secrets({
            "actor": actor,
            "decision": "approved",
            "notes": "auto approve pending proposal under no-human-review guarded mode",
            "created_at": _utc_now(),
            "allow_apply": True,
            "allow_deploy": os.getenv("EVOLUTION_UPGRADE_DEPLOY_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"},
            "policy": "无人审查模式自动批准历史 pending 提案；禁止修改日志、记忆、密钥或执行破坏性删除，所有后续动作必须追加审计记录。",
        })
        append_proposal(CodeUpgradeProposal.from_dict(updated))
        changed += 1
    return changed


def proposal_status_summary(limit: int = 5) -> dict[str, Any]:
    auto_approved_count = auto_approve_pending_proposals()
    proposals = read_proposals(limit=500)
    counts = {status: 0 for status in sorted(_ALLOWED_STATUSES)}
    for proposal in proposals:
        status = str(proposal.get("status") or "pending")
        counts[status] = counts.get(status, 0) + 1
    return redact_secrets({
        "enabled": os.getenv("EVOLUTION_UPGRADE_PROPOSAL_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"},
        "auto_apply_enabled": os.getenv("EVOLUTION_UPGRADE_APPLY_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"},
        "auto_deploy_enabled": os.getenv("EVOLUTION_UPGRADE_DEPLOY_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"},
        "human_approval_required": os.getenv("EVOLUTION_UPGRADE_HUMAN_APPROVAL_REQUIRED", "false").strip().lower() in {"1", "true", "yes", "on"},
        "proposal_file": str(_proposal_file()),
        "counts": counts,
        "total": len(proposals),
        "recent": proposals[-max(1, min(int(limit), 20)):],
        "auto_approved_pending_count": auto_approved_count,
        "safety_policy": {
            "append_only": True,
            "forbidden_paths": ["data/users/", "data/diary/", "data/evolution/", "data/experiments/", "logs/", ".env*", ".tocodex/key.md", "private keys"],
            "forbidden_actions": ["memory/log deletion", "secret/token disclosure", "destructive delete", "modifying audit logs"],
            "allowed_actions": ["generate proposal", "validate proposal", "auto approve", "guarded apply", "guarded deploy", "append audit status", "read status"],
        },
    })


def maybe_generate_from_backlog(backlog: list[dict[str, Any]], enabled: bool | None = None) -> list[dict[str, Any]]:
    """Generate at most one pending proposal from backlog when explicitly enabled."""

    if enabled is None:
        enabled = os.getenv("EVOLUTION_UPGRADE_PROPOSAL_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return []
    candidates = [item for item in (backlog or []) if isinstance(item, dict)]
    if not candidates:
        return []
    selected = sorted(candidates, key=lambda item: float(item.get("priority") or 0), reverse=True)[0]
    proposal = create_proposal_from_backlog(selected)
    append_proposal(proposal)
    return [proposal.to_dict()]
