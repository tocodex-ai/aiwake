"""Safe self-upgrade proposal primitives for AIwake.

Creates, validates, lists, records approval state, and performs guarded apply
of code-upgrade proposals. All records are written append-only under the
evolution data directory. Apply and deploy are gated by environment flags
and safety checks; destructive operations and protected paths are blocked.

Now includes LLM-driven patch generation: reads source files, asks the LLM to
produce search/replace edit blocks, parses them, validates safety, and applies
them with automatic snapshots. This closes the loop from "proposal" to actual
code modification.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from safety_guard import classify_command_risk, is_memory_path, is_sensitive_path, redact_secrets

_logger = logging.getLogger("evolution.self_upgrade")

# search/replace block markers (used by LLM patch generation)
_SEARCH_MARKER = "<<<<<<< SEARCH"
_SEPARATOR = "======="
_REPLACE_MARKER = ">>>>>>> REPLACE"

_ALLOWED_STATUSES = {"pending", "approved", "rejected", "blocked", "applied"}
_APPROVAL_TARGET_STATUSES = {"approved", "rejected"}
# Bounded retry for transient patch-generation failures (e.g. empty LLM
# response). Without a bound, an approved proposal whose patch generation keeps
# failing transiently stays 'approved' forever and the engine re-selects it on
# every tick (root cause of the zombie-approved idle loop observed online on
# 2026-06-16). Once attempts reach this bound the proposal is downgraded to
# 'blocked' so the engine stops re-selecting it.
MAX_PATCH_ATTEMPTS = 3
# 单次 proposal attempt 内仅对空响应重试一次；不会对异常、NO_PATCH 或格式错误
# 无限重试。补丁需要携带完整 search/replace 块，因此显式预留结构化输出预算。
PATCH_EMPTY_RESPONSE_ATTEMPTS = 2
PATCH_MAX_OUTPUT_TOKENS = 8192
_RISK_ORDER = {"low": 1, "medium": 2, "high": 3}
_SECRET_TEXT_RE = re.compile(
    r"(?i)(api[_-]?key\s*[=:]|access[_-]?token\s*[=:]|(?<![a-z_])secret\s*[=:]|password\s*[=:]|passwd\s*[=:]|pwd\s*[=:]|bearer\s+[A-Za-z0-9._-]+|sk-[a-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)"
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
    patch_attempt_count: int = 0

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
            patch_attempt_count=int(data.get("patch_attempt_count") or 0),
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
        checks.append(SafetyCheck("deploy_command", "warning", "提案包含部署命令；无人审查模式允许受控部署，必须完成脱敏检查并追加审计日志。"))
        risk = _risk_max(risk, "high")
    else:
        checks.append(SafetyCheck("deploy_command", "passed", "未发现部署命令。"))

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
        return ["tool_router.py"]
    if "乱码" in problem or "聊天" in problem or "失败回复" in problem or "chat" in text:
        return ["main.py"]
    if "反思" in problem or "主动" in problem or "心跳" in problem:
        return ["heartbeat.py"]
    if "记忆" in problem or "memory" in text:
        return ["user_memory.py"]
    if "安全" in problem or "secret" in text or "token" in text:
        return ["safety_guard.py"]
    return ["evolution/engine.py"]


def _suggest_patch_summary(problem: str, files: list[str]) -> str:
    return (
        "基于学习/观察结果定位问题，生成最小可验证补丁；"
        f"候选文件={files}；问题={problem[:180]}。"
        "在安全边界内受控应用与部署，必须追加审计日志并保留回滚线索。"
    )


# ─────────────────────────────────────────────────────────────────────
# LLM Patch Generation & Search/Replace Application
# ─────────────────────────────────────────────────────────────────────

_APP_ROOT = Path(os.getenv("APP_ROOT", "/app"))
_MAX_SOURCE_CHARS = int(os.getenv("UPGRADE_MAX_SOURCE_CHARS", "30000"))  # max chars to send per file to LLM (was 6000, increased to handle large files like tool_router.py ~73K chars)


def _read_source_file(rel_path: str) -> str | None:
    """Read a source file from the app root. Returns None if not found/protected."""
    norm = _normalize_path(rel_path)
    if _is_protected_path(norm):
        _logger.warning("[self_upgrade] skip protected path: %s", norm)
        return None
    # Try common prefixes
    candidates = [
        _APP_ROOT / norm,
        _APP_ROOT / norm.removeprefix("src/agent-mind/"),
        Path(norm),
    ]
    for p in candidates:
        try:
            if p.exists() and p.is_file():
                content = p.read_text(encoding="utf-8", errors="replace")
                return content[:_MAX_SOURCE_CHARS]
        except Exception:
            continue
    return None


def _resolve_file_path(rel_path: str) -> Path | None:
    """Resolve a relative path to an actual writable path under APP_ROOT."""
    norm = _normalize_path(rel_path)
    if _is_protected_path(norm):
        return None
    candidates = [
        _APP_ROOT / norm,
        _APP_ROOT / norm.removeprefix("src/agent-mind/"),
    ]
    for p in candidates:
        try:
            if p.exists() and p.is_file():
                return p
        except Exception:
            continue
    return None


@dataclass
class EditBlock:
    """A single search/replace edit parsed from LLM output."""
    file_path: str
    search: str
    replace: str


def parse_search_replace_blocks(llm_output: str) -> list[EditBlock]:
    """Parse LLM output containing search/replace edit blocks.

    Expected format per block (repeatable):
        FILE: path/to/file.py
        <<<<<<< SEARCH
        old code here
        =======
        new code here
        >>>>>>> REPLACE

    Returns a list of EditBlock. Tolerates extra whitespace and minor format variations.
    """
    blocks: list[EditBlock] = []
    lines = llm_output.split("\n")
    i = 0
    current_file = ""
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Detect file path line
        if stripped.upper().startswith("FILE:"):
            current_file = stripped[5:].strip().strip("`\"'")
            i += 1
            continue

        # Detect SEARCH marker
        if _SEARCH_MARKER in stripped:
            # Collect search content
            search_lines: list[str] = []
            i += 1
            while i < len(lines) and _SEPARATOR not in lines[i].strip():
                search_lines.append(lines[i])
                i += 1
            # Skip separator
            if i < len(lines):
                i += 1
            # Collect replace content
            replace_lines: list[str] = []
            while i < len(lines) and _REPLACE_MARKER not in lines[i].strip():
                replace_lines.append(lines[i])
                i += 1
            # Skip REPLACE marker
            if i < len(lines):
                i += 1

            search_text = "\n".join(search_lines)
            replace_text = "\n".join(replace_lines)
            if search_text.strip() and current_file:
                blocks.append(EditBlock(
                    file_path=current_file,
                    search=search_text,
                    replace=replace_text,
                ))
            continue
        i += 1

    return blocks


def _apply_single_edit(content: str, search: str, replace: str) -> tuple[str, bool]:
    """Apply a single search/replace edit to file content.

    Returns (new_content, success). Uses exact match first, then
    tries stripping trailing whitespace on each line for fuzzy match.
    """
    # Exact match
    if search in content:
        return content.replace(search, replace, 1), True

    # Fuzzy: strip trailing whitespace per line
    def strip_trailing(text: str) -> str:
        return "\n".join(line.rstrip() for line in text.split("\n"))

    stripped_content = strip_trailing(content)
    stripped_search = strip_trailing(search)
    if stripped_search in stripped_content:
        idx = stripped_content.index(stripped_search)
        new = stripped_content[:idx] + replace + stripped_content[idx + len(stripped_search):]
        return new, True

    return content, False


def apply_search_replace_patch(
    edits: list[EditBlock],
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Apply a list of search/replace edits to source files.

    Creates .snapshot backups before writing. Returns a result dict.
    """
    results: list[dict[str, Any]] = []
    files_modified: list[str] = []
    errors: list[str] = []

    for edit in edits:
        fpath = _resolve_file_path(edit.file_path)
        if fpath is None:
            errors.append(f"文件未找到或被保护: {edit.file_path}")
            results.append({"file": edit.file_path, "applied": False, "reason": "file_not_found_or_protected"})
            continue

        # Re-check safety on actual file content
        if _is_protected_path(str(fpath)):
            errors.append(f"受保护路径: {edit.file_path}")
            results.append({"file": edit.file_path, "applied": False, "reason": "protected_path"})
            continue

        # Check for secret content in replacement
        if _SECRET_TEXT_RE.search(edit.replace):
            errors.append(f"替换内容包含敏感信息: {edit.file_path}")
            results.append({"file": edit.file_path, "applied": False, "reason": "secret_in_replacement"})
            continue

        # Check for destructive commands in replacement
        if _DELETE_LINE_RE.search(edit.replace):
            errors.append(f"替换内容包含破坏性命令: {edit.file_path}")
            results.append({"file": edit.file_path, "applied": False, "reason": "destructive_command"})
            continue

        try:
            original = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            errors.append(f"读取文件失败 {edit.file_path}: {exc}")
            results.append({"file": edit.file_path, "applied": False, "reason": "read_error"})
            continue

        new_content, success = _apply_single_edit(original, edit.search, edit.replace)
        if not success:
            errors.append(f"SEARCH 块未匹配: {edit.file_path}")
            results.append({"file": edit.file_path, "applied": False, "reason": "search_not_found"})
            continue

        if dry_run:
            results.append({"file": edit.file_path, "applied": True, "dry_run": True})
            files_modified.append(edit.file_path)
            continue

        # Create snapshot before writing
        try:
            snapshot_path = fpath.with_suffix(fpath.suffix + f".snapshot-{int(time.time())}")
            snapshot_path.write_text(original, encoding="utf-8")
        except Exception as exc:
            _logger.warning("[self_upgrade] snapshot failed for %s: %s", fpath, exc)

        # Write modified content
        try:
            fpath.write_text(new_content, encoding="utf-8")
            files_modified.append(edit.file_path)
            results.append({"file": edit.file_path, "applied": True})
            _logger.info("[self_upgrade] patch applied to %s", fpath)
        except Exception as exc:
            errors.append(f"写入失败 {edit.file_path}: {exc}")
            results.append({"file": edit.file_path, "applied": False, "reason": "write_error"})

    return {
        "total_edits": len(edits),
        "applied_count": len(files_modified),
        "files_modified": files_modified,
        "errors": errors,
        "results": results,
    }


def _applied_patch_archive_dir() -> Path:
    """Directory (on the mounted persistent volume) where applied-patch copies
    are archived so they survive container restarts / re-deploys.

    Code edits land on the image layer (e.g. /app/tool_router.py), which is
    EPHEMERAL: a restart or re-deploy rebuilds the image from source and wipes
    those in-container edits. Without a persistent archive the self-upgrade
    loop "applies" patches that silently vanish — real work, zero durable
    effect. This archive lives under _data_dir() (the /app/data volume), giving
    every applied patch a durable, auditable, append-only record that can later
    be reconciled back into the source repository.
    """
    return _data_dir() / "applied_patches"


def archive_applied_patch(
    proposal_id: str,
    apply_result: dict[str, Any],
    *,
    actor: str = "evolution_engine_auto_apply",
) -> dict[str, Any]:
    """Persist copies of files modified by a successful patch to the volume.

    For each modified file we store its post-patch content plus a manifest
    entry under ``/app/data/evolution/applied_patches/<proposal_id>/``. This is
    additive and append-only; it never deletes or overwrites memory, diary,
    logs or secrets. Returns a small summary dict (also used in audit notes).
    """
    files_modified = apply_result.get("files_modified") or []
    if not files_modified:
        return {"archived": False, "reason": "no_files_modified", "archived_files": []}

    base = _applied_patch_archive_dir() / _normalize_path(proposal_id).replace("/", "_")
    archived_files: list[str] = []
    errors: list[str] = []
    try:
        base.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        return {"archived": False, "reason": f"mkdir_failed: {exc}", "archived_files": []}

    # De-duplicate while preserving order (same file can appear multiple times).
    seen: set[str] = set()
    unique_files = [f for f in files_modified if not (f in seen or seen.add(f))]

    for rel in unique_files:
        src = _resolve_file_path(rel)
        if src is None:
            errors.append(f"unresolved_or_protected: {rel}")
            continue
        try:
            content = src.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            errors.append(f"read_failed {rel}: {exc}")
            continue
        # Flatten the relative path into a single safe filename copy.
        safe_name = _normalize_path(rel).replace("/", "__")
        try:
            (base / safe_name).write_text(content, encoding="utf-8")
            archived_files.append(rel)
        except Exception as exc:
            errors.append(f"write_failed {rel}: {exc}")

    manifest = redact_secrets({
        "proposal_id": proposal_id,
        "actor": actor,
        "archived_at": _utc_now(),
        "files": archived_files,
        "errors": errors,
        "note": (
            "Durable copy of in-container edits on the /app/data volume; "
            "survives restart/redeploy and can be reconciled back to source."
        ),
    })
    try:
        with (base / "manifest.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest, ensure_ascii=False, default=str))
            handle.write("\n")
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        errors.append(f"manifest_write_failed: {exc}")

    return {
        "archived": bool(archived_files),
        "archive_dir": str(base),
        "archived_files": archived_files,
        "errors": errors,
    }


_PATCH_SYSTEM_PROMPT = (
    "你是 AIwake 的内部代码改进引擎。你的任务是根据问题描述和源代码，生成精确的 search/replace 编辑块。\n\n"
    "规则：\n"
    "1. 每个编辑块必须严格按照以下格式输出：\n"
    "   FILE: 文件的相对路径\n"
    "   <<<<<<< SEARCH\n"
    "   要被替换的精确原始代码（必须能在文件中找到完全匹配）\n"
    "   =======\n"
    "   替换后的新代码\n"
    "   >>>>>>> REPLACE\n\n"
    "2. SEARCH 块必须包含足够多的上下文行以确保唯一匹配\n"
    "3. 只修改解决问题必需的最小代码区域\n"
    "4. 不要修改日志文件、用户记忆、密钥、配置环境变量\n"
    "5. 不要添加破坏性删除命令（rm -rf, del /f 等）\n"
    "6. 不要泄露任何 API key、token 或密码\n"
    "7. 保持代码风格与现有代码一致\n"
    "8. 如果不确定如何修复，输出 NO_PATCH: 原因 而不是猜测\n\n"
    "重要：只输出编辑块，不要输出其他解释性文字。"
)


def _build_patch_prompt(problem: str, files: list[str], evidence: list[Any]) -> str:
    """Build user prompt with problem + source context for LLM patch generation."""
    parts = [f"## 问题\n{problem}\n"]

    if evidence:
        ev_text = "\n".join(str(e)[:300] for e in evidence[:5])
        parts.append(f"## 证据\n{ev_text}\n")

    parts.append("## 候选文件源代码\n")
    for fpath in files[:3]:  # max 3 files
        content = _read_source_file(fpath)
        if content:
            parts.append(f"### FILE: {fpath}\n```python\n{content}\n```\n")
        else:
            parts.append(f"### FILE: {fpath}\n（文件不可读或受保护）\n")

    parts.append("请生成 search/replace 编辑块来修复上述问题。")
    return "\n".join(parts)


async def generate_llm_patch(
    problem: str,
    files: list[str],
    evidence: list[Any] | None = None,
) -> dict[str, Any]:
    """Call LLM to generate a real code patch for the given problem.

    Returns dict with keys: success, edits (list[EditBlock]), raw_output, error.
    """
    from llm_gate import LLMGate

    evidence = evidence or []
    user_prompt = _build_patch_prompt(problem, files, evidence)

    try:
        gate = LLMGate()
        raw_output = ""
        for attempt in range(PATCH_EMPTY_RESPONSE_ATTEMPTS):
            raw_output = await gate.call_cloud(
                _PATCH_SYSTEM_PROMPT,
                user_prompt,
                profile_name="work",
                max_tokens_override=PATCH_MAX_OUTPUT_TOKENS,
            )
            if raw_output:
                break
            if attempt < PATCH_EMPTY_RESPONSE_ATTEMPTS - 1:
                _logger.info(
                    "[self_upgrade] LLM returned empty response; retrying once "
                    "within the current proposal attempt"
                )
    except Exception as exc:
        _logger.error("[self_upgrade] LLM patch generation failed: %s", exc)
        return {"success": False, "edits": [], "raw_output": "", "error": str(exc)}

    if not raw_output:
        _logger.info(
            "[self_upgrade] LLM returned empty response after %d calls (transient)",
            PATCH_EMPTY_RESPONSE_ATTEMPTS,
        )
        return {"success": False, "declined": False, "edits": [], "raw_output": "", "error": "LLM returned empty response"}

    if "NO_PATCH" in raw_output:
        # LLM judged the source already satisfies the requirement — this is a
        # terminal "no change needed" decision, NOT a transient failure.
        # Marking declined=True lets callers close the proposal instead of
        # endlessly retrying it (root cause of the NO_PATCH idle loop).
        _logger.info("[self_upgrade] LLM declined to patch (NO_PATCH): %s", raw_output[:200])
        return {"success": False, "declined": True, "edits": [], "raw_output": raw_output, "error": raw_output}

    edits = parse_search_replace_blocks(raw_output)
    if not edits:
        _logger.warning("[self_upgrade] LLM output contained no valid edit blocks")
        return {"success": False, "edits": [], "raw_output": raw_output[:500], "error": "no_valid_edit_blocks"}

    return {"success": True, "edits": edits, "raw_output": raw_output[:1000], "error": ""}


async def generate_and_apply_patch(
    proposal_id: str,
    *,
    dry_run: bool = False,
    actor: str = "evolution_engine_auto_apply",
) -> dict[str, Any]:
    """Full pipeline: read proposal -> LLM generates patch -> safety check -> apply.

    This is the key function that turns an approved proposal into actual code changes.
    Returns a result dict with apply status and details.
    """
    current = get_proposal(proposal_id)
    if not current:
        return {"applied": False, "reason": f"proposal not found: {proposal_id}"}

    status = str(current.get("status") or "")
    if status != "approved":
        return {"applied": False, "reason": f"proposal status is {status}, not approved"}

    files = current.get("proposed_files") or []
    problem = current.get("problem") or ""
    evidence = current.get("evidence") or []

    # Step 1: Generate patch via LLM
    _logger.info("[self_upgrade] generating LLM patch for proposal %s: %s", proposal_id, problem[:100])
    patch_result = await generate_llm_patch(problem, files, evidence)

    if not patch_result.get("success"):
        updated = dict(current)
        updated.setdefault("notes", [])
        if patch_result.get("declined"):
            # NO_PATCH: LLM judged the source already satisfies the requirement.
            # This is a terminal "no change needed" outcome — close the proposal
            # as rejected so the engine stops re-picking it (fixes NO_PATCH idle
            # loop). The dedup window then prevents immediate regeneration.
            updated["status"] = "rejected"
            updated["notes"] = list(updated.get("notes") or []) + [
                redact_secrets(f"no_patch_needed by {actor}: 源代码已实现该需求，无需改动。{str(patch_result.get('error', ''))[:200]}")
            ]
            updated["patch_generation_event"] = redact_secrets({
                "actor": actor,
                "attempted_at": _utc_now(),
                "success": False,
                "declined": True,
                "reason": "no_patch_needed",
                "detail": str(patch_result.get("error", ""))[:300],
            })
            append_proposal(CodeUpgradeProposal.from_dict(updated))
            return {"applied": False, "declined": True, "reason": "no_patch_needed: 源代码已实现该需求，提案已关闭"}
        # Transient generation failure (e.g. empty LLM response). Increment a
        # bounded attempt counter: stay 'approved' (retry on next tick) while
        # under MAX_PATCH_ATTEMPTS, then downgrade to 'blocked' so the engine
        # stops re-selecting a proposal whose patch generation keeps failing
        # transiently (root cause of the zombie-approved idle loop).
        try:
            attempts = int(updated.get("patch_attempt_count") or 0) + 1
        except (TypeError, ValueError):
            attempts = 1
        updated["patch_attempt_count"] = attempts
        downgraded = attempts >= MAX_PATCH_ATTEMPTS
        if downgraded:
            updated["status"] = "blocked"
            note = (
                f"transient generation failure exceeded max attempts "
                f"({attempts}/{MAX_PATCH_ATTEMPTS}) by {actor}: "
                f"{str(patch_result.get('error', 'unknown'))[:200]} — 已降级为 blocked，停止无限重选。"
            )
        else:
            note = (
                f"LLM patch generation failed by {actor} "
                f"(attempt {attempts}/{MAX_PATCH_ATTEMPTS}): "
                f"{str(patch_result.get('error', 'unknown'))[:200]} — 保持 approved，下次心跳重试。"
            )
        updated["notes"] = list(updated.get("notes") or []) + [redact_secrets(note)]
        updated["patch_generation_event"] = redact_secrets({
            "actor": actor,
            "attempted_at": _utc_now(),
            "success": False,
            "error": str(patch_result.get("error", ""))[:300],
            "patch_attempt_count": attempts,
            "max_patch_attempts": MAX_PATCH_ATTEMPTS,
            "downgraded_to_blocked": downgraded,
        })
        append_proposal(CodeUpgradeProposal.from_dict(updated))
        if downgraded:
            return {
                "applied": False,
                "blocked": True,
                "reason": (
                    f"transient generation failure exceeded max attempts "
                    f"({attempts}/{MAX_PATCH_ATTEMPTS}); proposal downgraded to blocked"
                ),
            }
        return {
            "applied": False,
            "reason": (
                f"LLM patch generation failed (attempt {attempts}/{MAX_PATCH_ATTEMPTS}): "
                f"{patch_result.get('error', '')[:200]}"
            ),
        }

    edits = patch_result["edits"]
    _logger.info("[self_upgrade] LLM generated %d edit blocks for proposal %s", len(edits), proposal_id)

    # Step 2: Safety re-check on actual patch content
    for edit in edits:
        if _is_protected_path(edit.file_path):
            updated = dict(current)
            updated["status"] = "blocked"
            updated.setdefault("notes", [])
            updated["notes"] = list(updated.get("notes") or []) + [
                redact_secrets(f"patch blocked: edit targets protected path {edit.file_path}")
            ]
            append_proposal(CodeUpgradeProposal.from_dict(updated))
            return {"applied": False, "reason": f"patch targets protected path: {edit.file_path}"}

        if _SECRET_TEXT_RE.search(edit.replace):
            updated = dict(current)
            updated["status"] = "blocked"
            updated.setdefault("notes", [])
            updated["notes"] = list(updated.get("notes") or []) + [
                "patch blocked: replacement content contains sensitive data"
            ]
            append_proposal(CodeUpgradeProposal.from_dict(updated))
            return {"applied": False, "reason": "replacement contains sensitive data"}

    # Step 3: Apply the patch
    apply_result = apply_search_replace_patch(edits, dry_run=dry_run)
    applied_count = apply_result.get("applied_count", 0)

    # Step 4: Record result
    updated = dict(current)
    archive_result: dict[str, Any] = {"archived": False}
    if applied_count > 0:
        updated["status"] = "applied"
        # Persist a durable copy of the modified files to the /app/data volume.
        # In-container edits on the image layer are wiped on restart/redeploy;
        # archiving here turns a previously ephemeral "apply" into a durable,
        # auditable record that survives restarts and can be reconciled back to
        # the source repository. Skipped for dry runs.
        if not dry_run:
            try:
                archive_result = archive_applied_patch(proposal_id, apply_result, actor=actor)
            except Exception as exc:  # pragma: no cover - defensive runtime guard
                archive_result = {"archived": False, "reason": f"archive_exception: {exc}"}
        updated.setdefault("notes", [])
        updated["notes"] = list(updated.get("notes") or []) + [
            redact_secrets(
                f"applied by {actor}: LLM 生成并应用了 {applied_count} 个编辑块。"
                f" 修改文件: {apply_result.get('files_modified', [])}。"
                f" 持久化归档: {'成功' if archive_result.get('archived') else '未归档'}"
                f"({len(archive_result.get('archived_files') or [])} 个文件)。"
                f" {'(dry_run)' if dry_run else ''}"
            )
        ]
        updated["apply_event"] = redact_secrets({
            "actor": actor,
            "applied_at": _utc_now(),
            "files_modified": apply_result.get("files_modified", []),
            "edit_count": len(edits),
            "applied_count": applied_count,
            "dry_run": dry_run,
            "raw_output_preview": patch_result.get("raw_output", "")[:300],
            "archive": {
                "archived": archive_result.get("archived", False),
                "archive_dir": archive_result.get("archive_dir", ""),
                "archived_files": archive_result.get("archived_files", []),
            },
            "policy": "LLM 自主生成 search/replace 补丁并受控应用；已通过安全检查，已创建 snapshot 备份，并将改动持久化归档到挂载卷以防重启丢失。",
        })
    else:
        updated.setdefault("notes", [])
        updated["notes"] = list(updated.get("notes") or []) + [
            redact_secrets(f"patch apply failed by {actor}: {apply_result.get('errors', [])}")
        ]
        updated["patch_generation_event"] = redact_secrets({
            "actor": actor,
            "attempted_at": _utc_now(),
            "success": False,
            "errors": apply_result.get("errors", []),
        })

    append_proposal(CodeUpgradeProposal.from_dict(updated))

    return {
        "applied": applied_count > 0,
        "proposal": redact_secrets(updated),
        "reason": f"applied {applied_count}/{len(edits)} edits" if applied_count > 0 else f"no edits applied: {apply_result.get('errors', [])}",
        "files_modified": apply_result.get("files_modified", []),
        "archived": bool(archive_result.get("archived")),
        "archived_files": archive_result.get("archived_files", []),
        "dry_run": dry_run,
    }


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
    # Collapse append-only history by latest status per id FIRST,
    # so that status filter sees the true latest status per proposal.
    latest: dict[str, dict[str, Any]] = {}
    for item in items:
        latest[str(item.get("id"))] = item
    collapsed = list(latest.values())
    if status:
        collapsed = [item for item in collapsed if item.get("status") == status]
    return collapsed[-max_count:]


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


def guarded_apply_proposal(proposal_id: str, actor: str = "auto_upgrade_policy") -> dict[str, Any]:
    """Apply an approved proposal in a guarded manner.

    Performs safety re-check before applying, records the apply event
    append-only, and returns the updated proposal record. This function
    does NOT deploy; deployment is a separate guarded step.

    Returns a dict with keys: applied (bool), proposal (dict), reason (str).
    """
    apply_enabled = os.getenv("EVOLUTION_UPGRADE_APPLY_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
    if not apply_enabled:
        return {"applied": False, "proposal": {}, "reason": "EVOLUTION_UPGRADE_APPLY_ENABLED is disabled"}

    current = get_proposal(proposal_id)
    if not current:
        return {"applied": False, "proposal": {}, "reason": f"proposal not found: {proposal_id}"}

    status = str(current.get("status") or "")
    if status != "approved":
        return {"applied": False, "proposal": current, "reason": f"proposal status is {status}, not approved"}

    # Re-run safety checks before applying
    files = current.get("proposed_files") or []
    summary = current.get("patch_summary") or ""
    evidence = current.get("evidence") or []
    checks, risk, blocked = check_proposal_safety(files, summary, evidence)

    if blocked:
        updated = dict(current)
        updated["status"] = "blocked"
        updated.setdefault("notes", [])
        updated["notes"] = list(updated.get("notes") or []) + [
            redact_secrets(f"apply blocked by {actor}: safety re-check failed - {[c.get('detail') for c in checks if c.get('status') == 'blocked']}")
        ]
        append_proposal(CodeUpgradeProposal.from_dict(updated))
        return {"applied": False, "proposal": updated, "reason": "safety re-check blocked the apply"}

    # Record apply event (append-only)
    updated = dict(current)
    updated["status"] = "applied"
    updated.setdefault("notes", [])
    updated["notes"] = list(updated.get("notes") or []) + [
        redact_secrets(f"applied by {actor}: 受控应用完成，已通过安全检查。补丁摘要: {summary[:200]}")
    ]
    updated["apply_event"] = redact_secrets({
        "actor": actor,
        "applied_at": _utc_now(),
        "files": files,
        "risk_level": risk,
        "safety_checks": checks,
        "allow_deploy": os.getenv("EVOLUTION_UPGRADE_DEPLOY_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"},
        "policy": "受控应用：已通过安全检查，禁止修改日志/记忆/密钥，所有动作已追加审计记录。",
    })
    append_proposal(CodeUpgradeProposal.from_dict(updated))
    return {"applied": True, "proposal": updated, "reason": "applied successfully"}


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


def _dedup_key(text: str) -> str:
    """Extract dedup key from problem/title text.

    Backlog titles follow the pattern ``"规则标题: 具体描述"``
    (e.g. "提升工具调用可靠性: 检测到工具调用失败").
    Different evaluation cycles produce different suffixes for the same
    rule category, causing exact-match dedup to miss them.

    This helper extracts the prefix before the first ``": "`` separator
    so that all variants of the same rule category share one dedup key.
    If no separator is found, the full (trimmed, lowered) text is used.
    """
    s = str(text).strip().lower()[:120]
    if ": " in s:
        return s.split(": ", 1)[0]
    if "：" in s:  # fullwidth colon variant
        return s.split("：", 1)[0]
    return s


def _collect_existing_proposal_keys(limit: int = 500) -> set[str]:
    """Collect dedup keys of existing proposals that should suppress regeneration.

    Mirrors the rule used by :func:`maybe_generate_from_backlog`:
    - ``approved``/``pending`` proposals always suppress same-category regeneration
      (no time window).
    - ``applied``/``rejected`` proposals suppress only within the dedup window
      (default 6h), so an improvement can be re-proposed after the window expires
      while NO_PATCH-closed proposals do not immediately respawn an idle loop.
    """

    existing_problems: set[str] = set()
    dedup_window_hours = float(os.getenv("UPGRADE_DEDUP_WINDOW_HOURS", "6"))
    dedup_cutoff = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(hours=dedup_window_hours)
    for prop in read_proposals(limit=limit):
        status = prop.get("status")
        if status in {"approved", "pending"}:
            key = _dedup_key(prop.get("problem", ""))
            if key:
                existing_problems.add(key)
        elif status in {"applied", "rejected"}:
            created = prop.get("created_at", "")
            try:
                prop_time = _dt.datetime.fromisoformat(str(created))
            except (ValueError, TypeError):
                prop_time = None
            if prop_time and prop_time > dedup_cutoff:
                key = _dedup_key(prop.get("problem", ""))
                if key:
                    existing_problems.add(key)
    return existing_problems


def classify_backlog_dedup(backlog: list[dict[str, Any]]) -> dict[str, Any]:
    """Explain why backlog items may not yield a fresh proposal this cycle.

    Returns a summary distinguishing backlog items that are *suppressed by dedup*
    (an equivalent proposal already exists/was recently handled — a normal, healthy
    state) from items that are genuinely *actionable* (no existing proposal covers
    them yet). This lets the upgrade-pressure gate avoid falsely reporting
    ``proposal_missing`` when the only reason no proposal was generated is that the
    backlog is already covered, which previously drove repeated reflection spinning.
    """

    candidates = [item for item in (backlog or []) if isinstance(item, dict)]
    result: dict[str, Any] = {
        "backlog_count": len(candidates),
        "actionable_count": 0,
        "deduped_count": 0,
        "deduped_keys": [],
        "all_deduped": False,
    }
    if not candidates:
        return result
    existing = _collect_existing_proposal_keys()
    deduped_keys: list[str] = []
    actionable = 0
    for item in candidates:
        key = _dedup_key(item.get("title") or item.get("problem", ""))
        if key and key in existing:
            deduped_keys.append(key)
        else:
            actionable += 1
    result["actionable_count"] = actionable
    result["deduped_count"] = len(deduped_keys)
    result["deduped_keys"] = sorted(set(deduped_keys))
    result["all_deduped"] = actionable == 0 and len(deduped_keys) > 0
    return result


def maybe_generate_from_backlog(backlog: list[dict[str, Any]], enabled: bool | None = None) -> list[dict[str, Any]]:
    """Generate at most one pending proposal from backlog when explicitly enabled.

    Includes deduplication: skips backlog items whose problem text already has
    an existing proposal (approved/pending) created within the last 6 hours,
    preventing identical proposals from accumulating.

    Dedup uses prefix matching: ``"提升工具调用可靠性: X"`` and
    ``"提升工具调用可靠性: Y"`` are treated as the same problem category.
    """

    if enabled is None:
        enabled = os.getenv("EVOLUTION_UPGRADE_PROPOSAL_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return []
    candidates = [item for item in (backlog or []) if isinstance(item, dict)]
    if not candidates:
        return []

    # ── 去重：收集已有提案的 problem 前缀 key ──
    existing_problems = _collect_existing_proposal_keys()

    selected = sorted(candidates, key=lambda item: float(item.get("priority") or 0), reverse=True)
    for item in selected:
        # backlog 条目用 "title"，create_proposal_from_backlog 将其转为 problem
        key = _dedup_key(item.get("title") or item.get("problem", ""))
        if key in existing_problems:
            continue  # 跳过已有相同问题类别的提案
        proposal = create_proposal_from_backlog(item)
        append_proposal(proposal)
        return [proposal.to_dict()]

    return []  # 所有候选项已有提案，无需生成
