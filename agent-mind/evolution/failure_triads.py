"""Failure triad evidence + other_error clustering.

目标：
1. 每次工具失败自动追加 failure_triads.jsonl 三元证据
   (failure_category / recovery_action / recovery_result)
2. 提供 degradation_evidence_miner：对 other_error 等失败做关键词聚类
3. evolution 周期在 other_error 占比过高时强制挖掘一次

只追加、不删除历史；密钥经 redact 后再落盘。
"""
from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from safety_guard import redact_secrets
except Exception:  # pragma: no cover
    def redact_secrets(value: Any) -> Any:  # type: ignore[misc]
        return value


def _data_dir() -> Path:
    base = os.getenv("EVOLUTION_DIR") or os.getenv("AIWAKE_EVOLUTION_DIR") or "/app/data/evolution"
    return Path(base)


def failure_triads_path() -> Path:
    return _data_dir() / "failure_triads.jsonl"


def failure_clusters_path() -> Path:
    return _data_dir() / "failure_clusters.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _append_jsonl(path: Path, item: dict[str, Any]) -> Path | None:
    try:
        _ensure_parent(path)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(item, ensure_ascii=False, default=str))
            fh.write("\n")
        return path
    except OSError:
        return None


def _read_jsonl(path: Path, limit: int = 200) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    items: list[dict[str, Any]] = []
    for line in lines[-max(1, int(limit)) :]:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            # 兼容 append_jsonl(name,item) 包裹形态
            if isinstance(obj.get("item"), dict) and "failure_category" not in obj:
                items.append(obj["item"])
            else:
                items.append(obj)
    return items


def map_failure_category(failure_type: str | None, tool_name: str = "", error_text: str = "") -> str:
    """把 tool_router.failure_type 映射到 metrics 友好的粗分类。"""
    ft = str(failure_type or "").strip().lower()
    text = f"{tool_name} {error_text} {ft}".lower()
    if not ft and not text.strip():
        return "other_error"
    if "command_missing" in ft or "not found" in text and "command" in text:
        return "command_missing"
    if "timeout" in ft or "timeout" in text or "超时" in text:
        return "timeout"
    if "rate_limited" in ft or "429" in text or "rate limit" in text:
        return "rate_limited"
    if "forbidden" in ft or "403" in text or "permission" in text:
        return "forbidden"
    if "not_found" in ft or "404" in text or "文件不存在" in text:
        return "not_found"
    if "config_blocked" in ft or "未被当前实验配置允许" in text:
        return "config_blocked"
    if "network" in ft or "connection refused" in text or "dns" in text:
        return "network_error"
    if "private_network" in ft or "ssrf" in text:
        return "forbidden"
    if "empty_result" in ft:
        return "empty_result"
    if ft in {
        "command_missing",
        "timeout",
        "rate_limited",
        "forbidden",
        "not_found",
        "config_blocked",
        "network_error",
        "other_error",
        "upstream_error",
        "unavailable",
    }:
        return ft
    # tool_router 细分类大多仍应进入 other_error 以便 miner 聚类
    if ft.startswith("aiwake_") or "/" in ft or ft.endswith("_failure"):
        return "other_error"
    return ft or "other_error"


def default_recovery_action(category: str, fallback_hint: str = "") -> str:
    cat = (category or "other_error").lower()
    if fallback_hint:
        return str(redact_secrets(fallback_hint))[:300]
    mapping = {
        "command_missing": "预检命令是否存在；换用已安装等价命令，停止重复同一缺失二进制",
        "timeout": "缩小范围后单次重试；仍超时则换轻量工具",
        "rate_limited": "退避后重试或切换等价网络工具",
        "forbidden": "停止绕过安全限制，改用允许的只读路径",
        "not_found": "核对真实路径/URL，不重试同一不存在资源",
        "config_blocked": "改用已允许工具名，不重复未授权工具",
        "network_error": "检查网络可达性后单次重试，失败则延后",
        "empty_result": "调整关键词/日期后单次复验，不当作基础设施故障",
        "other_error": "记录参数与错误摘要，聚类同类失败并选等价工具",
    }
    return mapping.get(cat, mapping["other_error"])


def append_failure_triad(
    *,
    tool_name: str,
    failure_type: str | None = None,
    error: Any = None,
    recovery_action: str | None = None,
    recovery_result: str = "recorded",
    status: str | None = None,
    http_status: Any = None,
    returncode: Any = None,
    diagnosis: Any = None,
    source: str = "tool_router",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """追加一条 failure triad。返回写入记录（含 path）。"""
    error_text = str(redact_secrets(error or ""))[:500]
    category = map_failure_category(failure_type, tool_name, error_text)
    action = recovery_action or default_recovery_action(category)
    record: dict[str, Any] = {
        "id": f"triad_{int(time.time())}_{abs(hash((tool_name, category, error_text[:40]))) % 10_000_000:07d}",
        "ts": time.time(),
        "created_at": _now_iso(),
        "event": "failure_triad",
        "source": source,
        "tool": tool_name,
        "failure_type": failure_type,
        "failure_category": category,
        "recovery_action": str(redact_secrets(action))[:400],
        "recovery_result": str(redact_secrets(recovery_result))[:300],
        "status": status,
        "error": error_text,
        "http_status": http_status,
        "returncode": returncode,
        "diagnosis": diagnosis,
    }
    if extra:
        # 只保留可序列化的小字段
        for k, v in list(extra.items())[:12]:
            if k in record:
                continue
            record[k] = redact_secrets(v) if isinstance(v, (str, int, float, bool)) or v is None else str(redact_secrets(v))[:200]

    path = _append_jsonl(failure_triads_path(), record)
    record["path"] = str(path) if path else None
    record["written"] = path is not None
    return record


def ensure_init_marker() -> dict[str, Any] | None:
    """若文件不存在则写入 INIT 标记（仅一次）。"""
    path = failure_triads_path()
    if path.exists():
        return None
    return append_failure_triad(
        tool_name="system",
        failure_type="init",
        error="INIT failure_triads store",
        recovery_action="initialize append-only evidence store",
        recovery_result="created",
        status="ok",
        source="failure_triads.init",
    )


def read_failure_triads(limit: int = 200) -> list[dict[str, Any]]:
    ensure_init_marker()
    return _read_jsonl(failure_triads_path(), limit=limit)


def _token_patterns(text: str) -> list[str]:
    text = (text or "").lower()
    patterns: list[str] = []
    rules = [
        (r"timeout|timed out|超时", "timeout_keyword"),
        (r"rate limit|too many requests|429", "rate_limit_keyword"),
        (r"permission denied|not permitted|forbidden|403", "permission_keyword"),
        (r"not found|no such file|404|文件不存在", "not_found_keyword"),
        (r"command not found|missing_binary|returncode.?127", "command_missing_keyword"),
        (r"connection refused|network|dns|no route", "network_keyword"),
        (r"json|decode|parse", "parse_keyword"),
        (r"invalid|validation|schema|required", "validation_keyword"),
        (r"empty|无结果|暂无", "empty_keyword"),
        (r"ssrf|private network|localhost|127\.0\.0\.1", "ssrf_keyword"),
        (r"traceback|exception|typeerror|valueerror|keyerror", "exception_keyword"),
        (r"budget|quota|limit exceeded|hourly", "budget_keyword"),
    ]
    for pat, name in rules:
        if re.search(pat, text, re.I):
            patterns.append(name)
    if not patterns:
        # 退化：取前几个词作为弱模式
        words = re.findall(r"[a-zA-Z_]{4,}|\w{2,}", text)
        for w in words[:3]:
            patterns.append(f"token:{w[:24]}")
    return patterns


def mine_other_error_patterns(
    *,
    limit: int = 200,
    min_count: int = 1,
    categories: set[str] | None = None,
    source: str = "degradation_evidence_miner",
) -> dict[str, Any]:
    """对 failure_triads 中的 other_error（默认）做模式聚类。"""
    ensure_init_marker()
    cats = categories or {"other_error", "aiwake_tool_failure"}
    items = read_failure_triads(limit=limit)
    selected = [
        it
        for it in items
        if str(it.get("failure_category") or it.get("failure_type") or "").lower() in {c.lower() for c in cats}
        or (
            "other_error" in cats
            and str(it.get("failure_type") or "").startswith("aiwake_")
        )
    ]
    # 排除 INIT
    selected = [it for it in selected if str(it.get("failure_type") or "").lower() != "init"]

    counter: Counter[str] = Counter()
    samples: dict[str, list[dict[str, Any]]] = {}
    for it in selected:
        blob = " ".join(
            str(it.get(k) or "")
            for k in ("error", "failure_type", "diagnosis", "tool", "fallback_hint")
        )
        for pat in _token_patterns(blob):
            counter[pat] += 1
            samples.setdefault(pat, [])
            if len(samples[pat]) < 3:
                samples[pat].append(
                    {
                        "tool": it.get("tool"),
                        "failure_type": it.get("failure_type"),
                        "error": str(it.get("error") or "")[:160],
                        "id": it.get("id"),
                    }
                )

    top = [
        {
            "pattern": name,
            "count": count,
            "samples": samples.get(name, []),
            "suggested_fix": _suggest_fix(name),
        }
        for name, count in counter.most_common(10)
        if count >= min_count
    ]

    result = {
        "status": "ok",
        "tool": "degradation_evidence_miner",
        "source": source,
        "created_at": _now_iso(),
        "scanned_triads": len(items),
        "matched_triads": len(selected),
        "categories": sorted(cats),
        "top_patterns": top,
        "triad_file": str(failure_triads_path()),
        "suggestion": (
            "对 count>=3 的 pattern 在 _classify_failure / metrics 分类中增加显式规则；"
            "每次新增后跑 pytest 并用 tool_failure_breakdown 复验。"
            if top
            else "样本不足：先让工具失败自动写入更多 triad，再聚类。"
        ),
    }

    # 追加聚类结果（append-only）
    cluster_record = {
        "event": "failure_cluster",
        "created_at": result["created_at"],
        "matched_triads": result["matched_triads"],
        "top_patterns": top[:5],
        "source": source,
    }
    path = _append_jsonl(failure_clusters_path(), cluster_record)
    result["cluster_path"] = str(path) if path else None
    return result


def _suggest_fix(pattern: str) -> str:
    mapping = {
        "timeout_keyword": "增加 timeout 子类与缩小请求范围的降级路径",
        "rate_limit_keyword": "统一 rate_limited 识别与备选工具切换",
        "permission_keyword": "明确 forbidden 分类并停止重试",
        "not_found_keyword": "not_found 止损，先 file_list/核对 URL",
        "command_missing_keyword": "shell 预检 missing_binary，换等价命令",
        "network_keyword": "network_error 分类 + 单次重试上限",
        "parse_keyword": "参数/JSON 解析失败单独分类，提示修参数",
        "validation_keyword": "preflight 参数内容校验增强",
        "empty_keyword": "empty_result 不当失败，调整查询条件",
        "ssrf_keyword": "私网限制归 forbidden，引导公开 HTTPS",
        "exception_keyword": "捕获异常类型写入 failure_type",
        "budget_keyword": "预算耗尽单独分类并切换 profile",
    }
    if pattern in mapping:
        return mapping[pattern]
    if pattern.startswith("token:"):
        return f"检查高频 token `{pattern[6:]}` 对应错误文本，补分类规则"
    return "基于样本补充显式 failure 分类与降级提示"


def maybe_force_mine_from_metrics(metrics: dict[str, Any] | None, *, min_other_error: int = 3) -> dict[str, Any] | None:
    """当 other_error 足够多时强制执行一次聚类，供 evolution 周期调用。"""
    metrics = metrics or {}
    breakdown = metrics.get("tool_failure_breakdown") or {}
    try:
        other = int(breakdown.get("other_error") or 0)
    except (TypeError, ValueError):
        other = 0
    if other < min_other_error:
        return None
    result = mine_other_error_patterns(
        limit=300,
        min_count=1,
        categories={"other_error"},
        source="evolution_forced_miner",
    )
    result["trigger"] = {
        "other_error_count": other,
        "tool_failure_count": metrics.get("tool_failure_count"),
        "tool_success_rate": metrics.get("tool_success_rate"),
    }
    return result


def triad_stats(limit: int = 500) -> dict[str, Any]:
    items = read_failure_triads(limit=limit)
    real = [it for it in items if str(it.get("failure_type") or "").lower() != "init"]
    by_cat: Counter[str] = Counter(str(it.get("failure_category") or "unknown") for it in real)
    return {
        "total": len(items),
        "real_count": len(real),
        "by_category": dict(by_cat),
        "path": str(failure_triads_path()),
    }
