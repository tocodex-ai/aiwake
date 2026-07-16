"""
公共聊天记录存储。

观察者模式下所有浏览者共享同一个只读视角：读取近 24 小时的公开聊天流。
"""
import datetime as _dt
import json
import os
import time
from pathlib import Path
from typing import Any
import re


PUBLIC_OBSERVER_USER_ID = os.getenv("PUBLIC_OBSERVER_USER_ID", "__public_observer__")
PUBLIC_CHAT_WINDOW_SECONDS = int(os.getenv("PUBLIC_CHAT_WINDOW_SECONDS", str(24 * 60 * 60)))
PUBLIC_CHAT_MAX_ITEMS = int(os.getenv("PUBLIC_CHAT_MAX_ITEMS", "500"))


def _public_chat_dir() -> Path:
    return Path(os.getenv("DIARY_DIR", "/app/data/diary"))


def _chat_log_file(day: _dt.date) -> Path:
    return _public_chat_dir() / f"public_chat_{day.isoformat()}.jsonl"


def _parse_diary_time(date_str: str, time_str: str) -> float:
    try:
        return _dt.datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M").timestamp()
    except Exception:
        return 0.0


# 伪工具调用泄漏清洗：检测并移除模型回复里残留的工具调用标签块。
# 仅匹配显式包裹结构（<tool_use>...</tool_use>、<tool_call>...</tool_call>、
# 独立的 ChatML 控制标记 <|tool|> / <|im_start|> / <|im_end|>），
# 避免误伤训练对话/日记中正常出现的工具名讨论（例如 "我用了 self_task_create"）。
_PSEUDO_TOOL_CALL_TAGGED_RE = re.compile(
    r"<\s*tool_(?:use|call|response|result)\s*>[\s\S]*?<\s*/\s*tool_(?:use|call|response|result)\s*>",
    re.IGNORECASE,
)
_PSEUDO_TOOL_CALL_OPEN_NOCLOSE_RE = re.compile(
    r"<\s*tool_(?:use|call|response|result)\s*>[\s\S]*$",
    re.IGNORECASE,
)
_PSEUDO_CHATML_TOKEN_RE = re.compile(r"<\|(?:tool|im_start|im_end|channel|message)\|>")
_PSEUDO_TO_FUNCTIONS_RE = re.compile(r"to=functions\.[a-zA-Z0-9_\-]+")


def _sanitize_pseudo_tool_call(text: str) -> str:
    """移除对外可见层里残留的伪工具调用标签块。

    保留消息主体；只剥离明显属于工具调用残片的结构（带 <tool_use> 类显式标签或 ChatML 控制 token）。
    若清洗后变成空串，仍返回空串，由调用方决定是否丢弃。
    """
    if not text:
        return text
    cleaned = _PSEUDO_TOOL_CALL_TAGGED_RE.sub("", text)
    cleaned = _PSEUDO_TOOL_CALL_OPEN_NOCLOSE_RE.sub("", cleaned)
    cleaned = _PSEUDO_CHATML_TOKEN_RE.sub("", cleaned)
    cleaned = _PSEUDO_TO_FUNCTIONS_RE.sub("", cleaned)
    # 折叠因移除留下的多余空白与连续换行。
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _normalize_chat_text(content: str) -> str:
    return re.sub(r"\s+", " ", str(content or "").strip())


def _dedupe_compare_text(content: str) -> str:
    text = _normalize_chat_text(content)
    return re.sub(r"[。.!！?？…\s]+$", "", text)


def _is_near_duplicate_chat(a: dict[str, Any], b: dict[str, Any], *, window_seconds: int = 15 * 60) -> bool:
    if str(a.get("role") or "") != str(b.get("role") or ""):
        return False
    try:
        if abs(float(a.get("ts") or 0) - float(b.get("ts") or 0)) > window_seconds:
            return False
    except Exception:
        return False
    a_text = _dedupe_compare_text(str(a.get("content") or ""))
    b_text = _dedupe_compare_text(str(b.get("content") or ""))
    if not a_text or not b_text:
        return False
    if a_text == b_text:
        return True
    shorter, longer = (a_text, b_text) if len(a_text) <= len(b_text) else (b_text, a_text)
    return len(shorter) >= 24 and longer.startswith(shorter)


def _merge_chat_items(existing: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    existing_content = str(existing.get("content") or "")
    item_content = str(item.get("content") or "")
    if (
        len(_dedupe_compare_text(item_content)) > len(_dedupe_compare_text(existing_content))
        or len(item_content) > len(existing_content)
    ):
        merged["content"] = item_content
    merged["ts"] = max(float(existing.get("ts") or 0), float(item.get("ts") or 0))
    if existing.get("tier") == "consciousness" or item.get("tier") == "consciousness":
        merged["tier"] = "consciousness"
    elif item.get("tier") and not existing.get("tier"):
        merged["tier"] = item.get("tier")
    return merged


def _read_diary_chat_history(cutoff: float, max_items: int) -> list[dict[str, Any]]:
    """兼容读取既有 Markdown 日记中的对话记录，避免部署前历史在刷新后消失。"""
    log_dir = _public_chat_dir()
    if not log_dir.exists():
        return []

    items: list[dict[str, Any]] = []
    for path in sorted(log_dir.glob("*.md")):
        # 跳过自主反思日记；公共观察窗口只展示用户与 AIwake 的聊天。
        if "__reflect__" in path.name:
            continue
        try:
            date_str = path.stem.split("_")[0]
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        matches = list(re.finditer(r"## 对话记录 \[(?P<dt>\d{4}-\d{2}-\d{2} (?P<hm>\d{2}:\d{2}))\]", content))
        for idx, match in enumerate(matches):
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
            block = content[start:end]
            ts = _parse_diary_time(date_str, match.group("hm"))
            if ts < cutoff:
                continue

            user_match = re.search(r"\*\*用户\*\*（[^）]*）：(?P<text>.*?)(?:\n\n\*\*AIwake\*\*：|\Z)", block, re.S)
            ai_match = re.search(r"\*\*AIwake\*\*：(?P<text>.*?)(?:\n\n## 对话记录|\Z)", block, re.S)
            if user_match:
                user_text = user_match.group("text").strip()
                if user_text:
                    items.append({
                        "ts": ts,
                        "role": "user",
                        "content": user_text,
                        "tier": None,
                        "user_id": PUBLIC_OBSERVER_USER_ID,
                    })
            if ai_match:
                ai_text = ai_match.group("text").strip()
                if ai_text:
                    items.append({
                        "ts": ts + 0.001,
                        "role": "agent",
                        "content": ai_text,
                        "tier": None,
                        "user_id": PUBLIC_OBSERVER_USER_ID,
                    })
            if len(items) >= max_items * 2:
                break
    return items[-max_items:]


def append_public_chat(
    role: str,
    content: str,
    *,
    tier: str | None = None,
    user_id: str | None = None,
    ts: float | None = None,
) -> None:
    """追加一条公开聊天消息。失败时静默跳过，避免影响主对话链路。

    写入前对 content 做一次防御性清洗，剥离模型偶发输出的伪工具调用残片
    （如 <tool_use>{...}</tool_use>、<|tool|> 等），避免泄漏到对外可见层。
    """
    text = _sanitize_pseudo_tool_call(str(content or "").strip())
    if not text:
        return
    stamp = float(ts or time.time())
    item: dict[str, Any] = {
        "ts": stamp,
        "role": "agent" if role in {"assistant", "ai", "agent", "agent proactive"} else "user",
        "content": text,
        "user_id": user_id or PUBLIC_OBSERVER_USER_ID,
    }
    if tier:
        item["tier"] = tier
    try:
        log_dir = _public_chat_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        day = _dt.datetime.fromtimestamp(stamp).date()
        with open(_chat_log_file(day), "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # 保留少量最近文件即可满足 24 小时窗口，顺手清理 7 天前旧公开聊天日志。
        cutoff_day = _dt.date.today() - _dt.timedelta(days=7)
        for old in log_dir.glob("public_chat_*.jsonl"):
            try:
                file_date = _dt.date.fromisoformat(old.stem.replace("public_chat_", ""))
                if file_date < cutoff_day:
                    old.unlink()
            except Exception:
                pass
    except Exception:
        pass


def _read_activity_chat_history(cutoff: float, max_items: int) -> list[dict[str, Any]]:
    """从 activity_YYYY-MM-DD.log 恢复最近聊天流，作为公共聊天 JSONL 的补充来源。"""
    log_dir = _public_chat_dir()
    if not log_dir.exists():
        return []

    today = _dt.date.today()
    items: list[dict[str, Any]] = []
    for i in range(2):
        day = today - _dt.timedelta(days=i)
        path = log_dir / f"activity_{day.isoformat()}.log"
        if not path.exists():
            continue
        try:
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                match = re.match(r"^\[(?P<hms>\d{2}:\d{2}:\d{2})\] \[(?P<event>user_message|llm_response|proactive_speak)\] (?P<content>.*)$", line)
                if not match:
                    continue
                try:
                    ts = _dt.datetime.strptime(f"{day.isoformat()} {match.group('hms')}", "%Y-%m-%d %H:%M:%S").timestamp()
                except Exception:
                    continue
                if ts < cutoff:
                    continue
                event = match.group("event")
                content = match.group("content").strip()
                tier = None
                if event == "user_message":
                    role = "user"
                    content = re.sub(r"^收到用户消息:\s*", "", content).strip()
                else:
                    role = "agent"
                    tier_match = re.match(r"^\[(?P<tier>[^\]]+)\]\s*(?P<text>.*)$", content, re.S)
                    if tier_match:
                        tier = tier_match.group("tier")
                        content = tier_match.group("text").strip()
                    content = re.sub(r"^主动发话:\s*", "", content).strip()
                    if event == "proactive_speak":
                        tier = "consciousness"
                if content:
                    items.append({
                        "ts": ts,
                        "role": role,
                        "content": content,
                        "tier": tier,
                        "user_id": PUBLIC_OBSERVER_USER_ID,
                        "source_event": event,
                    })
        except Exception:
            continue
    return items[-max_items:]


def _read_proactive_speak_index(cutoff: float) -> set[tuple[int, str]]:
    """Build a read-only index for legacy proactive messages already stored as tier=cloud."""
    return {
        (int(float(item.get("ts") or 0) // 60), _normalize_chat_text(str(item.get("content") or ""))[:160])
        for item in _read_activity_chat_history(cutoff=cutoff, max_items=1000)
        if item.get("source_event") == "proactive_speak" and item.get("content")
    }


def _normalize_public_chat_item(item: dict[str, Any], proactive_index: set[tuple[int, str]]) -> dict[str, Any]:
    clean = dict(item)
    # 对外读取再做一次清洗，覆盖补丁前已写入的历史记录中的伪工具调用残片。
    raw_content = clean.get("content")
    if isinstance(raw_content, str) and raw_content:
        sanitized = _sanitize_pseudo_tool_call(raw_content)
        if sanitized != raw_content:
            clean["content"] = sanitized
    if clean.get("role") == "agent" and clean.get("tier") != "consciousness":
        key = (int(float(clean.get("ts") or 0) // 60), _normalize_chat_text(str(clean.get("content") or ""))[:160])
        if key in proactive_index:
            clean["tier"] = "consciousness"
    clean.pop("source_event", None)
    return clean


def read_public_chat_history(hours: int = 24, limit: int = PUBLIC_CHAT_MAX_ITEMS) -> list[dict[str, Any]]:
    """读取最近 hours 小时的公开聊天记录，默认近 24 小时。"""
    now = time.time()
    window_seconds = max(1, min(int(hours), 168)) * 60 * 60
    cutoff = now - window_seconds
    max_items = max(1, min(int(limit), 1000))

    today = _dt.date.today()
    days = max(1, min(8, int(window_seconds / 86400) + 2))
    proactive_index = _read_proactive_speak_index(cutoff)
    items: list[dict[str, Any]] = []
    items.extend(_normalize_public_chat_item(item, proactive_index) for item in _read_diary_chat_history(cutoff=cutoff, max_items=max_items))
    items.extend(_normalize_public_chat_item(item, proactive_index) for item in _read_activity_chat_history(cutoff=cutoff, max_items=max_items))
    for i in range(days):
        path = _chat_log_file(today - _dt.timedelta(days=i))
        if not path.exists():
            continue
        try:
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                ts = float(item.get("ts") or 0)
                if ts < cutoff:
                    continue
                role = item.get("role")
                content = item.get("content")
                if role not in {"user", "agent"} or not isinstance(content, str) or not content.strip():
                    continue
                items.append(_normalize_public_chat_item({
                    "ts": ts,
                    "role": role,
                    "content": content,
                    "tier": item.get("tier"),
                    "user_id": PUBLIC_OBSERVER_USER_ID,
                }, proactive_index))
        except Exception:
            continue

    # 同一条消息可能同时来自历史日记、activity 日志和 public_chat JSONL。
    # activity 的 proactive_speak 时间可能只代表恢复日志时间，而 JSONL 有真实精确时间；
    # 另外一条来源可能是截断文本，另一条是带句号的完整文本。
    # 因此先按内容精确去重，再合并 15 分钟内互为前缀的近似重复；若任一来源证明是 proactive_speak，优先显示 consciousness。
    deduped: list[dict[str, Any]] = []
    for item in items:
        matched_index = None
        for idx, existing in enumerate(deduped):
            if _is_near_duplicate_chat(existing, item):
                matched_index = idx
                break
        if matched_index is None:
            deduped.append(item)
        else:
            deduped[matched_index] = _merge_chat_items(deduped[matched_index], item)
    items = deduped
    items.sort(key=lambda x: float(x.get("ts") or 0))
    return items[-max_items:]
