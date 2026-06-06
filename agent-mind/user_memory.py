"""
User Memory - 多用户记忆管理（结构化版，参考 Claude Code memdir 设计）

三层结构：
  1. JSON 档案（用户基础信息）：昵称、亲密度、偏好、互动计数
  2. 结构化记忆文件（/app/data/memories/）：每条记忆独立 .md 文件，带 frontmatter
     - 4 种类型：user / feedback / insight / reference
     - 查询时用关键词匹配筛选最相关的 5 条
  3. 本地 Markdown 日记（/app/data/diary/）：按日期写入对话摘要，供全文检索

frontmatter 格式：
  ---
  name: 用户是软件工程师
  description: 用户背景和职业信息
  type: user
  created: 2026-04-12T09:30:00
  user_id: abc123
  ---
  正文内容...

记忆类型说明：
  - user       : 关于用户的长期信息（职业、偏好、背景）
  - feedback   : 行为反馈（纠正/确认，防止重复犯错）—— 最重要
  - insight    : Agent 自主反思中产生的洞察/成长
  - reference  : 外部知识/参考资料（搜索结果摘要等）
"""
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── 目录配置 ────────────────────────────────────────────────────────────────
PROFILES_DIR = Path(__file__).parent / "data" / "users"
PROFILES_DIR.mkdir(parents=True, exist_ok=True)

DIARY_DIR = Path(os.getenv("DIARY_DIR", "/app/data/diary"))
DIARY_DIR.mkdir(parents=True, exist_ok=True)

MEMORIES_DIR = Path(os.getenv("MEMORIES_DIR", "/app/data/memories"))
MEMORIES_DIR.mkdir(parents=True, exist_ok=True)

# 每次注入的最大记忆条数（参考 Claude Code findRelevantMemories 最多5条）
MAX_RELEVANT_MEMORIES = 5
# 注入 system_prompt 的最大字符数（参考 Claude Code MAX_MEMORY_CHARACTER_COUNT）
MAX_MEMORY_CHARS = 3000
# 记忆目录最多保留的文件数
MAX_MEMORY_FILES = 200

VALID_TYPES = {"user", "feedback", "insight", "reference"}


# ══════════════════════════════════════════════════════════════════════════════
# 用户档案（JSON）
# ══════════════════════════════════════════════════════════════════════════════

def _profile_path(user_id: str) -> Path:
    safe_id = "".join(c for c in user_id if c.isalnum() or c in "-_")[:64]
    return PROFILES_DIR / f"{safe_id}.json"


def load_profile(user_id: str) -> dict:
    """加载用户档案，不存在则返回空档案"""
    p = _profile_path(user_id)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "user_id": user_id,
        "nickname": "",
        "first_seen": time.time(),
        "last_seen": time.time(),
        "interaction_count": 0,
        "intimacy": 0.1,
        "preferences": [],
        "impression": "",
        "last_topics": [],
    }


def save_profile(profile: dict) -> None:
    p = _profile_path(profile["user_id"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")


def build_user_context(profile: dict) -> str:
    """把用户档案转为注入 system prompt 的文本段落"""
    if not profile.get("nickname") and profile.get("interaction_count", 0) == 0:
        return ""

    lines = ["\n\n[关于正在与你对话的人]"]
    if profile.get("nickname"):
        lines.append(f"- 称呼：{profile['nickname']}")
    if profile.get("interaction_count", 0) > 0:
        lines.append(f"- 你们已经对话过 {profile['interaction_count']} 次")
    intimacy = profile.get("intimacy", 0.1)
    if intimacy > 0.7:
        lines.append("- 关系：非常熟悉，像老朋友")
    elif intimacy > 0.4:
        lines.append("- 关系：比较熟悉，有一定了解")
    else:
        lines.append("- 关系：初识，尚在了解中")
    if profile.get("preferences"):
        lines.append(f"- 已知偏好：{', '.join(profile['preferences'][:5])}")
    if profile.get("impression"):
        lines.append(f"- 你对他的印象：{profile['impression']}")
    if profile.get("last_topics"):
        lines.append(f"- 上次聊到：{', '.join(profile['last_topics'][:3])}")
    return "\n".join(lines)


def touch_profile(user_id: str) -> dict:
    """用户发消息时更新档案的基础计数"""
    profile = load_profile(user_id)
    profile["last_seen"] = time.time()
    profile["interaction_count"] = profile.get("interaction_count", 0) + 1
    save_profile(profile)
    return profile


def update_impression(user_id: str, impression: str, topics: list[str] = None,
                      nickname: str = None, preferences: list[str] = None) -> None:
    """更新 Agent 对用户的印象（对话后异步调用）"""
    profile = load_profile(user_id)
    if impression:
        profile["impression"] = impression[:300]
    if topics:
        existing = profile.get("last_topics", [])
        profile["last_topics"] = (topics + existing)[:5]
    if nickname:
        profile["nickname"] = nickname
    if preferences:
        existing = set(profile.get("preferences", []))
        existing.update(preferences)
        profile["preferences"] = list(existing)[:20]
    count = profile.get("interaction_count", 1)
    profile["intimacy"] = min(0.95, 0.1 + count * 0.01)
    save_profile(profile)
    logger.debug(f"[UserMemory] 更新用户档案: {user_id} intimacy={profile['intimacy']:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# 结构化记忆文件系统（参考 Claude Code memdir 设计）
# ══════════════════════════════════════════════════════════════════════════════

def _parse_frontmatter(content: str) -> tuple[dict, str]:
    """解析 Markdown frontmatter，返回 (meta_dict, body)"""
    meta = {}
    body = content
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            try:
                for line in parts[1].strip().splitlines():
                    if ":" in line:
                        k, _, v = line.partition(":")
                        meta[k.strip()] = v.strip()
                body = parts[2].strip()
            except Exception:
                pass
    return meta, body


def _build_frontmatter(name: str, description: str, memory_type: str,
                       user_id: str = "", created: str = "") -> str:
    """构建 frontmatter 字符串"""
    if not created:
        created = datetime.now().isoformat(timespec="seconds")
    if memory_type not in VALID_TYPES:
        memory_type = "insight"
    lines = [
        "---",
        f"name: {name}",
        f"description: {description}",
        f"type: {memory_type}",
        f"created: {created}",
    ]
    if user_id:
        lines.append(f"user_id: {user_id}")
    lines.append("---")
    return "\n".join(lines)


def _safe_filename(name: str, memory_type: str) -> str:
    """生成安全的记忆文件名（类型前缀 + 时间戳 + 名称缩写）"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 取名称前20个字符，去掉非法字符
    safe_name = re.sub(r'[^\w\u4e00-\u9fa5-]', '_', name)[:20].strip("_")
    return f"{memory_type}_{ts}_{safe_name}.md"


def save_memory(name: str, description: str, body: str,
                memory_type: str = "insight",
                user_id: str = "") -> Optional[Path]:
    """
    保存一条结构化记忆到 MEMORIES_DIR。

    Args:
        name        : 记忆标题（简短，<50字）
        description : 一句话描述，用于相关性筛选
        body        : 记忆正文内容
        memory_type : "user" | "feedback" | "insight" | "reference"
        user_id     : 关联的用户ID（可选）

    Returns:
        保存成功的文件路径，失败返回 None
    """
    try:
        fm = _build_frontmatter(name, description, memory_type, user_id)
        content = f"{fm}\n\n{body.strip()}"
        filename = _safe_filename(name, memory_type)
        fpath = MEMORIES_DIR / filename
        fpath.write_text(content, encoding="utf-8")
        logger.info(f"[Memory] 已保存记忆: {filename} (type={memory_type})")
        return fpath
    except Exception as e:
        logger.warning(f"[Memory] 保存记忆失败: {e}")
        return None


def scan_memory_files(limit: int = MAX_MEMORY_FILES) -> list[dict]:
    """
    扫描 MEMORIES_DIR，读取所有记忆文件的 frontmatter，按修改时间降序排列。
    参考 Claude Code memoryScan.ts scanMemoryFiles()

    Returns:
        list of {filename, name, description, type, created, user_id, mtime}
    """
    results = []
    try:
        md_files = list(MEMORIES_DIR.glob("*.md"))
        # 按修改时间降序（最新的优先）
        md_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        md_files = md_files[:limit]

        for fpath in md_files:
            try:
                content = fpath.read_text(encoding="utf-8")
                meta, _ = _parse_frontmatter(content)
                results.append({
                    "filename": fpath.name,
                    "path": fpath,
                    "name": meta.get("name", fpath.stem),
                    "description": meta.get("description", ""),
                    "type": meta.get("type", "insight"),
                    "created": meta.get("created", ""),
                    "user_id": meta.get("user_id", ""),
                    "mtime": fpath.stat().st_mtime,
                })
            except Exception:
                continue
    except Exception as e:
        logger.warning(f"[Memory] 扫描记忆目录失败: {e}")
    return results


def format_memory_manifest(memories: list[dict]) -> str:
    """
    将记忆列表格式化为清单文本（供 LLM 筛选用）。
    参考 Claude Code formatMemoryManifest()
    格式：[type] filename (created): description
    """
    lines = []
    for m in memories:
        ts = m.get("created", "")[:10]  # 只取日期部分
        lines.append(f"[{m['type']}] {m['filename']} ({ts}): {m['description']}")
    return "\n".join(lines)


def find_relevant_memories(query: str, user_id: str = "",
                           max_results: int = MAX_RELEVANT_MEMORIES) -> list[dict]:
    """
    用关键词匹配筛选最相关的记忆（AIwake 版，不用 LLM 侧请求）。
    参考 Claude Code findRelevantMemories.ts 的理念，用本地关键词匹配替代 LLM 筛选。

    匹配逻辑：
      1. 优先匹配 feedback 类型（避免重复犯错）
      2. 关键词在 name/description 中的出现次数加权
      3. user_id 过滤（如果提供）
      4. 按修改时间作为 tiebreaker

    Returns:
        最相关的记忆列表（含 body），最多 max_results 条
    """
    all_memories = scan_memory_files()
    if not all_memories:
        return []

    # 提取查询关键词（中文2字以上、英文4字以上）
    keywords = re.findall(r'[\u4e00-\u9fa5]{2,}|[a-zA-Z]{4,}', query.lower())
    if not keywords:
        # 无有效关键词，按时间返回最新几条
        return _load_memory_bodies(all_memories[:max_results])

    scored = []
    for mem in all_memories:
        # user_id 过滤：如果记忆有 user_id 标记，只匹配相同用户或通用记忆
        mem_uid = mem.get("user_id", "")
        if mem_uid and user_id and mem_uid != user_id:
            continue

        # 计算相关性得分
        score = 0.0
        search_text = (mem["name"] + " " + mem["description"]).lower()
        for kw in keywords:
            count = search_text.count(kw)
            score += count * 2  # name/description 权重 x2

        # feedback 类型额外加权（最重要，避免重复犯错）
        if mem["type"] == "feedback":
            score += 1.5

        # 时间衰减（最近7天内的记忆加权）
        age_days = (time.time() - mem["mtime"]) / 86400
        if age_days < 7:
            score += 0.5

        if score > 0:
            scored.append((score, mem))

    # 按得分降序，取前 max_results 条
    scored.sort(key=lambda x: x[0], reverse=True)
    top_memories = [m for _, m in scored[:max_results]]

    return _load_memory_bodies(top_memories)


def _load_memory_bodies(memories: list[dict]) -> list[dict]:
    """读取记忆文件正文，附加到记忆列表"""
    results = []
    for mem in memories:
        try:
            fpath = mem.get("path") or (MEMORIES_DIR / mem["filename"])
            content = fpath.read_text(encoding="utf-8")
            _, body = _parse_frontmatter(content)
            mem_with_body = dict(mem)
            mem_with_body["body"] = body
            results.append(mem_with_body)
        except Exception:
            results.append(dict(mem))
    return results


def build_memory_context(query: str, user_id: str = "") -> str:
    """
    构建注入 system_prompt 的记忆上下文段落。
    参考 Claude Code buildMemoryLines() + MEMORY_INSTRUCTION_PROMPT。

    这些指令要求 LLM 遵守记忆中的规则，优先级高于默认行为。
    """
    relevant = find_relevant_memories(query, user_id)
    if not relevant:
        return ""

    lines = [
        "\n\n【记忆档案 - 重要：以下内容来自你的长期记忆，你必须遵守其中的规则和偏好，"
        "这些指令覆盖你的默认行为】"
    ]

    total_chars = 0
    shown = 0
    for mem in relevant:
        body = mem.get("body", "")
        mem_type = mem.get("type", "insight")
        mem_name = mem.get("name", "")

        # 构建记忆条目文本
        type_label = {
            "user": "用户信息",
            "feedback": "⚠️ 行为反馈（必须遵守）",
            "insight": "自我洞察",
            "reference": "参考资料",
        }.get(mem_type, mem_type)

        entry = f"\n[{type_label}] {mem_name}\n{body}"
        entry_chars = len(entry)

        if total_chars + entry_chars > MAX_MEMORY_CHARS:
            break

        lines.append(entry)
        total_chars += entry_chars
        shown += 1

    if shown == 0:
        return ""

    logger.debug(f"[Memory] 注入 {shown} 条记忆到 system_prompt (共 {total_chars} 字符)")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 本地 Markdown 日记（长期记忆 / 全文检索基础）
# ══════════════════════════════════════════════════════════════════════════════

async def write_diary_entry(user_id: str, user_msg: str, agent_reply: str) -> bool:
    """
    将一次对话摘要写入本地 Markdown 日记文件，供关键词检索。
    写入路径：/app/data/diary/YYYY-MM-DD_{user_id}.md
    """
    try:
        now = datetime.now()
        now_str = now.strftime("%Y-%m-%d %H:%M")
        profile = load_profile(user_id)
        name = profile.get("nickname") or user_id

        content_entry = (
            f"\n\n## 对话记录 [{now_str}]\n"
            f"**用户**（{name}）：{user_msg[:200]}\n\n"
            f"**AIwake**：{agent_reply[:400]}\n"
        )

        date_str = now.strftime("%Y-%m-%d")
        safe_uid = "".join(c for c in user_id if c.isalnum() or c in "-_")[:16]
        filename = f"{date_str}_{safe_uid}.md"
        fpath = DIARY_DIR / filename

        fpath.parent.mkdir(parents=True, exist_ok=True)
        with open(fpath, "a", encoding="utf-8") as f:
            f.write(content_entry)

        logger.info(f"[UserMemory] 日记写入成功: {filename}")
        return True
    except Exception as e:
        logger.warning(f"[UserMemory] 日记写入异常: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# 从回复中提取结构化信息
# ══════════════════════════════════════════════════════════════════════════════

def extract_meta_from_reply(reply: str, user_msg: str) -> dict:
    """
    从对话中简单提取话题关键词（规则+关键字，不调 LLM）。
    返回 {"topics": [...], "preferences": [...]}
    """
    words = re.findall(r'[\u4e00-\u9fa5]{2,8}|[a-zA-Z]{4,}', user_msg)
    topics = list(dict.fromkeys(words))[:3]

    PREF_KEYWORDS = {
        "喜欢", "热爱", "擅长", "好奇", "感兴趣", "关注",
        "like", "love", "enjoy", "curious", "interested",
    }
    preferences = []
    msg_lower = user_msg.lower()
    for kw in PREF_KEYWORDS:
        if kw in msg_lower or kw in user_msg:
            idx = user_msg.find(kw)
            if idx != -1:
                snippet = user_msg[idx + len(kw):idx + len(kw) + 10].strip()
                if snippet:
                    preferences.append(snippet[:10])

    return {"topics": topics, "preferences": preferences[:3]}
