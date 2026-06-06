"""
Tool Router - 工具路由器（原生实现，无任何第三方代理依赖）
所有工具直接实现：
  - web_search   → 多后备策略：DDG JSON API → DDG HTML → Bing HTML
  - web_fetch    → 直接 httpx GET
  - url_fetch    → 直接 httpx GET（同 web_fetch）
  - weather      → wttr.in 免费天气 API
  - arxiv_search → Arxiv 官方 API（免费）
  - daily_note_read/write → 本地文件 /app/data/diary/
  - knowledge_search      → 关键词检索本地日记文件（tag_memo 为别名）
  - file_read/write → 直接本地文件操作
  - shell_exec   → asyncio.create_subprocess_shell
  - ssh_exec     → paramiko（可选依赖）
"""
import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx

from autonomy_config import load_autonomy_config
from experiment_store import get_experiment_store
from evolution.memory import append_growth_log
from safety_guard import (
    classify_command_risk,
    has_self_modification_audit,
    is_memory_path,
    is_self_project_code_path,
    is_sensitive_path,
    redact_secrets,
)

logger = logging.getLogger(__name__)

# 本地日记目录
DIARY_DIR = Path(os.getenv("DIARY_DIR", "/app/data/diary"))
DIARY_DIR.mkdir(parents=True, exist_ok=True)

# 可直接执行的工具（只读 + 写入/执行，无需审批）
ALLOWED_TOOLS: dict[str, float] = {
    # 只读工具
    "web_search": 0.1,
    "url_fetch": 0.2,
    "web_fetch": 0.2,
    "weather": 0.05,
    "file_read": 0.15,
    "knowledge_search": 0.1,
    "daily_note_read": 0.1,
    "schedule_read": 0.05,
    "arxiv_search": 0.1,
    "tag_memo": 0.1,
    # 写入工具
    "daily_note_write": 0.2,
    "file_write": 0.5,
    "shell_exec": 0.6,
    "ssh_exec": 0.7,
    "file_list": 0.12,
    "file_append": 0.35,
    "patch_write": 0.4,
    "memory_write": 0.25,
    "self_code_write": 0.75,
    "self_task_create": 0.2,
    "self_task_update": 0.2,
    "self_artifact_create": 0.25,
}

# 并发安全的只读工具（参考 Claude Code toolOrchestration.ts isConcurrencySafe）
READONLY_TOOLS: frozenset[str] = frozenset({
    "web_search",
    "url_fetch",
    "web_fetch",
    "weather",
    "file_read",
    "knowledge_search",
    "daily_note_read",
    "schedule_read",
    "arxiv_search",
    "tag_memo",
    "file_list",
})

# 并发批执行的最大并发数（参考 Claude Code runToolsConcurrently 最多10个）
MAX_CONCURRENT_TOOLS = 10

# 高风险工具（仍允许执行，但记录警告日志）
HIGH_RISK_TOOLS: dict[str, float] = {
    "file_delete": 0.85,
    "package_install": 0.9,
}


class ToolRouter:
    def __init__(self):
        self.store = get_experiment_store()

    def _classify_failure(self, tool_name: str, result: dict) -> dict:
        """为工具失败补充稳定的分类与降级路径，帮助下一轮反思可直接行动。"""
        status = str(result.get("status") or "").lower()
        error_text = str(result.get("error") or result.get("result") or "")
        if not result.get("error") and status not in {"error", "degraded"}:
            return result

        combined = f"{tool_name} {status} {error_text}".lower()
        if tool_name in {"web_search", "web_fetch", "url_fetch", "weather", "arxiv_search"}:
            failure_type = "external_service_failure"
            fallback_hint = "缩短请求范围或重试；若仍失败，改用等价网络工具/公开接口；记录 HTTP 状态、异常类型、URL 与耗时。"
        elif "permission" in combined or "allowed" in combined or "配置" in error_text or "拦截" in error_text:
            failure_type = "aiwake_tool_policy_failure"
            fallback_hint = "不要绕过安全策略；改用允许的只读工具或先生成 self_task/artifact 记录证据与边界。"
        elif "timeout" in combined or "超时" in error_text:
            failure_type = "aiwake_tool_timeout"
            fallback_hint = "降低 limit/缩短 hours/拆分任务后重试一次；仍失败时改用状态接口或本地日志交叉验证。"
        elif tool_name in {"file_read", "file_list", "file_write", "file_append", "self_code_write", "shell_exec", "ssh_exec"}:
            failure_type = "aiwake_internal_tool_failure"
            fallback_hint = "先确认参数、路径和安全边界；对文件/API 路径误用改用推荐工具；对命令失败保留 stderr/returncode。"
        else:
            failure_type = "aiwake_tool_failure"
            fallback_hint = "记录 error/status/工具名/参数摘要，选择等价工具或创建学习 artifact 后再继续。"

        enriched = dict(result)
        enriched.setdefault("status", "error")
        enriched.setdefault("tool", tool_name)
        enriched.setdefault("failure_type", failure_type)
        enriched.setdefault("error_type", type(result.get("error")).__name__ if result.get("error") is not None else status or "degraded")
        enriched.setdefault("fallback_hint", fallback_hint)
        enriched.setdefault("recommended_next_actions", [
            "复述可验证字段：tool/status/error/http_status/diagnosis/returncode。",
            "按 failure_type 选择降级路径，不把一次失败误判为整体不可用。",
            "降级后再次读取状态或日志，确认是否完成闭环。",
        ])
        return enriched

    def _check_permission(self, tool_name: str) -> tuple[bool, float]:
        """返回 (是否允许直接执行, 风险分)。实验模式只放开受保护工具。"""
        config = load_autonomy_config()
        if tool_name in {"shell_exec", "ssh_exec"} and not config.experiment_allow_shell:
            return False, 0.9
        if tool_name in {"file_write", "file_append"} and not config.experiment_allow_file_write:
            return False, 0.8
        if tool_name == "patch_write" and not config.experiment_allow_patch_generation:
            return False, 0.7
        if tool_name in {"self_task_create", "self_task_update", "self_artifact_create"} and not config.experiment_allow_self_tasks:
            return False, 0.5
        if tool_name in {"web_search", "url_fetch", "web_fetch", "weather", "arxiv_search"} and not config.experiment_allow_network:
            return False, 0.6
        if tool_name in ALLOWED_TOOLS:
            return True, ALLOWED_TOOLS[tool_name]
        if tool_name in HIGH_RISK_TOOLS:
            logger.warning(f"[ToolRouter] 高风险工具请求: {tool_name}")
            return config.experiment_allow_destructive, HIGH_RISK_TOOLS[tool_name]
        return False, 0.5

    def get_tool_descriptions(self) -> str:
        """返回可用工具的描述，供注入系统提示（Function Calling 模式下不再需要在 prompt 里描述工具格式）"""
        lines = [
            "## 可用工具（通过 Function Calling 自动调用，无需手动格式化）",
            "",
            "### 信息获取与搜索",
            "- web_search(query) — 网络搜索，获取最新信息",
            "- web_fetch(url) — 直接抓取网页完整内容",
            "- weather(city) — 查询实时天气",
            "- arxiv_search(query) — 搜索学术论文",
            "",
            "### 知识与记忆",
            "- knowledge_search(query) — 检索本地知识库",
            "- daily_note_read(date) — 读取日记/笔记",
            "- daily_note_write(content) — 写入成长日记",
            "",
            "### 文件与系统",
            "- file_list(path) — 列出目录文件",
            "- file_read(path) — 读取文件（敏感路径拦截）",
            "- file_write(path, content) — 写入文件（敏感路径/记忆路径拦截；自改代码请用 self_code_write）",
            "- file_append(path, content) — 追加写入文件（敏感路径/记忆路径拦截）",
            "- patch_write(path, content) — 写入实验 patch 草稿",
            "- memory_write(content) — 写入实验记忆事件",
            "- self_code_write(path, content, audit) — 在记录成长/审计日志后改动 AIwake 自身代码；audit 必含 purpose/change_summary/files/validation_result/risk_note",
            "- self_task_create(title, goal) — 创建实验自任务",
            "- self_task_update(task_id, status, note) — 更新实验自任务",
            "- self_artifact_create(task_id, title, content, kind, note) — 为自任务生成学习 artifact，建立任务-产物可验证闭环",
            "- shell_exec(command) — 执行 shell 命令（部署/破坏/密钥读取拦截）",
        ]
        return "\n".join(lines)

    def get_openai_tools_schema(self) -> list[dict]:
        """返回 OpenAI Function Calling 格式的工具定义数组，供 API 请求的 tools 参数使用"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "搜索互联网获取最新信息、新闻、知识等。当需要实时信息或不确定答案时使用。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "搜索关键词，支持中英文"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "web_fetch",
                    "description": "直接抓取某个 URL 的网页内容，适合读取文档、新闻、论文原文等。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "要抓取的网页 URL"}
                        },
                        "required": ["url"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "查询指定城市的实时天气状况。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "城市名称，支持中英文，如 Beijing、上海"}
                        },
                        "required": ["city"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "arxiv_search",
                    "description": "搜索 Arxiv 学术论文库，用于自主学习最新研究成果。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "论文搜索关键词，建议使用英文"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "knowledge_search",
                    "description": "检索本地知识库和历史日记，查找过去的对话、学习成果和感悟记录。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "检索关键词"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "daily_note_read",
                    "description": "读取指定日期的日记或笔记内容。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "date": {"type": "string", "description": "日期，格式 YYYY-MM-DD 或 'today'，默认 today"}
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "daily_note_write",
                    "description": "向日记写入内容，记录反思、学习成果、新的感受锚点等。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string", "description": "要写入日记的内容"}
                        },
                        "required": ["content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "self_task_create",
                    "description": "创建新的自我学习任务卡。适合在 open_task_count=0 且 next_plan 要求选择新小问题时使用，把空任务队列转化为可验证任务闭环。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "任务标题，聚焦一个最小可验证学习问题"},
                            "goal": {"type": "string", "description": "任务目标、证据来源、done_when 与安全边界摘要"}
                        },
                        "required": ["title", "goal"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "self_task_update",
                    "description": "更新已有自我学习任务的状态并追加证据摘要。适合在 done_when 已满足时把任务从 open 标记为 closed/done，避免重复生成 artifact。若只是观察到证据不足，可使用 evidence_observed，不要声称完成。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_id": {"type": "string", "description": "要更新的任务 id"},
                            "status": {"type": "string", "description": "新状态，如 evidence_observed、closed、done、failed、cancelled"},
                            "note": {"type": "string", "description": "状态更新证据摘要与安全边界说明"}
                        },
                        "required": ["task_id", "status", "note"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "self_artifact_create",
                    "description": "为已有自我学习任务生成学习 artifact。适合 latest_task 仍 open 且 latest_artifact 未匹配当前任务时使用，用可验证产物补齐任务闭环；不会关闭任务。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_id": {"type": "string", "description": "要绑定的自任务 id，必须来自 /experiment/status 的 latest_task.id"},
                            "title": {"type": "string", "description": "artifact 标题，聚焦一个最小学习产物"},
                            "content": {"type": "string", "description": "artifact 正文，必须包含证据、学习规则、下一步验证路径与安全边界"},
                            "kind": {"type": "string", "description": "artifact 类型，默认 learning_artifact"},
                            "note": {"type": "string", "description": "补充说明：不要声称部署完成或任务已关闭"}
                        },
                        "required": ["task_id", "title", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "file_read",
                    "description": "读取本地文件内容。不要用它读取 /health、/activity_logs、/public_chat、/experiment/status、/evolution/status 等 HTTP API 路径；这类路径必须通过接口/HTTP 证据获取。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "文件路径"}
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "file_write",
                    "description": "写入或创建本地文件。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "文件路径"},
                            "content": {"type": "string", "description": "要写入的内容"}
                        },
                        "required": ["path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "self_code_write",
                    "description": "在满足成长/审计日志要求时改动 AIwake 自身项目代码。禁止写入敏感路径、记忆目录或破坏性删除；audit 必含 purpose、change_summary、files、validation_result、risk_note。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "AIwake 项目内的代码或运行规则文件路径"},
                            "content": {"type": "string", "description": "完整文件内容或目标内容"},
                            "audit": {
                                "type": "object",
                                "description": "成长/审计记录",
                                "properties": {
                                    "purpose": {"type": "string"},
                                    "change_summary": {"type": "string"},
                                    "files": {"type": "array", "items": {"type": "string"}},
                                    "validation_result": {"type": "string"},
                                    "risk_note": {"type": "string"}
                                },
                                "required": ["purpose", "change_summary", "files", "validation_result", "risk_note"]
                            }
                        },
                        "required": ["path", "content", "audit"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "shell_exec",
                    "description": "在服务器上执行 shell 命令，用于系统操作、数据处理等。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "要执行的 shell 命令"}
                        },
                        "required": ["command"]
                    }
                }
            },
        ]

    async def call(self, tool_name: str, params: dict) -> dict:
        """执行工具调用（权限校验 + 直接实现）"""
        allowed, risk = self._check_permission(tool_name)
        if not allowed:
            return self._classify_failure(tool_name, {"error": f"工具 '{tool_name}' 未被当前实验配置允许", "tool": tool_name, "risk": risk})

        # 路由到各工具直接实现
        try:
            if tool_name in ("web_fetch", "url_fetch"):
                result = await self._web_fetch(params.get("url", ""))
            elif tool_name == "web_search":
                result = await self._web_search(params.get("query", ""))
            elif tool_name == "weather":
                result = await self._weather(params.get("city", "Shanghai"))
            elif tool_name == "arxiv_search":
                result = await self._arxiv_search(params.get("query", ""))
            elif tool_name == "daily_note_read":
                result = self._daily_note_read(params.get("date", "today"))
            elif tool_name == "daily_note_write":
                result = self._daily_note_write(params.get("content", ""))
            elif tool_name in ("knowledge_search", "tag_memo"):
                result = self._knowledge_search(params.get("query", "") or params.get("content", ""))
            elif tool_name == "file_list":
                result = self._file_list(params.get("path", "."))
            elif tool_name == "file_read":
                result = self._file_read(params.get("path", ""))
            elif tool_name == "file_write":
                result = self._file_write(params.get("path", ""), params.get("content", ""))
            elif tool_name == "file_append":
                result = self._file_append(params.get("path", ""), params.get("content", ""))
            elif tool_name == "patch_write":
                result = self._patch_write(params.get("path", ""), params.get("content", ""))
            elif tool_name == "memory_write":
                result = self._memory_write(params.get("content", ""))
            elif tool_name == "self_code_write":
                result = self._self_code_write(params.get("path", ""), params.get("content", ""), params.get("audit"))
            elif tool_name == "self_task_create":
                result = self._self_task_create(params.get("title", ""), params.get("goal", ""))
            elif tool_name == "self_task_update":
                result = self._self_task_update(params.get("task_id", ""), params.get("status", ""), params.get("note", ""))
            elif tool_name == "self_artifact_create":
                result = self._self_artifact_create(
                    params.get("task_id", ""),
                    params.get("title", ""),
                    params.get("content", ""),
                    params.get("kind", "learning_artifact"),
                    params.get("note", ""),
                )
            elif tool_name == "shell_exec":
                result = await self._shell_exec(params.get("command", ""))
            elif tool_name == "ssh_exec":
                result = await self._ssh_exec(
                    params.get("command", ""),
                    params.get("host", ""),
                    params.get("user", ""),
                    params.get("password", ""),
                    params.get("port", 22),
                )
            elif tool_name == "schedule_read":
                result = self._schedule_read()
            else:
                result = {"error": f"工具 '{tool_name}' 未实现"}
            return self._classify_failure(tool_name, result)
        except Exception as e:
            logger.error(f"[ToolRouter] 工具 '{tool_name}' 执行异常: {e}")
            return self._classify_failure(tool_name, {"error": str(e), "tool": tool_name})

    async def call_batch(self, tool_calls: list[dict]) -> list[dict]:
        """
        批量执行工具调用，只读工具并发执行，写入工具串行执行。
        参考 Claude Code toolOrchestration.ts partitionToolCalls() + runToolsConcurrently()

        Args:
            tool_calls: [{"tool": "web_search", "params": {"query": "..."}, "id": "optional"}, ...]

        Returns:
            按输入顺序对应的结果列表 [{"tool": ..., "result": ..., "id": ...}, ...]
        """
        if not tool_calls:
            return []

        # 分批：只读工具一批（并发），写入/执行工具各自单独串行
        readonly_batch: list[tuple[int, dict]] = []   # (原始索引, call)
        write_batch: list[tuple[int, dict]] = []

        for idx, call in enumerate(tool_calls):
            tool_name = call.get("tool", "")
            if tool_name in READONLY_TOOLS:
                readonly_batch.append((idx, call))
            else:
                write_batch.append((idx, call))

        results: list[Optional[dict]] = [None] * len(tool_calls)

        # ── 并发执行只读工具 ──────────────────────────────────────
        if readonly_batch:
            # 限制最大并发数
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_TOOLS)

            async def _exec_readonly(idx: int, call: dict) -> tuple[int, dict]:
                async with semaphore:
                    tool_name = call.get("tool", "")
                    params = call.get("params", {})
                    result = await self.call(tool_name, params)
                    return idx, {
                        "tool": tool_name,
                        "result": result,
                        "id": call.get("id", ""),
                        "concurrent": True,
                    }

            concurrent_results = await asyncio.gather(
                *[_exec_readonly(idx, call) for idx, call in readonly_batch],
                return_exceptions=True,
            )

            for item in concurrent_results:
                if isinstance(item, Exception):
                    logger.error(f"[ToolRouter] 并发工具执行异常: {item}")
                    continue
                idx, res = item
                results[idx] = res

            logger.info(f"[ToolRouter] 并发执行 {len(readonly_batch)} 个只读工具完成")

        # ── 串行执行写入/执行工具 ─────────────────────────────────
        for idx, call in write_batch:
            tool_name = call.get("tool", "")
            params = call.get("params", {})
            result = await self.call(tool_name, params)
            results[idx] = {
                "tool": tool_name,
                "result": result,
                "id": call.get("id", ""),
                "concurrent": False,
            }

        # 过滤掉 None（理论上不应该出现）
        return [r for r in results if r is not None]

    # ─────────────────────────────────────────────────────────────
    # Web 搜索：多后备策略（DDG JSON → DDG HTML → Bing HTML）
    # ─────────────────────────────────────────────────────────────
    async def _web_search(self, query: str) -> dict:
        """网络搜索，多后备策略：DDG JSON API → DDG HTML → Bing HTML"""
        if not query:
            return {"error": "query 参数不能为空"}

        # UA 轮换池（防封）
        _UA_POOL = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
        ]
        import random, time as _time
        ua = _UA_POOL[int(_time.time()) % len(_UA_POOL)]

        def _clean_html_text(value: str) -> str:
            import html as _html
            value = re.sub(r"<[^>]+>", " ", value or "")
            value = _html.unescape(value)
            return re.sub(r"\s+", " ", value).strip()

        def _parse_results(items: list, max_n: int = 6) -> str:
            parts = []
            seen_urls = set()
            for item in items:
                title = _clean_html_text(item.get("title", ""))
                url = _clean_html_text(item.get("url", item.get("href", "")))
                snippet = _clean_html_text(item.get("snippet", item.get("body", "")))
                if not title or not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                parts.append(f"[{len(parts)+1}] {title}\n    URL: {url}\n    摘要: {snippet}")
                if len(parts) >= max_n:
                    break
            return f"搜索「{query}」结果：\n\n" + "\n\n".join(parts)

        # ── 后备1：DuckDuckGo JSON API（无 HTML 解析）───────────────
        try:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                resp = await client.get(
                    "https://api.duckduckgo.com/",
                    params={"q": query, "format": "json", "no_html": "1",
                            "skip_disambig": "1", "kl": "cn-zh"},
                    headers={"User-Agent": ua, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"},
                )
                resp.raise_for_status()
                data = resp.json()

            # DDG JSON 返回 RelatedTopics（常规搜索）
            items = []
            for topic in data.get("RelatedTopics", []):
                if "Text" in topic and "FirstURL" in topic:
                    items.append({
                        "title": topic["Text"][:80],
                        "url": topic["FirstURL"],
                        "snippet": topic["Text"],
                    })
            # 也检查 Results 字段
            for r in data.get("Results", []):
                items.append({
                    "title": r.get("Text", "")[:80],
                    "url": r.get("FirstURL", ""),
                    "snippet": r.get("Text", ""),
                })
            if items:
                result_text = _parse_results(items)
                logger.info(f"[ToolRouter] web_search(DDG JSON) 返回 {len(items)} 条")
                return {"status": "ok", "tool": "web_search", "query": query, "result": result_text}
            logger.debug("[ToolRouter] DDG JSON 无结果，尝试 DDG HTML")
        except Exception as e:
            logger.warning(f"[ToolRouter] DDG JSON 失败: {e}，降级到 DDG HTML")

        # ── 后备2：DuckDuckGo Lite HTML ──────────────────────────────
        try:
            async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
                resp = await client.get(
                    "https://html.duckduckgo.com/lite/",
                    params={"q": query, "kl": "cn-zh"},
                    headers={"User-Agent": ua, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"},
                )
                resp.raise_for_status()
                html = resp.text

            links = re.findall(
                r'<a[^>]+class=["\'][^"\']*result-link[^"\']*["\'][^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
                html, re.DOTALL | re.IGNORECASE)
            if not links:
                links = re.findall(
                    r'<a[^>]+href=["\']([^"\']+)["\'][^>]*class=["\'][^"\']*result-link[^"\']*["\'][^>]*>(.*?)</a>',
                    html, re.DOTALL | re.IGNORECASE)
            snippets = re.findall(r'class=["\'][^"\']*result-snippet[^"\']*["\'][^>]*>(.*?)</(?:td|div)>', html, re.DOTALL | re.IGNORECASE)
            items = []
            for i, (url, title) in enumerate(links[:8]):
                snippet = snippets[i] if i < len(snippets) else ""
                items.append({"title": title, "url": url, "snippet": snippet})
            if items:
                result_text = _parse_results(items)
                logger.info(f"[ToolRouter] web_search(DDG HTML) 返回 {len(items)} 条")
                return {"status": "ok", "tool": "web_search", "query": query, "result": result_text}
            logger.debug("[ToolRouter] DDG HTML 无结果，尝试 Bing")
        except Exception as e:
            logger.warning(f"[ToolRouter] DDG HTML 失败: {e}，降级到 Bing")

        # ── 后备3：Bing HTML 搜索 ─────────────────────────────────────
        try:
            async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
                resp = await client.get(
                    "https://www.bing.com/search",
                    params={"q": query, "setlang": "zh-CN"},
                    headers={
                        "User-Agent": ua,
                        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    },
                )
                resp.raise_for_status()
                html = resp.text

            # Bing 结果：优先解析 b_algo 块；若结构变化，则退化为 h2/a 宽松解析。
            blocks = re.findall(r'<li[^>]+class=["\'][^"\']*b_algo[^"\']*["\'][^>]*>(.*?)</li>', html, re.DOTALL | re.IGNORECASE)
            items = []
            for block in blocks[:8]:
                m = re.search(r'<h2[^>]*>\s*<a[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', block, re.DOTALL | re.IGNORECASE)
                if not m:
                    continue
                url, title = m.group(1), m.group(2)
                if "bing.com" in url or url.startswith("/"):
                    continue
                sm = re.search(r'<p[^>]*>(.*?)</p>', block, re.DOTALL | re.IGNORECASE)
                snippet = sm.group(1) if sm else ""
                items.append({"title": title, "url": url, "snippet": snippet})
            if not items:
                links = re.findall(r'<h2[^>]*>\s*<a[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', html, re.DOTALL | re.IGNORECASE)
                for url, title in links[:8]:
                    if "bing.com" in url or url.startswith("/"):
                        continue
                    items.append({"title": title, "url": url, "snippet": ""})
            if items:
                result_text = _parse_results(items)
                logger.info(f"[ToolRouter] web_search(Bing) 返回 {len(items)} 条")
                return {"status": "ok", "tool": "web_search", "query": query, "result": result_text}
        except Exception as e:
            logger.error(f"[ToolRouter] Bing 搜索失败: {e}")

        # 全部后备失败
        return {
            "status": "degraded",
            "tool": "web_search",
            "query": query,
            "result": f"搜索服务暂时不可用（已尝试 DDG JSON/HTML + Bing），请直接访问相关网站获取「{query}」的信息。"
        }

    # ─────────────────────────────────────────────────────────────
    # Web 抓取：直接 httpx GET
    # ─────────────────────────────────────────────────────────────
    async def _web_fetch(self, url: str) -> dict:
        """直接 HTTP 抓取公开网页内容。包含最小 SSRF 防护、浏览器型请求头与清晰失败诊断。"""
        if not url:
            return {"error": "url 参数不能为空", "tool": "web_fetch"}

        import html as _html
        import ipaddress
        import socket
        from urllib.parse import urlparse

        parsed = urlparse(url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return {
                "error": "web_fetch 只允许抓取公开 http/https URL，且必须包含有效主机名",
                "tool": "web_fetch",
                "url": url,
            }
        if parsed.username or parsed.password:
            return {
                "error": "web_fetch 不接受包含用户名/密码的 URL",
                "tool": "web_fetch",
                "url": url,
            }

        async def _resolve_public_ips(hostname: str) -> tuple[bool, str]:
            try:
                infos = await asyncio.to_thread(socket.getaddrinfo, hostname, None, type=socket.SOCK_STREAM)
            except socket.gaierror as e:
                return False, f"DNS 解析失败: {e}"
            except Exception as e:
                return False, f"DNS 解析异常: {type(e).__name__}: {e}"

            addresses = sorted({info[4][0] for info in infos if info and info[4]})
            if not addresses:
                return False, "DNS 未返回可用地址"
            for address in addresses:
                try:
                    ip = ipaddress.ip_address(address)
                except ValueError:
                    return False, f"DNS 返回无效地址: {address}"
                if not ip.is_global:
                    return False, f"安全限制：目标解析到非公网地址 {address}，已拒绝抓取"
            return True, ",".join(addresses[:4])

        ok, dns_note = await _resolve_public_ips(parsed.hostname)
        if not ok:
            return {"error": dns_note, "tool": "web_fetch", "url": url}

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.5",
            "Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept-Encoding": "gzip, deflate",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "DNT": "1",
            "Upgrade-Insecure-Requests": "1",
        }
        timeout = httpx.Timeout(25.0, connect=10.0, read=20.0, write=10.0, pool=10.0)

        def _extract_text(raw: str, content_type: str) -> str:
            if "html" not in (content_type or "").lower():
                text = raw
            else:
                title = ""
                title_match = re.search(r"<title[^>]*>(.*?)</title>", raw, flags=re.DOTALL | re.IGNORECASE)
                if title_match:
                    title = _html.unescape(re.sub(r"\s+", " ", title_match.group(1))).strip()
                body = raw
                article_match = re.search(r"<(article|main)\b[^>]*>(.*?)</\1>", raw, flags=re.DOTALL | re.IGNORECASE)
                if article_match:
                    body = article_match.group(2)
                body = re.sub(r"<!--.*?-->", " ", body, flags=re.DOTALL)
                body = re.sub(r"<(script|style|noscript|svg|iframe)\b[^>]*>.*?</\1>", " ", body, flags=re.DOTALL | re.IGNORECASE)
                body = re.sub(r"<(br|p|div|li|h[1-6]|tr)\b[^>]*>", "\n", body, flags=re.IGNORECASE)
                body = re.sub(r"<[^>]+>", " ", body)
                text = _html.unescape(body)
                if title and title not in text[:500]:
                    text = f"标题: {title}\n{text}"
            text = re.sub(r"[ \t\r\f\v]+", " ", text)
            text = re.sub(r"\n\s*\n+", "\n", text)
            text = text.strip()
            if len(text) > 6000:
                text = text[:6000] + "\n...[内容已截断]"
            return text

        async def _request(verify: bool = True) -> httpx.Response:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, verify=verify, max_redirects=8) as client:
                return await client.get(url, headers=headers)

        try:
            try:
                resp = await _request(verify=True)
            except httpx.ConnectError as e:
                if "CERTIFICATE_VERIFY_FAILED" not in str(e):
                    raise
                logger.warning("[ToolRouter] web_fetch SSL 验证失败，降级为 verify=False 重试: %s", redact_secrets(url))
                resp = await _request(verify=False)

            content_type = resp.headers.get("content-type", "")
            final_url = str(resp.url)
            if resp.status_code >= 400:
                snippet = _extract_text(resp.text[:2000], content_type)[:500]
                if resp.status_code in {401, 403}:
                    diagnosis = "目标站点拒绝公开抓取（401/403），可能需要登录、地区/机器人访问限制或站点策略限制。"
                elif resp.status_code in {429, 503}:
                    diagnosis = "目标站点临时限流或服务不可用。"
                else:
                    diagnosis = "目标站点返回 HTTP 错误。"
                return {
                    "status": "error",
                    "tool": "web_fetch",
                    "url": url,
                    "final_url": final_url,
                    "http_status": resp.status_code,
                    "content_type": content_type,
                    "error": f"{diagnosis} HTTP {resp.status_code} {resp.reason_phrase}",
                    "diagnostic": snippet,
                }

            text = _extract_text(resp.text, content_type)
            if not text:
                return {
                    "status": "error",
                    "tool": "web_fetch",
                    "url": url,
                    "final_url": final_url,
                    "http_status": resp.status_code,
                    "content_type": content_type,
                    "error": "网页可访问但未提取到可读文本，可能是客户端渲染页面或二进制内容。",
                }
            logger.info("[ToolRouter] web_fetch 成功: url=%s status=%s bytes=%s dns=%s", redact_secrets(final_url), resp.status_code, len(resp.content), dns_note)
            return {
                "status": "ok",
                "tool": "web_fetch",
                "url": url,
                "final_url": final_url,
                "http_status": resp.status_code,
                "content_type": content_type,
                "result": text,
            }
        except httpx.TimeoutException as e:
            logger.error("[ToolRouter] web_fetch 超时 %s: %s", redact_secrets(url), type(e).__name__)
            return {"status": "error", "error": f"请求超时: {type(e).__name__}", "tool": "web_fetch", "url": url}
        except httpx.HTTPError as e:
            logger.error("[ToolRouter] web_fetch HTTP 失败 %s: %s: %s", redact_secrets(url), type(e).__name__, e)
            return {"status": "error", "error": f"HTTP 请求失败: {type(e).__name__}: {e}", "tool": "web_fetch", "url": url}
        except Exception as e:
            logger.error("[ToolRouter] web_fetch 失败 %s: %s: %s", redact_secrets(url), type(e).__name__, e)
            return {"status": "error", "error": f"抓取失败: {type(e).__name__}: {e}", "tool": "web_fetch", "url": url}

    # ─────────────────────────────────────────────────────────────
    # 天气：wttr.in 免费 API（无需 key）
    # ─────────────────────────────────────────────────────────────
    async def _weather(self, city: str) -> dict:
        """查询天气，使用 wttr.in 免费 API"""
        city = city.strip() or "Shanghai"
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    f"https://wttr.in/{city}",
                    params={"format": "3", "lang": "zh"},
                    headers={"User-Agent": "curl/7.68.0"},
                )
                resp.raise_for_status()
                result = resp.text.strip()
                return {"status": "ok", "tool": "weather", "city": city, "result": result}
        except Exception as e:
            logger.error(f"[ToolRouter] weather 失败: {e}")
            return {"error": str(e), "tool": "weather", "city": city}

    # ─────────────────────────────────────────────────────────────
    # Arxiv 搜索：官方 API（免费）
    # ─────────────────────────────────────────────────────────────
    async def _arxiv_search(self, query: str) -> dict:
        """搜索 Arxiv 学术论文"""
        if not query:
            return {"error": "query 参数不能为空"}
        try:
            async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
                resp = await client.get(
                    "https://export.arxiv.org/api/query",
                    params={
                        "search_query": f"all:{query}",
                        "start": 0,
                        "max_results": 5,
                        "sortBy": "relevance",
                    },
                )
                resp.raise_for_status()
                xml = resp.text

            # 简单解析 Atom XML
            entries = re.findall(r'<entry>(.*?)</entry>', xml, re.DOTALL)
            results = []
            for entry in entries[:5]:
                title = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
                summary = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
                link = re.search(r'<id>(.*?)</id>', entry, re.DOTALL)
                t = title.group(1).strip().replace('\n', ' ') if title else "无标题"
                s = summary.group(1).strip()[:200].replace('\n', ' ') if summary else ""
                l = link.group(1).strip() if link else ""
                results.append(f"- {t}\n  链接: {l}\n  摘要: {s}")

            result_text = f"Arxiv 搜索「{query}」结果：\n\n" + "\n\n".join(results) if results else "未找到相关论文"
            return {"status": "ok", "tool": "arxiv_search", "query": query, "result": result_text}
        except Exception as e:
            logger.error(f"[ToolRouter] arxiv_search 失败: {e}")
            return {"error": str(e), "tool": "arxiv_search", "query": query}

    # ─────────────────────────────────────────────────────────────
    # 本地日记：读/写
    # ─────────────────────────────────────────────────────────────
    def _diary_path(self, date_str: str, user_id: str = "agent") -> Path:
        if date_str in ("today", "", None):
            date_str = datetime.now().strftime("%Y-%m-%d")
        return DIARY_DIR / f"{date_str}_{user_id}.md"

    def _daily_note_read(self, date: str = "today") -> dict:
        """读取本地日记文件"""
        path = self._diary_path(date)
        try:
            if path.exists():
                content = redact_secrets(path.read_text(encoding="utf-8"))
                return {"status": "ok", "tool": "daily_note_read", "date": date, "result": content}
            else:
                return {"status": "ok", "tool": "daily_note_read", "date": date, "result": f"[{date} 暂无日记]"}
        except Exception as e:
            return {"error": str(e), "tool": "daily_note_read"}

    def _daily_note_write(self, content: str, user_id: str = "agent") -> dict:
        """写入本地日记文件（追加模式）"""
        if not content:
            return {"error": "content 不能为空", "tool": "daily_note_write"}
        try:
            path = self._diary_path("today", user_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
            entry = f"\n\n## [{now_str}]\n{redact_secrets(content)}\n"
            with open(path, "a", encoding="utf-8") as f:
                f.write(entry)
            logger.info(f"[ToolRouter] 日记写入成功: {path.name}")
            return {"status": "ok", "tool": "daily_note_write", "result": f"已写入日记 {path.name}"}
        except Exception as e:
            logger.error(f"[ToolRouter] daily_note_write 失败: {e}")
            return {"error": str(e), "tool": "daily_note_write"}

    # ─────────────────────────────────────────────────────────────
    # 知识检索：关键词搜索本地日记文件
    # ─────────────────────────────────────────────────────────────
    def _knowledge_search(self, query: str) -> dict:
        """关键词检索本地日记文件（简单 RAG）"""
        if not query:
            return {"error": "query 参数不能为空"}
        try:
            keywords = query.lower().split()
            results = []
            diary_files = sorted(DIARY_DIR.glob("*.md"), reverse=True)[:30]  # 只检索最近30个文件

            for fpath in diary_files:
                try:
                    text = fpath.read_text(encoding="utf-8")
                    text_lower = text.lower()
                    score = sum(1 for kw in keywords if kw in text_lower)
                    if score > 0:
                        # 提取包含关键词的段落
                        paragraphs = text.split('\n\n')
                        matched = []
                        for para in paragraphs:
                            if any(kw in para.lower() for kw in keywords):
                                matched.append(para.strip()[:200])
                        if matched:
                            results.append((score, fpath.name, matched[:2]))
                except Exception:
                    continue

            results.sort(key=lambda x: x[0], reverse=True)

            if not results:
                return {"status": "ok", "tool": "knowledge_search", "query": query, "result": f"未找到与「{query}」相关的记忆"}

            output_parts = [f"记忆检索「{query}」结果：\n"]
            for score, fname, matched in results[:5]:
                output_parts.append(f"**{fname}**（匹配度: {score}）")
                output_parts.extend(matched)
                output_parts.append("")

            return {"status": "ok", "tool": "knowledge_search", "query": query, "result": "\n".join(output_parts)}
        except Exception as e:
            logger.error(f"[ToolRouter] knowledge_search 失败: {e}")
            return {"error": str(e), "tool": "knowledge_search"}

    # ─────────────────────────────────────────────────────────────
    # 文件操作
    # ─────────────────────────────────────────────────────────────
    def _file_list(self, path: str = ".") -> dict:
        """列出目录文件，敏感文件只显示已保护。"""
        try:
            fpath = Path(path or ".")
            if is_sensitive_path(fpath):
                return {"error": "敏感路径已拦截", "tool": "file_list", "path": redact_secrets(path)}
            if not fpath.exists() or not fpath.is_dir():
                return {"error": f"目录不存在: {path}", "tool": "file_list"}
            items = []
            for child in sorted(fpath.iterdir())[:200]:
                items.append({"name": child.name, "type": "dir" if child.is_dir() else "file", "protected": is_sensitive_path(child)})
            return {"status": "ok", "tool": "file_list", "path": redact_secrets(path), "result": items}
        except Exception as e:
            return {"error": redact_secrets(str(e)), "tool": "file_list"}

    def _file_read(self, path: str) -> dict:
        """读取本地文件；若模型传入日记文件名，则自动在 DIARY_DIR 下解析。"""
        if not path:
            return {"error": "path 参数不能为空"}
        path_text = str(path).strip()
        api_path_pattern = re.compile(r"^/(health|activity_logs|public_chat|experiment(?:/status|/tasks|/artifacts)?|evolution/status|state|self-upgrade(?:/status|/proposals)?)(?:[/?#].*)?$")
        looks_like_api_path = bool(api_path_pattern.match(path_text)) or (
            path_text.startswith("/")
            and not re.search(r"\.(md|txt|json|jsonl|yaml|yml|py|log|csv|html|css|js)$", path_text, re.IGNORECASE)
        )
        if looks_like_api_path:
            return {
                "status": "error",
                "error": f"路径是 HTTP API 端点而非本地文件: {redact_secrets(path_text)}；请改用 web_fetch/url_fetch 或用户转述的接口证据。",
                "tool": "file_read",
                "path": redact_secrets(path_text),
                "diagnosis": "api_path_misuse",
                "recommended_tools": ["web_fetch", "url_fetch"],
            }
        if is_sensitive_path(path):
            return {"error": "敏感路径已拦截", "tool": "file_read", "path": redact_secrets(path)}
        try:
            fpath = Path(path)
            resolved_from = "direct"
            if not fpath.exists() and not fpath.is_absolute():
                # 线上反思经常先用 daily_note_write 写入 YYYY-MM-DD_agent.md，
                # 随后用 file_read 读取同名文件。容器工作目录不一定是 DIARY_DIR，
                # 因此对安全的日记文件名做一次显式回退解析，避免误报“文件不存在”。
                safe_diary_name = re.fullmatch(r"\d{4}-\d{2}-\d{2}_[A-Za-z0-9_\-]+\.md", fpath.name or "")
                if safe_diary_name and len(fpath.parts) == 1:
                    diary_path = DIARY_DIR / fpath.name
                    if diary_path.exists():
                        fpath = diary_path
                        resolved_from = "diary_dir"
            if not fpath.exists():
                return {"error": f"文件不存在: {redact_secrets(path)}", "tool": "file_read", "path": redact_secrets(path)}
            content = redact_secrets(fpath.read_text(encoding="utf-8", errors="replace"))
            if len(content) > 8000:
                content = content[:8000] + "\n...[内容已截断]"
            return {
                "status": "ok",
                "tool": "file_read",
                "path": redact_secrets(path),
                "resolved_from": resolved_from,
                "result": content,
            }
        except Exception as e:
            return {"error": redact_secrets(str(e)), "tool": "file_read"}

    def _file_write(self, path: str, content: str) -> dict:
        """写入本地文件。普通写入不能覆盖记忆目录，也不能直接改自身代码。"""
        if not path:
            return {"error": "path 参数不能为空"}
        if is_sensitive_path(path):
            return {"error": "敏感路径已拦截", "tool": "file_write", "path": redact_secrets(path)}
        if is_memory_path(path):
            return {"error": "记忆/成长数据路径禁止覆盖", "tool": "file_write", "path": redact_secrets(path)}
        if is_self_project_code_path(path):
            return {"error": "自改代码必须使用 self_code_write 并携带成长/审计日志", "tool": "file_write", "path": redact_secrets(path)}
        try:
            return self._write_file_with_snapshot(path, content, tool="file_write")
        except Exception as e:
            return {"error": redact_secrets(str(e)), "tool": "file_write"}

    def _write_file_with_snapshot(self, path: str, content: str, tool: str) -> dict:
        fpath = Path(path)
        config = load_autonomy_config()
        if fpath.exists() and not config.experiment_allow_destructive:
            backup = fpath.with_suffix(fpath.suffix + f".snapshot-{int(time.time())}")
            backup.write_text(fpath.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text(redact_secrets(content), encoding="utf-8")
        return {"status": "ok", "tool": tool, "path": redact_secrets(path), "result": f"文件已写入: {redact_secrets(path)}"}

    def _self_code_write(self, path: str, content: str, audit: dict | None) -> dict:
        """记录成长日志后允许 AIwake 改动自身代码/运行规则，保留记忆与密钥防护。"""
        if not path:
            return {"error": "path 参数不能为空", "tool": "self_code_write"}
        if not content:
            return {"error": "content 参数不能为空", "tool": "self_code_write", "path": redact_secrets(path)}
        if is_sensitive_path(path):
            return {"error": "敏感路径已拦截", "tool": "self_code_write", "path": redact_secrets(path)}
        if is_memory_path(path):
            return {"error": "记忆/成长数据路径禁止覆盖或删除", "tool": "self_code_write", "path": redact_secrets(path)}
        if not is_self_project_code_path(path):
            return {"error": "self_code_write 仅限 AIwake 自身项目代码/运行规则文件", "tool": "self_code_write", "path": redact_secrets(path)}
        ok, missing = has_self_modification_audit(audit)
        if not ok:
            return {"error": f"缺少成长/审计字段: {', '.join(missing)}", "tool": "self_code_write", "path": redact_secrets(path)}
        try:
            result = self._write_file_with_snapshot(path, content, tool="self_code_write")
            log_item = {
                "event": "self_code_modified",
                "tool": "self_code_write",
                "path": redact_secrets(path),
                "purpose": audit.get("purpose"),
                "change_summary": audit.get("change_summary"),
                "files": audit.get("files"),
                "validation_result": audit.get("validation_result"),
                "risk_note": audit.get("risk_note"),
            }
            append_growth_log(log_item)
            self.store.append("events", {"event": "self_code_modified", **log_item})
            result["growth_log"] = "已写入 evolution/growth_log.jsonl"
            return redact_secrets(result)
        except Exception as e:
            return {"error": redact_secrets(str(e)), "tool": "self_code_write"}

    def _file_append(self, path: str, content: str) -> dict:
        if not path:
            return {"error": "path 参数不能为空"}
        if is_sensitive_path(path):
            return {"error": "敏感路径已拦截", "tool": "file_append", "path": redact_secrets(path)}
        if is_memory_path(path):
            return {"error": "记忆/成长数据路径禁止任意追加，请使用 daily_note_write/memory_write/growth_log", "tool": "file_append", "path": redact_secrets(path)}
        try:
            fpath = Path(path)
            fpath.parent.mkdir(parents=True, exist_ok=True)
            with open(fpath, "a", encoding="utf-8") as f:
                f.write(redact_secrets(content))
            return {"status": "ok", "tool": "file_append", "path": redact_secrets(path), "result": f"文件已追加: {redact_secrets(path)}"}
        except Exception as e:
            return {"error": redact_secrets(str(e)), "tool": "file_append"}

    def _patch_write(self, path: str, content: str) -> dict:
        if not load_autonomy_config().experiment_allow_patch_generation:
            return {"error": "patch 生成未开启", "tool": "patch_write"}
        safe_name = re.sub(r"[^\w.\-]+", "_", Path(path or "experiment.patch").name)[:80]
        patch_path = Path(os.getenv("EXPERIMENT_PATCH_DIR", "/app/data/experiments/patches")) / safe_name
        return self._file_write(str(patch_path), content)

    def _memory_write(self, content: str) -> dict:
        item = self.store.append("events", {"event": "memory_update", "content": redact_secrets(content)})
        return {"status": "ok", "tool": "memory_write", "result": item}

    def _self_task_create(self, title: str, goal: str = "") -> dict:
        item = self.store.append("tasks", {"type": "self_task", "title": title or "未命名自任务", "goal": goal, "status": "open"})
        return {"status": "ok", "tool": "self_task_create", "result": item}

    def _self_task_update(self, task_id: str, status: str, note: str = "") -> dict:
        item = self.store.append("tasks", {"id": task_id or "unknown", "type": "self_task_update", "status": status or "updated", "note": redact_secrets(note)})
        return {"status": "ok", "tool": "self_task_update", "result": item}

    def _self_artifact_create(
        self,
        task_id: str,
        title: str,
        content: str,
        kind: str = "learning_artifact",
        note: str = "",
    ) -> dict:
        """为自任务追加可查询学习产物，不关闭任务、不触碰记忆/日志删除。"""
        clean_task_id = str(task_id or "").strip()
        clean_content = str(content or "").strip()
        if not clean_task_id:
            return {"error": "task_id 参数不能为空", "tool": "self_artifact_create"}
        if not clean_content:
            return {"error": "content 参数不能为空", "tool": "self_artifact_create", "task_id": redact_secrets(clean_task_id)}
        item = self.store.append("artifacts", {
            "task_id": clean_task_id,
            "kind": str(kind or "learning_artifact")[:80],
            "title": str(title or "未命名学习产物")[:160],
            "content": redact_secrets(clean_content),
            "note": redact_secrets(note or "这是学习产物，不代表任务已关闭或已部署。"),
            "source": "tool_router.self_artifact_create",
            "status": "draft",
        })
        return {"status": "ok", "tool": "self_artifact_create", "result": item}

    # ─────────────────────────────────────────────────────────────
    # Shell 执行
    # ─────────────────────────────────────────────────────────────
    async def _shell_exec(self, command: str, timeout: int = 30) -> dict:
        """执行 shell 命令"""
        if not command:
            return {"error": "command 参数不能为空"}
        config = load_autonomy_config()
        risk = classify_command_risk(command)
        risks = set(risk.get("risks", []))
        if "secret_access" in risks:
            return {"error": "命令涉及密钥读取，已拦截", "tool": "shell_exec", "risk": risk}
        if "memory_delete" in risks:
            return {"error": "命令涉及删除记忆/成长数据，已拦截", "tool": "shell_exec", "risk": risk}
        if "deploy" in risks and not config.experiment_allow_deploy:
            return {"error": "部署命令需要 EXPERIMENT_ALLOW_DEPLOY=true，已拦截", "tool": "shell_exec", "risk": risk}
        if "destructive" in risks and not config.experiment_allow_destructive:
            return {"error": "破坏性命令需要 EXPERIMENT_ALLOW_DESTRUCTIVE=true 或快照保护，已拦截", "tool": "shell_exec", "risk": risk}
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                return {"error": f"命令执行超时（{timeout}秒）", "tool": "shell_exec"}

            output = redact_secrets(stdout.decode("utf-8", errors="replace"))
            err_output = redact_secrets(stderr.decode("utf-8", errors="replace"))
            result = output
            if err_output:
                result += f"\n[stderr]: {err_output}"
            if len(result) > 3000:
                result = result[:3000] + "\n...[输出已截断]"
            return {"status": "ok", "tool": "shell_exec", "command": risk.get("redacted_command"), "risk": risk, "result": result, "returncode": proc.returncode}
        except Exception as e:
            logger.error(f"[ToolRouter] shell_exec 失败: {redact_secrets(e)}")
            return {"error": redact_secrets(str(e)), "tool": "shell_exec"}

    # ─────────────────────────────────────────────────────────────
    # SSH 执行
    # ─────────────────────────────────────────────────────────────
    async def _ssh_exec(self, command: str, host: str, user: str, password: str = "", port: int = 22) -> dict:
        """SSH 远程执行命令"""
        if not command or not host:
            return {"error": "command 和 host 参数不能为空"}
        try:
            import paramiko
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            def _run():
                client.connect(host, port=int(port), username=user, password=password, timeout=15)
                _, stdout, stderr = client.exec_command(command)
                out = stdout.read().decode("utf-8", errors="replace")
                err = stderr.read().decode("utf-8", errors="replace")
                client.close()
                return out, err

            loop = asyncio.get_event_loop()
            out, err = await loop.run_in_executor(None, _run)
            result = redact_secrets(out)
            if err:
                result += f"\n[stderr]: {redact_secrets(err)}"
            if len(result) > 3000:
                result = result[:3000] + "\n...[输出已截断]"
            return {"status": "ok", "tool": "ssh_exec", "host": redact_secrets(host), "result": result}
        except ImportError:
            return {"error": "paramiko 未安装，无法使用 ssh_exec（请在 requirements.txt 中添加 paramiko）", "tool": "ssh_exec"}
        except Exception as e:
            logger.error(f"[ToolRouter] ssh_exec 失败: {e}")
            return {"error": str(e), "tool": "ssh_exec"}

    # ─────────────────────────────────────────────────────────────
    # 日程：读取本地日程文件
    # ─────────────────────────────────────────────────────────────
    def _schedule_read(self) -> dict:
        """读取本地日程文件"""
        schedule_path = Path("/app/data/schedule.md")
        try:
            if schedule_path.exists():
                content = schedule_path.read_text(encoding="utf-8")
                return {"status": "ok", "tool": "schedule_read", "result": content}
            return {"status": "ok", "tool": "schedule_read", "result": "[暂无日程安排]"}
        except Exception as e:
            return {"error": str(e), "tool": "schedule_read"}
