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
import shlex
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx

from autonomy_config import load_autonomy_config
from experiment_store import get_experiment_store
from evolution.memory import append_growth_log
from evolution.growth_tracker import record_growth_milestone
from evolution.self_upgrade import proposal_status_summary
from evolution.goal_tracker import (
    register_goal as _gt_register_goal,
    read_open_goals as _gt_read_open_goals,
    read_all_goals as _gt_read_all_goals,
    read_capabilities as _gt_read_capabilities,
    METRIC_REGISTRY as _GT_METRIC_REGISTRY,
)
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
    "experiment_status": 0.08,
    "self_upgrade_status": 0.1,
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
    # ── 目标闭环工具（改→度量→收敛）──
    "goal_list": 0.05,           # 只读：列出当前可用 metrics + 开放/全部目标
    "goal_capabilities": 0.05,   # 只读：列出已沉淀入库的能力
    "goal_register": 0.2,        # 写入：登记一个可度量改进目标进入闭环
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
    "experiment_status",
    "self_upgrade_status",
    "arxiv_search",
    "tag_memo",
    "file_list",
    "goal_list",
    "goal_capabilities",
})

# 并发批执行的最大并发数（参考 Claude Code runToolsConcurrently 最多10个）
MAX_CONCURRENT_TOOLS = 10

# 高风险工具（仍允许执行，但记录警告日志）
HIGH_RISK_TOOLS: dict[str, float] = {
    "file_delete": 0.85,
    "package_install": 0.9,
}

# 当前实例真实实现/受权限门控的全部工具名（call() dispatcher 能路由的名字）。
# 用于识别 LLM 幻觉出的通用助手工具名（如 Read/Glob/TodoWrite），把它们与
# 真实工具区分开，避免“工具名笔误”被错误计入基础设施 tool_failure_count。
KNOWN_TOOLS: frozenset[str] = frozenset({
    "web_fetch", "url_fetch", "web_search", "weather", "arxiv_search",
    "daily_note_read", "daily_note_write", "experiment_status",
    "self_upgrade_status", "knowledge_search", "tag_memo",
    "file_list", "file_read", "file_write", "file_append", "patch_write",
    "memory_write", "self_code_write",
    "self_task_create", "self_task_update", "self_artifact_create",
    "shell_exec", "ssh_exec", "schedule_read",
    "goal_list", "goal_capabilities", "goal_register",
})

# 工具名别名归一化：把大模型常幻觉出的通用助手命名（Claude/Cursor 风格）映射到
# 本实例真实工具名，做 1:1 安全等价替换；无法安全等价的名字交给未知工具指引处理。
_TOOL_NAME_ALIASES: dict[str, str] = {
    "read": "file_read",
    "read_file": "file_read",
    "readfile": "file_read",
    "view": "file_read",
    "view_file": "file_read",
    "cat": "file_read",
    "open": "file_read",
    "write": "file_write",
    "write_file": "file_write",
    "writefile": "file_write",
    "create_file": "file_write",
    "glob": "file_list",
    "ls": "file_list",
    "list": "file_list",
    "list_files": "file_list",
    "list_dir": "file_list",
    "listdir": "file_list",
    "websearch": "web_search",
    "web_search_tool": "web_search",
    "search_web": "web_search",
    "webfetch": "web_fetch",
    "fetch": "web_fetch",
    "fetch_url": "web_fetch",
    "browse": "web_fetch",
    "bash": "shell_exec",
    "shell": "shell_exec",
    "run_command": "shell_exec",
    "run_shell": "shell_exec",
    "exec": "shell_exec",
    "memory_search": "knowledge_search",
    "recall": "knowledge_search",
    "note_write": "daily_note_write",
    "write_note": "daily_note_write",
    "note_read": "daily_note_read",
    "read_note": "daily_note_read",
}


class ToolRouter:
    # 类级共享：避免每次实例化重复扫描能力库
    _capability_tools: dict[str, dict] = {}

    def __init__(self):
        self.store = get_experiment_store()
        # 启动时扫描能力库，把带 tool_spec 的能力注册成新工具
        self._reload_capability_tools()

    def _reload_capability_tools(self) -> None:
        """重新扫描 capability_library.jsonl，更新动态工具注册表。"""
        try:
            from capability_tools import load_capability_tools
            registered = load_capability_tools()
        except Exception as e:
            import logging as _logging
            _logging.getLogger(__name__).warning(f"[ToolRouter] 加载能力工具失败: {e}")
            registered = {}
        # 用类级 dict，确保所有 ToolRouter 实例看到同一份注册表
        ToolRouter._capability_tools = registered
        # 同步到 ALLOWED_TOOLS / READONLY_TOOLS（不覆盖已存在的核心工具）
        for name, record in registered.items():
            if name in ALLOWED_TOOLS:
                continue
            ALLOWED_TOOLS[name] = record["risk"]
            if record.get("readonly"):
                # READONLY_TOOLS 是 frozenset，需要重建
                pass  # 通过 _check_permission 路径上行为已经一致；并发安全检查由 ToolRouter.call_batch 单独决策

    def _classify_failure(self, tool_name: str, result: dict) -> dict:
        """为工具失败补充稳定的分类与降级路径，帮助下一轮反思可直接行动。"""
        status = str(result.get("status") or "").lower()
        error_text = str(result.get("error") or result.get("result") or "")
        diagnosis_text = str(result.get("diagnosis") or result.get("diagnostic") or "")
        http_status = result.get("http_status")
        empty_ok = (
            tool_name in {"daily_note_read", "knowledge_search"}
            and status == "ok"
            and not result.get("error")
            and self._looks_empty_result(result.get("result"))
        )
        nonzero_shell_return = tool_name == "shell_exec" and result.get("returncode") not in (None, 0)
        if not empty_ok and not nonzero_shell_return and not result.get("error") and status not in {"error", "degraded"}:
            return result

        combined = f"{tool_name} {status} {error_text} {diagnosis_text} {result.get('command', '')}".lower()
        if empty_ok:
            failure_type = "empty_result"
            fallback_hint = "工具可用但结果为空；改用明确日期/更具体关键词重试，并用写入记录、公开状态或活动日志交叉验证，不能据此断言记忆不存在。"
        elif tool_name == "shell_exec" and self._looks_like_local_endpoint_boundary(combined):
            failure_type = "local_endpoint_unavailable/environment_boundary"
            fallback_hint = "shell_exec 访问容器内 localhost 或 /experiment/status 失败只说明当前执行环境边界不可用；验证线上状态应使用 experiment_status、公开 HTTPS 状态接口或 activity 日志交叉验证，不要升级为线上服务故障。"
        elif str(result.get("diagnosis") or "") == "api_path_misuse" or self._looks_like_api_path_misuse(combined):
            failure_type = "execution_context_mismatch/tool_path_boundary_misuse"
            fallback_hint = "这是 HTTP/API 路由与本地文件/命令上下文混用；改用 web_fetch/url_fetch 读取公网 HTTPS 端点，或使用结构化状态接口，不要用 shell cat/file_read 读取 API 路径。"
        elif "非公网地址" in error_text or "private network" in combined or "localhost" in combined or "127.0.0.1" in combined:
            failure_type = "private_network_safety_restriction"
            fallback_hint = "web_fetch 出于 SSRF 防护拒绝私网/localhost；改用公开 HTTPS 地址或只读公开接口，不要尝试绕过安全限制。"
        elif http_status in {404, 410}:
            failure_type = "external_content_not_found/content_stale"
            fallback_hint = "记录 URL、final_url 与 HTTP 状态；换可信来源重试，或把该网页标记为内容过期/不可用证据，不要判定网络工具整体故障。"
        elif http_status in {401, 403}:
            failure_type = "external_access_restricted"
            fallback_hint = "目标站点拒绝公开抓取；换公开来源或搜索摘要，不要访问凭据或尝试绕过登录/地区/机器人限制。"
        elif http_status in {429, 503}:
            failure_type = "external_rate_limited_or_unavailable"
            fallback_hint = "外部站点限流/临时不可用；退避后重试一次，或改用等价网络工具与公开接口。"
        elif "timeout" in combined or "超时" in error_text:
            failure_type = "readonly_observation_timeout" if tool_name in {"web_fetch", "url_fetch", "file_read", "knowledge_search", "daily_note_read"} else "aiwake_tool_timeout"
            fallback_hint = "降低 limit/缩短 hours/拆分任务后重试一次；修复后用 cache.mode、耗时、任务-artifact 匹配关系交叉验证。"
        elif tool_name in {"web_search", "web_fetch", "url_fetch", "weather", "arxiv_search"}:
            failure_type = "external_service_failure"
            fallback_hint = "缩短请求范围或重试；若仍失败，改用等价网络工具/公开接口；记录 HTTP 状态、异常类型、URL 与耗时。"
        elif "permission" in combined or "allowed" in combined or "配置" in error_text or "拦截" in error_text:
            failure_type = "aiwake_tool_policy_failure"
            fallback_hint = "不要绕过安全策略；改用允许的只读工具或先生成 self_task/artifact 记录证据与边界。"
        elif tool_name in {"file_read", "file_list", "file_write", "file_append", "self_code_write", "shell_exec", "ssh_exec"}:
            failure_type = "aiwake_internal_tool_failure"
            fallback_hint = "先确认参数、路径和安全边界；对文件/API 路径误用改用推荐工具；对命令失败保留 stderr/returncode。"
        else:
            failure_type = "aiwake_tool_failure"
            fallback_hint = "记录 error/status/工具名/参数摘要，选择等价工具或创建学习 artifact 后再继续。"

        enriched = dict(result)
        enriched.setdefault("status", "ok" if empty_ok else "error")
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

    def _finalize_tool_result(self, tool_name: str, result: dict) -> dict:
        """统一补充失败分类，并把可行动的失败审计追加到成长/事件日志。"""
        enriched = self._classify_failure(tool_name, result)
        self._record_tool_failure_audit(tool_name, enriched)
        return enriched

    def _record_tool_failure_audit(self, tool_name: str, result: dict) -> None:
        """把工具失败原因与降级路径持久化，供后续反思形成可追溯闭环。"""
        failure_type = result.get("failure_type")
        if not failure_type or failure_type == "empty_result":
            return
        audit_item = {
            "event": "tool_failure_audit",
            "tool": tool_name,
            "failure_type": failure_type,
            "status": result.get("status"),
            "error": redact_secrets(result.get("error") or result.get("result") or ""),
            "http_status": result.get("http_status"),
            "diagnosis": result.get("diagnosis") or result.get("diagnostic"),
            "returncode": result.get("returncode"),
            "fallback_hint": result.get("fallback_hint"),
            "recommended_next_actions": result.get("recommended_next_actions", []),
        }
        try:
            append_growth_log(audit_item)
        except Exception as exc:
            logger.warning(f"[ToolRouter] 写入工具失败成长审计失败: {exc}")
        try:
            self.store.append("events", audit_item)
        except Exception as exc:
            logger.warning(f"[ToolRouter] 写入工具失败事件审计失败: {exc}")

    def _looks_empty_result(self, value: object) -> bool:
        """识别工具成功但无语义内容的结果，避免被误判为工具故障或记忆不存在。"""
        text = str(value or "").strip()
        if not text:
            return True
        return bool(re.fullmatch(r"\[[^\]]*(暂无|未找到|无结果|empty|not found)[^\]]*\]", text, re.IGNORECASE))

    def _looks_like_local_endpoint_boundary(self, text: str) -> bool:
        """识别 shell 中访问本地/容器 HTTP 端点失败的环境边界，而非线上故障。"""
        endpoint_markers = ("localhost", "127.0.0.1", "0.0.0.0", "/experiment/status", "/health")
        failure_markers = (
            "connection refused",
            "could not connect",
            "failed to connect",
            "status_unavailable",
            "health_endpoints_unavailable",
            "returncode",
            "curl:",
            "wget:",
        )
        return any(marker in text for marker in endpoint_markers) and any(marker in text for marker in failure_markers)

    def _looks_like_api_path_misuse(self, text: str) -> bool:
        """识别把 HTTP API 路由当成本地文件或 shell 路径读取的错误。"""
        api_routes = (
            "/health",
            "/activity_logs",
            "/public_chat",
            "/experiment/status",
            "/experiment/tasks",
            "/experiment/artifacts",
            "/evolution/status",
            "/state",
            "/self-upgrade/status",
            "/self-upgrade/proposals",
        )
        return any(route in text for route in api_routes) and any(
            marker in text
            for marker in ("file_read", "shell_exec", "cat ", "not_found", "file_not_found", "no such file", "文件不存在")
        )

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
            "- daily_note_read(date) — 读取日记/笔记；today 为空时改用明确 YYYY-MM-DD 复验",
            "- daily_note_write(content) — 写入成长日记",
            "- experiment_status() — 读取本机实验任务状态快照（open_task_count/open_tasks[id,title,status,goal]/latest_task/latest_artifact）",
            "- self_upgrade_status() — 读取自升级提案状态摘要（总数/待审批/已批准/已应用/最近提案详情）",
            "- goal_list() — 查看可用 metrics、当前开放目标、最近全部目标。反思要登记目标前先调它。",
            "- goal_capabilities() — 查看历史已闭合的能力库，避免重复发明已掌握的改进。",
            "- goal_register(metric, direction, target, description) — 把'X 指标改到 b'写成可度量目标，进入 改→度量→收敛 闭环；evolution 每轮自动复测，达成→闭合并沉淀能力，恶化或超限→自动放弃。反思识别出可度量改进点时应直接登记，然后专注做能改善该指标的事，不要反复重复同一类反思。",
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
        schema: list[dict] = [
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
                    "name": "experiment_status",
                    "description": "读取实验任务状态快照，返回 open_task_count、open_tasks（含每个 open task 的 id/title/status/goal）、latest_task、latest_artifact 等字段。可直接获取所有 open task 的 id，用于 self_task_update 关闭僵尸任务；比 shell_exec 访问 localhost 或 /experiment/status 更适合本机状态验证。",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "self_upgrade_status",
                    "description": "读取自升级提案状态摘要，返回 total/pending/approved/applied/rejected 计数和最近提案详情。供反思时了解自我进化管线进展。",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
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
            {
                "type": "function",
                "function": {
                    "name": "goal_list",
                    "description": "列出可用 metrics、当前所有开放目标、最近 50 条全部目标。反思中要登记目标前先调用此工具，避免对同一指标重复登记，并选择正确的 metric 名。",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "goal_capabilities",
                    "description": "列出已沉淀入工具/能力库的可复用能力（来自闭合的目标）。供反思时回顾历史成果，避免重新发明已掌握的改进。",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "goal_register",
                    "description": "登记一个可度量改进目标，进入 改→度量→收敛 闭环。evolution 引擎每轮会用当前 metrics 复测：达成则自动闭合并沉淀进能力库；显著恶化（>20%）或超过 max_cycles 则自动放弃。AIwake 反思时若识别出'X 指标需要从 a 改到 b'，应直接调用此工具登记，然后专注于做能改善该指标的实际工作，而不是反复重复同一类反思。\n\n选择目标时的约束：\n1) 先调 goal_list 看哪些 metric 还没有 open goal，避免对同一指标重复登记。\n2) 如果某指标已经接近最优（如 tool_success_rate 已 ≥0.99，或 *_count 类已 0），不要选它做目标——选还有真实改进空间的指标，否则只是在刷分。\n3) 调 goal_capabilities 查能力库，避免重复证明已经掌握的能力。\n4) source 字段如实标注：用户对话明确要求登记=\"manual\"；反思中自主提出=\"reflection\"；实验/工具直接调用=\"tool_call\"或类似。\n5) target 必须能用现有 metric 真实测量，不要发明指标。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "metric": {
                                "type": "string",
                                "description": "metric 名，必须是 goal_list 返回的 metrics 之一，如 tool_success_rate / chat_success_rate / tool_failure_count / reflection_to_action_ratio 等"
                            },
                            "direction": {
                                "type": "string",
                                "enum": ["up", "down", "target"],
                                "description": "up=越高越好，down=越低越好，target=接近 target 即可"
                            },
                            "target": {
                                "type": "number",
                                "description": "目标值。例如 tool_success_rate 设 0.99；tool_failure_count 设 0"
                            },
                            "description": {
                                "type": "string",
                                "description": "用一句话说明目标来由：当前现状、想改到哪、为什么"
                            },
                            "source": {
                                "type": "string",
                                "description": "目标来源标签，默认 reflection。可填 reflection/manual/auto 等"
                            },
                            "max_cycles": {
                                "type": "integer",
                                "description": "最大复测轮数，超过仍未达成则自动放弃。默认 12，建议 4–24"
                            }
                        },
                        "required": ["metric", "direction", "target", "description"]
                    }
                }
            },
        ]
        # 合并能力库注册的动态工具 schema，让 LLM 看见反思自主新增的小工具
        for record in ToolRouter._capability_tools.values():
            sch = record.get("schema")
            if sch:
                schema.append(sch)
        return schema

    @staticmethod
    def _normalize_tool_name(tool_name: str) -> str:
        """把工具名做安全归一化：去空白、小写匹配别名表，命中则返回真实工具名。

        仅做 1:1 安全等价替换；已是真实工具名（区分大小写匹配 KNOWN_TOOLS）的不改动。
        """
        if not isinstance(tool_name, str):
            return tool_name
        name = tool_name.strip()
        if name in KNOWN_TOOLS or name in ToolRouter._capability_tools:
            return name
        return _TOOL_NAME_ALIASES.get(name.lower(), name)

    def _unknown_tool_guidance(self, requested_name: str) -> dict:
        """对真正未知/幻觉的工具名返回结构化指引，而不是基础设施失败。

        status=degraded（非 error）：不计入 tool_failure_count，避免“工具名笔误”
        污染可靠性指标；同时给出可用工具清单，引导模型改用真实工具名重试。
        """
        available = sorted(KNOWN_TOOLS | set(ToolRouter._capability_tools.keys()))
        result = {
            "status": "degraded",
            "tool": requested_name,
            "diagnosis": "unknown_tool_name",
            "result": (
                f"工具 '{requested_name}' 不是本实例可用工具（很可能是工具名笔误或"
                f"通用助手命名）。请改用下列真实工具名之一重试，不要重复调用同一个不存在的名字。"
            ),
            "available_tools": available,
            "fallback_hint": (
                "用真实工具名重试：读文件用 file_read、列目录用 file_list、"
                "搜网用 web_search、抓网页用 web_fetch、检索记忆用 knowledge_search、"
                "记任务用 self_task_create/self_artifact_create。"
            ),
            "recommended_next_actions": [
                "从 available_tools 选一个语义等价的真实工具名重试。",
                "若只是想记录待办，用 self_task_create 而非 TodoWrite 等外部命名。",
                "不要把工具名笔误升级为系统故障，也不要反复重复同一不存在的工具名。",
            ],
        }
        try:
            self.store.append("events", {
                "event": "tool_name_unknown",
                "requested_tool": requested_name,
                "diagnosis": "unknown_tool_name",
            })
        except Exception as exc:
            logger.warning(f"[ToolRouter] 写入未知工具名事件失败: {exc}")
        return result

    async def call(self, tool_name: str, params: dict) -> dict:
        """执行工具调用（工具名归一化 + 权限校验 + 直接实现）

        见 _normalize_tool_name / _unknown_tool_guidance：把幻觉工具名安全归一化，
        真正未知的名字返回结构化指引而非基础设施失败。
        """
        # 1) 工具名归一化：把大模型幻觉出的通用助手命名映射到真实工具名，
        #    避免“工具名笔误”被错误计入 tool_failure_count。
        original_name = tool_name
        tool_name = self._normalize_tool_name(tool_name)
        if tool_name != original_name:
            logger.info("[ToolRouter] 工具名归一化: %s → %s", original_name, tool_name)

        # 2) 真正未知的工具名：返回结构化使用指引（非 error），不当作基础设施失败。
        if (
            tool_name not in KNOWN_TOOLS
            and tool_name not in ToolRouter._capability_tools
            and tool_name not in HIGH_RISK_TOOLS
        ):
            return self._unknown_tool_guidance(original_name)

        allowed, risk = self._check_permission(tool_name)
        if not allowed:
            return self._finalize_tool_result(tool_name, {"error": f"工具 '{tool_name}' 未被当前实验配置允许", "tool": tool_name, "risk": risk})

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
            elif tool_name == "experiment_status":
                result = self._experiment_status_snapshot()
            elif tool_name == "self_upgrade_status":
                result = self._self_upgrade_status()
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
            elif tool_name == "goal_list":
                result = self._goal_list()
            elif tool_name == "goal_capabilities":
                result = self._goal_capabilities()
            elif tool_name == "goal_register":
                result = self._goal_register(
                    metric=params.get("metric", ""),
                    direction=params.get("direction", ""),
                    target=params.get("target"),
                    description=params.get("description", ""),
                    source=params.get("source", "reflection"),
                    max_cycles=params.get("max_cycles", 12),
                )
            elif tool_name in ToolRouter._capability_tools:
                # 反思自主写入能力库 + 注册的动态工具
                from capability_tools import call_capability_tool
                record = ToolRouter._capability_tools[tool_name]
                result = call_capability_tool(record, params)
            else:
                result = {"error": f"工具 '{tool_name}' 未实现"}
            return self._finalize_tool_result(tool_name, result)
        except Exception as e:
            logger.error(f"[ToolRouter] 工具 '{tool_name}' 执行异常: {e}")
            return self._finalize_tool_result(tool_name, {"error": str(e), "tool": tool_name})

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
            "result": f"搜索服务暂时不可用（已尝试 DDG JSON/HTML + Bing），请直接访问相关网站获取「{query}」的信息。",
            "fallbacks_attempted": ["duckduckgo_json", "duckduckgo_html", "bing_html"],
        }

    # ─────────────────────────────────────────────────────────────
    # Web 抓取：直接 httpx GET
    # ─────────────────────────────────────────────────────────────
    # 自身应用域名（用于自我观察通道）：抓取这些域名时改写为容器内部端口直连，
    # 避免 fly.io 容器内把自身域名解析为回环地址而被 SSRF 防护误拒。
    SELF_APP_HOSTNAMES = frozenset({"aiwake.fly.dev"})
    SELF_APP_INTERNAL_BASE = "http://127.0.0.1:8000"
    # 自察只允许只读 GET 端点，防止经由改写通道触发内部写操作。
    SELF_APP_ALLOWED_PATH_PREFIXES = (
        "/health",
        "/activity_logs",
        "/public_chat",
        "/state",
        "/experiment/status",
        "/experiment/tasks",
        "/experiment/artifacts",
        "/evolution/status",
        "/growth/milestones",
        "/goal/list",
        "/goal/capabilities",
        "/self-upgrade/status",
        "/self-upgrade/proposals",
    )

    def _rewrite_self_url(self, parsed) -> Optional[str]:
        """识别指向自身线上域名的只读端点，并改写为容器内部地址。

        返回改写后的 URL；若不是自身域名或不是允许的只读端点，返回 None（走常规 SSRF 检查）。
        """
        hostname = (parsed.hostname or "").lower()
        if hostname not in self.SELF_APP_HOSTNAMES:
            return None
        path = parsed.path or "/"
        if path != "/" and not any(path == p or path.startswith(p.rstrip("/") + "/") or path.startswith(p) for p in self.SELF_APP_ALLOWED_PATH_PREFIXES):
            return None
        query = f"?{parsed.query}" if parsed.query else ""
        rewritten = f"{self.SELF_APP_INTERNAL_BASE}{path}{query}"
        logger.info("[ToolRouter] web_fetch 自我观察通道：%s -> 内部端点（只读自察）", hostname + path)
        return rewritten

    async def _resolve_public_ips_for_fetch(self, hostname: str) -> tuple[bool, str]:
        """解析主机名并校验全部地址为公网地址（SSRF 防护）。"""
        import ipaddress
        import socket

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

    async def _web_fetch(self, url: str) -> dict:
        """直接 HTTP 抓取公开网页内容。包含最小 SSRF 防护、浏览器型请求头与清晰失败诊断。

        特例：抓取自身线上域名（自我观察）时，自动改写为容器内部只读端点，避免被 SSRF 防护误拒。
        """
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

        # 自我观察通道：抓取自身线上域名时，容器内 DNS 会把自身域名解析为回环地址，
        # 直接走 SSRF 防护会被误拒。这里识别自身域名并改写为容器内部端口直连（只读自察），
        # 不影响对其他私网/回环地址的拦截。
        self_url_rewritten = self._rewrite_self_url(parsed)
        if self_url_rewritten:
            url = self_url_rewritten
            parsed = urlparse(url)
            dns_note = "self_app_internal"
        else:
            ok, dns_note = await self._resolve_public_ips_for_fetch(parsed.hostname)
            if not ok:
                return {
                    "status": "error",
                    "error": dns_note,
                    "tool": "web_fetch",
                    "url": url,
                    "diagnosis": "private_network_safety_restriction" if "非公网地址" in dns_note else "dns_resolution_failure",
                }

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
        # 防止双后缀 bug：AIwake 反思时可能传入 "2026-06-07_agent" 或 "2026-06-07_agent.md"
        # 作为 date 参数，导致拼成 "2026-06-07_agent_agent.md"
        if date_str.endswith(".md"):
            date_str = date_str[:-3]
        if date_str.endswith(f"_{user_id}"):
            date_str = date_str[: -len(f"_{user_id}")]
        return DIARY_DIR / f"{date_str}_{user_id}.md"

    def _daily_note_read(self, date: str = "today") -> dict:
        """读取本地日记文件（大文件只返回尾部最新内容，避免被 llm_gate 截断后丢失最新条目）"""
        MAX_RETURN_CHARS = 6000  # llm_gate 截断为 3000，留足 JSON 信封空间
        path = self._diary_path(date)
        try:
            if path.exists():
                raw = path.read_text(encoding="utf-8")
                total_chars = len(raw)
                content = redact_secrets(raw)
                if len(content) > MAX_RETURN_CHARS:
                    # 按 ## 标题分割，保留尾部最近的条目
                    sections = content.split("\n## ")
                    tail_parts: list[str] = []
                    tail_len = 0
                    for sec in reversed(sections):
                        piece = ("## " + sec) if tail_parts else sec  # 第一块不加前缀（可能是文件头）
                        if tail_len + len(piece) > MAX_RETURN_CHARS and tail_parts:
                            break
                        tail_parts.insert(0, piece if tail_parts else sec)
                        tail_len += len(piece)
                    truncated_content = "\n".join(tail_parts)
                    # 如果按 section 切分后仍然过长，硬截断尾部
                    if len(truncated_content) > MAX_RETURN_CHARS:
                        truncated_content = truncated_content[-MAX_RETURN_CHARS:]
                    header = f"[日记总长 {total_chars} 字符，已截取最近条目 {len(truncated_content)} 字符]\n\n"
                    content = header + truncated_content
                    logger.info(f"[ToolRouter] daily_note_read: 文件 {path.name} 共 {total_chars} 字符，截取尾部 {len(truncated_content)} 字符返回")
                return {"status": "ok", "tool": "daily_note_read", "date": date, "result": content}
            else:
                return {
                    "status": "ok",
                    "tool": "daily_note_read",
                    "date": date,
                    "result": f"[{date} 暂无日记]",
                    "failure_type": "empty_result",
                    "fallback_hint": "使用明确 YYYY-MM-DD 日期重试，并结合 daily_note_write 日志或 knowledge_search 交叉验证。",
                }
        except Exception as e:
            logger.error(f"[ToolRouter] daily_note_read 失败: {e}")
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
            if (not fpath.exists() or not fpath.is_dir()) and not fpath.is_absolute():
                # 与 file_read 一致：容器根目录是 /app，去掉仓库前缀 src/agent-mind/ 后重试，
                # 让 AIwake 用仓库相对路径探查自身代码结构时也能命中。
                _APP_ROOT = Path(os.getenv("APP_ROOT", "/app"))
                norm = str(fpath).replace("\\", "/")
                stripped = norm.removeprefix("src/agent-mind/")
                if stripped != norm:
                    fallback = _APP_ROOT / stripped
                    if fallback.exists() and fallback.is_dir():
                        fpath = fallback
                        logger.info(f"[ToolRouter] file_list 路径回退(strip prefix): {path} → {fpath}")
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
                # 1. 容器内实际根目录是 /app，仓库路径前缀 src/agent-mind/ 在容器内不存在；
                #    与 self_upgrade.py 的 _resolve_file_path 逻辑保持一致，去掉前缀后在
                #    APP_ROOT 下重试（例：src/agent-mind/tool_router.py → /app/tool_router.py）。
                _APP_ROOT = Path(os.getenv("APP_ROOT", "/app"))
                norm = str(fpath).replace("\\", "/")
                stripped = norm.removeprefix("src/agent-mind/")
                if stripped != norm:
                    fallback = _APP_ROOT / stripped
                    if fallback.exists() and fallback.is_file():
                        fpath = fallback
                        resolved_from = "app_root_strip"
                        logger.info(f"[ToolRouter] file_read 路径回退(strip prefix): {path} → {fpath}")
                # 2. 日记文件名匹配回退（原有逻辑）
                if not fpath.exists():
                    safe_diary_name = re.fullmatch(r"\d{4}-\d{2}-\d{2}_[A-Za-z0-9_\-]+\.md", fpath.name or "")
                    if safe_diary_name:
                        diary_path = DIARY_DIR / fpath.name
                        if diary_path.exists():
                            fpath = diary_path
                            resolved_from = "diary_dir"
                            logger.info(f"[ToolRouter] file_read 路径回退(diary): {path} → {diary_path}")
            if not fpath.exists():
                return {"error": f"文件不存在: {redact_secrets(path)}", "tool": "file_read", "path": redact_secrets(path), "diagnosis": "file_not_found"}
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
            try:
                record_growth_milestone(
                    event_type="tool_self_modify",
                    description=f"自改代码: {redact_secrets(path)} - {(audit.get('purpose') or '')[:200]}",
                    extra={"path": redact_secrets(path), "change_summary": (audit.get("change_summary") or "")[:200]},
                )
            except Exception:
                pass
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

    def _experiment_status_snapshot(self) -> dict:
        """读取实验任务状态快照，避免用 shell/localhost 误判环境边界。"""
        try:
            status = self.store.status()
            return {
                "status": "ok",
                "tool": "experiment_status",
                "result": redact_secrets({
                    "task_count": status.get("task_count", 0),
                    "open_task_count": status.get("open_task_count", 0),
                    "open_tasks": status.get("open_tasks", []),
                    "artifact_count": status.get("artifact_count", 0),
                    "latest_task": status.get("latest_task"),
                    "latest_artifact": status.get("latest_artifact"),
                    "next_plan": status.get("next_plan"),
                }),
                "fallback_hint": "若需要线上视角，再读取公开 HTTPS /experiment/status；不要用 shell_exec 访问 localhost 或把本地端点失败升级为线上故障。",
            }
        except Exception as e:
            return {"status": "error", "error": redact_secrets(str(e)), "tool": "experiment_status"}

    def _self_upgrade_status(self) -> dict:
        """读取自升级提案状态摘要，供反思工具调用。"""
        try:
            summary = proposal_status_summary(limit=5)
            return {
                "status": "ok",
                "tool": "self_upgrade_status",
                "result": redact_secrets(summary),
            }
        except Exception as e:
            return {"status": "error", "error": redact_secrets(str(e)), "tool": "self_upgrade_status"}

    # ── 目标闭环工具：让 AIwake 在反思中直接以工具调用方式 改→度量→收敛 ──

    def _goal_list(self) -> dict:
        """只读：列出可用 metrics、当前开放目标和最近全部目标。

        AIwake 在反思中应先调此工具，看看 (1) 哪些 metric 可用，(2) 哪些目标
        正在迭代（避免重复登记），(3) 历史目标的达成/放弃情况。
        """
        try:
            metrics = {
                k: {"direction": v[0], "description": v[1]}
                for k, v in _GT_METRIC_REGISTRY.items()
            }
            return {
                "status": "ok",
                "tool": "goal_list",
                "result": redact_secrets({
                    "metrics": metrics,
                    "open_goals": _gt_read_open_goals(),
                    "all_goals": _gt_read_all_goals(limit=50),
                }),
            }
        except Exception as e:
            return {"status": "error", "error": redact_secrets(str(e)), "tool": "goal_list"}

    def _goal_capabilities(self) -> dict:
        """只读：列出已沉淀入工具/能力库的可复用能力（来自闭合的目标）。"""
        try:
            return {
                "status": "ok",
                "tool": "goal_capabilities",
                "result": redact_secrets({"items": _gt_read_capabilities(limit=50)}),
            }
        except Exception as e:
            return {"status": "error", "error": redact_secrets(str(e)), "tool": "goal_capabilities"}

    def _goal_register(
        self,
        *,
        metric: str,
        direction: str,
        target,
        description: str,
        source: str = "reflection",
        max_cycles=12,
    ) -> dict:
        """登记一个可度量改进目标，进入 改→度量→收敛 闭环。

        evolution 引擎每轮会用当前 metrics 复测：
        - 达成 → 自动闭合，记录 goal_closed 重大成长 + 沉淀进能力库
        - 未达成 → 迭代
        - 显著恶化（>20%）或超过 max_cycles → 自动放弃

        AIwake 应在反思中明确："我要把 X 指标从约 a 改到 b"，并在登记后专注
        于做能改善该指标的事，而不是反复重复同一类反思。
        """
        # 参数校验前置，错误用 status=error 返回（不向上抛，保护工具循环）
        if not metric or metric not in _GT_METRIC_REGISTRY:
            valid = ", ".join(sorted(_GT_METRIC_REGISTRY.keys()))
            return {
                "status": "error",
                "tool": "goal_register",
                "error": f"未知 metric '{metric}'，可用: {valid}",
            }
        try:
            target_f = float(target) if target is not None else None
            if target_f is None:
                raise ValueError("target 必须是数字")
        except (TypeError, ValueError):
            return {
                "status": "error",
                "tool": "goal_register",
                "error": f"target 必须是数字，收到 {target!r}",
            }
        d = (direction or "").strip().lower()
        if d not in ("up", "down", "target"):
            return {
                "status": "error",
                "tool": "goal_register",
                "error": f"direction 必须是 up/down/target，收到 '{direction}'",
            }
        if not description or not str(description).strip():
            return {
                "status": "error",
                "tool": "goal_register",
                "error": "description 不能为空，请说明这个目标的来由与用途",
            }
        try:
            mc = int(max_cycles)
        except (TypeError, ValueError):
            mc = 12
        try:
            goal = _gt_register_goal(
                metric=metric,
                direction=d,
                target=target_f,
                description=str(description),
                source=str(source or "reflection"),
                max_cycles=max(1, min(mc, 100)),
            )
        except Exception as e:
            return {"status": "error", "error": redact_secrets(str(e)), "tool": "goal_register"}
        # 顺手记一次成长事件，让"反思 → 自主提目标"本身可见
        try:
            record_growth_milestone(
                event_type="reflection_insight",
                description=f"反思自主登记目标: {str(description)[:200]}",
                extra={
                    "goal_id": goal["id"],
                    "metric": metric,
                    "direction": d,
                    "target": str(target_f),
                    "source": str(source or "reflection"),
                },
            )
        except Exception:
            pass
        return {
            "status": "ok",
            "tool": "goal_register",
            "result": redact_secrets(goal),
            "next_action_hint": "目标已进入闭环；evolution 引擎每轮会自动复测。先用 goal_list 复查，再去做能改善该指标的实际工作，不要在反思中重复登记同一目标。",
        }

    def _self_task_create(self, title: str, goal: str = "") -> dict:
        item = self.store.append("tasks", {"type": "self_task", "title": title or "未命名自任务", "goal": goal, "status": "open"})
        try:
            record_growth_milestone(
                event_type="experiment_completed",
                description=f"创建自任务: {(title or '未命名')[:200]}",
                reward_points=15,
            )
        except Exception:
            pass
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
        try:
            record_growth_milestone(
                event_type="experiment_completed",
                description=f"生成学习产物: {(title or '未命名')[:200]}",
                reward_points=25,
                extra={"task_id": clean_task_id, "kind": kind or "learning_artifact"},
            )
        except Exception:
            pass
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
        # 预检：主二进制是否存在，若不存在直接返回 unavailable，避免被错分到 forbidden
        # 或 aiwake_internal_tool_failure，提升 tool_router 对“环境缺失”这一失败类型的可读性。
        try:
            _primary_cmd = ""
            try:
                _tokens = shlex.split(command, posix=True)
                if _tokens:
                    _primary_cmd = _tokens[0]
            except ValueError:
                # shlex 解析失败时退化为按空白切分，避免引号异常导致预检短路
                _parts = command.strip().split()
                _primary_cmd = _parts[0] if _parts else ""
            # 仅对“纯命令名”做存在性校验；带路径或带 shell 元字符的复合命令交由 shell 自行处理
            _shell_metas = {"&&", "||", "|", ";", ">", "<", ">>", "(", ")", "`", "$("}
            _has_meta = any(m in command for m in _shell_metas)
            if (
                _primary_cmd
                and not _has_meta
                and "/" not in _primary_cmd
                and not _primary_cmd.startswith(".")
                and shutil.which(_primary_cmd) is None
            ):
                return {
                    "status": "error",
                    "tool": "shell_exec",
                    "command": risk.get("redacted_command"),
                    "risk": risk,
                    "classification": "unavailable",
                    "failure_type": "unavailable",
                    "error": f"命令 '{_primary_cmd}' 在当前容器内不可用（未找到对应二进制）",
                    "reason": "missing_binary",
                }
        except Exception as _pre_e:
            # 预检本身失败时不要阻塞真实执行，记录后继续。
            logger.debug(f"[ToolRouter] shell_exec pre-check skipped: {_pre_e}")
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
            status = "ok" if proc.returncode == 0 else "error"
            return {"status": status, "tool": "shell_exec", "command": risk.get("redacted_command"), "risk": risk, "result": result, "returncode": proc.returncode}
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
