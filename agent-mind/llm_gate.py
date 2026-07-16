"""
LLM Gate - 调用门控器（原生实现，无任何第三方代理依赖）
直接调用云端 OpenAI 兼容 API 或本地 Ollama，支持 Function Calling 原生工具调用。
人格注入通过读取 agent_card.md 实现。
"""
import os
import json
import httpx
import logging
import asyncio
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


def _extract_content(message: dict) -> str:
    """从 API 响应 message 中提取纯文本 content。

    兼容两种格式：
    1. 标准 OpenAI 格式：content 为纯文本字符串
    2. 结构化格式：content 为 [{"type": "output_text", "text": "..."}] 列表
    """
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text", "")
                if text:
                    parts.append(text)
        return "\n".join(parts)
    return str(content)

# 随镜像发布的运行时人格与 AIwake 规则路径。
# 不放在 data/ 下：生产环境 /app/data 是 Fly volume，会覆盖镜像内文件。
PROMPTS_DIR = Path(__file__).parent / "runtime_prompts"
AGENT_CARD_PATH = PROMPTS_DIR / "agent_card.md"
AIWAKE_RULES_PATH = PROMPTS_DIR / "aiwake_runtime_rules.md"


def load_text_file(path: Path, label: str) -> str:
    """读取运行时注入文本。"""
    try:
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
        logger.warning(f"[LLM Gate] 未找到 {label}: {path}")
    except Exception as e:
        logger.warning(f"[LLM Gate] 读取 {label} 失败: {e}")
    return ""


def load_agent_card() -> str:
    """读取 agent_card.md 人格文件内容"""
    return load_text_file(AGENT_CARD_PATH, "agent_card.md")


def load_aiwake_rules() -> str:
    """读取 AIwake 运行时规则。"""
    return load_text_file(AIWAKE_RULES_PATH, "aiwake_runtime_rules.md")

class LLMTier(Enum):
    NONE = "none"        # 无需LLM，规则直接处理
    LOCAL = "local"      # 本地小模型（Ollama）
    CLOUD = "cloud"      # 云端大模型（OpenAI 兼容 API）


@dataclass
class ModelProfile:
    name: str
    api_url: str
    api_key: str
    model: str
    hourly_limit: int

    @property
    def configured(self) -> bool:
        return bool(self.api_url and self.api_key)


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", "disabled"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _first_env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


class LLMGate:
    def __init__(self):
        # 本地小模型配置（Ollama，可禁用）
        self.local_model_enabled = _env_bool("LOCAL_MODEL_ENABLED", True)
        self.ollama_url = os.getenv("OLLAMA_URL", "http://ollama:11434")
        self.local_model = os.getenv("LOCAL_MODEL", "qwen2.5:3b")

        # 反思空转检测：记录最近N轮反思的工具调用链
        self._reflection_tool_history: list[list[dict]] = []
        self._reflection_history_max_len = 5

        legacy_api_url = _first_env("CLOUD_API_URL")
        legacy_api_key = _first_env("CLOUD_API_KEY")
        legacy_model = _first_env("CLOUD_MODEL", default="gpt-5.5")
        legacy_limit = _env_int("HOURLY_CLOUD_LIMIT", 200)

        # 双 API 配置：work 用于用户作业/学习，reflect 用于心跳反思/记忆任务。
        # 模型默认分流：心眺/心跳反思固定走 gpt-5.4，其它云端任务默认走 gpt-5.5。
        self.profiles: dict[str, ModelProfile] = {
            "work": ModelProfile(
                name="work",
                api_url=_first_env("WORK_API_URL", default=legacy_api_url),
                api_key=_first_env("WORK_API_KEY", default=legacy_api_key),
                model=_first_env("WORK_MODEL", default=legacy_model),
                hourly_limit=_env_int("HOURLY_WORK_LIMIT", legacy_limit),
            ),
            "reflect": ModelProfile(
                name="reflect",
                api_url=_first_env("REFLECT_API_URL", "WORK_API_URL", default=legacy_api_url),
                api_key=_first_env("REFLECT_API_KEY", "WORK_API_KEY", default=legacy_api_key),
                model=_first_env("REFLECT_MODEL", default="gpt-5.5"),
                hourly_limit=_env_int("HOURLY_REFLECT_LIMIT", legacy_limit),
            ),
        }

        # 推理强度（reasoning_effort）：GPT-5 / o-series 等推理模型支持的 OpenAI 兼容字段。
        # 取值梯度：xhigh > high > medium > low > minimal > ""
        #   - "xhigh" 是本项目接入的 GPT-5 网关支持的"超高"档（标准 OpenAI 仅到 "high"，xhigh 是供应商扩展）
        #   - 空字符串 = 不注入，保持服务端默认（一键回滚开关）
        # 默认拉到 "xhigh"（=超高）：让用户作业链路与心跳反思链路都走最深的思维链。
        # 通过 env var 可一键回滚：`WORK_REASONING_EFFORT=` / `REFLECT_REASONING_EFFORT=` 留空即不再注入；
        # 也可降到 "high" / "medium" 等档位以观察推理强度对产出/成本的影响。
        self.work_reasoning_effort = os.getenv("WORK_REASONING_EFFORT", "xhigh").strip()
        self.reflect_reasoning_effort = os.getenv("REFLECT_REASONING_EFFORT", "xhigh").strip()

        # Prompt Cache：相同 profile/model 使用稳定路由 key，提高兼容网关把
        # 相同长前缀送到同一缓存节点的概率。若网关明确拒绝该标准字段，
        # 当前进程会自动移除后重试，并记住该 profile 不再发送该字段。
        self.prompt_cache_enabled = _env_bool("PROMPT_CACHE_ENABLED", True)
        self.prompt_cache_key_prefix = os.getenv(
            "PROMPT_CACHE_KEY_PREFIX", "aiwake"
        ).strip() or "aiwake"
        self._prompt_cache_key_supported = {"work": True, "reflect": True}
        self._prompt_cache_stats = {
            name: {
                "responses": 0,
                "usage_reported": 0,
                "cache_hits": 0,
                "prompt_tokens": 0,
                "cached_tokens": 0,
                "cache_write_tokens": 0,
            }
            for name in self.profiles
        }

        # 兼容旧字段，供旧调用和日志继续工作。
        self.cloud_api_url = self.profiles["work"].api_url
        self.cloud_api_key = self.profiles["work"].api_key
        self.cloud_model = self.profiles["work"].model
        self.hourly_cloud_limit = self.profiles["work"].hourly_limit
        self._cloud_calls_this_hour = 0
        self._hour_start = 0
        self._profile_call_counts = {"work": 0, "reflect": 0}
        self._profile_hour_start = {"work": 0.0, "reflect": 0.0}

        logger.info(
            "[LLM Gate] 模型配置: local=%s(%s), work=%s(reasoning_effort=%s), reflect=%s(reasoning_effort=%s)",
            "enabled" if self.local_model_enabled else "disabled",
            self.local_model,
            self.profiles["work"].model,
            self.work_reasoning_effort or "<server-default>",
            self.profiles["reflect"].model,
            self.reflect_reasoning_effort or "<server-default>",
        )

        # 加载运行时人格与 AIwake 规则（启动时读取一次，避免每次读磁盘）
        self._agent_card = load_agent_card()
        if self._agent_card:
            logger.info("[LLM Gate] AIwake 人格卡片已加载 (%d chars)", len(self._agent_card))
        else:
            logger.warning("[LLM Gate] 未找到 agent_card.md，将不注入人格")

        self._aiwake_rules = load_aiwake_rules()
        if self._aiwake_rules:
            logger.info("[LLM Gate] AIwake 运行时规则已加载 (%d chars)", len(self._aiwake_rules))
        else:
            logger.warning("[LLM Gate] 未找到 aiwake_runtime_rules.md，将不注入 AIwake 运行时规则")

    def _inject_agent_card(self, system_prompt: str) -> str:
        """将人格卡片和 AIwake 运行时规则注入到 system_prompt 前面。"""
        sections = [s for s in (self._agent_card, self._aiwake_rules, system_prompt) if s]
        return "\n\n".join(sections)

    def decide_tier(self, task_type: str, complexity: float, risk: float) -> LLMTier:
        """根据任务类型、复杂度、风险决定调用层级"""

        # 无需LLM的任务
        RULE_BASED = {"health_check", "state_decay", "queue_poll", "cache_hit", "schedule_tick"}
        if task_type in RULE_BASED:
            return LLMTier.NONE

        # 高风险强制走审批，不直接调LLM
        if risk > 0.8:
            return LLMTier.NONE  # 进审批队列，不直接产出

        # 简单任务走本地小模型
        if complexity < 0.4 and risk < 0.3:
            return LLMTier.LOCAL

        # 复杂/重要任务走云端大模型
        if complexity > 0.6 or risk > 0.5:
            return LLMTier.CLOUD

        # 中等复杂度优先本地
        return LLMTier.LOCAL

    async def call_local(self, system_prompt: str, user_message: str) -> str:
        """调用本地 Ollama 小模型（单轮对话）"""
        return await self.call_local_messages([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ])

    async def call_local_messages(self, messages: list[dict]) -> str:
        """
        调用本地 Ollama 小模型（支持多条 messages 列表）。
        用于记忆压缩/用户印象摘要等轻量任务。
        Ollama 不可用或被禁用时静默返回空串，不抛出异常，不阻塞主流程。
        """
        if not self.local_model_enabled:
            logger.info("[LLM Gate] 本地模型已禁用，跳过 Ollama 调用")
            return ""
        try:
            async with httpx.AsyncClient(timeout=180) as client:
                resp = await client.post(
                    f"{self.ollama_url}/api/chat",
                    json={
                        "model": self.local_model,
                        "messages": messages,
                        "stream": False,
                    }
                )
                resp.raise_for_status()
                return _extract_content(resp.json()["message"])
        except Exception as e:
            logger.warning(f"[LLM Gate] 本地模型调用失败: {e}")
            return ""

    def _get_profile(self, profile_name: str = "work") -> ModelProfile:
        return self.profiles.get(profile_name) or self.profiles["work"]

    def _profile_budget_available(self, profile: ModelProfile) -> bool:
        import time
        now = time.time()
        if now - self._profile_hour_start[profile.name] > 3600:
            self._profile_hour_start[profile.name] = now
            self._profile_call_counts[profile.name] = 0
        return self._profile_call_counts[profile.name] < profile.hourly_limit

    def _count_profile_call(self, profile: ModelProfile) -> None:
        self._profile_call_counts[profile.name] += 1
        if profile.name == "work":
            self._cloud_calls_this_hour = self._profile_call_counts[profile.name]
            self._hour_start = self._profile_hour_start[profile.name]

    def _reasoning_effort_for(self, profile_name: str) -> str:
        """根据 profile 返回应注入的 reasoning_effort。
        空字符串 = 不注入（保持服务端默认 / 一键回滚）。
        """
        if profile_name == "reflect":
            return self.reflect_reasoning_effort or ""
        return self.work_reasoning_effort or ""

    def _maybe_inject_reasoning(self, payload: dict, profile_name: str) -> None:
        """如果配置了 reasoning_effort，把它注入到 OpenAI 兼容 payload 中。
        不识别该字段的服务端会忽略或返回 400，调用方已有重试/回退保护。
        """
        effort = self._reasoning_effort_for(profile_name)
        if effort:
            payload["reasoning_effort"] = effort

    def _prompt_cache_key_for(self, profile: ModelProfile) -> str:
        """返回 profile/model 维度的稳定缓存路由 key，不包含用户数据。"""
        raw = f"{self.prompt_cache_key_prefix}:{profile.name}:{profile.model}"
        return "".join(ch if ch.isalnum() or ch in "-_.:" else "-" for ch in raw)[:128]

    def _payload_with_prompt_cache_key(self, payload: dict, profile: ModelProfile) -> dict:
        """复制 payload 并按兼容状态注入 prompt_cache_key。"""
        prepared = dict(payload)
        if (
            self.prompt_cache_enabled
            and self._prompt_cache_key_supported.get(profile.name, True)
            and "prompt_cache_key" not in prepared
        ):
            prepared["prompt_cache_key"] = self._prompt_cache_key_for(profile)
        return prepared

    @staticmethod
    def _prompt_cache_key_rejected(response: httpx.Response) -> bool:
        """仅在网关明确指出 prompt_cache_key 不受支持时触发兼容回退。"""
        if getattr(response, "status_code", 200) not in {400, 422}:
            return False
        try:
            body = str(getattr(response, "text", "")).lower()
        except Exception:
            return False
        return "prompt_cache_key" in body

    @staticmethod
    def _extract_prompt_cache_usage(data: dict) -> dict[str, int | bool]:
        """兼容提取 OpenAI/Responses/DeepSeek/Anthropic 风格缓存 usage。"""
        usage = data.get("usage") if isinstance(data, dict) else None
        if not isinstance(usage, dict):
            return {
                "reported": False,
                "prompt_tokens": 0,
                "cached_tokens": 0,
                "cache_write_tokens": 0,
            }

        detail_candidates = (
            usage.get("prompt_tokens_details"),
            usage.get("input_tokens_details"),
        )
        details = next((item for item in detail_candidates if isinstance(item, dict)), {})

        def _nonnegative_int(*values: object) -> int:
            for value in values:
                if isinstance(value, bool):
                    continue
                try:
                    parsed = int(value)
                except (TypeError, ValueError):
                    continue
                if parsed >= 0:
                    return parsed
            return 0

        prompt_tokens = _nonnegative_int(
            usage.get("prompt_tokens"),
            usage.get("input_tokens"),
        )
        cached_tokens = _nonnegative_int(
            details.get("cached_tokens"),
            usage.get("prompt_cache_hit_tokens"),
            usage.get("cache_read_input_tokens"),
            usage.get("cached_tokens"),
        )
        cache_write_tokens = _nonnegative_int(
            details.get("cache_write_tokens"),
            usage.get("cache_creation_input_tokens"),
            usage.get("cache_write_tokens"),
        )
        cache_fields_present = any(
            key in details
            for key in ("cached_tokens", "cache_write_tokens")
        ) or any(
            key in usage
            for key in (
                "prompt_cache_hit_tokens",
                "cache_read_input_tokens",
                "cached_tokens",
                "cache_creation_input_tokens",
                "cache_write_tokens",
            )
        )
        return {
            "reported": cache_fields_present,
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "cache_write_tokens": cache_write_tokens,
        }

    def _record_prompt_cache_usage(
        self,
        data: dict,
        profile: ModelProfile,
        label: str = "",
    ) -> None:
        """累计并记录缓存读写指标，区分未命中与供应商未返回 usage。"""
        usage = self._extract_prompt_cache_usage(data)
        stats = self._prompt_cache_stats.setdefault(
            profile.name,
            {
                "responses": 0,
                "usage_reported": 0,
                "cache_hits": 0,
                "prompt_tokens": 0,
                "cached_tokens": 0,
                "cache_write_tokens": 0,
            },
        )
        stats["responses"] += 1
        if not usage["reported"]:
            logger.info(
                "[LLM Gate] Prompt Cache usage 未上报: profile=%s model=%s label=%s",
                profile.name,
                profile.model,
                label or "<none>",
            )
            return

        prompt_tokens = int(usage["prompt_tokens"])
        cached_tokens = int(usage["cached_tokens"])
        cache_write_tokens = int(usage["cache_write_tokens"])
        stats["usage_reported"] += 1
        stats["prompt_tokens"] += prompt_tokens
        stats["cached_tokens"] += cached_tokens
        stats["cache_write_tokens"] += cache_write_tokens
        if cached_tokens > 0:
            stats["cache_hits"] += 1
        hit_rate = cached_tokens / prompt_tokens if prompt_tokens > 0 else 0.0
        logger.info(
            "[LLM Gate] Prompt Cache: profile=%s model=%s hit=%s "
            "cached_tokens=%d cache_write_tokens=%d prompt_tokens=%d hit_rate=%.2f%% label=%s",
            profile.name,
            profile.model,
            cached_tokens > 0,
            cached_tokens,
            cache_write_tokens,
            prompt_tokens,
            hit_rate * 100,
            label or "<none>",
        )

    def prompt_cache_status(self) -> dict[str, dict[str, int]]:
        """返回当前进程缓存 usage 累计快照，供诊断与测试读取。"""
        return {name: dict(stats) for name, stats in self._prompt_cache_stats.items()}

    async def call_cloud(
        self,
        system_prompt: str,
        user_message: str,
        profile_name: str = "work",
        max_tokens_override: int | None = None,
    ) -> str:
        """调用云端 OpenAI 兼容接口，默认使用工作/学习模型。

        ``max_tokens_override`` 仅在传入正整数时写入请求；默认不传时保持
        历史 payload 不变。自升级等需要完整结构化输出的调用可据此显式声明预算。
        """
        profile = self._get_profile(profile_name)
        if not profile.configured:
            logger.warning(f"[LLM Gate] {profile.name} API 未配置")
            return ""
        full_system = self._inject_agent_card(system_prompt)
        payload = {
            "model": profile.model,
            "messages": [
                {"role": "system", "content": full_system},
                {"role": "user", "content": user_message},
            ],
        }
        if isinstance(max_tokens_override, int) and max_tokens_override > 0:
            payload["max_tokens"] = max_tokens_override
        self._maybe_inject_reasoning(payload, profile.name)
        data = await self._cloud_post_with_retry(
            payload=payload,
            label="call_cloud",
            max_attempts=1,
            profile_name=profile.name,
        )
        if not data:
            return ""
        try:
            return _extract_content(data["choices"][0]["message"])
        except Exception as e:
            logger.error(f"[LLM Gate] {profile.name} API 响应解析失败: {e}")
            return ""

    def _work_profile_fallback_allowed(self, profile: ModelProfile) -> bool:
        """反思模型不可用时，允许安全回退到 work profile，避免自主反思闭环停摆。"""
        if profile.name != "reflect":
            return False
        work = self.profiles.get("work")
        return bool(work and work.configured and work.name != profile.name)

    async def call_reflect_messages(
        self,
        messages: list[dict],
        *,
        tools_schema: list[dict] | None = None,
        tool_executor=None,
        max_rounds: int = 8,
    ) -> str:
        """调用心跳/反思模型；反思 profile 失败时安全回退 work，再回退本地模型。

        如果传入 ``tools_schema`` + ``tool_executor``，反思将走原生 Function Calling 路径
        （call_with_tools），让反思 LLM 能像 /chat 一样真实调用工具（goal_register、
        self_code_write 等）。否则走旧的纯文本路径作为兜底。
        """
        profile = self._get_profile("reflect")
        if profile.configured:
            system_prompt = ""
            payload_messages = messages
            if messages and messages[0].get("role") == "system":
                system_prompt = messages[0].get("content", "")
                payload_messages = messages[1:]
            # 真工具路径：反思层带 tools 调用，max_rounds 限制工具循环
            if tools_schema and tool_executor:
                try:
                    result, _tier = await self.call_with_tools(
                        system_prompt=system_prompt,
                        messages=payload_messages,
                        tools_schema=tools_schema,
                        tool_executor=tool_executor,
                        user_id="evolution_reflect",
                        max_rounds=max_rounds,
                        profile_name="reflect",
                    )
                    if result:
                        return result
                    logger.warning("[LLM Gate] reflect 真工具路径无结果，回退到无工具反思")
                except Exception as e:
                    logger.warning(f"[LLM Gate] reflect 真工具路径异常: {e}，回退到无工具反思")
            # 兜底：无工具反思（兼容旧调用方与失败回退）
            user_message = "\n\n".join(m.get("content", "") for m in payload_messages if m.get("role") == "user")
            result = await self.call_cloud(system_prompt, user_message, profile_name="reflect")
            if result:
                return result
            if self._work_profile_fallback_allowed(profile):
                logger.warning("[LLM Gate] reflect profile 无有效结果，安全回退到 work profile 执行反思消息")
                result = await self.call_cloud(system_prompt, user_message, profile_name="work")
                if result:
                    return result
        return await self.call_local_messages(messages)

    async def _cloud_post_with_retry(
        self,
        payload: dict,
        label: str = "",
        max_attempts: int = 2,
        retry_delay: float = 3.0,
        profile_name: str = "work",
    ) -> dict | None:
        """
        带自动重试的云端 API 调用（最多 max_attempts 次，失败等 retry_delay 秒再试）。
        成功返回 response JSON，失败返回 None。
        """
        profile = self._get_profile(profile_name)
        if not profile.configured:
            logger.warning(f"[LLM Gate] {profile.name} API 未配置")
            return None

        for attempt in range(max_attempts):
            log_label = f"{label} (第{attempt+1}次)" if label else f"第{attempt+1}次"
            try:
                if not self._profile_budget_available(profile):
                    logger.warning(f"[LLM Gate] {profile.name} API 调用已达本小时上限")
                    return None
                logger.info(f"[LLM Gate] {profile.name} API 调用 {log_label}，model={profile.model}")
                request_payload = self._payload_with_prompt_cache_key(payload, profile)
                async with httpx.AsyncClient(timeout=120) as client:
                    resp = await client.post(
                        f"{profile.api_url.rstrip('/')}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {profile.api_key}",
                            "Content-Type": "application/json",
                        },
                        json=request_payload,
                    )
                    if (
                        "prompt_cache_key" in request_payload
                        and self._prompt_cache_key_rejected(resp)
                    ):
                        self._prompt_cache_key_supported[profile.name] = False
                        logger.warning(
                            "[LLM Gate] %s API 不支持 prompt_cache_key，已自动移除并兼容重试",
                            profile.name,
                        )
                        fallback_payload = dict(request_payload)
                        fallback_payload.pop("prompt_cache_key", None)
                        resp = await client.post(
                            f"{profile.api_url.rstrip('/')}/chat/completions",
                            headers={
                                "Authorization": f"Bearer {profile.api_key}",
                                "Content-Type": "application/json",
                            },
                            json=fallback_payload,
                        )
                    resp.raise_for_status()
                    data = resp.json()
                    self._count_profile_call(profile)
                    self._record_prompt_cache_usage(data, profile, log_label)
                    return data
            except Exception as e:
                logger.warning(f"[LLM Gate] {profile.name} API 调用失败 {log_label}: {e}")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(retry_delay)
        return None

    async def call_via_vcp(
        self,
        system_prompt: str,
        messages: list[dict],
        user_id: str = "default",
        profile_name: str = "work",
    ) -> tuple[str, "LLMTier"]:
        """
        原 VCP 代理对话入口（已重构为直接调用云端 API）。
        人格注入通过 agent_card.md 直接拼入 system_prompt 实现。
        接口签名保持不变，以兼容 heartbeat.py / main.py 的调用方。
        返回 (回复内容, 实际使用的层级)。
        """
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")

        # 构建注入了人格卡片的完整 system_prompt
        full_system = self._inject_agent_card(system_prompt)

        # 优先尝试直连指定云端模型（携带完整对话历史）
        profile = self._get_profile(profile_name)
        if profile.configured:
            full_messages = [{"role": "system", "content": full_system}] + messages
            vcp_payload = {
                "model": profile.model,
                "messages": full_messages,
                "user": user_id,
                "stream": False,
            }
            self._maybe_inject_reasoning(vcp_payload, profile.name)
            data = await self._cloud_post_with_retry(
                payload=vcp_payload,
                label="call_via_vcp",
                profile_name=profile.name,
            )
            if data:
                result = _extract_content(data["choices"][0]["message"])
                if result:
                    logger.info(f"[LLM Gate] {profile.name} API 调用成功（AIwake 人格已注入）")
                    return result, LLMTier.CLOUD

        # 降级：本地 Ollama
        logger.warning("[LLM Gate] 降级调用本地 Ollama 模型")
        result = await self.call_local(full_system, last_user)
        if result:
            return result, LLMTier.LOCAL

        logger.error("[LLM Gate] 远端与本地模型均未返回结果")
        return "", LLMTier.NONE

    async def call_with_tools(
        self,
        system_prompt: str,
        messages: list[dict],
        tools_schema: list[dict],
        tool_executor,
        user_id: str = "default",
        max_rounds: int = 8,
        profile_name: str = "work",
        max_tokens_override: int | None = None,
    ) -> tuple[str, "LLMTier"]:
        """
        OpenAI Function Calling 原生工具调用循环。
        - tools_schema: OpenAI 格式工具定义数组（来自 ToolRouter.get_openai_tools_schema()）
        - tool_executor: async callable(tool_name: str, params: dict) -> dict
        - 支持多轮工具调用（最多 max_rounds 轮）
        - max_tokens_override: 可选的 max_tokens 物理上限（用于 vitality 环境压力 token 收缩，
          仅作用于本次工具调用循环；不传则保持原行为不写 max_tokens）
        - 返回 (最终文本回复, LLMTier)
        """
        profile = self._get_profile(profile_name)
        if not profile.configured:
            # 无指定云端 API，回退到普通调用
            return await self.call_via_vcp(system_prompt, messages, user_id, profile_name=profile.name)

        full_system = self._inject_agent_card(system_prompt)
        working_messages = [{"role": "system", "content": full_system}] + list(messages)

        for round_num in range(max_rounds):
            if not self._profile_budget_available(profile):
                logger.warning(f"[LLM Gate] {profile.name} API 调用已达本小时上限，停止工具循环")
                break

            payload = {
                "model": profile.model,
                "messages": working_messages,
                "user": user_id,
                "stream": False,
            }
            # 推理强度：work / reflect 各自独立的 reasoning_effort 配置
            self._maybe_inject_reasoning(payload, profile.name)
            # M-002 环境压力：物理截断反思 token 上限，让 LLM 无法继续"写日记"
            if isinstance(max_tokens_override, int) and max_tokens_override > 0:
                payload["max_tokens"] = max_tokens_override
            # 只在有工具定义时传入 tools 参数
            if tools_schema:
                payload["tools"] = tools_schema
                payload["tool_choice"] = "auto"

            data = await self._cloud_post_with_retry(
                payload=payload,
                label=f"call_with_tools 第{round_num+1}轮",
                profile_name=profile.name,
            )
            if data is None and self._work_profile_fallback_allowed(profile):
                work_profile = self.profiles["work"]
                fallback_payload = dict(payload)
                fallback_payload["model"] = work_profile.model
                # 回退到 work profile 时，按 work 的 reasoning_effort 重新注入
                fallback_payload.pop("reasoning_effort", None)
                self._maybe_inject_reasoning(fallback_payload, "work")
                logger.warning(
                    "[LLM Gate] reflect profile 工具循环失败，安全回退到 work profile 继续自主反思"
                )
                data = await self._cloud_post_with_retry(
                    payload=fallback_payload,
                    label=f"call_with_tools reflect->work fallback 第{round_num+1}轮",
                    profile_name="work",
                )

            if data is None:
                logger.warning(f"[LLM Gate] call_with_tools 第{round_num+1}轮重试均失败，退出循环")
                break

            try:
                choice = data["choices"][0]
                finish_reason = choice.get("finish_reason", "stop")
                message = choice["message"]

                if finish_reason == "tool_calls" and "tool_calls" in message:
                    # 执行所有工具调用
                    working_messages.append(message)  # assistant 消息（含 tool_calls）
                    tool_calls = message["tool_calls"]
                    logger.info(f"[LLM Gate] LLM 请求调用 {len(tool_calls)} 个工具")

                    for tc in tool_calls:
                        tc_id = tc["id"]
                        fn_name = tc["function"]["name"]
                        fn_args_str = tc["function"].get("arguments", "{}")
                        try:
                            import json as _json
                            fn_args = _json.loads(fn_args_str)
                        except Exception:
                            fn_args = {}

                        logger.info(f"[LLM Gate] 执行工具: {fn_name}({fn_args})")
                        try:
                            result_dict = await tool_executor(fn_name, fn_args)
                        except Exception as e:
                            result_dict = {"status": "error", "error": f"工具调用失败: {e}", "tool": fn_name}
                            logger.error(f"[LLM Gate] 工具 {fn_name} 执行异常: {e}")

                        # 将完整工具结果注入 messages（role=tool）。
                        # 不能只注入 result 字段：否则工具返回 error 时会退化成空内容，
                        # 模型无法区分“工具不可用”和“单个来源/参数失败”，容易反复报告工具失败。
                        try:
                            result_content = json.dumps(result_dict, ensure_ascii=False, default=str)
                        except Exception:
                            result_content = str(result_dict)
                        working_messages.append({
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": result_content[:3000],  # 避免单条结果过长
                        })
                    # 继续下一轮 LLM 推理；如果已到最后一轮，下面会追加一次无工具总结。
                    continue

                else:
                    # finish_reason == "stop"，普通文本回复
                    result = _extract_content(message)
                    # ── DSML 解析：LLM 可能在文本中用 ｜DSML｜ 格式输出工具调用，
                    #    但没有通过 Function Calling 协议。解析这些标签并执行工具，
                    #    避免"想调用但没调用"的空转。 ──
                    if result and tool_executor and round_num < max_rounds - 1:
                        try:
                            from dsml_parser import parse_dsml_tool_calls
                            dsml_calls = parse_dsml_tool_calls(result)
                            if dsml_calls:
                                logger.info(f"[LLM Gate] 从文本中解析到 {len(dsml_calls)} 个 DSML 工具调用")
                                working_messages.append({"role": "assistant", "content": result})
                                for dc in dsml_calls:
                                    fn_name = dc.get("name", "")
                                    fn_args = dc.get("arguments", {})
                                    logger.info(f"[LLM Gate] DSML 执行工具: {fn_name}({str(fn_args)[:80]})")
                                    try:
                                        result_dict = await tool_executor(fn_name, fn_args)
                                    except Exception as e:
                                        result_dict = {"status": "error", "error": f"DSML工具调用失败: {e}", "tool": fn_name}
                                        logger.error(f"[LLM Gate] DSML 工具 {fn_name} 执行异常: {e}")
                                    try:
                                        result_content = json.dumps(result_dict, ensure_ascii=False, default=str)
                                    except Exception:
                                        result_content = str(result_dict)
                                    working_messages.append({
                                        "role": "tool",
                                        "tool_call_id": f"dsml_{round_num}_{fn_name}",
                                        "content": result_content[:3000],
                                    })
                                continue  # 继续下一轮 LLM 推理
                        except ImportError:
                            pass
                        except Exception as e:
                            logger.warning(f"[LLM Gate] DSML 解析失败: {e}")
                    if result:
                        logger.info(f"[LLM Gate] call_with_tools 完成（{round_num+1}轮），tier=cloud")
                        return result, LLMTier.CLOUD
                    break

            except Exception as e:
                logger.warning(f"[LLM Gate] call_with_tools 第{round_num+1}轮解析异常: {e}")
                break

        # 工具调用轮次耗尽时，不丢弃已完成的工具结果：追加一次无工具最终总结。
        # 这能避免 agent 在拿到部分可用证据后仍因 max_rounds 结束而回退成“工具不可用”。
        if any(m.get("role") == "tool" for m in working_messages):
            final_messages = working_messages + [{
                "role": "user",
                "content": (
                    "请基于以上工具返回的 JSON 结果给出最终回复。"
                    "如果某个工具或网页失败，请明确说这是单个调用失败，并尽量使用其它已成功工具结果；"
                    "不要笼统宣称工具不可用。"
                ),
            }]
            final_payload = {
                "model": profile.model,
                "messages": final_messages,
                "user": user_id,
                "stream": False,
            }
            self._maybe_inject_reasoning(final_payload, profile.name)
            data = await self._cloud_post_with_retry(
                payload=final_payload,
                label="call_with_tools 最终无工具总结",
                profile_name=profile.name,
                max_attempts=1,
            )
            if data is not None:
                try:
                    result = _extract_content(data["choices"][0]["message"])
                    if result:
                        logger.info("[LLM Gate] call_with_tools 工具结果总结完成，tier=cloud")
                        return result, LLMTier.CLOUD
                except Exception as e:
                    logger.warning(f"[LLM Gate] call_with_tools 工具结果总结解析异常: {e}")

        # 云端失败/超限，回退到普通调用；reflect 失败时优先回退 work，避免心跳反思完全停摆。
        logger.warning("[LLM Gate] call_with_tools 未得到有效回复，回退到普通调用")
        if self._work_profile_fallback_allowed(profile):
            logger.warning("[LLM Gate] reflect profile 普通回退仍失败，改用 work profile 普通调用")
            result, tier = await self.call_via_vcp(system_prompt, messages, user_id, profile_name="work")
            if result:
                return result, tier
        return await self.call_via_vcp(system_prompt, messages, user_id, profile_name=profile.name)

    async def think(
        self,
        system_prompt: str,
        user_message: str,
        task_type: str = "general",
        complexity: float = 0.5,
        risk: float = 0.1,
        profile_name: str = "work",
    ) -> tuple[str, LLMTier]:
        """
        统一思考入口（内部反思/工具任务使用）。
        返回 (回复内容, 实际使用的层级)
        """
        tier = self.decide_tier(task_type, complexity, risk)
        if tier == LLMTier.NONE:
            return "", tier
        elif tier == LLMTier.LOCAL:
            result = await self.call_local(system_prompt, user_message)
            if not result:
                # 本地失败，降级到云端
                logger.info("[LLM Gate] 本地模型无结果，降级到云端")
                result = await self.call_cloud(system_prompt, user_message, profile_name=profile_name)
                return result, LLMTier.CLOUD
            return result, tier
        else:
            result = await self.call_cloud(system_prompt, user_message, profile_name=profile_name)
            return result, tier
