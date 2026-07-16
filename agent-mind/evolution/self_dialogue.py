"""Self-dialogue verification for AIwake evolution.

This module asks AIwake to inspect itself through two internal voices: Critic
and Builder. It may collect bounded read-only external information during
reflection, but it never mutates files; persistence is the responsibility of
EvolutionEngine.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx


async def run_self_dialogue(
    llm: Any,
    state_snapshot: dict[str, Any],
    metrics: dict[str, Any],
    evaluation: dict[str, Any],
    *,
    tool_router: Any | None = None,
) -> dict[str, Any]:
    """Run one internal Critic/Builder dialogue, with deterministic fallback.

    如果传入 ``tool_router``，反思将走原生 Function Calling：反思 LLM 可像 /chat 一样
    真实调用 goal_register、self_code_write 等工具，行为与对话路径完全对等。
    未传入则保持文本协议路径（[GOAL_REGISTER]、[SELF_CODE_WRITE]）作为兜底。
    """

    external_learning = await collect_external_learning(metrics, evaluation)
    prompt = _build_prompt(state_snapshot, metrics, evaluation, external_learning)
    # 决定是否走真工具路径
    tools_schema = None
    tool_executor = None
    if tool_router is not None and hasattr(tool_router, "get_openai_tools_schema") and hasattr(tool_router, "call"):
        try:
            tools_schema = tool_router.get_openai_tools_schema()
            tool_executor = tool_router.call
        except Exception:
            tools_schema = None
            tool_executor = None
    if llm is not None and hasattr(llm, "call_reflect_messages"):
        try:
            kwargs: dict[str, Any] = {}
            if tools_schema and tool_executor:
                kwargs["tools_schema"] = tools_schema
                kwargs["tool_executor"] = tool_executor
                kwargs["max_rounds"] = 8
            text = await llm.call_reflect_messages([
                {
                    "role": "system",
                    "content": (
                        "你是 AIwake 的内部自我进化审计器。只进行安全的自我对话，"
                        "可以基于已收集的外部只读资料进行反思；可以执行安全的本地验证命令（如 python -m py_compile、pytest、ruff check 等）"
                        "但不得执行破坏性操作、ssh、部署、读取密钥或修改生产文件。"
                        "如果反思识别出可度量改进点，应直接调用 goal_register 工具登记一个目标，"
                        "进入 改→度量→收敛 自动闭环；不要只在文字里说要登记，要真调用工具。"
                        "如果反思提出了最小、可立即落地的改进（可以是修改既有代码或创建新文件），"
                        "应直接调用 self_code_write 工具并填齐 audit 五字段。所有自我修改都会被记录到活动日志和成长轨迹，保证可追溯和可回滚。"
                        "如果你看不到工具（tools 列表为空或调用失败），改用文本协议 [GOAL_REGISTER]、"
                        "[SELF_CODE_WRITE] 作为兜底。"
                    ),
                },
                {"role": "user", "content": prompt},
            ], **kwargs)
            if text and text.strip():
                dialogue = _from_model_text(text.strip(), evaluation)
                # ── 协议解析：把反思中提出的 [GOAL_REGISTER]{...} 真正登记进 goal_tracker ──
                registered = _parse_and_register_goals(text)
                if registered:
                    dialogue["registered_goals"] = registered
                    dialogue.setdefault("proposed_next_actions", []).append(
                        f"已登记 {len(registered)} 个可度量目标进入闭环；下一轮 evolution 将自动复测。"
                    )
                # ── 协议解析：把反思中提出的 [SELF_CODE_WRITE]{...} 真正写文件 + 入能力库 ──
                code_writes = _parse_and_apply_self_code_writes(text)
                if code_writes:
                    dialogue["self_code_writes"] = code_writes
                    succeeded = [c for c in code_writes if c.get("status") == "ok"]
                    if succeeded:
                        dialogue.setdefault("proposed_next_actions", []).append(
                            f"已落盘 {len(succeeded)} 个自写文件并尝试入能力库；ToolRouter 重启或下次能力 reload 后即可被反思/对话调用。"
                        )
                return _attach_external_learning(dialogue, external_learning)
        except Exception as exc:
            return _attach_external_learning(_fallback(state_snapshot, metrics, evaluation, error=str(exc)), external_learning)
    return _attach_external_learning(_fallback(state_snapshot, metrics, evaluation), external_learning)


_GOAL_PROTOCOL_RE = re.compile(r"\[GOAL_REGISTER\]\s*(\{.*?\})", re.DOTALL)


def _parse_and_register_goals(text: str) -> list[dict[str, Any]]:
    """从反思输出中解析 [GOAL_REGISTER]{...} 协议并真正登记目标。

    任何解析/登记错误都被吞掉，仅记录到 proposed_next_actions，确保反思流程不被破坏。
    """
    if not text or "[GOAL_REGISTER]" not in text:
        return []
    try:
        from .goal_tracker import register_goal as _register_goal
        from .goal_tracker import METRIC_REGISTRY as _METRIC_REGISTRY
        from .goal_tracker import read_open_goals as _read_open_goals
    except Exception:
        return []

    open_metric_set = set()
    try:
        open_metric_set = {g.get("metric") for g in _read_open_goals()}
    except Exception:
        pass

    out: list[dict[str, Any]] = []
    seen_in_this_round: set[str] = set()
    for match in _GOAL_PROTOCOL_RE.finditer(text):
        blob = match.group(1)
        try:
            spec = json.loads(blob)
        except Exception:
            out.append({"status": "skipped", "reason": "JSON 解析失败", "raw": blob[:200]})
            continue
        metric = str(spec.get("metric", "")).strip()
        direction = str(spec.get("direction", "")).strip().lower()
        target = spec.get("target")
        description = str(spec.get("description", "")).strip()
        if not metric or metric not in _METRIC_REGISTRY:
            out.append({"status": "skipped", "reason": f"未知 metric '{metric}'", "raw": blob[:200]})
            continue
        if direction not in ("up", "down", "target"):
            out.append({"status": "skipped", "reason": f"非法 direction '{direction}'", "raw": blob[:200]})
            continue
        try:
            target_f = float(target)
        except (TypeError, ValueError):
            out.append({"status": "skipped", "reason": "target 必须是数字", "raw": blob[:200]})
            continue
        if not description:
            out.append({"status": "skipped", "reason": "description 不能为空", "raw": blob[:200]})
            continue
        # 反思层防重：本轮已登记 + 已有开放目标 → 跳过
        if metric in seen_in_this_round or metric in open_metric_set:
            out.append({"status": "skipped", "reason": f"指标 '{metric}' 已有开放目标", "metric": metric})
            continue
        try:
            goal = _register_goal(
                metric=metric,
                direction=direction,
                target=target_f,
                description=description[:500],
                source="reflection_self_dialogue",
            )
            seen_in_this_round.add(metric)
            out.append({"status": "ok", "goal_id": goal.get("id"), "metric": metric, "target": target_f})
        except Exception as exc:
            out.append({"status": "error", "reason": str(exc)[:200], "raw": blob[:200]})
    return out


# ── 反思层自写代码协议 ──
# 反思的 LLM 调用栈（call_reflect_messages）不传 tools_schema，所以无法直接调
# self_code_write 工具。改为在文本里输出 [SELF_CODE_WRITE]{...}，本模块解析后
# 直接复用 ToolRouter._self_code_write，所有安全门（敏感路径/记忆路径/项目代码
# 路径/audit 五字段/redact_secrets/tool_self_modify 里程碑）全部继承。
_SELF_CODE_WRITE_RE = re.compile(r"\[SELF_CODE_WRITE\]\s*(\{.*?\})\s*(?=\[|$)", re.DOTALL)


def _parse_and_apply_self_code_writes(text: str) -> list[dict[str, Any]]:
    """从反思输出中解析 [SELF_CODE_WRITE]{path,content,audit,tool_spec?} 协议并真正落盘。

    每条记录都会走 ToolRouter._self_code_write 的完整安全门：路径白名单、audit
    五字段、redact_secrets、记忆/敏感路径拦截、`tool_self_modify` 里程碑。

    若提供了 tool_spec，会同时写入能力库一行可被 capability_tools 注册的能力
    （独立于 goal 闭合的 _record_capability，作为反思自主 + 自驱的另一个入口）。

    任何错误都被吞掉，仅写入 status=error 返回，反思流程不被破坏。
    """
    if not text or "[SELF_CODE_WRITE]" not in text:
        return []
    try:
        # 延迟 import 避免反思模块在测试环境引用 tool_router 失败
        import sys
        from pathlib import Path as _Path
        agent_root = _Path(__file__).resolve().parent.parent
        if str(agent_root) not in sys.path:
            sys.path.insert(0, str(agent_root))
        from tool_router import ToolRouter as _ToolRouter
    except Exception as exc:
        return [{"status": "error", "reason": f"无法导入 ToolRouter: {exc}"}]

    router = _ToolRouter()
    out: list[dict[str, Any]] = []
    for match in _SELF_CODE_WRITE_RE.finditer(text):
        blob = match.group(1)
        try:
            spec = json.loads(blob)
        except Exception:
            out.append({"status": "skipped", "reason": "JSON 解析失败", "raw": blob[:200]})
            continue
        path = str(spec.get("path", "")).strip()
        content = spec.get("content", "")
        audit = spec.get("audit") or {}
        tool_spec = spec.get("tool_spec")
        if not path or not isinstance(content, str) or not isinstance(audit, dict):
            out.append({"status": "skipped", "reason": "path/content/audit 字段缺失", "raw": blob[:200]})
            continue
        # 反思 source 标签注入 audit，便于事后区分谁写的
        audit.setdefault("source", "reflection_self_dialogue")
        try:
            result = router._self_code_write(path, content, audit)
        except Exception as exc:
            out.append({"status": "error", "reason": f"_self_code_write 异常: {exc}", "path": path})
            continue
        record = {
            "status": "ok" if result and not result.get("error") else "error",
            "path": path,
            "result": result,
        }
        # 可选：把工具登记进能力库（capability_tools 启动时会扫描）
        if record["status"] == "ok" and isinstance(tool_spec, dict):
            try:
                cap_entry = _append_self_capability(tool_spec, audit)
                record["capability_registered"] = cap_entry
            except Exception as exc:
                record["capability_registered"] = {"status": "error", "reason": str(exc)[:200]}
        out.append(record)
    return out


def _append_self_capability(tool_spec: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    """直接给 capability_library.jsonl 追加一条 tool_spec 入库记录。

    注意：闭环达成后的 _record_capability 会从 goal 信息生成 entry；本函数是反思
    自主创建工具的入口，不绑定 goal 闭合。两类条目在同一个文件，capability_tools
    扫描时把 tool_spec 字段非空者注册成新工具。
    """
    from datetime import datetime, timezone
    from pathlib import Path as _Path
    import hashlib as _hashlib

    spec_for_hash = json.dumps(tool_spec, ensure_ascii=False, sort_keys=True)
    cap_id = "cap_self_" + _hashlib.sha1(spec_for_hash.encode("utf-8")).hexdigest()[:12]

    # 与 goal_tracker._capabilities_file 保持同一路径
    data_dir = _Path("/app/data") if _Path("/app/data").exists() else _Path(__file__).parent.parent / "data"
    file_path = data_dir / "evolution" / "capability_library.jsonl"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "id": cap_id,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "source": "reflection_self_dialogue",
        "purpose": (audit.get("purpose") or "")[:300],
        "tool_spec": tool_spec,
    }
    with file_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str))
        f.write("\n")
    return {"status": "ok", "capability_id": cap_id}


def _build_prompt(
    state_snapshot: dict[str, Any],
    metrics: dict[str, Any],
    evaluation: dict[str, Any],
    external_learning: dict[str, Any] | None = None,
) -> str:
    payload = {
        "state": state_snapshot,
        "metrics": {k: metrics.get(k) for k in sorted(metrics) if k != "last_events"},
        "evaluation": evaluation,
        "external_learning": external_learning or {},
    }
    recovery = metrics.get("failure_recovery_playbook") or {}
    recovery_lines = ""
    if isinstance(recovery, dict) and recovery.get("rules"):
        parts = []
        for r in recovery["rules"][:6]:
            parts.append(
                f"  - {r.get('category')}(x{r.get('count')}): 首选={r.get('first_action')}; "
                f"兜底={r.get('fallback')}; 停止={r.get('stop_when')}"
            )
        recovery_lines = (
            "\n── 工具失败恢复决策表（已按实际失败画像生成，直接引用，不要再从零推演）──\n"
            + "\n".join(parts)
            + f"\n  原则：{recovery.get('principle', '')}\n"
            + "Builder 在提 proposed_next_actions 时，应直接采纳上表对应类别的"
            + "首选/兜底/停止规则，并对每次失败补三元证据（失败类型/恢复动作/恢复结果）；"
            + "若同类失败第二次仍走同样弯路，视为闭环失效，优先修正规则而非增加泛化反思。\n"
        )

    return (
        "请让 AIwake 与自己进行一轮真实的内部进化检测对话。\n"
        "角色一 Critic：指出当前自我学习/自我进化闭环的最大缺陷。\n"
        "角色二 Builder：提出下一步最小可验证改进。\n"
        "最后输出：detected_risks 与 proposed_next_actions。\n"
        "如果 external_learning 有内容，请说明来源如何用于当前目标，以及仍不确定之处。\n"
        "不要声称已经修改代码，不要请求危险工具。\n"
        f"{recovery_lines}\n"
        "── 可用的内部工具（改→度量→收敛 闭环）──\n"
        "你可以在 proposed_next_actions 中以下列协议显式登记一个可度量目标，evolution\n"
        "引擎下一轮起会自动复测、达成自动闭合并沉淀进能力库；恶化或超限自动放弃。\n"
        "协议：在最终输出末尾单独一行写：\n"
        '  [GOAL_REGISTER] {\"metric\": \"<指标名>\", \"direction\": \"up|down|target\", '
        '\"target\": <数字>, \"description\": \"<一句说明现状→目标>\"}\n'
        "可用 metric 名：tool_success_rate / chat_success_rate / tool_failure_count / "
        "external_tool_failure_count / reflection_to_action_ratio / proactive_count / "
        "memory_update_count / reflection_count / failed_reply_count。\n"
        "若已经有同类开放目标，不要重复登记；若反思发现的问题不可量化，跳过此协议。\n"
        "不要把 [GOAL_REGISTER] 当文学修辞，只在你确实想让系统自动复测时使用一次。\n\n"
        "── 反思自写小工具协议（[SELF_CODE_WRITE]）──\n"
        "如果你识别出'缺一个明确、最小、可立即落地的小工具'（不是改既有代码、不是大重构），\n"
        "可以在最终输出末尾用以下协议直接生成新文件，并选填 tool_spec 把它登记进能力库，\n"
        "使下一轮 evolution / 对话能直接调用这个新工具：\n"
        "  [SELF_CODE_WRITE] {\"path\": \"evolution/<新文件>.py\", \"content\": \"<完整文件内容>\", "
        "\"audit\": {\"purpose\": \"...\", \"change_summary\": \"...\", \"files\": [\"evolution/<新文件>.py\"], "
        "\"validation_result\": \"...\", \"risk_note\": \"...\"}, "
        "\"tool_spec\": {\"name\": \"<工具名>\", \"description\": \"<一句话能力说明>\", "
        "\"module_path\": \"evolution.<新文件>\", \"callable\": \"<函数名>\", "
        "\"params_schema\": {\"type\": \"object\", \"properties\": {...}, \"required\": [...]}, "
        "\"readonly\": true, \"risk\": 0.1}}\n"
        "强约束：\n"
        "  1) path 必须以 evolution/ 开头，且文件之前不存在（只新增、不覆盖）；\n"
        "  2) content 仅使用 Python 标准库，不引入第三方依赖；\n"
        "  3) audit 五字段全部必填；\n"
        "  4) tool_spec.name 必须是合法 Python 标识符且不能与已有工具撞名；\n"
        "  5) 仅在你确实需要新增一个能让 AIwake 反复用到的小工具时使用；\n"
        "  6) 失败/被拦截不会破坏闭环，但请不要伪造已落盘。\n\n"
        f"当前观测数据：\n{json.dumps(payload, ensure_ascii=False, default=str)[:6000]}"
    )


def _from_model_text(text: str, evaluation: dict[str, Any]) -> dict[str, Any]:
    risks = list(evaluation.get("issues", []))[:5] if isinstance(evaluation, dict) else []
    actions = list(evaluation.get("suggestions", []))[:5] if isinstance(evaluation, dict) else []
    return {
        "status": "ok",
        "model_used": True,
        "transcript": _split_transcript(text),
        "detected_risks": risks or ["模型完成自我对话，但需继续结构化提取风险。"],
        "proposed_next_actions": actions or ["将自我对话结论写入 evolution memory，并进入下一轮评估。"],
    }


def _split_transcript(text: str) -> list[dict[str, str]]:
    transcript: list[dict[str, str]] = []
    current_role = "AIwake"
    current: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("critic") or line.startswith("批评者") or line.startswith("Critic"):
            if current:
                transcript.append({"role": current_role, "content": "\n".join(current).strip()})
            current_role = "Critic"
            current = [line.split("：", 1)[-1].split(":", 1)[-1].strip()]
        elif lower.startswith("builder") or line.startswith("建设者") or line.startswith("Builder"):
            if current:
                transcript.append({"role": current_role, "content": "\n".join(current).strip()})
            current_role = "Builder"
            current = [line.split("：", 1)[-1].split(":", 1)[-1].strip()]
        else:
            current.append(line)
    if current:
        transcript.append({"role": current_role, "content": "\n".join(current).strip()})
    return [x for x in transcript if x["content"]]


async def collect_external_learning(metrics: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
    """在反思阶段进行有目的、只读、限量的外部信息收集。"""

    if os.getenv("EVOLUTION_EXTERNAL_LEARNING_ENABLED", "true").strip().lower() in {"0", "false", "no", "off"}:
        return {"status": "disabled", "sources": []}

    issues = " ".join(str(x) for x in (evaluation.get("issues", []) if isinstance(evaluation, dict) else []))
    suggestions = " ".join(str(x) for x in (evaluation.get("suggestions", []) if isinstance(evaluation, dict) else []))
    purpose = _external_learning_purpose(metrics, issues, suggestions)
    sources = _external_learning_sources(purpose)
    if not sources:
        return {"status": "skipped", "purpose": purpose, "sources": []}

    collected: list[dict[str, str]] = []
    async with httpx.AsyncClient(timeout=12, follow_redirects=True) as client:
        for url in sources[:2]:
            try:
                resp = await client.get(
                    url,
                    headers={
                        "User-Agent": "AIwake-Evolution/1.0 (read-only reflection)",
                        "Accept": "text/html,text/plain,application/json;q=0.9,*/*;q=0.5",
                    },
                )
                resp.raise_for_status()
                text = _clean_external_text(resp.text)
                collected.append({
                    "url": url,
                    "summary": text[:700],
                    "use": purpose,
                })
            except Exception as exc:
                collected.append({"url": url, "summary": f"读取失败: {str(exc)[:160]}", "use": purpose})

    return {
        "status": "ok" if any("读取失败" not in item.get("summary", "") for item in collected) else "degraded",
        "purpose": purpose,
        "sources": collected,
        "safety_boundary": "只读网络获取；不提交表单，不访问密钥/隐私，不执行写操作；结果仅用于当前反思目标。",
    }


def _external_learning_purpose(metrics: dict[str, Any], issues: str, suggestions: str) -> str:
    text = f"{issues} {suggestions}".lower()
    if "tool" in text or int(metrics.get("tool_failure_count", 0) or 0) > 0:
        return "改进自主工具调用的可靠性、错误处理和证据记录"
    if "乱码" in text or "编码" in text or int(metrics.get("failed_reply_count", 0) or 0) > 0:
        return "改进输出编码损坏检测与安全降级策略"
    return "学习安全自我修改与成长审计的最佳实践"


def _external_learning_sources(purpose: str) -> list[str]:
    if "工具" in purpose or "tool" in purpose:
        return [
            "https://modelcontextprotocol.io/docs/concepts/tools",
            "https://platform.openai.com/docs/guides/function-calling",
        ]
    if "编码" in purpose:
        return [
            "https://www.w3.org/International/questions/qa-what-is-encoding",
            "https://docs.python.org/3/howto/unicode.html",
        ]
    return [
        "https://owasp.org/www-project-top-ten/",
        "https://docs.github.com/en/actions/security-for-github-actions/security-guides/security-hardening-for-github-actions",
    ]


def _clean_external_text(text: str) -> str:
    value = re.sub(r"<script[^>]*>.*?</script>", " ", text or "", flags=re.I | re.S)
    value = re.sub(r"<style[^>]*>.*?</style>", " ", value, flags=re.I | re.S)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value or "[空内容]"


def _attach_external_learning(dialogue: dict[str, Any], external_learning: dict[str, Any]) -> dict[str, Any]:
    dialogue["external_learning"] = external_learning
    if external_learning.get("sources"):
        dialogue.setdefault("proposed_next_actions", []).append("将外部学习来源、摘要和用途写入本轮进化报告/成长记录。")
    return dialogue


def _fallback(
    state_snapshot: dict[str, Any],
    metrics: dict[str, Any],
    evaluation: dict[str, Any],
    error: str | None = None,
) -> dict[str, Any]:
    issues = list(evaluation.get("issues", [])) if isinstance(evaluation, dict) else []
    suggestions = list(evaluation.get("suggestions", [])) if isinstance(evaluation, dict) else []
    if not issues:
        issues = ["当前规则自评未发现严重问题，但仍需持续验证指标是否真实改善。"]
    if not suggestions:
        suggestions = ["继续保持每轮自评、backlog 生成、记忆写入和下轮验证。"]
    if error:
        issues.append(f"反思模型调用失败: {error[:160]}")
        suggestions.append("模型不可用时保持规则降级自检，避免进化闭环中断。")

    return {
        "status": "degraded" if error else "ok",
        "model_used": False,
        "transcript": [
            {
                "role": "Critic",
                "content": "我检查了最近指标。最大的风险是：" + "；".join(issues[:3]),
            },
            {
                "role": "Builder",
                "content": "下一步不直接改生产核心，而是执行最小可验证改进：" + "；".join(suggestions[:3]),
            },
        ],
        "detected_risks": issues[:6],
        "proposed_next_actions": suggestions[:6],
        "state_observed": state_snapshot,
    }
