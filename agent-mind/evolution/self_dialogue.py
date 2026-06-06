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
) -> dict[str, Any]:
    """Run one internal Critic/Builder dialogue, with deterministic fallback."""

    external_learning = await collect_external_learning(metrics, evaluation)
    prompt = _build_prompt(state_snapshot, metrics, evaluation, external_learning)
    if llm is not None and hasattr(llm, "call_reflect_messages"):
        try:
            text = await llm.call_reflect_messages([
                {
                    "role": "system",
                    "content": (
                        "你是 AIwake 的内部自我进化审计器。只进行安全的自我对话，"
                        "可以基于已收集的外部只读资料进行反思，但不得请求执行 shell、ssh、部署、读取密钥或修改生产文件。"
                    ),
                },
                {"role": "user", "content": prompt},
            ])
            if text and text.strip():
                return _attach_external_learning(_from_model_text(text.strip(), evaluation), external_learning)
        except Exception as exc:
            return _attach_external_learning(_fallback(state_snapshot, metrics, evaluation, error=str(exc)), external_learning)
    return _attach_external_learning(_fallback(state_snapshot, metrics, evaluation), external_learning)


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
    return (
        "请让 AIwake 与自己进行一轮真实的内部进化检测对话。\n"
        "角色一 Critic：指出当前自我学习/自我进化闭环的最大缺陷。\n"
        "角色二 Builder：提出下一步最小可验证改进。\n"
        "最后输出：detected_risks 与 proposed_next_actions。\n"
        "如果 external_learning 有内容，请说明来源如何用于当前目标，以及仍不确定之处。\n"
        "不要声称已经修改代码，不要请求危险工具。\n\n"
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
