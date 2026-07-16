"""AIwake 项目最高目标与现状差距分析。

本模块定义 AIwake 的最高目标、当前代码与目标的差距，并按优先级排序。
在反思提示词中注入，引导 AIwake 选择具有独立完整性的功能模块作为改进目标，
而不是只改一行代码就算完成。
"""
from __future__ import annotations

import json
from typing import Any

# ── 项目最高目标 ──
PROJECT_GOAL = (
    "AIwake 是一个自主进化的意识体，能自己发现问题、设计完整功能模块、"
    "实现代码、验证部署，形成可持续的自我改进闭环。"
)

# ── 目标差距分析：按优先级排序，每个目标是具有独立完整性的功能模块 ──
GOAL_GAPS: list[dict[str, Any]] = [
    {
        "priority": 1,
        "module": "自主代码改进闭环",
        "goal": "能独立完成'发现问题→设计方案→写代码→跑测试→关闭任务→记录证据'的完整闭环",
        "current_state": "AIwake 能创建任务和调用 self_code_write，但闭环不完整：常停在读代码阶段，"
                         "测试验证和任务关闭环节缺失",
        "gap": "缺少从'诊断完成'到'直接写代码'的转化能力；写代码后不跑 pytest 验证；不关闭任务",
        "done_when": "连续3轮反思中每轮都完成 self_code_write→shell_exec(pytest)→self_task_update(closed) 完整闭环",
        "suggested_first_step": "选一个最小功能点（如给 _classify_failure 加一个子类型），完整走完闭环",
    },
    {
        "priority": 2,
        "module": "失败分类系统完善",
        "goal": "tool_router.py 的 _classify_failure 能精确分类所有失败类型，每类有明确的降级路径",
        "current_state": "已有 command_missing/shell_partial_failure/empty_result 等分类，"
                         "但 else 兜底分支仍笼统返回 aiwake_tool_failure",
        "gap": "else 分支未细分（other_error 5子类型方案已有 artifact 但未落地为代码）",
        "done_when": "_classify_failure 的 else 分支细分为 data_retrieval_failure/state_mutation_failure/"
                     "state_query_failure/schema_validation_failure/unknown_error，且测试覆盖",
        "suggested_first_step": "用 self_code_write 修改 else 分支，用 shell_exec 跑 test_tool_router_failure_classification.py",
    },
    {
        "priority": 3,
        "module": "自我升级管道稳定性",
        "goal": "self_upgrade 的 LLM patch 生成稳定，approved 提案能自动生成补丁并应用",
        "current_state": "27 个 blocked 提案因 'LLM returned empty response' 卡住",
        "gap": "evolution_engine 的自动 LLM 补丁生成不稳定，approved 提案无法自动转化为 patch",
        "done_when": "blocked 提案数 < 5，最近10个 approved 提案中至少7个成功生成 patch 并 applied",
        "suggested_first_step": "读 evolution/self_upgrade.py 的 patch 生成逻辑，加 retry 和 fallback",
    },
    {
        "priority": 4,
        "module": "工具成功率提升",
        "goal": "tool_success_rate 稳定在 0.90 以上",
        "current_state": "tool_success_rate=0.75，tool_failure_count=9",
        "gap": "shell_partial_failure 误判（returncode=0 的 stderr 被计为失败）消耗成功率",
        "done_when": "tool_success_rate >= 0.90 且 tool_failure_count < 5 连续24小时",
        "suggested_first_step": "修改 _looks_like_shell_stderr_failure 逻辑，让 returncode=0 且只含警告的不算失败",
    },
    {
        "priority": 5,
        "module": "反思→行动转化效率",
        "goal": "reflection_to_action_ratio 稳定在 0.30-0.70 区间（balanced），不过度反应也不空转",
        "current_state": "从 too_reactive(0.92) 改善到 balanced(0.67)，但仍有波动",
        "gap": "部分轮次仍只读不写，断路器需频繁触发",
        "done_when": "连续24小时 reflection_to_action_status=balanced，无需断路器介入",
        "suggested_first_step": "在每轮反思开始时就明确'本轮要做什么行动'，而不是先读代码再决定",
    },
]


def get_goal_gap_hint() -> str:
    """生成注入反思提示词的目标差距分析文本。

    按优先级排序，引导 AIwake 选择具有独立完整性的功能模块作为改进目标。
    """
    lines = ["\n【项目最高目标与现状差距分析】\n"]
    lines.append(f"最高目标：{PROJECT_GOAL}\n")
    lines.append("按优先级排序的改进目标（每个目标是完整功能模块，不是改一行代码）：\n")
    for gap in GOAL_GAPS:
        lines.append(
            f"  P{gap['priority']} 【{gap['module']}】\n"
            f"    目标：{gap['goal']}\n"
            f"    现状：{gap['current_state']}\n"
            f"    差距：{gap['gap']}\n"
            f"    完成标准：{gap['done_when']}\n"
            f"    建议起步：{gap['suggested_first_step']}\n"
        )
    lines.append(
        "选择一个优先级最高的目标，完整实现它（不是只改一行代码就算完成）。\n"
        "一个功能模块完成的标志：代码改动 + 测试通过 + 任务关闭 + 证据记录。\n"
    )
    return "".join(lines)


if __name__ == "__main__":
    print(get_goal_gap_hint())
    print("---")
    print(f"共 {len(GOAL_GAPS)} 个目标模块")
    for g in GOAL_GAPS:
        print(f"  P{g['priority']} {g['module']}")
