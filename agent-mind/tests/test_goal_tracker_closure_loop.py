"""Self-checks for the goal-closure loop (改→度量→收敛).

Covers:
1. Register goal → first cycle sets baseline (no false closure).
2. Subsequent cycle that meets target → closed + growth milestone "goal_closed"
   + capability_library entry + "capability_registered" milestone.
3. Subsequent cycle that does not yet meet target → iterating.
4. Significant regression → auto abandoned (regression).
5. Max cycles exceeded → auto abandoned (max_cycles_exceeded).
6. Same description registered twice produces distinct goal IDs (dedup-safe).

Run from repository root:
    python src/agent-mind/tests/test_goal_tracker_closure_loop.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path


def _find_agent_dir() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "heartbeat.py").exists() and (parent / "tool_router.py").exists():
            return parent
        if parent.name == "agent-mind":
            return parent
        candidate = parent / "src" / "agent-mind"
        if candidate.exists():
            return candidate
        candidate = parent / "agent-mind"
        if candidate.exists():
            return candidate
    raise RuntimeError("Cannot locate agent-mind directory")


AGENT_DIR = _find_agent_dir()
sys.path.insert(0, str(AGENT_DIR))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _latest_per_id(records: list[dict]) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for rec in records:
        gid = rec.get("id")
        if gid:
            latest[gid] = rec
    return latest


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["EVOLUTION_DIR"] = tmp
        # 强制重新加载，避免之前测试缓存的 _data_dir
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("evolution.goal_tracker") or mod_name.startswith("evolution.growth_tracker"):
                del sys.modules[mod_name]

        from evolution.goal_tracker import (  # type: ignore
            register_goal,
            evaluate_open_goals,
            read_open_goals,
            read_all_goals,
            read_capabilities,
            DEFAULT_MAX_CYCLES,
        )

        goals_file = Path(tmp) / "goals.jsonl"
        capabilities_file = Path(tmp) / "capability_library.jsonl"
        milestones_file = Path(tmp) / "growth_milestones.jsonl"

        # ── Case 1: baseline_set on first evaluation, no false closure ──
        goal = register_goal(
            metric="tool_success_rate",
            direction="up",
            target=0.99,
            description="把工具成功率从约 0.97 提升到 0.99",
            source="reflection",
            max_cycles=4,
        )
        assert goal["status"] == "open", f"刚登记的目标应为 open，实为 {goal['status']}"
        assert goal["baseline"] is None, "未提供基线时初始 baseline 应为 None"

        # 第一轮复测（设基线）：当前值 0.97
        results = evaluate_open_goals({"tool_success_rate": 0.97})
        assert len(results) == 1
        first = results[0]
        assert first["status"] == "baseline_set", f"首轮应 baseline_set，实为 {first['status']}"
        assert abs(first["baseline"] - 0.97) < 1e-9
        # 即便首轮值已经达到目标，也只设基线不闭合，避免“伪达成”
        results_fake = evaluate_open_goals({"tool_success_rate": 0.999})
        # 此时 case-1 目标已经设过基线，再来一轮如果直接达标应被允许闭合
        assert results_fake[0]["status"] == "closed", \
            f"基线已设、第二轮已达标应闭合，实为 {results_fake[0]['status']}"
        # 闭合后应触发能力库 + 重大成长里程碑
        caps = read_capabilities(limit=10)
        assert len(caps) == 1, f"闭合后应沉淀 1 条能力，实为 {len(caps)}"
        assert caps[0]["metric"] == "tool_success_rate"
        assert caps[0]["achieved"] >= 0.99 - 1e-9
        ms_records = _read_jsonl(milestones_file)
        ms_types = [m.get("event_type") for m in ms_records]
        assert "goal_closed" in ms_types, f"应记录 goal_closed 里程碑，已记录 {ms_types}"
        assert "capability_registered" in ms_types, f"应记录 capability_registered 里程碑，已记录 {ms_types}"

        # ── Case 2: iterating（达标失败）──
        goal2 = register_goal(
            metric="tool_failure_count",
            direction="down",
            target=0,
            description="把工具失败数降到 0",
            source="reflection",
            max_cycles=DEFAULT_MAX_CYCLES,
        )
        # 首轮：基线 5
        evaluate_open_goals({"tool_failure_count": 5})
        # 第二轮：仍是 4，未达标，应迭代
        r2 = evaluate_open_goals({"tool_failure_count": 4})
        match2 = [r for r in r2 if r.get("goal_id") == goal2["id"]]
        assert match2, "应能匹配到 goal2 的本轮结果"
        assert match2[0]["status"] == "iterating", \
            f"目标未达成应 iterating，实为 {match2[0]['status']}"

        # ── Case 3: regression abandon ──
        goal3 = register_goal(
            metric="tool_success_rate",
            direction="up",
            target=0.999,
            description="工具成功率回归恶化样例",
            source="reflection",
            max_cycles=DEFAULT_MAX_CYCLES,
        )
        # 基线 0.95
        evaluate_open_goals({"tool_success_rate": 0.95})
        # 当前 0.70，远低于 baseline*(1-0.20)=0.76，应自动放弃
        r3 = evaluate_open_goals({"tool_success_rate": 0.70})
        match3 = [r for r in r3 if r.get("goal_id") == goal3["id"]]
        assert match3 and match3[0]["status"] == "abandoned", \
            f"显著恶化应自动放弃，实为 {match3 and match3[0]['status']}"
        ms_records = _read_jsonl(milestones_file)
        abandon_count = sum(1 for m in ms_records if m.get("event_type") == "goal_abandoned")
        assert abandon_count >= 1, "应记录至少 1 条 goal_abandoned 里程碑"

        # ── Case 4: max cycles exceeded ──
        goal4 = register_goal(
            metric="tool_failure_count",
            direction="down",
            target=0,
            description="超限自动放弃样例",
            source="reflection",
            max_cycles=2,
        )
        # 基线 3，然后再两轮都不达标
        evaluate_open_goals({"tool_failure_count": 3})  # baseline_set, cycle_count=1
        evaluate_open_goals({"tool_failure_count": 2})  # iterating, cycle_count=2 -> hits max
        # 因 cycle_count >= max_cycles（在第二次复测时已达 max=2），目标应被放弃
        all_goals_now = {g["id"]: g for g in read_all_goals(limit=200)}
        g4_state = all_goals_now.get(goal4["id"], {})
        assert g4_state.get("status") in ("abandoned", "iterating"), \
            f"max_cycles 边界态应为 abandoned 或 iterating，实为 {g4_state.get('status')}"
        # 再追加一轮，确保达到 max 后必然放弃
        evaluate_open_goals({"tool_failure_count": 2})
        all_goals_now = {g["id"]: g for g in read_all_goals(limit=200)}
        g4_final = all_goals_now.get(goal4["id"], {})
        assert g4_final.get("status") == "abandoned", \
            f"超过 max_cycles 必须 abandoned，实为 {g4_final.get('status')}"

        # ── Case 5: same description twice → distinct goal IDs (no silent merge) ──
        time.sleep(0.001)  # 让时间戳推进，hash 不冲突
        ga = register_goal(
            metric="tool_success_rate",
            direction="up",
            target=0.99,
            description="重复描述但应得新 goal_id",
            source="reflection",
        )
        time.sleep(0.001)
        gb = register_goal(
            metric="tool_success_rate",
            direction="up",
            target=0.99,
            description="重复描述但应得新 goal_id",
            source="reflection",
        )
        assert ga["id"] != gb["id"], "重复登记应产生不同 goal_id"

        # ── Case 5b: trivial 目标（登记即达标）应 abandoned，绝不闭合刷分 ──
        # 方向 up：现状已远超目标（baseline=27 >= target=1）
        caps_before = len(read_capabilities(limit=100))
        ms_before = _read_jsonl(milestones_file)
        closed_before = sum(1 for m in ms_before if m.get("event_type") == "goal_closed")
        cap_reg_before = sum(1 for m in ms_before if m.get("event_type") == "capability_registered")

        g_trivial_up = register_goal(
            metric="proactive_count",
            direction="up",
            target=1,
            description="伪目标：现状27已远超目标1（应被如实放弃）",
            source="manual",
            max_cycles=4,
        )
        r_tu = evaluate_open_goals({"proactive_count": 27})
        match_tu = [r for r in r_tu if r.get("goal_id") == g_trivial_up["id"]]
        assert match_tu and match_tu[0]["status"] == "abandoned", \
            f"登记即达标(up)应 abandoned，实为 {match_tu and match_tu[0]['status']}"

        # 方向 down：现状已低于目标（baseline=2 <= target=10）
        g_trivial_down = register_goal(
            metric="tool_failure_count",
            direction="down",
            target=10,
            description="伪目标：现状2已低于目标10（应被如实放弃）",
            source="manual",
            max_cycles=4,
        )
        r_td = evaluate_open_goals({"tool_failure_count": 2})
        match_td = [r for r in r_td if r.get("goal_id") == g_trivial_down["id"]]
        assert match_td and match_td[0]["status"] == "abandoned", \
            f"登记即达标(down)应 abandoned，实为 {match_td and match_td[0]['status']}"

        # 伪目标绝不能沉淀能力，也不能记 goal_closed / capability_registered
        caps_after = len(read_capabilities(limit=100))
        assert caps_after == caps_before, "伪目标不得写入能力库刷分"
        ms_after = _read_jsonl(milestones_file)
        closed_after = sum(1 for m in ms_after if m.get("event_type") == "goal_closed")
        cap_reg_after = sum(1 for m in ms_after if m.get("event_type") == "capability_registered")
        assert closed_after == closed_before, "伪目标不得记 goal_closed 刷分"
        assert cap_reg_after == cap_reg_before, "伪目标不得记 capability_registered 刷分"
        # 应如实记录为 trivial_target 放弃
        trivial_abandons = [
            m for m in ms_after
            if m.get("event_type") == "goal_abandoned"
            and (m.get("extra") or {}).get("reason") == "trivial_target"
        ]
        assert len(trivial_abandons) >= 2, \
            f"应记录至少 2 条 trivial_target 放弃里程碑，实为 {len(trivial_abandons)}"

        # ── Case 6: read_open_goals 不应再返回已闭合 / 已放弃的目标 ──
        opens = read_open_goals()
        open_ids = {g["id"] for g in opens}
        assert goal["id"] not in open_ids, "已闭合的目标不应出现在 open 列表"
        assert goal3["id"] not in open_ids, "已放弃的目标不应出现在 open 列表"

        # ── Case 7: 文件落盘审计 ──
        all_records = _read_jsonl(goals_file)
        assert len(all_records) >= 8, f"goals.jsonl 应有≥8 条 append-only 记录，实为 {len(all_records)}"
        latest = _latest_per_id(all_records)
        assert latest[goal["id"]]["status"] == "closed"
        assert latest[goal3["id"]]["status"] == "abandoned"
        assert latest[goal4["id"]]["status"] == "abandoned"

        cap_records = _read_jsonl(capabilities_file)
        assert len(cap_records) >= 1, "capability_library.jsonl 应至少有 1 条沉淀"

        print("OK goal_tracker closure loop self-check passed.")
        print(f" - goals.jsonl entries:        {len(all_records)}")
        print(f" - capability_library entries: {len(cap_records)}")
        print(f" - growth milestones recorded: {len(_read_jsonl(milestones_file))}")


if __name__ == "__main__":
    main()
