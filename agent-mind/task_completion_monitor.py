"""任务完成压力监控模块。

监控 open task 增长与 closed task 增长比例。当 AIwake 持续创建新任务但
不完成任务（open_task_count 持续增长而 closed 不增长）时，生成强制指令
注入到反思提示词，推动 agent 关闭僵尸任务。

背景：线上观察到 task_count 从 217 增长到 222，但 open_task_count 也从 1
增长到 3，说明 AIwake 在创建任务但不完成任务。本模块在 open task 持续增长
但 closed 不增长时生成压力信号。

完全只读：仅读取 experiment_store.status()，不写入任何数据，不包含敏感信息。
"""

from __future__ import annotations

import datetime as _dt
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)

# 触发警告的 open task 数量阈值
_OPEN_COUNT_THRESHOLD = 2
# 任务创建后多久仍未关闭才视为"僵尸任务"（秒）
_STALE_SECONDS = 3600  # 1 小时


def _parse_created_at(ts_raw: str) -> _dt.datetime | None:
    """解析 ISO 格式时间戳，失败返回 None。

    与 evolution/vitality.py 保持一致的解析方式：兼容带 'Z' 后缀和
    无时区信息的 ISO 字符串，统一归一化为 UTC。
    """
    if not ts_raw:
        return None
    try:
        ts = _dt.datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=_dt.timezone.utc)
    return ts


def check_task_completion_pressure(experiment_store) -> dict[str, Any]:
    """检查任务完成压力，判断是否需要注入完成任务的压力信号。

    从 experiment_store.status() 获取 task_count、open_task_count、open_tasks，
    当 open_task_count >= 2 且有任务创建超过 1 小时仍未关闭时，返回
    should_warn=True 及压力提示文本，用于注入反思提示词。

    Args:
        experiment_store: ExperimentStore 实例，需提供 status() 方法。

    Returns:
        dict 包含:
        - should_warn: bool — 是否需要注入完成任务的压力信号
        - open_count: int — 当前 open task 数量
        - oldest_open_task: dict — 最老的 open task 信息（id, title, goal, created_at）；
          无 open task 时为 None
        - pressure_hint: str — 注入到反思提示词的压力文本（should_warn=True 时非空，
          否则为空字符串）
    """
    result: dict[str, Any] = {
        "should_warn": False,
        "open_count": 0,
        "oldest_open_task": None,
        "pressure_hint": "",
    }

    try:
        status = experiment_store.status()
    except Exception as e:  # noqa: BLE001 — 只读监控，任何异常都安全降级
        logger.warning("[TaskCompletionMonitor] 读取 experiment_store.status() 失败: %s", e)
        return result

    open_tasks = status.get("open_tasks") or []
    open_count = int(status.get("open_task_count", len(open_tasks)))
    result["open_count"] = open_count

    if not open_tasks:
        return result

    now = _dt.datetime.now(_dt.timezone.utc)

    # 为每个 open task 解析创建时间，用于判断年龄和排序
    enriched: list[tuple[_dt.datetime | None, int, dict[str, Any]]] = []
    for idx, task in enumerate(open_tasks):
        created_dt = _parse_created_at(task.get("created_at", ""))
        enriched.append((created_dt, idx, task))

    # 按 created_at 升序排序，取最老的任务（无法解析时间的视为最新，排最后）
    def _sort_key(item: tuple[_dt.datetime | None, int, dict[str, Any]]):
        created_dt, idx, _ = item
        if created_dt is None:
            return (_dt.datetime.max.replace(tzinfo=_dt.timezone.utc), idx)
        return (created_dt, idx)

    enriched.sort(key=_sort_key)
    _, _, oldest_task = enriched[0]

    result["oldest_open_task"] = {
        "id": oldest_task.get("id", ""),
        "title": oldest_task.get("title", ""),
        "goal": oldest_task.get("goal", ""),
        "created_at": oldest_task.get("created_at", ""),
    }

    # 触发条件 1: open_task_count >= 2
    if open_count < _OPEN_COUNT_THRESHOLD:
        return result

    # 触发条件 2: 有任务创建超过 1 小时仍未关闭
    has_stale = False
    for created_dt, _, _ in enriched:
        if created_dt is not None and (now - created_dt).total_seconds() >= _STALE_SECONDS:
            has_stale = True
            break

    if not has_stale:
        return result

    # 生成压力提示：列出所有 open task 的 id 和 title
    task_lines = []
    for task in open_tasks:
        tid = task.get("id", "")
        title = task.get("title", "")
        task_lines.append(f"- [{tid}] {title}")
    task_list_text = "\n".join(task_lines)

    pressure_hint = (
        "⚠️ 任务完成压力信号：检测到 open task 持续增长但未关闭。\n"
        f"当前有 {open_count} 个 open task 长时间未完成：\n"
        f"{task_list_text}\n\n"
        "请立即选择一个最简单的任务，用 self_task_update 工具关闭它。"
        "完成任务本身也是一种行动——关闭一个已达成目标或已无意义的任务，"
        "能让 open_task_count 下降，证明你具备闭环能力。"
        "不要只创建新任务而不关闭旧任务。"
    )
    result["should_warn"] = True
    result["pressure_hint"] = pressure_hint

    return result


if __name__ == "__main__":
    # 自检测试
    # Windows 终端默认 GBK 编码无法输出 emoji，重定向 stdout 为 UTF-8
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass
    print("=== task_completion_monitor 自检 ===\n")

    class _FakeStore:
        """模拟 experiment_store，仅提供 status() 方法。"""

        def __init__(self, status_dict: dict[str, Any]):
            self._status = status_dict

        def status(self) -> dict[str, Any]:
            return self._status

    _now = _dt.datetime.now(_dt.timezone.utc)
    _old_ts = (_now - _dt.timedelta(hours=2)).isoformat()
    _recent_ts = (_now - _dt.timedelta(minutes=10)).isoformat()

    # 测试 1: open_count=0，不应警告
    r1 = check_task_completion_pressure(_FakeStore({
        "task_count": 5,
        "open_task_count": 0,
        "open_tasks": [],
    }))
    assert r1["should_warn"] is False, "测试1失败: open_count=0 不应警告"
    assert r1["open_count"] == 0
    assert r1["oldest_open_task"] is None
    print("测试1 通过: open_count=0 → should_warn=False")

    # 测试 2: open_count=1，不应警告
    r2 = check_task_completion_pressure(_FakeStore({
        "task_count": 5,
        "open_task_count": 1,
        "open_tasks": [{"id": "t1", "title": "任务1", "goal": "g1", "created_at": _old_ts}],
    }))
    assert r2["should_warn"] is False, "测试2失败: open_count=1 不应警告"
    assert r2["open_count"] == 1
    assert r2["oldest_open_task"] is not None
    print("测试2 通过: open_count=1 → should_warn=False")

    # 测试 3: open_count>=2 但全部 1 小时内创建，不应警告
    r3 = check_task_completion_pressure(_FakeStore({
        "task_count": 5,
        "open_task_count": 2,
        "open_tasks": [
            {"id": "t1", "title": "任务1", "goal": "g1", "created_at": _recent_ts},
            {"id": "t2", "title": "任务2", "goal": "g2", "created_at": _recent_ts},
        ],
    }))
    assert r3["should_warn"] is False, "测试3失败: 全部1小时内不应警告"
    assert r3["open_count"] == 2
    print("测试3 通过: open_count=2 但全部1小时内 → should_warn=False")

    # 测试 4: open_count>=2 且有任务超过1小时，应警告（模拟线上场景）
    r4 = check_task_completion_pressure(_FakeStore({
        "task_count": 222,
        "open_task_count": 3,
        "open_tasks": [
            {"id": "t1", "title": "学习工具证据", "goal": "g1", "created_at": _old_ts},
            {"id": "t2", "title": "状态乱码训练", "goal": "g2", "created_at": _recent_ts},
            {"id": "t3", "title": "路径核对训练", "goal": "g3", "created_at": _old_ts},
        ],
    }))
    assert r4["should_warn"] is True, "测试4失败: 有僵尸任务应警告"
    assert r4["open_count"] == 3
    assert r4["oldest_open_task"] is not None
    assert "self_task_update" in r4["pressure_hint"], "测试4失败: 压力提示应包含 self_task_update"
    assert "t1" in r4["pressure_hint"], "测试4失败: 压力提示应包含 t1"
    assert "t2" in r4["pressure_hint"], "测试4失败: 压力提示应包含 t2"
    assert "t3" in r4["pressure_hint"], "测试4失败: 压力提示应包含 t3"
    print("测试4 通过: open_count=3 且有僵尸任务 → should_warn=True")
    print(f"  oldest_open_task: {r4['oldest_open_task']}")
    print(f"  pressure_hint 预览:\n{r4['pressure_hint']}\n")

    # 测试 5: status() 异常，应安全返回默认值
    class _ErrorStore:
        def status(self) -> dict[str, Any]:
            raise RuntimeError("模拟异常")

    r5 = check_task_completion_pressure(_ErrorStore())
    assert r5["should_warn"] is False, "测试5失败: 异常应安全降级"
    assert r5["open_count"] == 0
    assert r5["oldest_open_task"] is None
    print("测试5 通过: status() 异常 → 安全降级 should_warn=False")

    # 测试 6: created_at 缺失/格式错误，不应崩溃
    r6 = check_task_completion_pressure(_FakeStore({
        "task_count": 5,
        "open_task_count": 2,
        "open_tasks": [
            {"id": "t1", "title": "任务1", "goal": "g1", "created_at": ""},
            {"id": "t2", "title": "任务2", "goal": "g2", "created_at": "not-a-date"},
        ],
    }))
    assert r6["should_warn"] is False, "测试6失败: 无法解析时间不应警告"
    assert r6["open_count"] == 2
    print("测试6 通过: created_at 缺失/格式错误 → 不崩溃，should_warn=False")

    print("\n=== 全部自检通过 ===")
