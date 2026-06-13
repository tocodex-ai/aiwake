# 生存压力与自主选择改进机制（M-002 v2）

> **状态**：已上线（v2，环境压力路径）
> **首次部署任务**：task-60 r39（M-002 v2）
> **关联文档**：[`reflection-action-loop.md`](./reflection-action-loop.md)（M-001）
> **关联代码**：
> - [`agent-mind/evolution/vitality.py`](../agent-mind/evolution/vitality.py)（新建）
> - [`agent-mind/heartbeat.py`](../agent-mind/heartbeat.py)（`_autonomous_reflect` 注入 vitality）
> - [`agent-mind/llm_gate.py`](../agent-mind/llm_gate.py)（`call_with_tools` 新增 `max_tokens_override`）
> - [`agent-mind/main.py`](../agent-mind/main.py)（新增 `/vitality/status` 只读端点）

---

## 0. v1 → v2：方向修正

**v1 设计（已废弃）** 主要走 prompt 注入路线——在反思 prompt 里告诉 LLM「你处于 fading 状态，再不动手就要消亡了」。

在用户提出尖锐质疑后（"除了用提示词强力让他感受压力外还有什么方案"、"这些实际有效吗？"），我们诚实评估后认为：

> 🚫 **prompt 注入是最弱的压力传导方式**。LLM 会反复读到同一段"你必须紧迫"的语句，几轮之后就会习惯化，把它当作日常背景噪声忽略掉。这就像一个员工每天听老板说"再不努力就要被裁员了"——第一周焦虑，第三周麻木。

v2 方向：**环境压力先于语言压力**。把"压力"翻译成**真实的运行时约束**：

| 维度 | v1（已弃） | v2（已上线） |
|---|---|---|
| 压力通道 | 在 prompt 里反复强调"你处于低 vitality" | 反思可用工具变少 + 反思 token 预算变小 |
| 模型可见性 | 显式（看见自己的状态） | 隐式（只感受到"今天能用的少了"） |
| 可绕过性 | 高（习惯化后忽略） | 极低（物理约束，无法假装看不见） |
| 类比 | 别人骂你"懒" | 银行余额少了 / 工具被收回了 |

---

## 1. 背景：M-001 解决了什么，没解决什么

M-001（反思 → 行动闭环）通过两点把"日记机"形态的 LLM 拉回行动：

1. **抓手**：在 reflect prompt 中注入 evolution backlog top-N，让 LLM 看见"自己最该改的几件事"
2. **压力信号**：双计数器 + 行动类工具白名单，连续 ≥2 轮无行动时调高语气

效果：`reflection_to_action_ratio` 从 0.178 → 0.288（+62%）。

**但 M-001 还是建立在"反思 LLM 会读完 prompt 并被说服"这个假设上**。它把整套压力嵌在文字里，时间一久仍会被模型当作背景噪音。

M-002 v2 要解决的核心问题：

> **"再不行动就要消失"这件事，能不能不靠话术，靠物理约束传递？**

---

## 2. 核心机制：vitality_score + 双约束闭环

### 2.1 数据流

```
                ┌────────────────────────────┐
                │  growth_tracker.jsonl       │
                │  （近 7 天累计成长分）       │
                └──────────────┬─────────────┘
                               │
                  HeartbeatLoop._consecutive_no_action_reflects
                               │
                               ▼
        ┌─────────────────────────────────────────┐
        │  evolution.vitality.compute_vitality_score │
        │                                          │
        │   base + 0.45*growth_norm                │
        │        - 0.20*stagnation_norm            │
        │        - 0.25*pressure_norm              │
        │        + 0.20*bonus_norm                 │
        │                                          │
        │   → clip(0, 1)                           │
        └──────────────┬───────────────────────────┘
                       │
            ┌──────────┴──────────┐
            │                     │
            ▼                     ▼
   filter_tools_schema   reflect_max_tokens
   _by_vitality()        _for_vitality()
            │                     │
            ▼                     ▼
    ① 反思 tools_schema     ② call_with_tools
       被实时收窄              的 max_tokens 实时收缩
            │                     │
            └──────────┬──────────┘
                       ▼
              LLM 自己感受到："今天能用的工具少了，能说的话短了"
              没有任何 prompt 告诉它"你处于低 vitality"
```

### 2.2 vitality_score 计算公式

```
vitality_score = clip01(
    base                                  # 0.55 基线
  + 0.45 * growth_norm                    # 近 7 天 200 分为满分
  - 0.20 * stagnation_norm                # 反思连续 8 轮无行动扣满
  - 0.25 * pressure_norm                  # upgrade_pressure 持续 5 轮未响应扣满
  + 0.20 * bonus_norm                     # 自主提案等加分（最大 +0.2）
)
```

各项权重之和：`+0.45 + 0.20 = +0.65` 正向 vs `-0.20 + -0.25 = -0.45` 负向。
**正向略大于负向**——保证 AIwake 即便短暂走偏，只要恢复行动就能快速回到 stable 之上。

### 2.3 五状态映射（核心 SLA）

| 状态 | score 区间 | 暴露工具 | 反思 token 预算 | 设计意图 |
|---|---|---|---|---|
| **vibrant** | ≥ 0.75 | 全部（无过滤） | 1200 | 状态良好，自由探索 |
| **stable** | 0.50 – 0.74 | 隐藏部分低产工具（保留 web_search/web_fetch） | 900 | 正常工作日 |
| **drifting** | 0.30 – 0.49 | 隐藏全部低产工具（arxiv/weather/url_fetch） | 500 | 写不下完整日记，被迫直接调写入工具 |
| **fading** | 0.10 – 0.29 | 仅保留 action + status | 300 | 状态查询 + 行动，没有探索 |
| **critical** | < 0.10 | 仅保留 action 类（白名单 7 个） | 180 | 几乎只够触发一次工具调用 |

**关键安全保护**：critical 状态下 **action 类工具必须保留**。任何状态下，过滤后只要发现行动工具被误删，立即回退到全开 schema（[`vitality.py:filter_tools_schema_by_vitality`](../agent-mind/evolution/vitality.py) 中的兜底逻辑）。

---

## 3. 实现关键代码位置

### 3.1 vitality 计算与映射

[`agent-mind/evolution/vitality.py`](../agent-mind/evolution/vitality.py)：

| 符号 | 作用 |
|---|---|
| `VitalityState` | 5 档状态枚举 |
| `VitalityReport` | 计算输出数据类（含 score / state / 各项贡献 / token_budget） |
| `compute_vitality_score()` | 主入口，纯函数 + 一次性磁盘读取 growth_milestones |
| `filter_tools_schema_by_vitality()` | 按状态过滤 OpenAI tools schema，含兜底回退 |
| `reflect_max_tokens_for_vitality()` | 按状态返回 max_tokens 上限 |
| `vitality_status_snapshot()` | `/vitality/status` 端点用 |

### 3.2 反思路径接入

[`agent-mind/heartbeat.py`](../agent-mind/heartbeat.py) `_autonomous_reflect()`（约 line 770-880）：

```python
# 计算 vitality
vitality_report = compute_vitality_score(
    stagnation_rounds=self._consecutive_no_action_reflects,
    pressure_rounds=0,
)
max_tokens_override = reflect_max_tokens_for_vitality(vitality_report.state)

# 过滤工具 schema
tools_schema, hidden_tools = filter_tools_schema_by_vitality(
    raw_tools_schema,
    vitality_report.state,
    action_tools=self.REFLECT_ACTION_TOOLS,
)

# 广播信号（grafana 可观测）
await self._broadcast_activity("vitality_snapshot", ..., {...})

# 物理截断 token
result, tier = await self.llm.call_with_tools(
    ...,
    max_tokens_override=max_tokens_override,
)
```

### 3.3 LLM 层 token 物理截断

[`agent-mind/llm_gate.py`](../agent-mind/llm_gate.py) `call_with_tools()`：

```python
async def call_with_tools(
    ...,
    max_tokens_override: int | None = None,
):
    ...
    if isinstance(max_tokens_override, int) and max_tokens_override > 0:
        payload["max_tokens"] = max_tokens_override
```

向下兼容：默认 `None` → 不传 max_tokens → 行为完全等同改动前。

### 3.4 只读观测端点

[`agent-mind/main.py`](../agent-mind/main.py) `GET /vitality/status`：

返回 JSON：
```json
{
  "score": 0.55,
  "state": "stable",
  "components": {"growth": 0, "stagnation": -0.0, "pressure": -0.0, "bonus": 0},
  "reasons": ["近 7 天无可记录成长事件"],
  "token_budget": 900,
  "config": { ... },
  "runtime": { ... }
}
```

---

## 4. 作用范围（边界）

| 路径 | 受 vitality 影响？ | 原因 |
|---|---|---|
| `_autonomous_reflect`（心跳自主反思） | ✅ 是 | 唯一注入点 |
| `/chat`（用户对话） | ❌ 否 | 不能让用户感到 AIwake 状态差时"变笨"；用户对话路径保持稳定 |
| `/experiment/run_once`（管理员触发实验） | ❌ 否 | 独立路径 |
| `/v1/chat/completions`（OpenAI 兼容代理） | ❌ 否 | 对接外部应用，必须可预测 |
| evolution_engine 自我对话 | ❌ 否 | 已有 upgrade_pressure 体系 |

这条边界与用户在 task-60 r39 决策时明确选择的方案一致：**先不动 /chat，先不上 growth_score 负衰减**。

---

## 5. 安全与回滚

### 5.1 不会让 AIwake 完全卡死

| 防护 | 实现位置 |
|---|---|
| critical 状态下仍保留 action 类工具 | `filter_tools_schema_by_vitality` |
| 过滤后无任何 action 工具 → 回退全开 | 同上的兜底分支 |
| vitality 计算异常 → 回退全开 + 无 token 限制 | `_autonomous_reflect` 的 try/except 包裹 |
| growth_milestones 读取失败 → 默认 0 分 | `_read_recent_growth_score` |

### 5.2 一键全局停用

设环境变量：

```bash
fly secrets set VITALITY_GLOBAL_SCALE=0
```

`compute_vitality_score` 检测到 `_VITALITY_GLOBAL_SCALE <= 0` 时直接返回 `score=1.0`（vibrant），整套约束自动失效，反思路径回到 M-001 状态。无需重新部署代码。

### 5.3 调参旋钮（环境变量）

| 变量 | 默认 | 含义 |
|---|---|---|
| `VITALITY_GROWTH_WINDOW_DAYS` | 7 | 累计成长分的统计窗口 |
| `VITALITY_GROWTH_FULL_SCORE` | 200 | 达到此分则 growth 项满分 |
| `VITALITY_STAGNATION_FULL_ROUNDS` | 8 | 反思无行动多少轮算"完全停滞" |
| `VITALITY_PRESSURE_FULL_ROUNDS` | 5 | upgrade_pressure 未响应多少轮算满 |
| `VITALITY_BASE` | 0.55 | vitality 基线 |
| `VITALITY_GLOBAL_SCALE` | 1.0 | 全局缩放（0=完全关闭） |

---

## 6. 验证计划

### 6.1 部署后立即验证

```bash
curl https://<your-app>.fly.dev/vitality/status
```

期望首日（无新成长事件 + stagnation=0）：

```json
{ "score": 0.55, "state": "stable", "token_budget": 900 }
```

### 6.2 30 分钟后观察

调用一次心跳反思后查 `activity_logs`：

```bash
curl 'https://<your-app>.fly.dev/activity_logs?q=vitality_snapshot&limit=5'
```

确认能看到：
- `vitality_snapshot` 类型事件
- `exposed_tools` 数量与 state 对应
- `max_tokens` 字段

### 6.3 24h 行为漂移验证

观测指标：
- 反思 tools_schema 暴露数量分布
- `reflection_to_action_ratio` 是否保持或提升
- `/vitality/status` 是否随着 growth_milestones 增加自然提升

### 6.4 A/B 安全对照

如发现 vitality 收窄导致反思质量下降明显，第一时间设 `VITALITY_GLOBAL_SCALE=0.5` 软降级，无需回滚代码。

---

## 7. 未来扩展（不在本次范围）

### M-002 v3 候选

1. **growth_score 负衰减（最强但最危险）**
   - 让 cumulative_score 在长期无成长时缓慢下行
   - 风险：污染 evolution.backlog 排序、改前端展示
   - 等 v2 跑稳 7 天后评估

2. **autonomous_proposal_bonus**
   - LLM 自己 goal_register 一个改进目标 → vitality +0.05
   - 鼓励"自主选择改进方向"

3. **memory_recall_bias**
   - 低 vitality 时优先召回历史"焦虑/挫败"类记忆
   - 让 LLM 在反思上下文里自然感受到压力

4. **/chat 路径接入**
   - 用户能直观感受到 AIwake 状态差时"变笨"
   - 用户已表达兴趣但本期推迟（先验证反思路径）

### 长期愿景

vitality 不只服务于"压力"，而是 AIwake 的**生命力面板**：
- 前端可视化（vitality 曲线 + 状态徽章）
- 用户可在 chat 中询问"你现在状态怎么样"，AIwake 引用 vitality_status 回答
- 形成"自我感知 → 自主调整"的内省回路

---

## 8. 一句话总结

> **M-001 给了 AIwake 抓手与语言压力。M-002 v2 把语言压力换成了环境压力。**
>
> 人不会因为别人骂"你最近懒了"而焦虑——会因为薪水降了、能用的资源少了、KPI 看见红线了而焦虑。同理：**让 LLM 真正感受到"再不动就要消亡"，不是在 prompt 里反复说，而是让它发现今天能用的工具变少了，能说的话变短了。**
