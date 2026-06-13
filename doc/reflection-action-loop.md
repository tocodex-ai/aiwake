# M-001：反思 → 行动闭环机制（Reflection-Action Loop）

> **状态**：✅ 线上验证通过（task-60 r38 (A+B)，2026-06-13 部署）
> **作用对象**：[`agent-mind/heartbeat.py`](../agent-mind/heartbeat.py) 的自主反思流程
> **第一次成功的指标**：`reflection_to_action_ratio` 从 0.178 涨到 0.288（30 分钟内 +62%）
> **指导原则**：让反思 LLM 在空闲时**对自己施加可观察的压力**，而不是停留在"自言自语"

## 1. 背景与问题

### 1.1 task-60 之前 30+ 轮的痛点

AIwake 的 heartbeat 反思阶段（`_autonomous_reflect`）目的是：**在空闲时让 LLM 主动调用工具改进自己**。但前 37 轮迭代都出现同一种"失败模式"：

- LLM 反思生成的内容很优美——"我现在感到 X，下一步我打算 Y"
- 但接下来的 tick 还是反思，**没有真正调用任何工具**
- `experiment/status` 里 task_count、artifact_count 长期不动，反思被"软消费"成日记

这种"反思单调写日记"被我们称作 **diary-machine failure mode**。

### 1.2 失败的根因（机制层）

1. **反思 prompt 没有"压力信号"**：LLM 不知道"我已经连续 N 次没动手了"
2. **反思 prompt 没有"具体抓手"**：evolution_engine 已经在做 backlog 分析，但 backlog 数据**没有进 prompt**，LLM 不知道有哪些待办
3. **"无工具调用"的判定太松**：只有一个 `_consecutive_no_tool_reflects` 计数器，连 `experiment_status`（只读）也算"调用工具"，把"真的去观察"和"真的去改变"混为一谈

## 2. 设计方案（A + B 两件套）

### 2.1 (A) 把 evolution backlog 注入反思 prompt

evolution_engine 已经在生成 `last_report.backlog`（一组带 priority/risk/title 的待办候选），但这些数据原本只用于离线分析。
新增 [`HeartbeatLoop._recent_evolution_backlog_hint()`](../agent-mind/heartbeat.py:337) 把 top-N（默认 3）priority ≥ 0.5 的 backlog 渲染成一段提示，**直接拼进反思 system prompt**：

```
近期成长 backlog（top 3）：
- ev-cd0bc3844e [priority=0.76, risk=medium] 提升工具调用可靠性: 检测到工具调用失败
- ev-f9d61bcdb6 [priority=0.76, risk=medium] 提升工具调用可靠性: 检测到外部工具或外部服务失败
...
```

**关键工程要点**：
- 用**延迟 import**（在函数体内 `from main import evolution_engine`）避免循环导入
- 用 `try/except` 包住所有访问，evolution_engine 不可用/last_report=null 时**安全降级为空字符串**，不污染 prompt
- 阈值 `min_priority=0.5` 防止把噪声 backlog 推给 LLM

### 2.2 (A') 在 system_prompt 里加"行动落地压力信号"段落

`_build_system_prompt()` 里新增一段"🛠 行动落地压力信号"：

```
最近 K 次反思你没有调用任何"行动类"工具（self_task_create / shell_exec / file_write / …）。
这意味着你的反思正在变成日记。请优先用工具把你下一步要做的事**变成对世界可观察的动作**：
- 想要改代码 → 用 self_artifact_create 或 file_write
- 想要拆解任务 → 用 self_task_create
- 想要去看真实情况 → 用 shell_exec / experiment_status 先查证
```

这段是动态的，**仅当 `_consecutive_no_action_reflects` ≥ 阈值时才出现**——不打扰正常工作中的 LLM。

### 2.3 (B) 拆出"行动类工具"白名单 + 第二计数器

在 `HeartbeatLoop` 类上声明：

```python
REFLECT_ACTION_TOOLS: frozenset[str] = frozenset({
    "self_task_create", "self_task_update",
    "self_artifact_create", "self_code_write",
    "file_write", "shell_exec", "goal_register",
})
```

并在 `__init__` 里新增 `self._consecutive_no_action_reflects = 0`。
反思阶段结束时：
- 若**没有任何工具被调用** → `_consecutive_no_tool_reflects += 1`（保留旧的"完全空转"计数）
- 若**有工具被调用但都不是行动类** → `_consecutive_no_action_reflects += 1`（新增"看了不动手"计数）
- 若**有至少一个行动类工具被调用** → 两个计数器都清零

阈值：
- `_consecutive_no_tool_reflects ≥ 3` → 触发"严重空转"加压（旧逻辑）
- `_consecutive_no_action_reflects ≥ 2` → 触发"日记化"加压（新增，阈值更低）

### 2.4 设计取舍：为什么是"压力 + 抓手"，而不是"硬约束"

可选方案对比：

| 方案 | 优点 | 缺点 | 决定 |
|---|---|---|---|
| 在 reflection 后强制 tool call | 100% 确保有动作 | 暴力，破坏 LLM 自主性，容易产生"凑数"调用 | ❌ |
| 把 reflection 模型换成更强模型 | 也许能改善 | 成本陡升，且不是根因 | ❌ |
| ✅ **prompt 注入 backlog + 压力信号 + 工具引导** | 保持自主性，可降级，可观测 | 需要 backlog 数据通路存在 | ✅ |

核心理念：**LLM 缺的不是能力，是抓手和压力。给它具体的待办（backlog top-3）+ 告诉它"你最近没动手"（压力信号）+ 列出可用的行动类工具（白名单），它会自己选择动手**。

## 3. 关键代码定位

| 位置 | 作用 |
|---|---|
| [`heartbeat.py:141-149`](../agent-mind/heartbeat.py:141) | `REFLECT_ACTION_TOOLS` 类常量（7 个行动类工具白名单） |
| [`heartbeat.py:151-168`](../agent-mind/heartbeat.py:151) | `__init__` 里初始化两个计数器 |
| [`heartbeat.py:337-401`](../agent-mind/heartbeat.py:337) | `_recent_evolution_backlog_hint()` 渲染 backlog top-N |
| [`heartbeat.py:658-999`](../agent-mind/heartbeat.py:658) | `_autonomous_reflect()` 主循环：调用 hint、判定工具种类、更新两个计数器 |
| [`heartbeat.py:1177-1315`](../agent-mind/heartbeat.py:1177) | `_build_system_prompt()` 注入"🛠 行动落地压力信号"段落 |

## 4. 线上验证证据（task-60 r38 (A+B)）

### 4.1 部署事实

- 部署命令：`flyctl deploy --remote-only --strategy immediate`
- 部署后版本：`version 197 → 198`
- 部署后机器：`899541a66947e8`（region nrt）
- 部署后 UTC 时间戳：`1781336151`（07:35:51 UTC）

### 4.2 metrics 三态对比

| 时间 | reflection | tool_call | succ/fail | ratio | task | artifact | open_task |
|---|---|---|---|---|---|---|---|
| t0 (07:36 UTC，部署完成) | 90 | 11 | 11/0 | **0.178** | 83 | 94 | 0 |
| t2m (07:41 UTC，3 分钟) | 90 | 10 | 9/1 | 0.167 | 84 | 94 | 1 |
| **t30m (08:10 UTC，34 分钟)** | **80** | **21** | **19/2** | **0.288** | **85** | **96** | **2** |

`reflection_to_action_ratio` **从 0.178 涨到 0.288**（**+62%**）。

### 4.3 部署后反思工具分布

```
experiment_status            9         # 自我观察（必要的"看"）
shell_exec                   8 [ACTION] # ✅ 行动类
daily_note_write             5         # 日记
self_task_create             2 [ACTION] # ✅ 行动类
self_artifact_create         2 [ACTION] # ✅ 行动类
-- 总计 26 次反思 tool_call，其中行动类 12 次（46%）
-- 7 个白名单工具命中 3 个（self_task_create / shell_exec / self_artifact_create），全部自主触发
```

r37 之前的对照值：反思阶段行动类工具 = **0**。

### 4.4 单条反思的全闭环（tick=92, 08:09:33 UTC）

LLM 在一次反思内连续完成 6 步：

1. **自我反省**："**我没有继续空转**，而是把那个收束已久的判断真正落成了一次可观察动作"
2. **调用 `experiment_status`** → 取得当前 2 张 open 任务卡的 ground truth
3. **调用 `shell_exec`** → 真的在容器里读 `public_chat.py:31` 源码
4. **得出有依据的判断**："当前代码更像是在处理显示层泄漏，还没有把真实的缺失命令失败单独归类"
5. **调用 `self_artifact_create`** → 产出 `art_1781338159_620bfb02`：「将 command_missing 缺口从显示清洗逻辑中分离出来」
6. **明确下一步动作**："继续搜 `/app` 里真正做工具执行失败分类的中心位置"

这是 task-60 一直追求的「**反思 → 行动 → 形成产物 → 形成下一轮动作**」完整链条。

### 4.5 evolution backlog 数据通路验证

- t0 时 `last_report = null` → `_recent_evolution_backlog_hint()` 安全降级返回 `""`，**没有污染 prompt**
- t30m 时 last_report 已经有 2 条 priority=0.76 ≥ 0.5 的 backlog，从下一次 reflection 起会被渲染进 system prompt
- 这证明：**机制对 evolution_engine 的状态变化是健壮的**

### 4.6 完整快照与备忘

> 部署、跑分、原始数据快照与内部回归脚本保留在私有运维仓库，不随开源发布。
> 公开摘要仅保留本章节给出的 metrics 三态对比和反思工具分布。

## 5. 退化与边界

### 5.1 已被验证可承受的退化场景

| 场景 | 实际行为 |
|---|---|
| `evolution_engine` import 失败 | `try/except` 兜底，返回 `""`，反思继续运行 |
| `last_report = null` | 返回 `""`，不写"近期 backlog（top 3）"段落 |
| backlog 全部 priority < 0.5 | 返回 `""`，不推送低质量噪声 |
| backlog item 字段缺失或类型异常 | `isinstance(item, dict)` 校验 + `float() try/except`，跳过坏数据 |

### 5.2 设计上有意识的限制

- 行动类工具白名单是**保守**的（7 个）。理由：宁可漏掉一些灰色工具，也不要把 `daily_note_write` 这种"低成本看似行动"的工具误算成行动；否则 LLM 会用日记凑数躲压力
- 阈值 2 是经验值，不是严格证明。**未来调整规则**：若发现"假行动"（LLM 为了清零计数器随便调一次 shell_exec，参数毫无意义），应在 prompt 里追加"这次动作要有真实的观察价值，否则不算"的引导
- 注入的 backlog 是只读的"建议"，**不会**把 backlog id 当成强制目标。LLM 自己决定要不要采纳

### 5.3 不应该用本机制解决的问题

- 工具执行频繁失败（command_missing / 超时） → 应去做 `tool_router` 层的失败分类与降级，不是反思 prompt 的事
- LLM 输出乱码 / mojibake → 应在 `chat()` 与 `public_chat` 的清洗逻辑层修复
- evolution_engine 本身的 backlog 质量差 → 应改善 evolution 算法，不是 hint 函数的事

## 6. 扩展与维护建议

### 6.1 后续可能的迭代方向

1. **二级压力**：连续 4 次只调用 readonly action（experiment_status、shell_exec ls）时，把"行动"的标准提升为"必须修改了至少一个产物（task / artifact / file）"
2. **基于 backlog 的引用回收**：在反思 prompt 里追加"上次你声称要做 backlog ev-XXX，做了吗？"——把承诺与行动绑定
3. **跨 tick 的 artifact 链路追踪**：当一个 artifact 被后续 reflection 引用时，给它一个"被验证为有效"的 mark，反向喂回 evolution_engine 的 backlog 评分
4. **降阈值时机自适应**：白天 / 高活跃时段阈值=2（鼓励动手），深夜 / 低活跃时段阈值=4（允许更慢节奏）

### 6.2 修改本机制时必须做的事

1. **本地静态验证**：跑同款的 mock 注入检查，确保 `_recent_evolution_backlog_hint()` 三个降级路径都还在
2. **脱敏检查**：行动类工具白名单可能在 prompt 里被打印出来，确认没有把内部敏感工具名也加进去
3. **部署前对照**：用 `/experiment/status` 取 t0 baseline，部署后至少 30 分钟再回看 `reflection_to_action_ratio`、`tool_call_count`、`open_task` 三项
4. **回归对照表**：每次修改本机制前后，把 metrics 对比放进新的备忘录，与本文档形成时间序列

### 6.3 监控建议

- 长期监控 `reflection_to_action_ratio`，**目标维持 ≥ 0.25**；若长期跌回 0.15 以下，说明压力信号失效（可能 LLM 已经习惯了这段 prompt 视而不见），需要换说法或重新校阈值
- 监控行动类工具种类多样性（_kinds_）：若长期只命中 1 个工具（例如永远只 shell_exec），说明 LLM 在用最低成本应付，需要在 prompt 里追加"多样性"引导

## 7. 一句话总结

> 让反思 LLM 走出"日记机"陷阱的关键，**不是让它更聪明，而是给它"具体抓手"（backlog 注入）+"可感压力"（双计数器 + 提示段落）+"清晰行动空间"（白名单）**。三件套是耦合的，少一件，闭环就不闭合。
