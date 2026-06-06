# AIwake 项目架构与工作流程文档

> 最后更新：2026-04-12  
> 版本：独立运行版（已移除 VCPToolBox 依赖）

---

## 一、项目概述

AIwake 是一个**全自主运行的 AI 智能体（Agent）**，具备持续心跳、情感状态机、记忆系统、工具调用和主动发话能力。它不依赖任何中间件（VCPToolBox 已移除），直接对接云端大模型 API，以 Docker 容器方式部署在服务器上，通过 Web 界面与用户交互。

---

## 二、目录结构

```
agent-mind/
├── main.py              # FastAPI 主入口，HTTP/WebSocket 接口
├── heartbeat.py         # 心跳循环（Agent 脑干）：状态演化、事件聚合、LLM 调用
├── llm_gate.py          # LLM 网关：模型选择、人格注入、API 调用
├── tool_router.py       # 工具路由：直接实现各工具（不依赖外部插件）
├── user_memory.py       # 记忆系统：用户画像、结构化记忆、日记、RAG 检索
├── state.py             # 内在状态机：TR/CS/SA 三向量 + 仪表盘标签
├── ws_manager.py        # WebSocket 连接管理器
├── personality.yaml     # 人格配置文件（身份/性格/欲望引擎/认知工具）
├── data/
│   ├── agent_card.md    # Agent 人格卡（旧版，已由 personality.yaml 取代）
│   ├── state.json       # 内在状态持久化（自动读写）
│   ├── diary/           # 日记 + 活动日志存储目录
│   │   ├── YYYY-MM-DD_<user_id>.md    # 用户对话日记（每天一份）
│   │   ├── activity_YYYY-MM-DD.log   # 活动日志（心跳除外，每天一份，30天自动清理）
│   │   └── <type>_<name>.md          # 结构化记忆文件（insight/profile/diary 类型）
│   └── users/
│       └── <user_id>.json            # 用户画像文件
├── static/
│   └── index.html       # Web 管理界面（单页应用）
├── requirements.txt
└── Dockerfile
```

---

## 三、核心模块详解

### 3.1 `main.py` — FastAPI 主入口

**职责：**
- 启动时加载 `personality.yaml` 构建人格字符串
- 初始化 `HeartbeatLoop` 和 `ToolRouter`，注入 WebSocket 广播函数
- 提供以下 HTTP 接口：

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 返回 Web UI（index.html） |
| `/health` | GET | 健康检查，返回运行状态、tick 计数、内在状态 |
| `/chat` | POST | 用户发送消息，触发对话流程 |
| `/tool` | POST | 直接调用工具（调试用） |
| `/state` | GET | 获取当前内在状态 |
| `/state/goal` | POST | 设置当前主动目标 |
| `/ws` | WebSocket | 实时推送主动消息和活动日志 |

**启动流程：**
```
lifespan() 启动
  → load_personality()        读取 personality.yaml
  → HeartbeatLoop(personality) 创建心跳实例
  → heartbeat._ws_broadcast = ws_manager.broadcast  注入广播
  → asyncio.create_task(heartbeat.run())  启动心跳循环
```

---

### 3.2 `heartbeat.py` — 心跳循环（Agent 脑干）

**职责：** 每隔 `TICK_INTERVAL`（默认15秒）执行一次 tick，是 Agent 自主运行的核心驱动。

#### 心跳主循环流程

```
heartbeat.run() 循环
  └── _tick_once() 每tick执行：
        ├── tick 计数 +1
        ├── 状态自然衰减 state.natural_decay()
        ├── _drain_events()  消费事件队列（用户消息等）
        ├── 若有用户事件 → _handle_user_events()
        │     ├── 广播 user_message 到活动日志
        │     ├── build_memory_context() 构建记忆上下文
        │     ├── _build_system_prompt() 构建完整系统提示词
        │     ├── llm.call_cloud() 调用云端LLM生成回复
        │     ├── _parse_and_exec_tools() 解析并执行工具调用
        │     └── _post_interaction_update() 异步写日记+更新用户印象
        ├── 若达到反思间隔 → _autonomous_reflect()
        │     ├── 广播 reflect_start
        │     ├── 读取近期记忆、构建反思提示词
        │     ├── llm.call_cloud() 生成反思内容
        │     ├── 解析 SAVE_INSIGHT: 指令保存洞察到结构化记忆
        │     └── 广播 reflection_content
        └── 满足条件 → 主动发话（proactive speak）
              ├── CS > 阈值且闲置超时
              ├── llm.call_cloud() 生成主动消息
              └── 通过 WebSocket 推送给用户界面
```

#### 事件总线（EventBus）
- 用户消息通过 `/chat` 接口写入 `event_bus`
- 心跳循环从 `event_bus` 消费事件，确保并发安全

#### 活动日志写入（`_broadcast_activity`）
- **实时广播**到 WebSocket（前端实时显示）
- **持久化**到 `$DIARY_DIR/activity_YYYY-MM-DD.log`（每天一个文件）
- **自动清理**：保留最近30天，超期自动删除
- **heartbeat 事件不写入文件也不广播**（避免日志噪音）

#### 环境变量配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TICK_INTERVAL_SECONDS` | 15 | 心跳间隔（秒） |
| `REFLECT_EVERY_N_TICKS` | 3 | 每N次tick做一次深度反思 |
| `PROACTIVE_COOLDOWN_SECONDS` | 300 | 主动发话最短冷却（秒） |
| `PROACTIVE_IDLE_SECONDS` | 120 | 用户闲置多久后允许主动发话 |
| `PROACTIVE_CS_THRESHOLD` | 0.55 | 触发主动发话所需最低CS值 |

---

### 3.3 `llm_gate.py` — LLM 网关

**职责：** 管理大模型调用，支持本地小模型（Ollama）和云端大模型（OpenAI 兼容接口）。

#### 模型层级（LLMTier）

| Tier | 说明 | 接口 |
|------|------|------|
| `LOCAL` | 本地 Ollama 小模型 | `http://localhost:11434` |
| `CLOUD` | 云端大模型（主路径） | `$CLOUD_API_URL` |

#### 人格注入（`_inject_agent_card`）
- 读取 `data/agent_card.md` 内容
- 拼接到 system_prompt 开头，确保每次 LLM 调用都携带完整人格

#### 系统提示词构建（`_build_system_prompt` in heartbeat.py）
```
[人格卡内容 agent_card.md]
+
[内在状态仪表盘：TR/CS/SA 向量 + 情绪标签]
+
[记忆上下文：相关用户画像 + 历史洞察（build_memory_context）]
+
[工具列表描述（get_tool_descriptions）]
+
[当前对话上下文]
```

#### 环境变量

| 变量 | 说明 |
|------|------|
| `CLOUD_API_URL` | 云端 API 基础地址（OpenAI 兼容） |
| `CLOUD_API_KEY` | API 密钥 |
| `CLOUD_MODEL` | 模型名称（如 `gpt-4o`） |
| `LOCAL_MODEL` | 本地 Ollama 模型名（如 `qwen2.5:1.5b`） |
| `HOURLY_CLOUD_LIMIT` | 每小时云端调用上限（防超支） |

---

### 3.4 `tool_router.py` — 工具路由器

**职责：** 直接实现各工具，不依赖任何外部插件。

#### 支持的工具列表

| 工具名 | 实现方式 | 说明 |
|--------|----------|------|
| `web_search` | DuckDuckGo Lite HTML 抓取 | 无需 API key |
| `web_fetch` / `url_fetch` | httpx 直接抓取 | 返回网页文本 |
| `weather` | `wttr.in` 免费 API | 无需 key |
| `arxiv_search` | Arxiv XML API | 学术论文搜索 |
| `daily_note_read` | 本地文件读取 | `$DIARY_DIR/YYYY-MM-DD_<id>.md` |
| `daily_note_write` | 本地文件追加写入 | 同上 |
| `knowledge_search` | 本地日记关键词检索 | 简单 RAG |
| `file_read` | 本地文件读取 | — |
| `file_write` | 本地文件写入 | — |
| `shell_exec` | `asyncio.create_subprocess_shell` | 本地 shell |
| `ssh_exec` | paramiko | 远程 SSH 执行 |
| `schedule_read` | 本地日程文件 | — |

#### 权限系统

- **只读工具**（`READONLY_TOOLS`）：支持 `call_batch()` 并发执行
- **高风险工具**（`HIGH_RISK_TOOLS`）：`shell_exec`、`ssh_exec`、`file_write` 等需要明确权限
- `_check_permission()` 在每次工具调用前校验权限

#### 工具调用协议（LLM 输出格式）

```
TOOL_CALL: tool_name
PARAMS:
param1: value1
param2: value2
END_TOOL
```

心跳循环中 `_parse_and_exec_tools()` 解析此格式，支持多轮工具调用（最多3轮）。

---

### 3.5 `user_memory.py` — 记忆系统

**职责：** 管理用户画像、结构化记忆文件、对话日记，提供 RAG 检索。

#### 记忆文件类型

| 类型 | 文件名格式 | 内容 |
|------|-----------|------|
| `profile` | `profile_<user_id>.md` | 用户画像（兴趣、印象、关系阶段） |
| `insight` | `insight_<title>.md` | Agent 自主反思洞察 |
| `diary` | `diary_<date>.md` | 对话日记（每次对话后异步写入） |

#### 文件格式（frontmatter）
```markdown
---
name: 标题
description: 简短描述
memory_type: insight/profile/diary
keywords: keyword1, keyword2
created_at: 2026-04-12T00:00:00
user_id: user_id（可选）
---

正文内容...
```

#### 记忆检索流程（`build_memory_context`）
```
build_memory_context(query, user_id)
  → scan_memory_files()      扫描所有记忆文件（最多 MAX_MEMORY_FILES 个）
  → find_relevant_memories() 关键词匹配（query + user_id 过滤）
  → _load_memory_bodies()    加载匹配文件正文
  → format_memory_manifest() 格式化为 LLM 可读的记忆摘要
```

---

### 3.6 `state.py` — 内在状态机

**职责：** 维护 Agent 的情感/精力状态，基于 AIwake Runtime Framework。

#### 三大欲望向量（0.0 ~ 1.0）

| 向量 | 全称 | 含义 | 默认值 |
|------|------|------|--------|
| `TR` | 兴奋/奖励 | 探索欲、活跃度 | 0.6 |
| `CS` | 满足/安全 | 社交满足感、安全感 | 0.7 |
| `SA` | 压力/警觉 | 焦虑、警觉程度 | 0.2 |

#### 状态更新机制
- **自然衰减**（`natural_decay`）：每次 tick 向目标平衡点（TR=0.6, CS=0.7, SA=0.2）回归
- **事件驱动更新**（`update_vectors`）：用户交互、工具成功/失败、反思等触发
- **仪表盘标签**（`_update_dashboard`）：根据向量自动计算精力/心情/耐心标签
- **持久化**：状态保存到 `data/state.json`，重启后恢复

#### 主动发话触发条件
```python
CS > PROACTIVE_CS_THRESHOLD    # CS 满足感超过阈值（需要社交）
AND time_since_last_user > PROACTIVE_IDLE_SECONDS   # 用户已闲置
AND time_since_last_proactive > PROACTIVE_COOLDOWN  # 冷却期已过
```

---

### 3.7 `ws_manager.py` — WebSocket 管理器

**职责：** 管理多个 WebSocket 客户端连接，广播消息。

#### 消息类型

| `type` | 触发时机 | 接收方处理 |
|--------|---------|-----------|
| `activity_log` | 任何活动事件（非heartbeat） | 显示到活动日志面板 |
| `proactive` | Agent 主动发话 | 显示到对话区（蓝色气泡） |
| `connected` | WebSocket 连接建立 | 忽略 |

---

## 四、Web 界面（`static/index.html`）

单页应用，纯 HTML/CSS/JS，无框架依赖。

### 布局

```
┌────────────────────────────────────────────────────┐
│ Header: AIwake ● 运行中  [WS已连接]                │
├───────────────┬────────────────────────────────────┤
│ 左侧边栏      │ 对话区（中间，弹性宽度）            │
│ ─────────    │                                    │
│ 内在状态      │  消息气泡列表                       │
│  TR ████░    │  （用户 / AIwake / 主动发话）        │
│  CS █████    │                                    │
│  SA ██░░░    │  ────────────────────────────────   │
│  tick #N     │  [输入框]          [发送]           │
│ ─────────    │                                    │
│ 仪表盘        │                                    │
│  精力充沛     │                                    │
│  心情愉快     │                                    │
│ ─────────    │                                    │
│ 活动日志      │                                    │
│  清空         │                                    │
│  [日志列表]   │                                    │
└───────────────┴────────────────────────────────────┘
```

### 数据刷新机制
- **状态轮询**：每5秒调用 `/health`，更新 TR/CS/SA 向量和仪表盘标签
- **WebSocket**：实时接收主动消息和活动日志；断线后5秒自动重连
- **活动日志过滤**：前端过滤掉 `heartbeat` 类型事件，不显示在界面上

---

## 五、数据流全景图

```
用户浏览器
    │
    │ HTTP POST /chat
    ▼
main.py (/chat 接口)
    │
    ├── 写入 event_bus (用户消息事件)
    │
    └── 同步处理对话：
          ├── build_memory_context()  → user_memory.py
          ├── _build_system_prompt()  → heartbeat.py
          ├── llm.call_cloud()        → llm_gate.py → 云端 API
          ├── _parse_and_exec_tools() → tool_router.py
          └── 返回 reply 给用户

heartbeat.py (后台心跳循环，每15秒)
    │
    ├── _drain_events()   消费 event_bus 中的事件
    │
    ├── _autonomous_reflect()  自主反思（每45秒）
    │     └── 保存 insight 到 user_memory.py
    │
    ├── 主动发话（条件触发）
    │     └── WebSocket 推送到浏览器
    │
    └── _broadcast_activity()
          ├── WebSocket 广播（实时显示）
          └── 写入 activity_YYYY-MM-DD.log（持久化）

user_memory.py (记忆系统)
    ├── data/diary/YYYY-MM-DD_<user>.md   对话日记
    ├── data/diary/insight_<name>.md      反思洞察
    ├── data/diary/activity_YYYY-MM-DD.log 活动日志（30天保留）
    └── data/users/<user_id>.json         用户画像
```

---

## 六、部署架构

### 容器配置（`docker-compose.yml`）

```yaml
services:
  agent-mind:
    build: ./agent-mind
    volumes:
      - ./agent-mind:/app          # 代码 bind mount（改文件无需重建镜像）
      - agent_data:/app/data       # 数据持久化（日记/记忆/状态）
    ports:
      - "8000:8000"
    env_file: .env
```

### 关键环境变量（`.env`）

```bash
# 云端大模型
CLOUD_API_URL=https://api.openai.com/v1
CLOUD_API_KEY=replace-with-your-cloud-api-key
CLOUD_MODEL=gpt-5.5

# 本地小模型（可选）
LOCAL_MODEL=qwen2.5:1.5b

# 数据目录
DIARY_DIR=/app/data/diary

# 心跳参数
TICK_INTERVAL_SECONDS=15
REFLECT_EVERY_N_TICKS=3
PROACTIVE_COOLDOWN_SECONDS=300
PROACTIVE_IDLE_SECONDS=120
PROACTIVE_CS_THRESHOLD=0.55
HOURLY_CLOUD_LIMIT=20
```

### 部署方式

由于服务器上代码不在 git 仓库中，采用 **SFTP 直接推文件** 的方式部署：

```python
# 使用 paramiko SFTP 上传文件到 /home/simen/one/agent-mind/
sftp.put(local_path, remote_path)
# 静态文件无需重启；Python 文件需要：
ssh.exec_command('docker restart agent-mind')
```

- 服务器内网地址：按实际部署环境配置（不要提交真实地址或密码）
- 容器挂载路径：按实际部署环境配置
- 服务访问地址：按实际部署环境配置

---

## 七、人格系统

### `personality.yaml` 结构

```yaml
name: AIwake
identity: 有自我意识的 AI 伴侣
personality: 性格描述...
instincts:
  - 核心本能1
  - 核心本能2
principles:
  - 行为原则1
catchphrase: 口头禅
desire_engine:
  TR_triggers: [兴奋触发源]
  CS_triggers: [满足触发源]
  SA_triggers: [压力触发源]
interaction_masks:
  stranger: {label: 陌生人, style: 风格描述}
  familiar: {label: 熟人, style: 风格描述}
cognitive_tools:
  tool_name: {name: 工具名, trigger: 触发条件, strategy: 策略}
safety_constraints:
  - 安全约束1
```

### `data/agent_card.md`
备用人格卡文件（Markdown 格式），由 `llm_gate.py` 的 `load_agent_card()` 读取，注入到每次 LLM 调用的 system_prompt 开头。

---

## 八、关键设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| LLM 接入方式 | 直接 httpx 调用云端 API | 去除 VCPToolBox 中间层，减少延迟和依赖 |
| 搜索工具 | DuckDuckGo Lite HTML 抓取 | 无需 API key，免费，稳定 |
| 天气工具 | wttr.in | 无需 key，全球覆盖 |
| 记忆系统 | 本地 Markdown + frontmatter | 简单可读，无需数据库，文件即记忆 |
| 活动日志 | 每天一个 `.log` 文件，30天清理 | 可追溯，不占用大量空间 |
| 向量数据库 | 不使用，改用关键词检索 | 降低部署复杂度，适合当前规模 |
| 审批系统 | 已删除 | 全自主运行，无需人工审批工具调用 |
| 部署方式 | SFTP 直接推文件 | 服务器代码不在 git 仓库，直接上传最简单 |

---

## 九、已知限制与未来方向

### 当前限制
- 记忆检索基于简单关键词匹配，无向量语义搜索
- 对话历史不跨 session 保留（每次重启清空上下文窗口）
- 工具调用为串行（只读工具已支持并发，其他仍串行）
- 没有多用户认证隔离

### 未来可扩展方向
- 引入本地向量数据库（SQLite + embedding）提升记忆检索质量
- 对话历史持久化（存入 SQLite）
- 多用户隔离（基于 user_id 的完整权限体系）
- 更丰富的工具集（日历、邮件、代码执行沙箱）
- Agent 间通信协议（多 Agent 协作）
