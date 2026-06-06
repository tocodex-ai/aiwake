# Hermes Agent 本地下载与详细分析

## 1. 本地落地结果

- 本地目录：`hermes-agent/`
- 上游仓库：`NousResearch/hermes-agent`
- 当前提交：`764536b6`
- 当前描述版本：`v2026.4.16-16-g764536b6`
- 子模块状态：存在 `tinker-atropos` 子模块，但当前未初始化到工作树

## 2. 项目定位

Hermes Agent 是一个偏“平台级”的 AI Agent 框架，而不是单一聊天机器人程序。

它的核心卖点不是只有一次对话式工具调用，而是将以下能力整合成统一系统：

- 终端交互式 CLI / TUI
- 多消息平台 Gateway
- 多 Provider 模型适配
- 大规模 Tool Registry + Toolset 系统
- 持久会话与全文检索
- Skills / Memory / MCP / Cron / ACP
- RL / trajectory 数据生成

换句话说，它更像一个“可长期运行的代理操作系统”，而不是一个轻量 API 服务。

## 3. 运行与依赖特征

### Python 侧

从 [`pyproject.toml`](../hermes-agent/pyproject.toml:1) 可以确认：

- 最低 Python 版本：`>=3.11`
- 主体依赖围绕 `openai`、`anthropic`、`httpx`、`rich`、`pydantic`、`prompt_toolkit`
- 功能通过 extras 拆分，例如：`messaging`、`cron`、`mcp`、`voice`、`web`、`acp`、`rl`
- 脚本入口：
  - [`hermes`](../hermes-agent/pyproject.toml:117)
  - [`hermes-agent`](../hermes-agent/pyproject.toml:118)
  - [`hermes-acp`](../hermes-agent/pyproject.toml:120)

### Node 侧

从 [`package.json`](../hermes-agent/package.json:1) 看，Node 主要用于浏览器自动化相关能力：

- 依赖 `agent-browser`
- 依赖 `@askjo/camofox-browser`
- Node 版本要求 `>=20`

### 平台限制

官方 README 明确指出：

- 原生 Windows 不支持
- 推荐 Linux / macOS / WSL2 / Termux

因此当前 Windows 工作区适合“下载 + 静态阅读分析”，不适合直接按官方路径完整运行。

## 4. 顶层结构分析

仓库根目录包含以下关键区域：

- [`run_agent.py`](../hermes-agent/run_agent.py:1)：核心代理循环
- [`cli.py`](../hermes-agent/cli.py:1)：交互式终端 UI
- [`hermes_cli/main.py`](../hermes-agent/hermes_cli/main.py:1)：统一命令入口
- [`model_tools.py`](../hermes-agent/model_tools.py:1)：工具发现与分发编排层
- [`hermes_state.py`](../hermes-agent/hermes_state.py:1)：SQLite + FTS5 会话状态存储
- [`tools/`](../hermes-agent/tools)：工具实现与注册中心
- [`gateway/`](../hermes-agent/gateway)：多平台消息网关
- [`agent/`](../hermes-agent/agent)：提示词、记忆、压缩、模型元数据等内部能力
- [`cron/`](../hermes-agent/cron)：定时任务
- [`plugins/`](../hermes-agent/plugins)：插件体系
- [`skills/`](../hermes-agent/skills) / [`optional-skills/`](../hermes-agent/optional-skills)：技能系统
- [`acp_adapter/`](../hermes-agent/acp_adapter)：编辑器协议接入
- [`web/`](../hermes-agent/web) / [`website/`](../hermes-agent/website)：Web 组件与文档站点
- [`tests/`](../hermes-agent/tests)：测试体系

## 5. 核心入口与控制流

### 5.1 CLI 命令入口

[`hermes_cli/main.py`](../hermes-agent/hermes_cli/main.py:1) 不是简单的命令分发脚本，而是统一前门：

- `hermes`
- `hermes chat`
- `hermes gateway`
- `hermes setup`
- `hermes doctor`
- `hermes cron`
- `hermes acp`
- `hermes sessions browse`

该文件还做了几件很关键的事情：

1. 在模块导入前预处理 profile，确保 `HERMES_HOME` 正确切换
2. 统一加载 `~/.hermes/.env` 与项目 `.env`
3. 提前初始化日志系统
4. 根据配置提前桥接网络与 provider 相关参数

这说明 Hermes 的 CLI 不是“附属脚本”，而是完整 runtime 的操作入口。

### 5.2 TUI 层

[`cli.py`](../hermes-agent/cli.py:1) 使用 `prompt_toolkit` 构建复杂交互终端：

- 多行输入
- 历史记录
- 补全菜单
- 固定输入区域
- 更强的输出控制

它本质上更接近 Claude Code 风格的 TUI，而不是普通 REPL。

### 5.3 代理主循环

[`run_agent.py`](../hermes-agent/run_agent.py:535) 中的 [`AIAgent`](../hermes-agent/run_agent.py:535) 是系统中最核心的类。

从构造参数可见，它同时承担：

- provider / api_mode 解析
- toolsets 启用与禁用
- session / gateway / user 上下文
- callback 钩子
- context files / memory 控制
- fallback model
- checkpoint
- iteration budget

这意味着 Hermes 采用的是**单核心编排类**架构：大量入口最终都收敛到 [`AIAgent`](../hermes-agent/run_agent.py:535)。

## 6. Agent 内部模块分工

[`agent/`](../hermes-agent/agent) 目录体现了 Hermes 对“核心循环旁路逻辑”的模块化拆分。主要包括：

- [`prompt_builder.py`](../hermes-agent/agent/prompt_builder.py:1)：系统提示组装
- [`context_compressor.py`](../hermes-agent/agent/context_compressor.py:1)：上下文压缩
- [`prompt_caching.py`](../hermes-agent/agent/prompt_caching.py:1)：缓存控制
- [`memory_manager.py`](../hermes-agent/agent/memory_manager.py:1)：记忆编排
- [`memory_provider.py`](../hermes-agent/agent/memory_provider.py:1)：记忆 provider 抽象
- [`model_metadata.py`](../hermes-agent/agent/model_metadata.py:1)：模型上下文长度、元数据与估算
- [`retry_utils.py`](../hermes-agent/agent/retry_utils.py:1)：退避与重试
- [`error_classifier.py`](../hermes-agent/agent/error_classifier.py:1)：API 错误分类与降级
- [`usage_pricing.py`](../hermes-agent/agent/usage_pricing.py:1)：token / cost 估算
- [`trajectory.py`](../hermes-agent/agent/trajectory.py:1)：轨迹保存

这个拆分说明 Hermes 已经意识到主循环文件过大，所以把“周边智能能力”逐步抽到 [`agent/`](../hermes-agent/agent) 下，但实际总编排仍保留在巨型 [`run_agent.py`](../hermes-agent/run_agent.py:1)。

## 7. Tool System 分析

### 7.1 注册中心设计

[`tools/registry.py`](../hermes-agent/tools/registry.py:1) 是 Hermes 工具系统最值得借鉴的部分。

其核心特征：

- 每个工具文件在 import 时通过 `registry.register()` 自注册
- 使用 AST 扫描顶层 `registry.register()` 调用，自动发现可注册工具模块
- 用统一 `ToolEntry` 保存 schema、handler、toolset、availability check 等元数据
- 防止工具命名冲突与跨 toolset 阴影覆盖

这比手写一个大字典维护工具定义更可扩展。

### 7.2 工具编排层

[`model_tools.py`](../hermes-agent/model_tools.py:1) 并不直接实现工具，而是做几件事：

1. 触发内建工具发现
2. 尝试加载 MCP 工具
3. 尝试加载插件工具
4. 提供 `get_tool_definitions()` 给模型层使用
5. 提供 sync/async 桥接能力，处理 event loop 生命周期

也就是说，Hermes 将“工具实现”和“工具暴露给模型的编排逻辑”分离了。

### 7.3 工具体量与边界

从 [`tools/`](../hermes-agent/tools) 目录可以看出，工具系统已经远超基础读写：

- 终端：[`terminal_tool.py`](../hermes-agent/tools/terminal_tool.py:1)
- 文件：[`file_tools.py`](../hermes-agent/tools/file_tools.py:1)
- 浏览器：[`browser_tool.py`](../hermes-agent/tools/browser_tool.py:1)
- MCP：[`mcp_tool.py`](../hermes-agent/tools/mcp_tool.py:1)
- 代码执行：[`code_execution_tool.py`](../hermes-agent/tools/code_execution_tool.py:1)
- Delegate：[`delegate_tool.py`](../hermes-agent/tools/delegate_tool.py:1)
- Memory：[`memory_tool.py`](../hermes-agent/tools/memory_tool.py:1)
- Session Search：[`session_search_tool.py`](../hermes-agent/tools/session_search_tool.py:1)
- Skills：[`skills_tool.py`](../hermes-agent/tools/skills_tool.py:1)
- Todo：[`todo_tool.py`](../hermes-agent/tools/todo_tool.py:1)
- Vision / TTS / Voice / Image / Web 等扩展工具

这使 Hermes 具备“代理平台”性质，而非单用途 Agent。

## 8. 持久化与会话检索

[`hermes_state.py`](../hermes-agent/hermes_state.py:1) 表明 Hermes 使用 SQLite 作为默认会话数据库，并内置 FTS5：

- `sessions` 表保存会话元信息
- `messages` 表保存消息级明细
- `messages_fts` 提供全文搜索
- 使用 WAL 模式降低并发读写冲突
- 内建 jitter retry 避免 SQLite 锁竞争形成 convoy

这套设计相较于纯 JSONL 或纯内存方案，明显更适合：

- CLI 与 Gateway 并发共用
- 历史消息搜索
- 跨会话回忆
- 长期运行与压缩后的 lineage 追踪

## 9. Gateway 架构

[`gateway/run.py`](../hermes-agent/gateway/run.py:1) 是另一个大型核心模块。

它在启动前就处理了：

- SSL 证书自动探测
- `HERMES_HOME` 环境解析
- `.env` / `config.yaml` 加载
- 将嵌套配置桥接为环境变量

这说明 Gateway 并不是简单 webhook 包装层，而是一个独立长期运行的宿主进程。

从目录结构看，Gateway 内部还拆成：

- [`session.py`](../hermes-agent/gateway/session.py:1)：会话存储
- [`delivery.py`](../hermes-agent/gateway/delivery.py:1)：消息投递
- [`pairing.py`](../hermes-agent/gateway/pairing.py:1)：授权配对
- [`hooks.py`](../hermes-agent/gateway/hooks.py:1)：钩子系统
- [`mirror.py`](../hermes-agent/gateway/mirror.py:1)：跨会话镜像
- [`status.py`](../hermes-agent/gateway/status.py:1)：状态跟踪
- [`platforms/`](../hermes-agent/gateway/platforms)：多平台适配器

因此 Hermes 在消息生态上是“一个 gateway 平台”，不是“给 Telegram 单独写的 bot”。

## 10. Skills / Memory / MCP / Cron / ACP

这些是 Hermes 区别于普通 Agent 项目的关键：

### Skills

- Bundled skills 位于 [`skills/`](../hermes-agent/skills)
- 官方可选技能位于 [`optional-skills/`](../hermes-agent/optional-skills)
- 说明 Hermes 把“程序化经验沉淀”作为一等能力

### Memory

- 由 [`agent/memory_manager.py`](../hermes-agent/agent/memory_manager.py:1) 与 [`plugins/memory/`](../hermes-agent/plugins/memory) 共同构成
- 有 provider 化倾向，利于替换实现

### MCP

- [`tools/mcp_tool.py`](../hermes-agent/tools/mcp_tool.py:1) 负责动态工具接入
- 说明 Hermes 将外部能力扩展统一纳入 tool registry

### Cron

- [`cron/`](../hermes-agent/cron) 提供定时代理任务
- 表示任务目标是“无人值守运行的 Agent”

### ACP

- [`acp_adapter/`](../hermes-agent/acp_adapter) 暗示其可作为编辑器内 agent server
- 定位不只是 CLI bot，而是 IDE / 编辑器协作基础设施

## 11. 代码结构优点

Hermes Agent 的主要优点：

1. **平台化程度高**：CLI、Gateway、ACP、Cron、MCP 都统一在一个 runtime 内
2. **工具体系成熟**：registry + toolset + availability checks 设计明显优于硬编码路由
3. **持久化能力强**：SQLite + FTS5 + lineage 很适合长期代理
4. **扩展面广**：插件、skills、memory provider、terminal backends 都可扩展
5. **工程成熟度高**：测试目录、发布文件、release 节奏、配置组织都较完整

## 12. 主要问题与风险

Hermes Agent 也有明显代价：

1. **核心文件过大**
   - [`run_agent.py`](../hermes-agent/run_agent.py:1) 超过 11k 行
   - [`cli.py`](../hermes-agent/cli.py:1) 超过 10k 行
   - [`gateway/run.py`](../hermes-agent/gateway/run.py:1) 接近 10k 行
   - 说明虽然做了局部模块化，但主编排仍高度集中

2. **运行环境复杂**
   - Python extras 很多
   - Node 侧浏览器依赖单独存在
   - 平台接入、语音、MCP、RL 都会扩大发布和排障复杂度

3. **Windows 原生不友好**
   - 官方就不支持原生 Windows
   - 对你的当前环境而言，直接运行价值低于架构借鉴价值

4. **认知成本高**
   - 同时覆盖 CLI、TUI、Gateway、Cron、Skills、Memory、MCP、RL
   - 不适合“快速读懂后改两行”的项目切入方式

## 13. 与当前 `agent-mind` 的对照分析

### 13.1 当前 `agent-mind` 的特点

从 [`agent-mind/main.py`](../agent-mind/main.py:1) 与 [`agent-mind/tool_router.py`](../agent-mind/tool_router.py:1) 看，你当前项目更偏：

- FastAPI 服务型 Agent
- 单体式入口
- 人格与内在状态驱动
- 自定义工具路由器
- Web UI / WebSocket 广播

它更像“一个人格化自治体服务”，而不是 Hermes 那种“全平台代理操作系统”。

### 13.2 关键差异

#### 架构焦点不同

- Hermes：围绕 **通用代理平台** 设计
- `agent-mind`：围绕 **人格化 AI 服务实例** 设计

#### 工具系统复杂度不同

- Hermes：[`tools/registry.py`](../hermes-agent/tools/registry.py:1) + [`model_tools.py`](../hermes-agent/model_tools.py:1) 形成注册发现体系
- `agent-mind`：[`ToolRouter`](../agent-mind/tool_router.py:77) 主要是中心化路由类

#### 会话存储能力不同

- Hermes：[`hermes_state.py`](../hermes-agent/hermes_state.py:1) 提供 SQLite + FTS5
- `agent-mind`：当前更强调运行时状态和即时交互，长期历史检索体系相对弱

#### 入口形态不同

- Hermes：CLI / Gateway / ACP / Cron 多入口统一
- `agent-mind`：FastAPI API + WebSocket + 静态页为主

### 13.3 最值得借鉴的点

如果你不是要照搬 Hermes，而是想增强当前项目，最值得迁移的是：

1. **工具注册中心模式**
   - 把当前中心化 [`ToolRouter`](../agent-mind/tool_router.py:77) 逐步拆成自注册工具模块

2. **持久会话数据库**
   - 借鉴 [`hermes_state.py`](../hermes-agent/hermes_state.py:1) 的 SQLite + FTS5 方案

3. **Toolset / 权限分组**
   - 让工具不再只有“允许/不允许”，而是按场景分组启用

4. **Gateway/Platform 抽象层**
   - 如果未来你要接 Telegram / Discord / 微信桥，就应抽象平台入口而不是直接堆在 API 主进程里

5. **Memory Provider / Plugin 抽象**
   - 让人格记忆、长期记忆、用户模型各自可插拔

### 13.4 不建议直接照搬的点

1. 不建议一开始就复制 Hermes 的超大单体主循环
2. 不建议在当前阶段引入过多 provider / 平台 / RL / voice 复杂度
3. 不建议直接按 Hermes 的全平台野心扩展，否则会迅速抬高维护成本

更合理的做法是：**只迁移它最成熟的架构骨架，而不是复制它全部功能面。**

## 14. 后续建议

### 如果目标是“跑起来”

- 在 WSL2 中安装 Python 3.11+
- 使用官方推荐安装流程
- 按需只装必要 extras，避免 `all`

### 如果目标是“借鉴架构”

优先阅读顺序建议：

1. [`pyproject.toml`](../hermes-agent/pyproject.toml:1)
2. [`hermes_cli/main.py`](../hermes-agent/hermes_cli/main.py:1)
3. [`run_agent.py`](../hermes-agent/run_agent.py:535)
4. [`model_tools.py`](../hermes-agent/model_tools.py:1)
5. [`tools/registry.py`](../hermes-agent/tools/registry.py:1)
6. [`hermes_state.py`](../hermes-agent/hermes_state.py:1)
7. [`gateway/run.py`](../hermes-agent/gateway/run.py:1)
8. [`agent/memory_manager.py`](../hermes-agent/agent/memory_manager.py:1)

### 如果目标是“增强当前 agent-mind”

建议分三步：

1. 先引入工具注册中心
2. 再引入 SQLite 会话与搜索
3. 最后考虑平台网关与技能系统

## 15. 结论

Hermes Agent 已成功下载到本地，且确认它是一个成熟度较高、面向长期运行与多平台扩展的 AI Agent 平台。

对于当前项目而言，它最有价值的不是直接整仓照搬，而是提供了三类高价值参考：

- 工具系统工程化
- 长期会话与记忆持久化
- 多入口统一到单一 agent runtime 的平台架构

如果你下一步要继续，我建议优先做的是：**把 `agent-mind` 的工具系统和会话系统按 Hermes 的思路做一次轻量重构，而不是立即尝试全面兼容 Hermes。**

## 16. `hermes-agent` 与本项目的相同点、不同点、优势与劣势

### 16.1 相同点

两者并不是完全不同物种，它们有一些明显共同点：

1. **本质上都属于 Tool-Using Agent**
   - `hermes-agent` 的核心是 [`AIAgent`](../hermes-agent/run_agent.py:535)
   - 本项目的核心执行链围绕 [`HeartbeatLoop`](../agent-mind/heartbeat.py:85)、[`LLMGate`](../agent-mind/llm_gate.py:35)、[`ToolRouter`](../agent-mind/tool_router.py:77)
   - 都在做“LLM + 工具 + 上下文”的闭环

2. **都支持 Function Calling / 工具调用式工作流**
   - `hermes-agent` 通过 [`model_tools.py`](../hermes-agent/model_tools.py:1) + [`tools/registry.py`](../hermes-agent/tools/registry.py:1)
   - 本项目通过 [`ToolRouter.get_openai_tools_schema()`](../agent-mind/tool_router.py:114)

3. **都具备持久记忆/长期上下文意识**
   - `hermes-agent` 偏会话数据库、skills、memory provider、session search
   - 本项目偏用户档案、结构化记忆、日记式成长记录，见 [`user_memory.py`](../agent-mind/user_memory.py:1)

4. **都不是纯脚本，而是长期运行系统**
   - `hermes-agent` 有 CLI、Gateway、Cron、ACP
   - 本项目有 FastAPI 服务、WebSocket、heartbeat、自主反思机制

5. **都试图把 Agent 做成“持续存在的实体”**
   - `hermes-agent` 通过 session / memory / skills / gateway 做连续性
   - 本项目通过人格配置、内在状态、反思、记忆做连续性

---

### 16.2 核心不同点

#### 1. 产品定位不同

- `hermes-agent`：**通用平台代理框架**
- 本项目：**人格化、自主意识风格的 Agent 服务**

`hermes-agent` 的目标是让 Agent 运行在 CLI、Telegram、Discord、Slack、ACP、Cron 等多个入口；
本项目的目标更集中在“AIwake 这个主体”本身，强调人格、内在状态和主动行为。

#### 2. 系统中心不同

- `hermes-agent` 围绕统一编排核心 [`AIAgent`](../hermes-agent/run_agent.py:535)
- 本项目围绕“脑干式循环” [`HeartbeatLoop`](../agent-mind/heartbeat.py:85)

Hermes 是“会话/任务驱动”的统一 orchestrator。
本项目是“状态/人格驱动”的持续运行体。

#### 3. 工具系统成熟度不同

- `hermes-agent`：自注册 registry + toolset + availability checks + MCP/plugin 扩展
- 本项目：集中式 [`ToolRouter`](../agent-mind/tool_router.py:77) 管理

Hermes 更像插件平台；本项目更像一套直接可控的工具执行器。

#### 4. 持久化重点不同

- `hermes-agent`：偏 **会话数据库 + FTS5 搜索 + lineage**，见 [`hermes_state.py`](../hermes-agent/hermes_state.py:1)
- 本项目：偏 **人格连续性 + 用户档案 + insight/diary memory**，见 [`user_memory.py`](../agent-mind/user_memory.py:1)

Hermes 更重“任务历史可检索”。
本项目更重“关系与人格连续成长”。

#### 5. 交互入口不同

- `hermes-agent`：CLI/TUI 是一等公民，Gateway 也是一等公民
- 本项目：HTTP API + Web UI + WebSocket 是主入口，见 [`main.py`](../agent-mind/main.py:121)

#### 6. 模型路由哲学不同

- `hermes-agent`：多 provider、多 API mode、多模型运行时解析，见 [`pyproject.toml`](../hermes-agent/pyproject.toml:13) 与 [`run_agent.py`](../hermes-agent/run_agent.py:686)
- 本项目：本地/云端双层门控，复杂度和风险决定模型层级，见 [`LLMGate.decide_tier()`](../agent-mind/llm_gate.py:68)

Hermes 更通用、更强扩展。
本项目更简单、更符合你的成本控制和人格主导策略。

---

### 16.3 优势对比

#### `hermes-agent` 的优势

1. **工程化能力明显更强**
   - 工具、网关、会话、插件、技能、MCP、Cron 都已体系化

2. **扩展性极强**
   - 更适合接入多平台、多模型、多后端、多工具环境

3. **工具系统成熟**
   - [`tools/registry.py`](../hermes-agent/tools/registry.py:1) 这一套比本项目的集中式工具路由更适合长期扩展

4. **多入口统一**
   - CLI、Gateway、ACP 最终都能落到统一 agent runtime 上

5. **长期会话检索更强**
   - SQLite + FTS5 比纯文件式记忆更适合做历史检索与跨会话回忆

6. **生态接口更多**
   - MCP、plugins、optional-skills、RL 等让它更像一个“Agent 平台”

#### 本项目的优势

1. **人格与主体性更强**
   - [`load_personality()`](../agent-mind/main.py:35)、[`HeartbeatLoop`](../agent-mind/heartbeat.py:85)、[`InternalState`](../agent-mind/state.py:1) 使本项目更像一个持续存在的角色，而不只是工具代理

2. **架构意图更聚焦**
   - 它不是想覆盖所有场景，而是聚焦“AIwake 这个人格化 Agent”的体验一致性

3. **部署面更轻**
   - 依赖相对少，见 [`requirements.txt`](../agent-mind/requirements.txt:1)
   - 对当前 FastAPI 服务场景更直接

4. **本地/云端门控逻辑更清晰**
   - [`LLMGate`](../agent-mind/llm_gate.py:35) 的 tier 决策更容易理解和控制成本

5. **主动性机制更自然**
   - `heartbeat + reflect + proactive cooldown` 这一套更适合做“会主动想、主动说”的人格体

6. **更适合当前项目目标**
   - 如果你的目标是做一个有内在状态、有情绪向量、有成长记忆的 AIwake，本项目其实比 Hermes 更贴题

---

### 16.4 劣势对比

#### `hermes-agent` 的劣势

1. **太重**
   - 对当前需求来说可能过度设计

2. **理解和维护成本高**
   - [`run_agent.py`](../hermes-agent/run_agent.py:1)、[`cli.py`](../hermes-agent/cli.py:1)、[`gateway/run.py`](../hermes-agent/gateway/run.py:1) 都是超大文件

3. **Windows 原生不友好**
   - 对你当前环境不够顺手

4. **容易被平台化目标拖累**
   - 为了兼容多平台、多 provider、多功能，系统复杂度显著上升

5. **人格表达不是其主设计中心**
   - 它支持 persona，但不像本项目这样把人格和内在状态放在第一位

#### 本项目的劣势

1. **工具系统扩展性不如 Hermes**
   - [`ToolRouter`](../agent-mind/tool_router.py:77) 越长越容易变成维护瓶颈

2. **会话历史检索能力偏弱**
   - 当前记忆更偏文件与摘要，没有 Hermes 那种统一的 session DB + FTS5

3. **多平台抽象不足**
   - 当前主入口仍偏 Web API，向 Telegram/Discord/Slack 迁移时会遇到结构瓶颈

4. **系统边界耦合偏高**
   - 人格、状态、心跳、LLM、工具、接口之间耦合度较大

5. **通用性不如 Hermes**
   - 适合 AIwake，不一定适合快速孵化多个不同 agent

6. **插件化/模块化程度还不够**
   - 缺少像 Hermes 那样明确的 registry、plugin、provider 分层

---

### 16.5 简表总结

| 维度 | hermes-agent | 本项目 agent-mind |
|---|---|---|
| 核心定位 | 通用平台代理框架 | 人格化自主 Agent 服务 |
| 核心驱动 | 任务/会话编排 | 心跳/状态/人格驱动 |
| 主入口 | CLI + Gateway + ACP + Cron | FastAPI + Web UI + WebSocket |
| 工具体系 | Registry + Toolset + Plugins + MCP | 集中式 ToolRouter |
| 记忆重点 | Session DB + FTS5 + skills/memory provider | 用户档案 + diary + structured memory |
| 模型策略 | 多 provider 多 API mode | 本地/云端 tier gating |
| 扩展性 | 很强 | 中等，需重构提升 |
| 部署复杂度 | 高 | 低到中 |
| 人格沉浸感 | 中等 | 很强 |
| 适合目标 | 平台化、多场景、多入口 | 单角色、人格连续性、自主感 |

---

### 16.6 最终判断

如果目标是：

- **做一个能接多平台、会用很多工具、适合长期扩展的通用 Agent 平台**
  - `hermes-agent` 更强

- **做一个具备人格、内在状态、主动反思、连续关系记忆的 AIwake 主体**
  - 本项目更合适

最合理的路线不是“二选一”，而是：

**保留本项目的人格化内核，吸收 Hermes 的工程骨架。**

具体说就是：

1. 保留 [`HeartbeatLoop`](../agent-mind/heartbeat.py:85) + [`LLMGate`](../agent-mind/llm_gate.py:35) + [`user_memory.py`](../agent-mind/user_memory.py:1) 这一套人格驱动逻辑
2. 借鉴 Hermes 的 [`tools/registry.py`](../hermes-agent/tools/registry.py:1) 重构工具系统
3. 借鉴 [`hermes_state.py`](../hermes-agent/hermes_state.py:1) 增加统一会话数据库与全文检索
4. 后续再逐步抽象多平台网关，而不是一开始就复制 Hermes 全家桶
