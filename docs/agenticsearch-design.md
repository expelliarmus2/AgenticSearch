# AgenticSearch 平台设计文档（v2.0 · 不含 RL）

> 版本：v2.0｜v2.0 变更：重组为四部分交付结构——新增「第一部分：系统定位与核心设计」「第二部分：系统总体架构」，详细设计与落地章节据此对齐重排
> 版本历史：v1.0 初稿；v1.1 知识记忆库对接收敛为 CRUD 四类接口（Agent 工具化 + 内部更新机制直插）；v2.0 结构化重组（本版）
> 范围：基于现有 s3 / AgenticSearch 代码仓演进，**v1 阶段不包含 RL 训练**，但保留与 RL（verl）的接口兼容性，便于 v2 平滑接入。
> 阅读对象：架构、算法、后端工程师。

**文档导读**：

- **第一部分（§1–§3）系统定位与核心设计**：为什么做、做什么、核心设计思想（原则/概念/决策）——评审入口；
- **第二部分（§4–§5）系统总体架构**：系统如何组织——分层架构、模块职责、项目结构、关键数据流、运行与部署；
- **第三部分（§6–§10）详细设计**：数据模型、知识记忆库 CRUD 接口、对外接口、轨迹存储、自进化 Skill 与索引；
- **第四部分（§11–§14）视图与落地**：4+1 视图、迁移路线、里程碑验收、附录。

---

# 第一部分：系统定位与核心设计

## 1. 背景与问题定义

### 1.1 现状：s3 智能体搜索框架

现有代码仓（s3）是一个面向检索增强生成的搜索智能体框架，核心是"搜索助手（Search Copilot）"：模型输出 `<query>` / `<search_complete>` 等结构化动作，外部检索服务返回 `<information>` 观测，循环多轮直至给出答案。当前实现围绕 verl + PPO 的 RL 训练闭环（`verl/trainer/ppo/ray_trainer.py` 调用 `s3/llm_agent/generation_s3.py`），训练与推理共享同一套循环代码。

### 1.2 现状代码盘点（与设计的映射基础）

| 现有代码 | 作用 | 设计中的去向 |
|---|---|---|
| `s3/llm_agent/generation_s3.py` | Search-C1 智能体循环（RL 训练用），含 `run_llm_loop`、`execute_predictions`、`postprocess_predictions`、`batch_search` | 拆解为 `agent/loop`（循环）+ `agent/policies`（策略）+ `agent/actions`（动作解析），v1 复用其动作协议 |
| `s3/llm_agent/generation_searchr1.py` | Search-R1 风格循环（观测/奖励与 s3 略有差异） | 同一动作协议下的另一个循环策略实现，v1 保留兼容 |
| `s3/llm_agent/tensor_helper.py` | 张量拼接/掩码工具（RL 专用） | 移入 `agent/loop/rl/` 适配层（v2），v1 不依赖 |
| `s3/search/retrieval_server.py` | FastAPI 检索服务 `POST /retrieve`（BM25 / Dense faiss，支持远程 embedding API） | 保留，演化为 `retrieval/server.py` + `retrieval/backends/*` |
| `s3/search/retrieval.py` | 检索逻辑（可导入的库形式） | `retrieval/backends/dense.py` / `bm25.py` |
| `s3/search/serp_search_server.py` | 在线搜索（SERP）`POST /retrieve` | `retrieval/backends/serp.py`，统一到同一检索接口 |
| `s3/search/rerank_server.py`、`retrieval_rerank_server.py` | 重排服务 | `retrieval/rerank.py` |
| `s3/search/index_builder.py` | faiss 索引构建 | 保留于 `scripts/index_build/` |
| `generator_llms/*` | 生成器 LLM 适配器（qwen / gpt-azure / claude / together / deepinfra / local vLLM） | 保留为"环境层"，智能体循环中作为只读反馈源 |
| `scripts/data_process/*`、`scripts/dataset_construct/*` | 数据构造（parquet，含 `<question>` 模板） | 保留；新增轨迹数据生成入口 |
| `scripts/baselines/*` | RAG / DeepRetrieval / IRCoT / Search-o1 / Search-R1 基线 | 保留评估对比；v1 智能体可作为统一替代 |
| `scripts/evaluation/*` | 评估（context、zeroshot、eval_only） | 保留，并新增轨迹级评估组件 |
| `verl/*` | RL 训练框架（PPO） | **v1 不启用**；`agent/loop/rl/` 预留适配接口，动作协议保持兼容 |
| `utils/embedding_client.py` | OpenAI 兼容 embedding API 客户端 | 保留，供知识层（图谱实体/向量/轨迹摘要）向量化复用 |
| `data/agentic/` | agentic 数据样例（s3_api_results.json 等） | 作为轨迹数据的雏形，转为标准轨迹格式 |

### 1.3 核心问题定义

| # | 问题 | 现状表现 |
|---|---|---|
| P1 | 循环与训练耦合 | 智能体循环嵌在 RL 训练框架（verl）内，无法作为独立推理服务部署与复用 |
| P2 | 知识不沉淀 | 只有语料检索（BM25/Dense），无知识记忆层：跨会话经验、结构化知识、图谱关系无法写入与复用 |
| P3 | 经验不可追溯 | 搜索过程不落盘：无法回放一次搜索的完整决策过程，无法诊断失败，无法复用成功经验 |
| P4 | 策略不自进化 | 决策质量改进依赖人工迭代提示词与检索参数，无系统化的"归纳→验证→发布"机制 |
| P5 | 接口不完整 | 对外仅 `/retrieve` 检索接口，缺智能体、轨迹、知识、技能等服务化接口 |

**v1 的使命**：在不动 RL（verl）的前提下，把这套搜索智能体演进为一个"可服务化、可沉淀、可自进化"的平台，并为 v2 的 RL 训练保留数据与接口兼容。

---

## 2. 系统定位

### 2.1 定位陈述

> **AgenticSearch 是一个可沉淀、可进化的智能体搜索平台**：以兼容 s3 动作协议的搜索智能体为核心，把每一次搜索产生的**轨迹、知识、经验**沉淀为平台资产（知识记忆库 / 轨迹库 / Skill 库），并通过**离线自进化闭环**持续改进检索与决策质量。

三个关键词（分别回应 §1.3 的问题）：

- **服务化**（P1/P5）：智能体循环脱离训练框架，成为可独立部署、可水平扩展的推理服务；
- **资产化**（P2/P3）：轨迹、知识、经验全部落库，成为可查询、可复用、可训练的数据资产；
- **自进化**（P4）：非梯度式（统计归纳 + LLM 提炼 + A/B 验证）的持续改进闭环，RL 留给 v2。

### 2.2 目标用户与核心场景

| 用户 | 场景 | 使用入口 |
|---|---|---|
| 算法/评测工程师 | 多跳问答评测、基线对比、批量实验 | `/v1/agent/tasks` + 轨迹评估回填 |
| 应用开发者 | 把智能体搜索嵌入自有产品（客服、研究助手） | `/v1/agent/search` + `/retrieve` |
| 平台算法工程师 | 知识库运营（导入/聚合/溯源）、Skill 进化运营 | Ingestor 脚本 + `/v1/skills` |
| v2 训练（预留） | 用轨迹数据训练 RL 搜索策略 | 轨迹导出（parquet 兼容，§9.4） |

### 2.3 与周边方案的关系

| 方案 | 差异 |
|---|---|
| 朴素 RAG（单次检索 + 生成） | AgenticSearch 是多轮循环决策，检索是"动作"而非固定流程；且检索质量与搜索经验可沉淀、可进化 |
| 通用 Agent 框架（LangChain / LangGraph 等） | 本平台面向"搜索"垂直域：动作协议与检索栈深度绑定；轨迹格式面向搜索评估/训练定制；自带非梯度自进化闭环 |
| s3 / verl（RL 训练框架） | 本平台 v1 = s3 的服务化演进，**不含训练**；动作协议与数据格式保持兼容，v2 可回接 verl 训练 |

### 2.4 范围与非目标

**范围内（v1）**：智能体推理服务；知识记忆库 CRUD 对接（多后端）；轨迹存储与语义检索；离线 Skill 自进化；三层自进化索引；对外 REST API。

**非目标（v1）**：RL/策略梯度训练（v2，`agent/loop/rl/` 预留）；在线重建倒排/向量索引（离线构建 + 热桶增量留 v2）；多智能体协作编排；图形界面。

---

## 3. 核心设计

### 3.1 设计原则（后续章节的设计依据）

| # | 原则 | 含义 | 主要落点 |
|---|---|---|---|
| D1 | 协议兼容优先 | `<query>`/`<search_complete>`/`<information>` 动作协议与 `/retrieve` 签名不变，存量调用方零破坏 | §8.1、§12 |
| D2 | 读写分离 | 知识"读"对 Agent 开放为工具；"写"统一收口内部 Ingestor（受控 add 工具除外）；对外不暴露写 API | §7.2/§7.3 |
| D3 | 一切皆轨迹 | 每轮动作/观测/决策/耗时全量落盘，评估、进化、诊断、训练共用同一数据底座 | §9 |
| D4 | 进化非梯度 | v1 用"统计归纳 + LLM 提炼 + A/B 验证"替代 RL 获得持续改进，可解释、可回滚 | §10 |
| D5 | 稳定索引 + 活参数 | 索引本体离线构建保稳定；在线只进化参数（重排/融合/改写）与图谱活索引 | §10.6 |
| D6 | 可插拔后端 | 知识与轨迹存储统一 Protocol，多后端（图谱/向量/SQL/BM25/记忆）可替换、可组合 | §7.4、§9.2 |

### 3.2 核心概念模型

五个核心概念（可序列化的数据模型见 §6）：

| 概念 | 一句话定义 |
|---|---|
| **Agent Loop** | 状态机式搜索循环：Policy 每轮决策动作（search / knowledge_query / answer…），工具执行产生观测，直至终止 |
| **KnowledgeBase** | 知识记忆库：多后端统一 CRUD 协议的读写层，"读"服务检索增强，"写"沉淀平台经验 |
| **Trajectory** | 一次搜索的完整记录（turns 全量），是评估、进化、诊断、训练的统一数据底座 |
| **Skill** | 可复用的搜索经验单元（触发条件 + 执行体 + 统计），由离线进化管线生产与淘汰 |
| **三层索引** | 基础索引（离线，稳）+ 经验层参数（在线自适应）+ 图谱活索引（随轨迹进化） |

概念关系：

```
在线主链路:
  question → Agent Loop ──(Policy 决策)──→ Action ──(Tool 执行)──→ Observation → … → answer
                 │ 读                                      │ 全程记录
                 ▼                                        ▼
        KnowledgeBase / Retrieval                    Trajectory（轨迹库）

离线进化闭环:
  Trajectory ──评估──→ 归纳/聚类 ──LLM 提炼──→ Skill(draft) ──A/B 验证──→ active ──→ 反哺 Policy
      │                                                        │
      ├── Ingestor 沉淀（实体/关系/经验片段）──→ KnowledgeBase（图谱活索引）
      └── 进化总结知识直接 add ──────────────→ KnowledgeBase
```

### 3.3 关键设计决策

| 决策点 | 选择 | 理由 | 被否方案 |
|---|---|---|---|
| 知识库交互 | CRUD 四接口收敛（query/add/update/delete + batch_upsert） | 语义最小完备，Agent 工具化与内部更新机制共用一套契约 | 读/写/事件三类接口（事件流引入时序耦合，v1 无消费者） |
| 知识写入路径 | 对外不暴露写 API，统一走 Ingestor（Agent 的 add 工具受控开启） | 保证可追溯（source）与幂等，防知识污染 | 开放写 API（溯源与质量不可控） |
| 进化方式 | 离线非梯度（归纳 + 提炼 + A/B） | 无训练基础设施依赖；可解释、可回滚；v1 即可跑通闭环 | RL（依赖 verl 训练栈，v2）；在线即时进化（无验证闸门，风险高） |
| 索引进化 | 三层：离线基础 + 在线参数 + 图谱活索引 | 兼顾稳定性（运行期索引只读）与进化性（参数与图谱可变） | 在线重建索引（延迟/一致性风险）；纯静态索引（无进化能力） |
| 循环运行形态 | 独立推理服务（FastAPI），无 verl 依赖 | 服务化复用；RL 只是 v2 的一种 loop 实现 | 继续嵌在训练框架内 |
| 轨迹存储 | JSONL + SQLite 元数据 + faiss 语义索引 | 零依赖起步、可回放、可语义召回、可转 parquet 训练格式 | 单一数据库（大文本检索弱）；纯日志文件（不可查询） |

### 3.4 v1 设计目标

**v1 设计目标**（不依赖 RL）：

1. 把"智能体循环"从 RL 训练代码中解耦，成为可独立运行的推理服务；
2. 支持**知识记忆库对接层**：以增删改查（CRUD）四类接口对接多种知识后端（知识图谱、向量库、SQL、BM25、短期记忆）；Agent 将 CRUD 当工具调用，自进化总结知识/多条记忆经内部更新机制直接插入；
3. 支持**轨迹存储（Trajectory Storage）**：完整记录 agentic search 的每一步动作、观测、决策与评估，可回放、可检索；
4. 支持**自进化 Skill**：从成功/失败轨迹中归纳、提炼、验证、发布技能，形成离线自进化闭环（非梯度式进化）；
5. 定义清晰的**对外接口**（检索、智能体、轨迹、Skill、健康检查），与现状代码保持动作协议兼容；
6. 支持**自进化索引（三层策略）**：基础索引离线批量构建保质量，经验层做轻量自适应（重排权重/融合系数/改写规则），图谱作为"活索引"随轨迹进化（§10.6）。

---

# 第二部分：系统总体架构

## 4. 总体架构

### 4.1 分层架构图

```
                    ┌─────────────────────────────────────────┐
                    │            外部调用方 (App/评测/用户)      │
                    └───────────────┬─────────────────────────┘
                                    │ REST / gRPC
                    ┌───────────────▼─────────────────────────┐
                    │                api/ 对外接口层            │
                    │  /v1/agent  /v1/retrieve  /v1/trajectory │
                    │  /v1/skills  /health  /v1/config        │
                    └───────────────┬─────────────────────────┘
                                    │
      ┌─────────────────────────────┼──────────────────────────────┐
      │                agent/ 智能体核心层                          │
      │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────┐  │
      │  │  loop     │ │  policy   │ │ actions  │ │  memory       │  │
      │  │ (执行循环) │ │ (决策策略) │ │(动作解析) │ │(知识CRUD/轨迹) │  │
      │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └──────┬────────┘  │
      │       │            │            │             │            │
      │  ┌────▼────────────▼────────────▼─────────────▼──────┐      │
      │  │                 tools/ 工具集                       │      │
      │  │ retriever | knowledge_query/add | answer | calc  │      │
      │  └────┬────────────┬────────────┬───────────────────┘      │
      └───────┼────────────┼────────────┼──────────────────────────┘
              │            │            │
   ┌──────────▼──┐  ┌──────▼──────┐ ┌───▼───────────────────┐
   │ retrieval/  │  │ generator_  │ │  skills/ 自进化 Skill  │
   │ 检索层       │  │ llms/ 环境层 │ │  registry + evolution │
   │ /retrieve   │  │ (只读反馈源) │ └───┬───────────────────┘
   └──────────┬──┘  └──────┬──────┘     │
              │           │            │
   ┌──────────▼───────────▼────────────▼───────────────────┐
   │ 知识/存储层：图谱 | 向量 | SQL | BM25 | 记忆 | 轨迹 |   │
   │ 索引 | 配置（基础索引离线构建；知识写入走 Ingestor）   │
   └──────────────────────────────────────────────────────┘
```

分层职责与设计原则（§3.1）的对应：

- **api/ 对外接口层**：统一 REST 信封与鉴权，隔离外部协议与内部实现；D1 协议兼容（`/retrieve` 原样透传）在此层落地；
- **agent/ 智能体核心层**：loop / policy / actions / memory 四件套，平台中枢；动作协议兼容（D1）在 actions/parser 落地；
- **tools/ 工具集**：把检索与知识 CRUD 包装为 Policy 可调用的动作；D2 读写分离在此层守门（工具开关与审计）；
- **retrieval/ 检索层 + generator_llms/ 环境层**：语料检索与生成反馈，对 agent 只读；
- **skills/ 自进化层**：消费轨迹、生产 Skill 与检索参数，反哺 policy 与检索（D4/D5）；
- **知识/存储层**：图谱/向量/SQL/BM25/记忆/轨迹/索引，统一经接口访问（D6 可插拔）。

### 4.2 模块职责一览

| 模块 | 职责 | 关键接口/文件 |
|---|---|---|
| `agent/loop` | 驱动搜索循环：调 Policy 取动作、执行工具、记录 Turn、判终止 | `AgentLoop`、`sequential.py` |
| `agent/policies` | 决策：`llm_policy`（LLM 直出）/ `skill_policy`（Skill 触发优先，llm 兜底） | `Policy.decide()` |
| `agent/actions` | 动作模型与解析（兼容 `<query>`/`<search_complete>` 协议） | `parser.py`、`registry.py` |
| `agent/memory` | 知识记忆库 CRUD 契约、Ingestor、后端适配、工作记忆、轨迹存储 | `KnowledgeBase`、`Ingestor` |
| `retrieval/` | 语料检索（BM25/Dense/SERP/hybrid）、重排、经验层自适应 | `Retriever.search()` |
| `skills/` | Skill 注册/执行/评估/自进化 | `SkillRegistry`、`evolution.py` |
| `api/` | 对外 REST 服务装配（agent/trajectory/skill/retrieval） | FastAPI `app.py` |
| `evaluation/` | EM/F1/LLM-judge、轨迹级评估、Skill A/B | `metrics.py` |
| `generator_llms/` | 生成器 LLM 适配（答案生成、反馈），只读环境层 | 现状保留 |

> 模块依赖与开发约束（禁止反向依赖等）见 §11.1 逻辑视图。

### 4.3 目标项目结构

```
AgenticSearch/
├── docs/                              # 设计文档（本文件）
├── agent/                             # 智能体核心（由 s3/llm_agent 演化）
│   ├── core/                          # 核心数据模型
│   │   ├── trajectory.py              # 轨迹模型（Turn / Trajectory / Evaluation）
│   │   ├── observation.py             # 观测（信息块、错误、终止）
│   │   └── config.py                  # AgentConfig
│   ├── loop/
│   │   ├── base.py                    # AgentLoop 抽象（与模型无关）
│   │   ├── sequential.py              # v1 顺序循环（无 RL）
│   │   └── rl/                        # v2 预留：RL 循环适配（verl 兼容）
│   ├── policies/
│   │   ├── base.py                    # Policy 抽象：state -> action
│   │   ├── llm_policy.py              # 直接 LLM 决策（复用 generation_s3 的解析）
│   │   └── skill_policy.py            # Skill 驱动决策（触发 + 执行 skill）
│   ├── actions/
│   │   ├── action.py                  # Action 类型：search / knowledge_query / knowledge_add / answer ...
│   │   ├── parser.py                  # 从模型输出解析动作（<query>/<search_complete> 兼容）
│   │   └── registry.py                # 动作注册表
│   ├── memory/                        # 知识记忆库对接层（§7）
│   │   ├── working_memory.py          # 工作记忆（当前轮次上下文滚动窗口）
│   │   ├── knowledge_base.py          # CRUD 接口契约（query/add/update/delete/batch_upsert）
│   │   ├── ingestor.py                # 内部更新机制：解析→路由→幂等写入（后台直插）
│   │   ├── backends/                  # 可插拔后端适配器
│   │   │   ├── graph.py               # 知识图谱（Neo4j/JsonGraph/Sqlite/Remote）
│   │   │   ├── vector.py              # 向量库（faiss/milvus）
│   │   │   ├── sql.py                 # SQL/结构化知识（sqlite/postgres）
│   │   │   ├── bm25.py                # BM25/全文（复用现状语料索引）
│   │   │   └── episodic.py            # 短期记忆（TTL）
│   │   └── trajectory_store.py        # 轨迹存储（§9）
│   └── skills/                        # Skill 系统（§10）
│       ├── skill.py                   # Skill 模型与序列化
│       ├── registry.py                # Skill 注册/检索/版本
│       ├── executor.py                # Skill 执行器（提示词/工具序列）
│       ├── evaluator.py               # 轨迹评估（EM/LLM-judge/规则）
│       └── evolution.py               # 自进化引擎（归纳→提炼→验证→发布）
├── retrieval/
│   ├── server.py                      # FastAPI 检索服务（兼容现状 /retrieve）
│   ├── schema.py                      # 请求/响应模型（QueryRequest 等）
│   ├── backends/
│   │   ├── base.py                    # Retriever 抽象（search/batch_search）
│   │   ├── bm25.py                    # pyserini BM25
│   │   ├── dense.py                   # faiss + e5/bge（本地或远程 embedding）
│   │   ├── serp.py                    # 在线搜索引擎
│   │   └── hybrid.py                  # 混合检索（稀疏+稠密+图谱+SQL 融合）
│   ├── rerank.py                      # 重排服务
│   ├── adaptive.py                    # 经验层自适应（重排权重/融合系数/改写规则，检索型 Skill 驱动，§10.6）
│   └── knowledge_adapter.py           # 知识层适配（将知识命中转检索结果）
├── generator_llms/                    # 环境层（现状保留）
│   ├── qwen_api.py / gpt_azure.py / claude_api.py / together_api.py / deepinfra.py / local.py
│   └── query_rewrite.py
├── tools/
│   ├── registry.py                    # 工具注册
│   ├── retriever_tool.py              # 检索工具（search）
│   ├── knowledge_tool.py              # 知识工具（CRUD 工具化：knowledge_query / knowledge_add，§7.2）
│   └── answer_tool.py                 # 作答工具（调用 generator LLM）
├── api/                               # 对外服务层
│   ├── app.py                         # FastAPI 应用装配
│   ├── agent_api.py                   # 智能体服务（§8.2）
│   ├── trajectory_api.py              # 轨迹服务（§8.3）
│   ├── skill_api.py                   # Skill 服务（§8.4）
│   ├── retrieval_api.py               # 检索服务（§8.1，复用现状）
│   └── schemas.py                     # Pydantic 请求/响应
├── evaluation/
│   ├── metrics.py                     # EM / F1 / LLM-judge
│   ├── trajectory_eval.py             # 轨迹级评估（成功率、轮次、检索质量）
│   └── skill_ab.py                    # Skill A/B 验证
├── scripts/                           # 运维与数据脚本（现状保留 + 新增）
│   ├── deploy_retriever/  data_process/  dataset_construct/
│   ├── index_build/  evaluation/  baselines/
│   ├── run_agent_server.sh            # 启动智能体服务
│   ├── run_skill_evolution.sh         # 触发离线 skill 进化
│   └── export_trajectories.sh         # 轨迹导出/统计
├── data/                              # 数据、索引、轨迹
│   ├── indexes_wiki/  wiki-18.jsonl ...
│   └── trajectories/                  # 轨迹存储（JSONL / 元数据 sqlite）
├── utils/                             # embedding_client、java_init（保留）
├── verl/                              # v2 预留（v1 不使用）
└── s3/                                # 现状代码保留（过渡期），v1 目标迁入 agent/
```

### 4.4 关键数据流

**F1 在线问答流（同步，低延迟路径）**

```
Client → api/ → AgentLoop
  → Policy.decide（skill_policy 优先，llm_policy 兜底）
  → Action{search}          → retrieval/（POST /retrieve，D1 兼容）  → Observation
  → Action{knowledge_query} → KnowledgeBase.query（多后端融合）       → Observation
  → Action{answer}          → generator_llms（生成器 LLM）            → 最终答案
  全程每轮 Turn 异步写入 TrajectoryStore（不阻塞主循环）
```

**F2 知识写入流（异步/后台，不经 Agent 主链路）**

```
Trajectory 结束  → Ingestor：实体/关系/经验片段 → add（图谱+向量）
Skill 进化产出   → Ingestor：总结知识 → add/update（直接插入）
定时聚合任务     → Ingestor：碎片记忆合并 → batch_upsert + 旧条目软删
外部导入脚本     → Ingestor：批量知识 → batch_upsert
（全部幂等、按 source 可追溯；契约见 §7.3）
```

**F3 离线进化流（定时批处理）**

```
TrajectoryRepository 拉取窗口期轨迹
  → evaluator 打标（EM / LLM-judge）
  → 归纳聚类（问题类型 / 动作序列 / 改写模式；失败模式反向归纳）
  → LLM 提炼 Skill 草稿（trigger + body）
  → A/B 验证（vs 基线，达阈值）
  → 发布 active → 注入 skill_policy 与 retrieval/adaptive
  → 监控 stats → 淘汰 / 迭代（详见 §10）
```

## 5. 运行与部署架构

### 5.1 运行时视图（服务与进程）

| 运行单元 | 形态 | 扩展方式 |
|---|---|---|
| Agent Service | FastAPI（api/ + agent loop，异步任务池） | 无状态，多实例水平扩展 |
| Retriever Service | FastAPI `/retrieve` | 多实例 + 只读索引共享 |
| Knowledge Backends | 图谱/向量/SQL/BM25/记忆，可合可拆 | 按后端独立扩缩 |
| Trajectory Store | JSONL 写入 + SQLite 查询 + faiss 召回 | 单写者 + 读副本 |
| Generator LLM | 外部 API 或本地 vLLM | 现成服务 |
| 离线管道 | 定时任务（评估/进化/沉淀/聚合） | 独立节点，错峰执行 |

> 完整进程交互与数据流见 §11.3 进程视图。

### 5.2 部署拓扑

v1 支持两种部署形态，配置驱动切换、代码路径不变：

- **单机一体化**（默认，零外部依赖）：JsonGraph + faiss + sqlite + JSONL 轨迹，全部进程同机；
- **分布式**：检索/知识/存储/离线分节点部署，图谱用 Neo4j、SQL 用 postgres、轨迹归档走对象存储。

> 节点级拓扑示例见 §11.4 物理视图。

### 5.3 技术选型与理由

| 选型 | 理由 |
|---|---|
| FastAPI + Pydantic | 现状已是 FastAPI（`/retrieve`），异步友好，Schema 即文档 |
| faiss + pyserini(Lucene) | 复用现状索引产物与脚本（`index_builder.py`），零迁移成本 |
| sqlite → postgres | 开发/单机零依赖，生产平滑升级（SQL 后端同协议） |
| Neo4j（可选图谱） | 生产大规模图谱；v1 默认 Json/Sqlite 图谱零依赖起步 |
| JSONL + SQLite + faiss（轨迹） | 零依赖、可回放、可语义召回、可转 parquet 训练格式 |
| vLLM / 云 API（LLM） | 决策与生成解耦，按需替换；现状 `generator_llms/` 适配器全保留 |

---

# 第三部分：详细设计

## 6. 核心概念与数据模型

本章把第一部分 §3.2 的核心概念（Loop / KnowledgeBase / Trajectory / Skill / KnowledgeItem）落为可序列化的数据模型，是所有子系统的公共语言。

### 6.1 智能体循环（Agent Loop）

循环沿用现状动作协议，但脱离 verl/RL 独立运行：

```
循环输入: question + 初始上下文
   ┌──────────────────────────────┐
   │ Policy(state) -> Action       │   ← llm_policy / skill_policy
   │   Action: search | knowledge_ │
   │   query/add | answer | ...    │
   └──────────────┬───────────────┘
                  ▼
   ┌──────────────────────────────┐
   │ Tool 执行:                    │   ← tools/（检索、知识CRUD、作答）
   │   observation 生成            │
   └──────────────┬───────────────┘
                  ▼
   ┌──────────────────────────────┐
   │ 轨迹记录: 每一轮 (action, obs, │   ← trajectory_store（异步写）
   │   decision, latency) 入轨迹    │
   └──────────────┬───────────────┘
                  ▼
   终止条件: answer 产出 / 最大轮次 / 超时 → 生成 Trajectory
```

### 6.2 轨迹模型（Trajectory）

```python
@dataclass
class Action:
    type: str                    # "search" | "knowledge_query" | "knowledge_add" | "answer" | ...
    payload: Dict[str, Any]      # search: {"query": "...", "topk": 3}

@dataclass
class Observation:
    type: str                    # "information" | "error" | "terminate"
    content: str                 # <information> 块文本
    docs: List[DocRef]           # 命中文档（doc_id/title/score）
    meta: Dict[str, Any]         # 检索器、耗时等

@dataclass
class Turn:
    turn_index: int
    action: Action
    observation: Observation
    decision: Decision            # policy 名、skill_id、reason
    latency_ms: int
    token_usage: Dict[str, int]
    state_snapshot: Optional[str] # 决策时刻的状态摘要（供分析）

@dataclass
class Trajectory:
    trajectory_id: str
    session_id: str
    question: str
    created_at: str
    finished_at: Optional[str]
    status: str                   # success | failed | timeout
    answer: Optional[str]
    turns: List[Turn]
    evaluation: Evaluation        # EM/F1/LLM-judge/用户反馈
    skills_used: List[str]
    knowledge_updates: List[Dict] # 经 Ingestor 写入知识库的知识项（可追溯）
```

### 6.3 Skill 模型（§10 详述）

```python
@dataclass
class Skill:
    id: str
    name: str
    description: str
    version: int
    trigger: SkillTrigger         # 触发规则或 LLM 判断
    body: SkillBody               # 提示词模板 / 工具序列 / 代码
    stats: SkillStats             # usage_count / success_rate / avg_score
    status: str                   # draft | active | deprecated
    source_trajectories: List[str]# 归纳来源轨迹
```

### 6.4 知识项模型（KnowledgeItem / KnowledgeMeta）

```python
@dataclass
class KnowledgeMeta:              # 写入元数据（溯源与生命周期）
    source: str                   # trajectory_id / skill_id / import:批次号
    confidence: float = 1.0
    priority: str = "normal"      # high | normal | low
    ttl: Optional[int] = None     # 秒；None = 永久
    tags: List[str] = field(default_factory=list)

@dataclass
class KnowledgeItem:              # 知识记忆库统一写入单元（"增"的载荷）
    kind: str                     # entity | relation | document | row | snippet | fact
    key: str                      # 幂等键（实体ID / 主键 / 文档ID）
    content: Dict[str, Any]       # 载荷：图谱字段 / 文档正文 / SQL行 / 经验片段
    meta: KnowledgeMeta = None
```

## 7. 知识记忆库对接接口（CRUD + 工具化 + 更新机制）

> 对应设计原则 **D2 读写分离**、**D6 可插拔后端**（§3.1）。

现状代码只有检索后端（BM25/Dense）。v1 引入**知识记忆库对接层**：知识记忆库是独立子系统（支持多后端：知识图谱、向量库、SQL、BM25/全文、短期记忆，也可对接外部已有知识系统）。AgenticSearch 与它的全部交互收敛为**增删改查（CRUD）四类接口**，使用方式有两种：

1. **工具化**：Agent 在决策循环中把 CRUD 接口当**工具**调用（读为主，写受控）；
2. **更新机制（内部）**：自进化总结知识、多条记忆的聚合结果，由后台机制**直接插入**，不经 Agent。

### 7.1 CRUD 接口契约

```python
# agent/memory/knowledge_base.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Protocol

@dataclass
class KnowledgeQuery:                # "查"的参数
    query: str                       # 自然语言/检索文本
    mode: str = "hybrid"             # semantic | graph | sql | bm25 | hybrid
    kinds: List[str] = field(default_factory=list)   # 限定知识类型
    seed_entities: List[str] = field(default_factory=list)
    topk: int = 10
    filters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class KnowledgeHit:                  # "查"的返回
    key: str
    kind: str
    content: Dict[str, Any]
    score: float
    evidence: str                    # 可读化证据文本（可直接进 <information>）

class KnowledgeBase(Protocol):
    """知识记忆库对接接口：增删改查。后端适配器实现此协议。"""

    # ---- 查 (Read) ----
    async def query(self, q: KnowledgeQuery) -> List[KnowledgeHit]: ...

    # ---- 增 (Create) ----
    async def add(self, items: List[KnowledgeItem]) -> "IngestReport": ...

    # ---- 改 (Update) ----
    async def update(self, keys: List[str], patch: Dict[str, Any]) -> int: ...

    # ---- 删 (Delete) ----
    async def delete(self, keys: List[str], soft: bool = True) -> int: ...

    # ---- 批量/管理 ----
    async def batch_upsert(self, items: List[KnowledgeItem]) -> "IngestReport": ...
    async def backends(self) -> List[str]: ...
    async def health(self) -> Dict[str, Any]: ...
```

> `add` 与 `batch_upsert` 的区别：`add` 面向单条/小批量（Agent 工具或轨迹沉淀即时写入）；`batch_upsert` 面向更新机制的大批量（离线导入、记忆聚合、自进化产物），按 key 幂等合并。

### 7.2 工具化：Agent 把 CRUD 当工具调用

| 工具 | 对应接口 | 默认状态 | 说明 |
|---|---|---|---|
| `knowledge_query` | query | **开启** | Agent 检索知识的主工具；结果 `evidence` 直接作为观测 |
| `knowledge_add` | add | 受控开启 | Agent 搜索中发现高价值知识可即时写入，强制携带 source/confidence |
| `knowledge_update` | update | **关闭** | 防止 Agent 误改知识；仅管理员/进化管线可用 |
| `knowledge_delete` | delete | **关闭** | 防误删；仅管理员可用（软删优先） |

工具参数与 CRUD 契约一致，外层统一做**权限校验 + 审计日志 + 限流**；每次调用记录进 Turn（含命中/写入条数），保证可追溯。

### 7.3 更新机制：后台直接插入（不经 Agent）

| 机制 | 触发时机 | 调用 | 说明 |
|---|---|---|---|
| 轨迹沉淀 | 轨迹结束（异步） | `add` | 成功轨迹抽取实体/关系/经验片段 → 图谱+向量 |
| **自进化总结知识插入** | Skill 进化产出（§10） | `add`/`update` | 进化管线归纳出的成功模式总结、查询改写经验、领域知识摘要，**直接插入**知识库，供后续 `knowledge_query` 命中 |
| 多条记忆聚合 | 定时（离线） | `batch_upsert` + `update`/`delete` | 碎片记忆合并为总结性知识后批量写入；旧条目标记合并来源或软删 |
| 外部导入 | 离线脚本 | `batch_upsert` | 语料/结构化知识批量入库 |

所有更新机制共用 `Ingestor`（`agent/memory/ingestor.py`）：解析 → 按路由规则分发后端（可 fan-out）→ 幂等 upsert → 返回 IngestReport（写入数/跳过数/失败死信，均可按 source 追溯）。

### 7.4 后端适配器

| 后端类型 | 适配器 | 存储 | 适用场景 |
|---|---|---|---|
| 图谱 | `JsonGraphBackend` | 本地 JSON（零依赖） | v1 默认，开发/单机 |
| 图谱 | `SqliteGraphBackend` | sqlite 三表（entity/relation/emb） | 小规模持久化 |
| 图谱 | `Neo4jGraphBackend` | Neo4j（Cypher 查询） | 生产大规模图谱 |
| 图谱 | `RemoteGraphBackend` | HTTP/gRPC 转发 | 对接已有外部图谱服务 |
| 向量 | `FaissVectorBackend` | faiss + jsonl 载荷 | 单机，复用现有索引产物 |
| SQL | `SqliteBackend` / `PostgresBackend` | sqlite / postgres | 结构化知识、统计 |
| BM25 | `LuceneBm25Backend` | pyserini Lucene | 全文检索（复用现状索引） |
| 记忆 | `EpisodicMemoryBackend` | 内存/redis + TTL | 会话级短期记忆 |

> 路由规则（配置驱动）：`entity/relation`→图谱(+向量)；`document/snippet`→向量(+BM25)；`row`→SQL；会话片段→episodic（TTL）。向量化复用 `utils/embedding_client.APIEncoder`。BM25 属重索引型后端：在线只写暂存区，索引更新走离线构建路径（三层策略见 §10.6）。

### 7.5 智能体如何使用知识层

1. **读**：`knowledge_query` 工具（graph/sql/bm25/hybrid 按需）+ `retrieval/backends/hybrid.py` 混合检索（BM25 稀疏 + e5 稠密 + 图谱子图 + SQL 结果融合进 `<information>`）；
2. **写**：一律走内部 Ingestor（§7.3），对外不暴露写 API（如需人工导入走默认关闭的管理端点）。

```python
# 沉淀示例（伪代码，全部走内部 Ingestor）
report = await kb.add(items_from(traj))
#  → 图谱: entity(qid) / entity(doc_i) / relation(qid, doc_i, "retrieved_by")
#  → 向量: snippet(question, answer, evidence, emb)
#  → SQL : row(stats: hits, turns, latency)

# 自进化总结知识插入（进化管线直接调用）
await kb.add([KnowledgeItem(kind="fact", key=f"skill:{sid}:summary",
                            content={"text": summary}, meta=KnowledgeMeta(source=sid))])
```

---

## 8. 对外接口（External APIs）

> 对应设计原则 **D1 协议兼容优先**（§3.1）。

统一走 REST（FastAPI），鉴权 v1 用 API-Key 头（`X-Api-Key`），后续可加 OAuth。所有服务返回标准信封：`{"code": 0, "data": ..., "message": "ok"}`。

### 8.1 检索接口（兼容现状，不破坏既有调用方）

```
POST /retrieve
Request:  {"queries": ["..."], "topk": 3, "return_scores": true}
Response: {"result": [[{"document": {"title":..., "text":..., "contents":...}, "score": 0.9}]]}
```

新增可选字段：`mode`（`sparse|dense|hybrid|graph|sql`）、`knowledge`（知识层参数：`kinds`、`seed_entities`、`filters`）。

### 8.2 智能体服务（Agent API）

```
POST /v1/agent/search          # 同步单轮问答
  Request:  {"question": "...", "session_id": "s1",
             "options": {"max_turns": 3, "topk": 3, "policies": ["skill_policy"], "knowledge": {"tools": ["knowledge_query", "knowledge_add"]}}}
  Response: {"trajectory_id": "...", "answer": "...", "status": "success",
             "turns": 2, "skills_used": ["s-001"]}

POST /v1/agent/tasks           # 异步批量任务（评测/离线批量）
  Request:  {"questions": [...], "options": {...}, "callback_url": "..."}
  Response: {"task_id": "..."}
GET  /v1/agent/tasks/{task_id} # 查询任务状态与结果
POST /v1/agent/sessions/{sid}/continue   # 多轮会话续接
```

### 8.3 轨迹服务（Trajectory API）

```
POST /v1/trajectories          # 写入轨迹（由智能体内部调用或外部导入）
GET  /v1/trajectories/{id}     # 读取单条轨迹（完整 turns）
GET  /v1/trajectories?question=&status=&from=&to=&page=   # 列表查询
POST /v1/trajectories/search   # 语义检索历史轨迹（按问题相似度召回，供 reuse）
POST /v1/trajectories/{id}/evaluate   # 对轨迹执行评估并回填 evaluation
DELETE /v1/trajectories/{id}
```

### 8.4 Skill 服务（Skill API）

```
GET    /v1/skills?status=active
POST   /v1/skills              # 手动创建 skill
PUT    /v1/skills/{id}         # 更新（版本递增）
POST   /v1/skills/{id}/publish # draft → active
POST   /v1/skills/{id}/deprecate
POST   /v1/skills/evolution    # 触发离线自进化一轮（§10）
GET    /v1/skills/{id}/stats   # 使用统计/成功率
```

### 8.5 系统接口与知识查询

```
GET  /health               # 存活与依赖探活（retriever/knowledge backends/store）
GET  /v1/config            # 运行配置（模型、检索、知识后端与工具开关）
POST /v1/knowledge/query   # （可选）知识库统一只读查询，聚合多后端
```

> **写接口约定**：知识记忆库的 add/update/delete **不提供对外 API**（§7），由 Agent 工具（受控）与内部更新机制（Ingestor）调用；外部批量导入走默认关闭的管理端点。

### 8.6 接口契约总览

| 接口 | 方法 | 用途 | 鉴权 | 现状兼容 |
|---|---|---|---|---|
| `/retrieve` | POST | 检索 | 公开（可选 Key） | ✅ 直接复用 |
| `/v1/agent/search` | POST | 智能体单轮问答 | 用户 | 新 |
| `/v1/agent/tasks` | POST | 异步批量 | 用户 | 新 |
| `/v1/trajectories` | GET/POST | 轨迹读写检索 | 用户（DELETE 管理员） | 新 |
| `/v1/skills` | GET/POST/PUT | Skill 管理 | 用户（发布/下线管理员） | 新 |
| `/v1/skills/evolution` | POST | 自进化触发 | 管理员 | 新 |
| `/v1/knowledge/query` | POST | 知识库只读查询（写仅内部/工具） | 用户 | 新（可选） |
| `/health` | GET | 探活 | 公开 | 新 |

> **接口治理**：任何新接口必须归入上述某一类（检索/智能体/轨迹/Skill/知识查询/系统）；若不 fit，先在本文档扩展契约再实现。对外响应统一标准信封 `{code, data, message}`，鉴权用 `X-Api-Key` 头。

## 9. 轨迹存储设计

> 对应设计原则 **D3 一切皆轨迹**（§3.1）。

### 9.1 存储形态

| 数据 | 存储 | 说明 |
|---|---|---|
| 轨迹正文（turns 全量） | JSONL（按日分文件）+ 可选 parquet | 可回放、可分析、可入训练（v2） |
| 轨迹元数据索引 | SQLite / DuckDB | id、question、status、时间、skills_used 快速查询 |
| 语义索引 | faiss（轨迹摘要 embedding） | `/v1/trajectories/search` 相似召回 |
| 归档 | S3/OSS 兼容 | 冷数据迁移 |

### 9.2 Repository 接口

```python
class TrajectoryRepository(Protocol):
    async def save(self, traj: Trajectory) -> str: ...
    async def get(self, trajectory_id: str) -> Optional[Trajectory]: ...
    async def query(self, filter: TrajFilter, page: int, size: int) -> List[Trajectory]: ...
    async def search_similar(self, question: str, topk: int) -> List[Trajectory]: ...
    async def stats(self) -> Dict[str, Any]: ...
```

实现：`FileTrajectoryRepository`（v1 默认，JSONL+SQLite）、`S3TrajectoryRepository`、`MemoryTrajectoryRepository`（测试用）。

### 9.3 写入时序

```
Policy 每产生一个动作 → Turn 记录进内存缓冲
工具执行返回观测   → Turn.observation 补齐
循环结束          → Trajectory 组装完成
                   → 异步写入轨迹存储（不阻塞响应）
                   → 异步触发知识沉淀(Ingestor) + skill 评估打标
```

### 9.4 轨迹的用途

1. **回放与调试**：`trajectory_id` 定位任何一次搜索的完整过程；
2. **评估**：批量回填 EM/LLM-judge，产出成功率、平均轮次、检索命中质量等指标；
3. **Skill 进化输入**（§10）：归纳源；
4. **知识沉淀输入**（§7.3）：经 Ingestor 写入知识库；
5. **v2 训练数据源**：格式与现有 `scripts/data_process` 的 parquet 兼容，可直接转 RL 训练集。

## 10. 自进化 Skill 设计

> 对应设计原则 **D4 进化非梯度**、**D5 稳定索引 + 活参数**（§3.1）。

### 10.1 定义

Skill = 一组"触发条件 + 执行体（提示词模板/工具序列/代码）+ 统计"，用于把反复成功的搜索经验固化为可复用组件。**v1 的自进化是离线、增量、基于轨迹统计与 LLM 归纳的**，不做策略梯度优化（RL 留给 v2）。

### 10.2 Skill 生命周期

```
采集(轨迹) → 评估(打标) → 归纳(聚类成功模式) → 提炼(LLM 生成 skill 草稿)
   → 验证(A/B 对比基线) → 发布(active) → 运行(被 policy 触发)
   → 监控(stats) → 淘汰(deprecated) 或 迭代(v+1)
```

### 10.3 自进化循环（离线批处理）

**Step 1 采集**：从 `TrajectoryRepository` 拉取最近 N 条轨迹（如 7 天 / 5000 条）。

**Step 2 评估**：`skills/evaluator.py` 为每条轨迹打标：
- 自动指标：EM / F1 / 命中率；
- LLM-judge：对开放性问题的答案打分（1-5）；
- 用户反馈（若有）：点赞/踩。

**Step 3 归纳**：对成功轨迹做模式聚类：
- 按问题类型（多跳、对比、时效性、定义类…）；
- 按动作序列模式（如 `search → knowledge_query → search → answer`）；
- 按查询改写模式（追问式、拆解式、同义扩展式）；
- 失败轨迹反向归纳：标注常见失败模式（过早 answer、查询过泛、未利用图谱）。

**Step 4 提炼**：LLM 将聚类结果转为 skill 草稿：
- `trigger`：规则（关键词/问题分类器）或 LLM 触发判断；
- `body`：提示词模板（注入到决策 prompt）或工具序列（如"先 knowledge_query 再 search"）；
- 草稿默认 `status=draft`，可人工审核。

**Step 5 验证（A/B）**：在验证集（未参与归纳的轨迹对应问题）上对比：
- 基线（无 skill / 旧版本 skill）vs 新 skill；
- 指标：成功率提升、平均轮次下降、检索成本下降；
- 达到阈值（如成功率 ≥ 基线 +2pt 且无副作用）→ 发布。

**Step 6 发布与监控**：
- 发布为 `active`，注入 `skill_policy` 可用集；
- 每次被触发回写 `stats`（usage_count、success_rate、avg_score、last_used）；
- 连续低分（如成功率 < 阈值且使用次数足够）→ `deprecated`；
- 高分 skill 可触发再迭代（基于新轨迹优化版本号）。

### 10.4 Policy 如何消费 Skill

```python
class SkillPolicy(Policy):
    async def decide(self, state) -> Action:
        # 1) 候选 skill：registry.match(state.question, state.memory)
        # 2) 若有高置信 skill → 按其 body 执行/注入 prompt
        # 3) 否则回退 llm_policy（默认决策）
        # 4) 记录 decision.skill_id 到 Turn（供统计与进化）
```

### 10.5 与知识记忆库的联动

- Skill 触发时优先 `knowledge_query` 复用既有知识证据；
- 高置信 skill 命中的轨迹，其新证据经 Ingestor 优先沉淀；
- 进化管线产出的**总结知识（成功模式、改写经验）直接 add 进知识库**，供后续查询命中；
- 进化归纳时读取图谱子图与 SQL 统计（成功率、命中率），辅助聚类与验证。

### 10.6 自进化索引（三层策略）与检索型 Skill

索引不采用"在线重建"路线，而是三层分工，兼顾稳定性与进化性：

1. **基础索引层（离线，保质量）**：语料本体（faiss/BM25）走离线批量构建（`scripts/index_build/`），构建产物可验证、可回滚，运行期只读；
2. **经验层（在线参数，轻量自适应）**：不重建索引，只进化"索引之上的参数"——重排权重、混合检索融合系数、查询扩展/改写词典与规则；这些参数由**检索型 Skill** 承载（`retrieval/adaptive.py`），复用 §10.3 的进化闭环（从轨迹成败归纳 → A/B 验证 → 发布）；高频新增语料先进"热桶"小索引，定期离线合并进主索引；
3. **图谱活索引层（随轨迹进化）**：运行经验（实体/关系/证据）经 Ingestor 持续沉淀进图谱，查询时经混合检索融合图谱子图——图谱本身就是随轨迹自进化的"活索引"。

**检索型 Skill 清单（示例）**：
- 后端选择策略：何时 hybrid / 何时图谱优先 / 何时 SQL；
- 查询改写模式：追问式、拆解式、同义扩展、实体限定；
- 融合系数与重排权重的调整规则。

**落地节奏**：v1 先做 基础索引（现状）+ 图谱活索引（§7）+ 检索型 Skill 起步（零重建风险）；热桶增量索引留 v2（时效性需求出现时）。

---

# 第四部分：视图与落地

## 11. 4+1 视图

> 本章与第二部分的关系：§4 给出架构总览，本章按 4+1 视图规范完整刻画系统；逻辑视图的依赖规则、进程/物理视图的细节是 §4/§5 的规范化展开。

### 11.1 逻辑视图（Logical View）

```
api/ ──→ agent/ (loop → policy → actions)
  │          │
  │          ├─→ memory/ (knowledge_base CRUD | backends | ingestor | trajectory)
  │          ├─→ tools/ (retriever | knowledge_query/add | answer)
  │          └─→ skills/ (registry | executor | evolution)
  │
  ├─→ retrieval/ (server → backends → rerank | knowledge_adapter)
  └─→ generator_llms/（环境层，被 tools/answer 与 loop 反馈引用）
```

依赖规则（开发约束）：
- `agent/` 不直接依赖 `verl/`（RL 相关仅存在于 `agent/loop/rl/` 适配层）；
- `api/` 只依赖 `agent/`、`retrieval/`、存储接口，不依赖具体实现；
- 知识/存储访问一律通过接口（`KnowledgeBase` CRUD / `TrajectoryRepository`），后端可插拔替换；
- 现状 `s3/` 代码在过渡期保留，新代码禁止反向依赖 `s3/`（除 `s3/search/retrieval_server.py` 先被 `retrieval/server.py` 包装复用）。

### 11.2 开发视图（Development View）

```
包: agent / retrieval / generator_llms / tools / api / evaluation / scripts / utils
测试: 单元(actions/parser) · 集成(loop+stub tools) · 端到端(api) · 契约(pytest + httpx)
配置: YAML + 环境变量（检索 url、知识后端、模型、阈值）
CI:  lint → unit → integration → contract
```

模块间通过类型注解 + Protocol 依赖，便于 mock 与并行开发。

### 11.3 进程视图（Process View）

```
进程1 Agent Service (api/agent_api + agent loop)
  └─ 线程/异步任务池：每个 question 一个循环任务
进程2 Retriever Service (FastAPI /retrieve)      ← 可多实例
进程3 Knowledge Backends（图谱 Neo4j/Json/Sqlite + 向量 faiss + SQL + BM25，可拆多实例）
进程4 Trajectory Store（JSONL 写入 + SQLite 查询 + embedding 索引）
进程5 Generator LLM（外部 API 或本地 vLLM）
进程6 离线管道：
  ├─ trajectory_eval（打标）
  ├─ skill_evolution（归纳→验证→发布，总结知识 add 进知识库）
  └─ knowledge_ingest（轨迹沉淀 / 记忆聚合 / 批量导入）
```

数据流（一次搜索）：
```
Agent Service → Policy → Retriever Service(/retrieve)
Agent Service → Knowledge Base(query 工具调用 / add 受控写入)
Agent Service → Generator LLM(feedback/answer)
Agent Service → Trajectory Store(写轨迹)
Trajectory Store / Skill进化 → Ingestor → Knowledge Base(add/batch_upsert)
```

### 11.4 物理视图（Physical View）

```
┌─ 部署拓扑（示例，可单机全合一）─────────────────────────┐
│ 节点A 检索节点    : retrieval/server.py + faiss/BMM +   │
│                   embedding API(可远程)               │
│ 节点B Agent 节点  : api/app.py + agent loop           │
│ 节点C 知识节点    : 图谱(Neo4j/json/sqlite)+向量+SQL+   │
│                   BM25（可拆分部署，写入走 Ingestor）   │
│ 节点D 存储节点    : 轨迹 JSONL + SQLite + 对象存储      │
│ 节点E 模型服务    : vLLM（生成器 LLM）或外部云 API      │
│ 节点F 离线节点    : 定时跑 skill_evolution / 评估       │
└─────────────────────────────────────────────────────┘
资源：Agent 节点无 GPU 依赖（LLM 走远程/vLLM），检索/向量节点需 GPU 或远程 embedding；
    图谱、SQL、BM25 为 CPU 型后端；轨迹存储可挂载同一磁盘或走对象存储。
```

### 11.5 场景视图（Scenarios / 用例）

| 用例 | 主流程 |
|---|---|
| **U1 单轮多跳问答** | 用户提问 → `skill_policy` 匹配 skill → 多轮 search/knowledge_query → answer → 轨迹入库 |
| **U2 知识增强检索** | 问题命中知识库（图谱/向量/SQL）→ 证据混入 `<information>` → 回答更精准 → 轨迹经 Ingestor 沉淀回写 |
| **U3 历史轨迹复用** | 新问题与历史轨迹语义相似 → `/v1/trajectories/search` 召回 → 注入 prompt 作为参考 |
| **U4 Skill 自进化** | 离线批处理：轨迹评估 → 聚类 → LLM 提炼 skill → A/B 验证 → 发布/淘汰（§10） |
| **U5 批量评测** | `/v1/agent/tasks` 提交 N 个问题 → 异步跑完 → 评估回填 → 报告 |
| **U6 失败诊断** | 定位 `trajectory_id` → 回放 turns → 定位失败轮（过早 answer / 查询失败 / 知识未命中） |
| **U7 自进化索引** | 轨迹评估发现检索弱项 → 归纳检索型 Skill（改写/融合权重）→ A/B 验证 → 发布生效；图谱证据随轨迹持续沉淀 |

## 12. 与现状代码的迁移路线（不含 RL）

| 阶段 | 内容 | 交付物 |
|---|---|---|
| M0（现状） | s3 代码仓，RL 训练闭环 | 已有 |
| M1 骨架化 | 新建 `agent/`、`api/`、`retrieval/backends`；复用 `generation_s3.postprocess_predictions` 作为 `actions/parser`；复用 `retrieval_server.py` 为 `retrieval/server.py` | v1 可运行的单轮搜索服务（无 RL） |
| M2 轨迹 | 实现 `trajectory_store` + `/v1/trajectories`；智能体循环全量记录 turns | 轨迹可存可查可回放 |
| M3 知识层 | 实现 `KnowledgeBase` CRUD + Ingestor + 图谱/向量/SQL/BM25 适配器（先 Json/Sqlite/faiss/pyserini）+ `knowledge_query/add` 工具 + 混合检索 | 知识记忆库接入智能体（CRUD 工具化） |
| M4 Skill | 实现 skill 模型/registry/executor + 评估 + 自进化管线 + A/B + 检索型 Skill（经验层自适应，§10.6） | 自进化闭环（含检索自适应） |
| M5 打磨 | 异步任务、鉴权、监控、评测脚本统一、文档 | 生产可用 v1 |
| M6（v2，预留） | 基于轨迹数据 + `agent/loop/rl/` 适配 verl 接入 RL；`s3/` 代码退役 | RL 训练闭环 |

**兼容性保证**：
- `/retrieve` 接口签名不变，现有脚本/基线不破坏；
- 动作协议（`<query>` JSON、`<search_complete>`、`<information>`）不变；
- 轨迹数据可直接转换为现有 parquet 训练格式（`scripts/data_process` 复用）；
- verl 侧 `LLMGenerationManager.run_llm_loop` 逻辑可整体迁移为 v2 的一个 loop 实现，接口签名保留。

## 13. 里程碑与验收

| 里程碑 | 验收标准 |
|---|---|
| M1 | `POST /v1/agent/search` 可完成多跳搜索并返回 answer；无 verl 依赖 |
| M2 | 搜索后 `/v1/trajectories/{id}` 可回放完整 turns；`/search` 可相似召回 |
| M3 | 知识项 add 后可被 `knowledge_query` 命中（图谱/向量/SQL/BM25 至少各一）；自进化总结知识可插入并被查询命中 |
| M4 | 跑完一轮 evolution 后 skill 库新增 ≥1 个 active skill 且验证集成功率不降；检索型 Skill 生效后混合检索命中率不降 |
| M5 | 全链路压测、鉴权、监控就绪；评测脚本统一 |

---

## 14. 附录

### 14.1 关键接口契约示例（JSON）

```json
// POST /v1/agent/search
{ "question": "Which director won the most Oscars?", "options": { "max_turns": 3 } }

// Response
{
  "code": 0,
  "data": {
    "trajectory_id": "tr-0001",
    "answer": "John Ford...",
    "status": "success",
    "turns": 3,
    "skills_used": ["skill-entity-disambiguation-v1"]
  }
}
```

```json
// 轨迹片段
{
  "trajectory_id": "tr-0001",
  "question": "...",
  "turns": [{
    "turn_index": 1,
    "action": {"type": "search", "payload": {"query": "{\"query\":\"...\"}", "topk": 3}},
    "observation": {"type": "information", "docs": [{"doc_id": "d42", "score": 0.87}],
                    "content": "<information>Doc 1... </information>"},
    "decision": {"policy": "skill_policy", "skill_id": "skill-entity-disambiguation-v1"},
    "latency_ms": 412
  }]
}
```

### 14.2 配置示例（config/agent.yaml）

```yaml
agent:
  max_turns: 3
  topk: 3
  policies: [skill_policy, llm_policy]
  default_policy: llm_policy
retriever:
  url: http://127.0.0.1:3000/retrieve
  mode: hybrid            # sparse | dense | hybrid | graph
knowledge_base:
  backends: [graph, vector, sql, bm25, episodic]   # 按需启用
  graph:    { backend: sqlite, neo4j: { uri: bolt://localhost:7687 } }
  vector:   { backend: faiss, index_dir: data/indexes_knowledge }
  sql:      { backend: sqlite, dsn: data/knowledge.db }
  bm25:     { index_dir: data/indexes_wiki, offline_rebuild: true }
  episodic: { ttl_hours: 24 }
  tools: { query: on, add: guarded, update: off, delete: off }   # CRUD 工具开关
  ingest_routing: { entity: [graph, vector], document: [vector, bm25], row: [sql], snippet: [vector, episodic] }
trajectory_store:
  backend: file           # file | s3
  dir: data/trajectories
skill:
  evolution_interval_days: 7
  publish_threshold: 0.02 # 相对基线成功率提升阈值
```

### 14.3 设计框架总结与接口设计 Prompt

#### 14.3.1 设计框架总结（v2.0）

**一句话**：AgenticSearch v1 = 无 RL 的智能体搜索平台 = 「兼容 s3 动作协议的执行循环」+「CRUD 式知识记忆库」+「全量轨迹存储」+「离线自进化 Skill」，以**服务化、资产化、自进化**回应现状五大问题（§1.3）。

**文档结构**：第一部分 定位与核心设计（§1–§3）→ 第二部分 总体架构（§4–§5）→ 第三部分 详细设计（§6–§10）→ 第四部分 视图与落地（§11–§14）。

**分层与关键决策**：

1. agent/：loop（顺序循环，无 verl 依赖）+ policy（llm_policy / skill_policy）+ actions（兼容 `<query>` / `<search_complete>` / `<information>` / `<important_info>`）；
2. 知识层：知识记忆库是独立子系统，多后端（图谱/向量/SQL/BM25/episodic）实现同一 CRUD 协议；Agent 工具化调用（读开、写受控）；自进化知识与记忆聚合经内部 Ingestor 直插，不经 Agent；
3. 轨迹层：全量 turns 记录（action/observation/decision/latency），可回放、可评估、可转 v2 训练集；
4. Skill 层：离线闭环 采集→评估→归纳→提炼→A/B 验证→发布→监控→淘汰；检索策略类知识作为"检索型 Skill"复用同一管线；
5. 对外 API：`/retrieve` 向后兼容 + `/v1/agent` + `/v1/trajectories` + `/v1/skills` + `/v1/knowledge/query`（只读，写不对外）；
6. 索引策略：基础索引离线批量构建（保质量），经验层做轻量自适应（重排权重/融合系数/改写规则），图谱作为"活索引"随轨迹进化（详见 §10.6）。

#### 14.3.2 接口设计 Prompt（可复用）

> 用途：后续实现/扩展本系统时，可直接将以下 prompt 投给 LLM 生成或评审接口定义；也可作为人工评审 checklist。约定与正文 §7–§10 一致，冲突时以正文为准。

```text
你是 AgenticSearch 平台的接口设计者。请严格遵循以下契约输出接口定义（Python Protocol + REST 路由），不要引入契约之外的概念。

[范围与红线]
- 分层：agent(loop/policy/actions) / retrieval / knowledge(backends+ingestor) / skills / api / evaluation。
- v1 不含 RL；动作协议必须兼容 <query>/<search_complete>/<information>/<important_info>。
- /retrieve 接口签名向后兼容：{queries, topk, return_scores}；所有对外响应用标准信封 {code, data, message}，鉴权 X-Api-Key。
- agent/ 不依赖 verl/；api/ 只依赖接口抽象；新代码不反向依赖 s3/。

[知识记忆库 CRUD 契约]
1. 对接收敛为四接口：query(查)/add(增)/update(改)/delete(删)，另加 batch_upsert(批量幂等合并)。
2. 统一模型：
   - KnowledgeItem{kind: entity|relation|document|row|snippet|fact, key(幂等键), content, meta{source, confidence, priority, ttl, tags}}
   - KnowledgeQuery{query, mode: semantic|graph|sql|bm25|hybrid, kinds, seed_entities, topk, filters}
   - KnowledgeHit{key, kind, content, score, evidence(可直接进 <information> 的可读文本)}
3. 多后端实现同一协议：graph(Neo4j/Json/Sqlite/Remote)、vector(faiss/milvus)、sql(sqlite/postgres)、bm25(pyserini)、episodic(redis+TTL)。
4. 写入路由配置驱动、可 fan-out：entity/relation→graph(+vector)；document/snippet→vector(+bm25)；row→sql；会话片段→episodic。BM25 重索引：在线只写暂存，重建走离线。

[工具化规则]
- knowledge_query：默认开启，主读工具，evidence 直接作为观测。
- knowledge_add：受控开启，强制携带 source/confidence。
- knowledge_update / knowledge_delete：默认关闭，仅管理员/进化管线。
- 每次工具调用记入 Turn（含命中/写入条数），外层统一权限+审计+限流。

[更新机制（不经 Agent，后台直插）]
- 轨迹沉淀→add；自进化总结知识→add/update；多条记忆聚合→batch_upsert+旧条目软删；外部导入→batch_upsert。
- 全部经 Ingestor：解析→路由→幂等 upsert→IngestReport{written, skipped, failed}（按 source 可追溯）。

[轨迹契约]
- Trajectory{trajectory_id, session_id, question, status, answer, turns[], evaluation, skills_used, knowledge_updates}。
- Turn{turn_index, action, observation, decision{policy, skill_id, reason}, latency_ms, token_usage}。
- 异步写入不阻塞主循环；存储：JSONL(按日) + SQLite 元数据 + faiss 语义索引。

[Skill 自进化契约]
- 生命周期 draft→active→deprecated，版本递增；stats 持续回写。
- 进化闭环：采集→评估(EM/LLM-judge)→归纳(聚类)→提炼(LLM 生成 trigger+body)→A/B 验证(达阈值)→发布→监控→淘汰。
- Skill 命中轨迹的新证据经 Ingestor 优先沉淀；进化产物（总结知识）直接 add 进知识库。

[输出要求]
- 先给 Protocol/Pydantic 定义，再给 REST 路由表（方法/路径/请求/响应/鉴权级别），最后列配置项与兼容性说明。
- 任何新接口必须归入上述某一类；若不 fit，先在文档中扩展契约再实现。
```

---

*（本文档为 v2.0 交付件：重组为「第一部分 系统定位与核心设计 → 第二部分 系统总体架构 → 第三部分 详细设计 → 第四部分 视图与落地」四部分结构；知识记忆库以 CRUD 四类接口对接、Agent 工具化调用、自进化知识/记忆聚合经内部更新机制直接插入。欢迎评审迭代。）*
