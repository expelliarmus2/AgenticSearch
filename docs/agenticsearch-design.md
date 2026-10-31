# AgenticSearch 设计文档（v3.0 · 精炼版）

> 版本：v3.0 精炼版。本文档为权威设计文档，以**总体架构**与**对外/交互接口**为核心，供本分支（`dev/agenticsearch`）后续代码开发直接使用。v2.1 冗长版（含 4+1 视图、配置示例、接口设计 Prompt 等细节）保留于 `main` 分支与 git 历史，可按需查阅。
>
> **一句话定位**：v1 = 无 RL 的智能体搜索平台 =「Agent Runtime 循环 + 四网关能力编排 + CRUD 式知识记忆库 + 事件化全量轨迹 + 候选审核制自进化」，并预留 v2 接入 verl RL 的接口。
>
> **参考方案**：s3（动作协议与 `/retrieve` 服务直接复用）、Search-R1（interleaved 推理-搜索循环）、evoGraph 系（知识图谱作"活索引/持续证据记忆"，随轨迹自进化、失败感知维护）——借鉴与差异见 §9。

---

## 1. 定位与设计原则

### 1.1 现状问题与定位

| 问题 | 现状表现 | 应对 |
|---|---|---|
| P1 循环与训练耦合 | 智能体循环嵌在 verl 训练框架内，无法独立部署 | **服务化**：Runtime 脱离 verl，成为独立推理服务 |
| P2 知识不沉淀 | 只有语料检索（BM25/Dense），无知识记忆层 | **资产化**：知识记忆库（CRUD 多后端，§6） |
| P3 经验不可追溯 | 搜索过程不落盘，无法回放/诊断/复用 | **资产化**：轨迹全量落库（§7.1） |
| P4 策略不自进化 | 决策改进靠人工迭代提示词与检索参数 | **自进化**：非梯度闭环（归纳→提炼→A/B，§7.2） |
| P5 接口不完整 | 对外仅 `/retrieve` | **服务化**：智能体/轨迹/Skill/知识 REST API（§5） |

### 1.2 设计原则

| # | 原则 | 主要落点 |
|---|---|---|
| D1 | 协议兼容优先 | `<query>`/`<search_complete>`/`<information>`/`<important_info>` 动作协议与 `/retrieve` 签名不变，存量调用方零破坏 |
| D2 | 读写分离 | 知识"读"对 Agent 开放为工具；"写"统一收口内部 Ingestor，对外不暴露写 API |
| D3 | 一切皆轨迹 | 每轮动作/观测/决策/耗时经 Bounded Event Queue 旁路全量落库，评估/进化/训练共用 |
| D4 | 进化非梯度 | 统计归纳 + LLM 提炼 + A/B 验证；产物「候选 + 审核」制，可解释、可回滚 |
| D5 | 稳定索引 + 活参数 | 基础索引离线构建只读；在线仅进化重排/融合/改写参数与图谱活索引 |
| D6 | 可插拔后端 | 知识/轨迹统一 Protocol，多后端（图谱/向量/SQL/BM25/记忆）可替换 |

### 1.3 范围

- **v1（本分支）**：智能体推理服务；知识记忆库 CRUD 对接；轨迹存储与语义检索；Skill 离线自进化；三层索引；对外 REST API。
- **非目标（v1）**：RL/策略梯度训练（v2，`agent/runtime/rl/` 预留，§8）；在线重建索引；多智能体编排；GUI。

---

## 2. 总体架构

### 2.1 架构图（七层，文本版）

```
调用方: 用户/应用 · 上层 Agent · 管理与评估人员
  │
  ▼
[ Edge  api/ ] API Facade → 鉴权/配额/审计 → 统一信封 {code, data, message}
  │
  ▼
[ Core  agent/ ] Agent Runtime（循环中枢）
  ├─ Planner 改写/拆解 · Policy(llm/skill) 决策 · actions 动作解析
  ├─ Evidence Buffer(证据) · Answer Composer(答案) · Budget(终止判定)
  │
  ├──▶ [ Capabilities  gateways/ ] 四网关：Retrieval / Knowledge / Skill / Model
  │        │                          （tool_switch 守门：权限/审计/限流）
  │        ▼
  │     [ DataPlane  retrieval/ ] fusion→rerank→backends(vector/bm25/sql/graph/legacy_s3)
  │        │                        knowledge/: CRUD 契约 + Ingestor（§6）
  │        ▼
  │     [ External ] LLM 服务 · 知识/记忆后端 · 索引与检索服务
  │
  └─ events（旁路, 不阻塞在线链路）──▶ [ Obs  obs/ ]
                                       Bounded Event Queue → 轨迹存储(JSONL+SQLite+faiss)
                                       → 评估(EM/LLM-judge) → 进化(归纳→提炼→A/B→候选/审核)
```

### 2.2 分层职责与目录映射

| 层 | 目录 | 模块 | 职责 |
|---|---|---|---|
| Edge | `api/` | `app/auth/envelope` + agent/trajectory/skill/retrieval/knowledge_api | 唯一入口：鉴权、配额、审计、统一信封、路由 |
| Core | `agent/` | `runtime/`、`planner.py`、`policies/`、`actions/`、`evidence_buffer.py`、`answer_composer.py`、`budget.py` | 搜索循环中枢与决策（六组件，§4.1） |
| Capabilities | `gateways/` | `retrieval_gateway` / `knowledge_gateway` / `skill_runtime` / `model_gateway` / `tool_switch` | 把下层能力封装为 Runtime 可调用的统一入口 |
| DataPlane | `retrieval/` | `fusion.py`、`rerank.py`、`adaptive.py`、`backends/*`、`knowledge/*` | 多源检索归一化/融合/重排；知识 CRUD 与 Ingestor |
| Obs | `obs/` | `eventbus.py`、`trajectory_store.py`、`evaluation/`、`evolution/` | 旁路观测：轨迹落库、评估打标、离线进化 |
| External | `generator_llms/` | qwen / gpt-azure / claude / together / deepinfra / local 适配器 | LLM provider（Model Gateway 的后端实现） |

### 2.3 关键约束（三条旁路/回流线）

1. **观测旁路**：`Runtime -.events.-> Bounded Event Queue`——有界队列（如 maxsize=10k），满时降级（丢 turn 明细、保轨迹摘要），**绝不阻塞在线链路**（D3）；
2. **Skill 候选制**：`Evolution -.candidate only.-> Skill Runtime`——进化产物仅以 draft 进入 Skill Registry，审核/A-B 达标后才 active（D4）；
3. **知识审核制**：`Evolution -.reviewed knowledge.-> Knowledge Gateway`——总结知识审后由 Ingestor 写入，未经审核不进知识库（D2，防污染）。

---

## 3. 核心数据流

**F1 在线问答（同步，低延迟路径）**

```
question → Edge(鉴权/信封) → Agent Runtime
  每轮: Planner.rewrite → Policy.decide → Action{search|knowledge_query|answer|…}
      → 经四网关执行 → Observation → Evidence Buffer（去重/引用追踪）
      → Budget 检查（轮次/Token/时延/成本）→ 终止 or 继续
  终止: Answer Composer → Model Gateway 生成最终答案（含引用）→ 返回 trajectory_id + answer
```

**F2 事件与轨迹（旁路异步，观测链路）**

```
Runtime 每轮发 TurnEvent / 结束发 TrajectoryEvent
  → Bounded Event Queue（有界、满则降级丢明细保摘要）
  → 异步消费者: 轨迹落库（JSONL + SQLite 元数据 + faiss 语义索引）
              → 知识沉淀（Knowledge Gateway → Ingestor，§6.3）
              → Evaluation Pipeline 打标回填
```

**F3 离线进化（定时批处理，全部"审后生效"）**

```
轨迹窗口 → 评估打标（EM/LLM-judge）→ 归纳聚类（成功模式/失败模式）
  → LLM 提炼（skill 草稿 trigger+body / 总结知识）
  → A/B 验证（vs 基线，达阈值）→ 产物分流:
      candidate only     → Skill Registry（draft → 审核 → active → skill_policy 可用）
      reviewed knowledge → Knowledge Gateway → Ingestor 入库
  → 监控 stats → 淘汰 / 迭代
```

---

## 4. 核心模块与数据模型

### 4.1 Agent Runtime 六组件（Core 层）

| 组件 | 文件 | 职责 |
|---|---|---|
| Agent Runtime | `agent/runtime/sequential.py` | 循环骨架 `run(question) → answer`，编排其余组件；`runtime/base.py` 抽象，`runtime/rl/` 为 v2 预留（§8） |
| Planner | `agent/planner.py` | 查询改写/拆解/同义扩展 |
| Policy | `agent/policies/` | 决策：`llm_policy`（LLM 直出）/ `skill_policy`（Skill 优先，llm 兜底，§7.2） |
| Evidence Buffer | `agent/evidence_buffer.py` | 证据工作记忆：滚动窗口 + 去重 + 引用追踪 |
| Answer Composer | `agent/answer_composer.py` | 证据筛选 → 答案生成 → 引用对齐 |
| Budget | `agent/budget.py` | 轮次/Token/时延/成本预算与终止判定 |

### 4.2 动作协议（兼容 s3，D1）

沿用现状 `s3/llm_agent/generation_s3.py` 协议，v1 直接复用其 `postprocess_predictions` 解析逻辑（落为 `agent/actions/parser.py`）：

- 模型输出含 `<query>{json}</query>`（动作查询）、`<search_complete>true|false</search_complete>`（终止信号）、`<important_info>…</important_info>`（证据标记）；
- 检索结果以 `<information>…</information>` 作为下一轮观测注入；
- 动作类型收敛为：`search` / `knowledge_query` / `knowledge_add` / `answer`（`agent/actions/action.py` + `registry.py`）。

### 4.3 数据模型（精简）

```python
@dataclass
class Turn:
    turn_index: int
    action: Action            # {type: search|knowledge_query|... , payload}
    observation: Observation  # {type: information|error|terminate, content, docs[]}
    decision: Decision        # {policy, skill_id, reason}
    latency_ms: int
    token_usage: Dict[str, int]

@dataclass
class Trajectory:
    trajectory_id: str; session_id: str; question: str
    status: str               # success | failed | timeout
    answer: Optional[str]
    turns: List[Turn]
    evaluation: Evaluation    # EM/F1/LLM-judge/用户反馈
    skills_used: List[str]
    knowledge_updates: List[Dict]   # 经 Ingestor 写入的知识项（可追溯）

@dataclass
class Skill:
    id: str; name: str; description: str; version: int
    trigger: SkillTrigger     # 规则或 LLM 判断
    body: SkillBody           # 提示词模板 / 工具序列
    stats: SkillStats         # usage_count / success_rate / avg_score
    status: str               # draft | active | deprecated
    source_trajectories: List[str]

@dataclass
class KnowledgeItem:          # 知识记忆库统一写入单元（"增"的载荷）
    kind: str                 # entity | relation | document | row | snippet | fact
    key: str                  # 幂等键（实体ID / 主键 / 文档ID）
    content: Dict[str, Any]
    meta: KnowledgeMeta       # {source, confidence, priority, ttl, tags}
```

---

## 5. 对外接口（REST，Edge 层）

约定：全部经 API Facade → 鉴权（`X-Api-Key`）→ 统一信封 `{"code": 0, "data": ..., "message": "ok"}`；任何新接口必须归入下表某一类。

| 接口 | 方法 | 用途 | 鉴权 | 现状兼容 |
|---|---|---|---|---|
| `/retrieve` | POST | 检索 `{queries, topk, return_scores}`；新增可选字段 `mode(sparse\|dense\|hybrid\|graph\|sql)`、`knowledge(kinds, seed_entities, filters)` | 公开（可选 Key） | ✅ 直接复用 |
| `/v1/agent/search` | POST | 同步单轮问答，返回 `{trajectory_id, answer, status, turns, skills_used}` | 用户 | 新 |
| `/v1/agent/tasks` | POST/GET | 异步批量评测任务 | 用户 | 新 |
| `/v1/agent/sessions/{sid}/continue` | POST | 多轮会话续接 | 用户 | 新 |
| `/v1/trajectories` | GET/POST | 轨迹读写；`/search` 语义召回；`/{id}/evaluate` 评估回填 | 用户（DELETE 管理员） | 新 |
| `/v1/skills` | GET/POST/PUT | Skill 管理；`/{id}/publish\|deprecate`；`/evolution` 触发自进化 | 用户（发布/进化管理员） | 新 |
| `/v1/knowledge/query` | POST | 知识库只读查询（多后端聚合） | 用户 | 新（可选） |
| `/health` `/v1/config` | GET | 探活 / 运行配置与工具开关 | 公开 | 新 |

**写接口约定**：知识记忆库 add/update/delete **不提供对外 API**（D2），仅由 Agent 工具（受控）与内部 Ingestor 调用；外部批量导入走默认关闭的管理端点。

---

## 6. 知识记忆库对接接口（CRUD）

> 与知识记忆库的全部交互收敛为 CRUD 四接口 + `batch_upsert`，统一经 Knowledge Gateway 接入；两种使用方式：**工具化**（Agent 调用，读开写受控）与**内部更新机制**（后台直插，不经 Agent）。

### 6.1 契约（retrieval/knowledge/knowledge_base.py）

```python
class KnowledgeBase(Protocol):
    async def query(self, q: KnowledgeQuery) -> List[KnowledgeHit]: ...   # 查
    async def add(self, items: List[KnowledgeItem]) -> IngestReport: ...  # 增
    async def update(self, keys: List[str], patch: Dict[str, Any]) -> int:# 改
    async def delete(self, keys: List[str], soft: bool = True) -> int: ...# 删
    async def batch_upsert(self, items: List[KnowledgeItem]) -> IngestReport: ...  # 批量幂等合并

# KnowledgeQuery{query, mode: semantic|graph|sql|bm25|hybrid, kinds, seed_entities, topk, filters}
# KnowledgeHit{key, kind, content, score, evidence}   # evidence 为可读文本，可直接进 <information>
# IngestReport{written, skipped, failed}              # 按 source 可追溯
```

### 6.2 工具化（Agent 侧，Knowledge Gateway + tool_switch 守门）

| 工具 | 接口 | 默认 | 说明 |
|---|---|---|---|
| `knowledge_query` | query | **开** | 主读工具，结果 `evidence` 直接作为观测 |
| `knowledge_add` | add | 受控 | 强制携带 source/confidence |
| `knowledge_update` | update | **关** | 仅管理员/进化管线可用 |
| `knowledge_delete` | delete | **关** | 防误删，软删优先，仅管理员 |

每次工具调用记入 Turn（含命中/写入条数）；外层统一权限 + 审计 + 限流。

### 6.3 内部更新机制（不经 Agent，全部经 Ingestor 收口）

| 机制 | 触发时机 | 调用 |
|---|---|---|
| 轨迹沉淀 | 轨迹结束（异步） | `add`（实体/关系/经验片段 → 图谱+向量） |
| 自进化总结知识 | 进化产出（§7.2） | `add`/`update`，**reviewed 审核后**写入 |
| 多条记忆聚合 | 定时（离线） | `batch_upsert` + 旧条目标记/软删 |
| 外部导入 | 离线脚本 | `batch_upsert` |

`Ingestor`（`retrieval/knowledge/ingestor.py`）：解析 → 按路由规则分发（可 fan-out）→ 幂等 upsert → IngestReport。

### 6.4 后端适配器与路由

| 后端 | 适配器 | 适用场景 |
|---|---|---|
| 图谱 | JsonGraph / SqliteGraph / Neo4j / Remote | 实体-关系、"活索引"（§7.3）；v1 默认 Json/Sqlite 零依赖 |
| 向量 | FaissVector | 语义召回，复用现有索引产物 |
| SQL | Sqlite / Postgres | 结构化知识与统计 |
| BM25 | LuceneBm25 | 全文检索（复用现状索引） |
| 记忆 | EpisodicMemory（内存/redis + TTL） | 会话级短期记忆 |

路由规则（配置驱动）：`entity/relation→graph(+vector)`；`document/snippet→vector(+bm25)`；`row→sql`；会话片段→episodic。向量化复用 `utils/embedding_client.py`。BM25 属重索引型后端：在线只写暂存区，索引更新走离线构建路径（§7.3）。

---

## 7. 轨迹存储与自进化 Skill

### 7.1 轨迹存储（obs/trajectory_store.py）

| 数据 | 存储 | 说明 |
|---|---|---|
| 轨迹正文（turns 全量） | JSONL（按日分文件） | 可回放、可分析、可转 parquet 训练集（v2） |
| 元数据索引 | SQLite | id/question/status/时间/skills_used 查询 |
| 语义索引 | faiss（轨迹摘要 embedding） | `/v1/trajectories/search` 相似召回 |
| 归档 | S3/OSS（可选） | 冷数据迁移 |

```python
class TrajectoryRepository(Protocol):
    async def save(self, traj) -> str
    async def get(self, trajectory_id) -> Trajectory
    async def query(self, filter, page, size) -> List[Trajectory]
    async def search_similar(self, question, topk) -> List[Trajectory]
```
实现：`FileTrajectoryRepository`（v1 默认）/ `S3TrajectoryRepository` / `MemoryTrajectoryRepository`（测试）。

### 7.2 Skill 生命周期（自进化闭环，全部"审后生效"）

```
采集(轨迹) → 评估(EM/LLM-judge 打标) → 归纳(成功模式/失败模式聚类)
→ 提炼(LLM 生成 trigger+body 草稿, status=draft) → A/B 验证(vs 基线, 阈值如成功率 ≥ +2pt)
→ 审核(自动阈值 + 人工抽检) → 发布(active, 注入 skill_policy 可用集)
→ 运行 → 监控(stats 回写) → 淘汰(deprecated) / 迭代(v+1)
```

`skill_policy.decide`：1) `registry.match(question, memory)` → 2) 高置信 skill 按 body 执行/注入 prompt → 3) 否则回退 `llm_policy`；命中记录 `decision.skill_id` 到 Turn（供统计与再进化）。

### 7.3 三层索引策略

1. **基础索引（离线，只读保质量）**：语料 faiss/BM25 批量构建（现状 `scripts/index_build/` 保留），可验证、可回滚；
2. **经验层（在线参数，轻量自适应）**：不重建索引，只进化重排权重/融合系数/改写规则——由**检索型 Skill** 承载（`retrieval/adaptive.py`，复用 §7.2 闭环）；高频新增语料先入"热桶"小索引，定期离线合并（热桶留 v2）；
3. **图谱活索引（随轨迹进化）**：运行经验（实体/关系/证据）经 Ingestor 持续沉淀进图谱，查询时融合图谱子图——图谱即"活索引"（借鉴 evoGraph，§9）。

v1 节奏：先做 1 + 3 + 检索型 Skill 起步（零重建风险）；热桶增量索引留 v2。

---

## 8. 与 RL（verl）的解耦与 v2 预留

| 关注点 | v1（本分支） | v2（预留；RL 开发继续在 main/其他分支） |
|---|---|---|
| 循环实现 | `agent/runtime/sequential.py`（FastAPI 推理服务，**无 verl 依赖**） | `agent/runtime/rl/` 适配层；`generation_s3.run_llm_loop` 可整体迁移为一个 loop 实现，接口签名保留 |
| 动作协议 | `<query>/<search_complete>/<information>/<important_info>` 不变 | 不变 |
| 训练数据 | 轨迹 JSONL 可直接转现有 parquet 格式（`scripts/data_process` 复用） | 轨迹直接作为 RL 训练集 |
| 检索服务 | 现状 `s3/search/retrieval_server.py` 经 `retrieval/backends/legacy_s3.py`（Legacy S3 Retriever Adapter）接入 | 独立 `retrieval/server.py` + backends |
| 依赖约束 | 新代码**禁止依赖 `verl/`**、**禁止反向依赖 `s3/`**（仅 legacy_s3 适配器接入） | 仅 `runtime/rl/` 与 verl 交互 |

---

## 9. 参考方案借鉴

| 方案 | 借鉴点 | 差异 |
|---|---|---|
| **s3**（本仓基础，EMNLP 2025） | 动作协议与 `/retrieve` 检索服务直接复用；"只优化搜索组件、不动生成器"的思路 → v1 决策（Policy）与生成（Model Gateway）解耦 | s3 专注 RL 训练；本平台 v1 无 RL，做服务化/资产化/自进化 |
| **Search-R1**（基于 veRL） | interleaved 推理-搜索循环形态 → 作为另一类 loop 策略保留（`generation_searchr1.py` 兼容） | Search-R1 以 RL 训练出搜索策略；本平台 v1 以 Skill + 提示词策略替代 |
| **evoGraph 系**（EvoGraph-R1 / CoEvoKG / EvoGraph-Mem / EoG） | ① 知识图谱作为"活索引/持续证据记忆"随搜索经验自进化 → §7.3 图谱活索引 + 轨迹沉淀；② 记忆会污染（过时/过度泛化有害）→ 知识「reviewed 审核制」+ Skill「候选制」双闸门；③ 图谱可产生训练任务 → v2 用图谱子图生成 RL 训练任务 | evoGraph 聚焦图谱记忆维护与 RL 自博弈；本平台 v1 为非梯度进化 + 审核闸门，图谱只是知识多后端之一 |

---

## 10. 里程碑（迁移路线摘要）

| 阶段 | 内容 | 验收标准 |
|---|---|---|
| M1 骨架化 | `agent/`（runtime/policies/actions + planner/evidence_buffer/answer_composer/budget）、`api/`（Facade+Auth+信封）、`gateways/`、`retrieval/backends/legacy_s3.py`；复用 `postprocess_predictions` 为 `actions/parser` | `/v1/agent/search` 可完成多跳搜索并返回 answer，**无 verl 依赖** |
| M2 轨迹 | `obs/eventbus`（Bounded Event Queue）+ `trajectory_store` + `/v1/trajectories` | 轨迹可存、可查、可回放、可相似召回 |
| M3 知识层 | Knowledge Gateway + `KnowledgeBase` CRUD + Ingestor + 适配器（先 Json/Sqlite/faiss/pyserini）+ 工具 + 混合检索 | add 后可被 `knowledge_query` 命中（≥1 后端）；reviewed 总结知识经 Knowledge Gateway 入库并可被命中 |
| M4 Skill | Skill Runtime + Evaluation Pipeline + 自进化管线（A/B + 候选/审核）+ 检索型 Skill | 一轮 evolution 产出 ≥1 个经审核转 active 的 skill 且验证集成功率不降；检索型 Skill 生效后命中率不降 |
| M5 打磨 | 异步任务、鉴权、监控、评测脚本统一 | 全链路可用 |
| M6（v2，另分支） | 轨迹数据 + `runtime/rl/` 接入 verl | RL 训练闭环 |

**兼容性保证**：`/retrieve` 签名不变，现有脚本/基线不破坏；动作协议不变；轨迹可转现有 parquet 训练格式；verl 侧 `run_llm_loop` 逻辑可整体迁移为 v2 的一个 loop 实现（接口签名保留）。
