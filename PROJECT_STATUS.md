# 项目现状说明文档

更新时间：2026-06-23

## 1. 项目定位

本项目是一个面向患者端的中文导诊分诊助手 Demo。用户输入自然语言症状描述后，系统会先进行急症识别，再进行医学领域路由，只在命中的领域知识库内做 RAG 检索，最终生成结构化中文导诊回答。

当前项目核心能力：

- 急症优先拦截：基于规则识别胸痛、呼吸困难、晕厥、意识不清、大出血等高危信号。
- 分领域路由：基于关键词规则优先判断科室，必要时尝试调用 DeepSeek。
- 混合检索：按候选领域和 shared 共享知识库同时执行向量检索与 BM25 检索。
- 融合重排：使用 RRF 融合向量/BM25 结果，再用 BGE reranker 或默认分数排序重排。
- 证据驱动回答：返回 answer、evidence、route_reason、debug_info，便于解释和调试。
- 双入口展示：FastAPI 提供 `/chat` 接口，Streamlit 提供简单前端页面。

## 2. 当前目录结构

实际源码位于外层目录下的同名子目录：

```text
Patient-Triage-Assistant-main/
└─ Patient-Triage-Assistant-main/
   ├─ app/                 # 后端应用、工作流、检索、LLM、数据处理
   ├─ data/                # 原始知识库与处理后的 chunk
   ├─ frontend/            # Streamlit 前端
   ├─ indexes/             # 已生成的 Qdrant/BM25 索引
   ├─ reports/             # RAG 评估结果与失败案例
   ├─ scripts/             # 启动、索引构建、评估脚本
   ├─ tests/               # smoke test
   ├─ requirements.txt
   ├─ docker-compose.yml
   ├─ .env.example
   ├─ .env                 # 本地环境配置，包含敏感配置风险
   └─ README.md
```

注意：当前目录不是 Git 仓库，未发现 `.git`。如果后续要重构，建议先初始化版本控制，并补充 `.gitignore`，避免提交 `.env`、`__pycache__`、大体积索引文件或本地 IDE 配置。

## 3. 技术栈与依赖

主要依赖来自 `requirements.txt`：

- Web 服务：FastAPI、uvicorn
- 前端展示：Streamlit、requests
- 工作流：LangGraph
- LLM 接入：openai SDK、langchain-openai，实际配置 DeepSeek 兼容接口
- 检索：qdrant-client、rank-bm25、jieba、scikit-learn、sentence-transformers
- 配置：pydantic-settings、python-dotenv
- 测试：pytest

外部服务与模型：

- DeepSeek：通过 `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL` 配置。
- Qdrant：默认 `QDRANT_URL=http://127.0.0.1:6333`，`docker-compose.yml` 提供 qdrant 服务。
- Embedding：默认 `BAAI/bge-small-zh-v1.5`，失败时回退到 TF-IDF。
- Reranker：默认 `BAAI/bge-reranker-base`，失败时回退到默认排序。

## 4. 核心运行链路

请求入口：

```text
POST /chat
  -> app.api.routes.chat
  -> app.graph.workflow.get_workflow().invoke(...)
```

LangGraph 工作流：

```text
emergency_check_node
  ├─ 如果命中急症：domain = emergency，直接进入检索
  └─ 如果非急症：进入 domain_route_node

domain_route_node
  -> 抽取症状字段
  -> 查询改写
  -> 领域路由

retrieve_node
  -> 候选领域 + shared 检索
  -> 向量检索 + BM25
  -> RRF 融合
  -> rerank
  -> 构造证据摘要和 debug 信息

generate_answer_node
  -> 优先根据证据片段拼装结构化回答
  -> 无可用证据摘要时再尝试 DeepSeek
  -> 失败则模板兜底
```

返回结构：

- `query`
- `domain`
- `candidate_domains`
- `route_reason`
- `route_method`
- `is_emergency`
- `emergency_reason`
- `answer`
- `evidence`
- `debug_info`

## 5. 模块职责

### `app/main.py`

创建 FastAPI 应用，注册 CORS 和 API 路由。启动时如果索引缺失，会调用 `build_all_indexes()` 自动构建演示索引。

现状风险：

- FastAPI 使用 `@app.on_event("startup")`，新版本 FastAPI 推荐 lifespan。
- 启动时自动构建索引可能导致服务冷启动很慢，也会把运行时职责和离线构建职责混在一起。

### `app/config.py`

使用 Pydantic Settings 读取 `.env`。全局通过 `get_settings()` 缓存配置。

现状风险：

- 部分配置有环境变量 alias，部分配置没有 alias，风格不统一。
- 路径默认值和环境变量路径混用，重构时需要明确“项目根目录”和“运行工作目录”的关系。

### `app/graph/`

包含 LangGraph 状态定义、工作流编排和节点实现。

关键文件：

- `state.py`：定义 `TriageState`。
- `workflow.py`：编排四个节点。
- `nodes.py`：包含急症识别、字段抽取、查询改写、证据整理和回答生成。

现状风险：

- `nodes.py` 逻辑较集中，混合了规则、LLM、检索结果格式化、回答生成策略，后续可拆分。
- 急症规则和领域关键词硬编码在代码里，不利于维护和评估。
- 证据拼装逻辑和医疗安全策略耦合，建议独立成 answer composer 或 safety policy。

### `app/retrieval/`

检索层负责领域路由、向量检索、BM25 检索、融合和重排。

关键文件：

- `domain_router.py`：关键词路由优先，必要时调用 LLM。
- `hybrid_retriever.py`：对候选领域和 shared 并发执行向量/BM25 检索。
- `vector_store.py`：调用 Qdrant 查询。
- `bm25_store.py`：加载 pickle 格式 BM25 索引。
- `fusion.py`：RRF 融合。
- `reranker.py`：BGE reranker 或默认排序。
- `qdrant_utils.py`：创建并缓存 Qdrant client。

现状风险：

- 领域关键词、优先级、急症关键词分散且硬编码。
- BM25 使用 pickle 持久化，存在兼容性和安全风险，只适合本地可信环境。
- Qdrant 默认读取 URL；如果本地 Qdrant 未启动，向量检索会失败并返回空结果。
- Embedding fallback 到 TF-IDF 时，索引构建和查询必须共享同一个已 fit 的进程内 vectorizer，否则可能出现查询不可用问题。

### `app/ingestion/`

数据加载、切分、共享知识抽取和索引构建。

关键文件：

- `loader.py`：读取 `data/raw/{domain}` 下 `.md` / `.txt` 文件。
- `splitter.py`：按标题和递归文本切分 chunk，并推断 `intent_type`。
- `knowledge_layout.py`：把通用模板类知识归并到 shared 知识层。
- `build_indexes.py`：构建 Qdrant collection、docs json、encoder json、BM25 pkl。

现状风险：

- 数据处理规则依赖中文标题、文件名后缀和模板文本，后续扩展知识库时容易隐性失效。
- 构建索引直接重建 collection，缺少版本化索引、增量更新和构建产物校验。
- `data/processed` 中已有 chunks，但当前构建链路主要从 `data/raw` 重新生成。

### `app/llm/`

DeepSeek 接入层。

关键文件：

- `deepseek_client.py`：直接用 OpenAI SDK 调 DeepSeek 兼容接口。
- `langchain_deepseek.py`：封装 LangChain `ChatOpenAI`。

现状风险：

- 同时存在直接 OpenAI SDK 和 LangChain 两套调用方式，错误处理、超时、重试和日志不统一。
- 配置里有 `api_timeout_seconds`，但当前 LLM client 未显式使用该超时配置。

### `frontend/streamlit_app.py`

Streamlit Demo 页面，允许输入 API 地址和症状描述，展示摘要、回答、证据片段和 debug 信息。

现状风险：

- 前端是演示性质，几乎没有输入校验、会话状态、历史记录和错误分类。
- UI 和后端返回结构耦合较紧，接口调整时前端容易同步破裂。

### `scripts/`

脚本入口：

- `run_api.py`：启动 FastAPI。
- `run_streamlit.py`：启动 Streamlit。
- `ingest.py`：构建索引。
- `eval*.py`：评估脚本。

现状风险：

- `eval.py`、`eval_clean.py` 中曾存在明显编码损坏和疑似语法损坏片段，不宜作为可信评估入口。
- reports 中已有部分评估报告，但也存在编码不一致情况。`rag_eval_report_random50.md` 可正常显示中文摘要。

## 6. 数据与索引现状

原始知识库覆盖 10 个领域：

| 领域 | 文件数 |
| --- | ---: |
| cardiology | 9 |
| dermatology | 9 |
| emergency | 10 |
| gastroenterology | 9 |
| general | 9 |
| gynecology | 9 |
| neurology | 9 |
| orthopedics | 9 |
| pediatrics | 9 |
| respiratory | 9 |

索引产物：

- `indexes/bm25/{domain}.pkl`
- `indexes/qdrant/{domain}_docs.json`
- `indexes/qdrant/{domain}_encoder.json`
- `indexes/qdrant/collection/patient_triage_{domain}/storage.sqlite`
- 额外包含 `shared` 共享知识索引

索引策略：

- 每个医学领域独立建索引。
- 通用 FAQ、检查准备、复诊建议、症状记录等模板类内容抽为 shared。
- 查询时总是检索候选领域，并附加 shared 领域。

## 7. API 与前端启动方式

安装依赖：

```bash
pip install -r requirements.txt
```

构建索引：

```bash
python scripts/ingest.py
```

启动 Qdrant：

```bash
docker compose up -d qdrant
```

启动后端：

```bash
python scripts/run_api.py
```

启动前端：

```bash
python scripts/run_streamlit.py
```

健康检查：

```bash
GET http://127.0.0.1:8000/health
```

导诊接口：

```bash
POST http://127.0.0.1:8000/chat
Content-Type: application/json

{"query":"胸痛伴出汗，活动后更明显"}
```

## 8. 测试与验证现状

现有测试：

- `tests/smoke_test.py`
  - `test_health`
  - `test_chat_response_shape`

本次验证结果：

- `pytest -q` 未能运行：当前终端中找不到 `pytest` 命令。
- `python -m pytest -q` 未能运行：当前终端中找不到 `python` 命令。
- 因此本机当前无法完成自动化测试验证。

已有评估报告：

- `reports/rag_eval_report_random50.md` 可读，记录了 50 条随机样本评估。
- 摘要指标包括：
  - 急症召回率：78.9%
  - 非急症路由准确率：100.0%
  - Recall@3：83.9%
  - MRR@3：0.715
  - 忠实度：44.7%
  - 幻觉率：55.3%

注意：这些指标来自既有报告文件，不是本次重新运行得到的结果。

## 9. 当前主要问题清单

### P0：重构前基础设施问题

- 当前不是 Git 仓库，缺少版本控制基线。
- `.env` 存在于项目目录，后续必须避免提交。
- `__pycache__`、`.idea`、索引产物和报告产物混在源码目录中，缺少清晰的提交边界。
- 当前终端找不到 `python` / `pytest`，无法验证项目是否可运行。

### P1：代码结构问题

- `app/graph/nodes.py` 职责过重，集中承载规则、抽取、证据整理、回答生成。
- 急症规则、领域关键词、优先级、意图推断规则散落在多个文件中，缺少统一配置层。
- LLM 调用存在 LangChain 和 OpenAI SDK 两条路径，行为不完全一致。
- 启动时自动构建索引，容易影响服务启动稳定性。

### P1：数据与索引问题

- 索引构建不具备版本化、增量更新、构建校验和回滚能力。
- BM25 pickle 不适合跨版本长期持久化。
- TF-IDF fallback 依赖进程内 fit 状态，部署时需要谨慎。
- Qdrant 本地服务与本地 path 模式边界不清晰，目前默认 URL 模式。

### P2：评估与质量问题

- `scripts/eval.py` 和 `scripts/eval_clean.py` 出现编码损坏或脚本损坏痕迹。
- README 尾部 Docker/Qdrant 段落存在乱码。
- 测试覆盖很浅，只验证接口形状，不验证路由、急症、检索、回答安全边界。
- 既有报告中幻觉率较高，回答生成和忠实度评估应作为后续重构重点。

## 10. 建议重构顺序

### 第一阶段：建立可控基线

1. 初始化 Git 仓库。
2. 新增 `.gitignore`，排除 `.env`、`__pycache__`、`.idea`、本地索引、临时报告等。
3. 修复 README 和评估脚本编码问题。
4. 固定 Python 环境，例如 `.python-version`、`requirements.lock` 或 `pyproject.toml`。
5. 恢复本地 `python` / `pytest` 可用性，确保 smoke test 能跑通。

### 第二阶段：拆分核心业务边界

1. 把急症规则、领域关键词、路由优先级迁移到配置文件或独立 policy 模块。
2. 拆分 `nodes.py`：
   - emergency policy
   - query extraction / rewrite
   - route service
   - retrieval orchestration
   - evidence composer
   - answer composer
3. 统一 LLM client，明确超时、重试、日志、fallback 行为。
4. 把启动时建索引改为显式离线任务或管理员任务。

### 第三阶段：增强检索与评估

1. 重建评估脚本，使用 UTF-8 数据和稳定样本集。
2. 增加针对急症召回、领域路由、证据召回、回答忠实度的单元测试和回归测试。
3. 为索引构建增加 manifest，记录数据版本、模型版本、chunk 参数、构建时间和校验信息。
4. 明确 Qdrant 部署模式：本地 path、Docker URL、远程服务三者择一或通过配置显式切换。

### 第四阶段：产品化改造

1. 增加多轮对话状态和用户画像字段：年龄、性别、病史、孕产状态、用药史。
2. 引入更严格的医疗安全策略：高危问题优先、免责声明、不可替代医生诊断。
3. 扩展前端：历史会话、证据展开、风险提示醒目展示、错误状态分类。
4. 完善日志、审计和脱敏策略。

## 11. 重构时建议保留的能力

- 分领域 RAG 架构。
- 急症优先拦截。
- `candidate_domains` + `shared` 的检索策略。
- `evidence` 和 `debug_info` 的可解释返回。
- 无 DeepSeek Key 时的规则/模板 fallback。
- Streamlit 快速演示入口。

## 12. 重构时建议优先删除或隔离的内容

- 已损坏或不可信的评估脚本副本。
- 源码目录内的 `__pycache__`。
- 本地 IDE 配置 `.idea`。
- 已生成索引是否入库需要重新决策；若保留，应放到明确的数据产物目录并记录版本。
- `.env` 应保留本地使用，但必须从版本控制中排除。

## 13. 推荐的目标结构草案

```text
app/
├─ api/
├─ core/               # config, logging, dependencies
├─ domain/
│  ├─ triage/           # 急症、安全策略、领域路由规则
│  ├─ retrieval/        # 检索接口与实现
│  └─ answering/        # 证据组织与回答生成
├─ ingestion/
├─ llm/
└─ workflows/

configs/
├─ emergency_rules.yaml
├─ domain_keywords.yaml
└─ retrieval.yaml

data/
├─ raw/
├─ processed/
└─ manifests/

tests/
├─ unit/
├─ integration/
└─ regression/
```

## 14. 总结

项目当前已经具备完整 Demo 闭环：知识库、索引构建、FastAPI、Streamlit、LangGraph 工作流、混合检索和证据化回答都存在。但它更像课程/演示项目，而不是可维护的生产级服务。

后续重构应先建立可运行、可测试、可版本控制的基线，再拆业务边界。医疗导诊场景里，最值得优先稳住的是急症召回、路由可解释性、证据忠实度和 fallback 行为。
