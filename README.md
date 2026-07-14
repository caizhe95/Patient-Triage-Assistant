# 患者端普通门诊导诊 Agentic RAG Demo

这是一个面向简历展示的轻量医疗导诊 Demo。项目定位是“小病、常见病、普通门诊导诊助手”，不做疾病诊断，也不替代医生面诊。

项目将 RAG 设计为 Agent 的 retrieval tool：LLM 先理解用户症状并规划推荐科室与检索问题，再调用分领域检索链路返回证据，最后生成结构化导诊回答。

## 核心能力

- LLM-first planner：使用 DeepSeek + LangChain 完成语义理解、科室选择和检索 query 生成。
- LangGraph workflow：编排 `safety guard -> LLM planner -> retrieval tool -> answer composer`。
- Retrieval-as-tool：RAG 不直接决定业务流程，只作为 Agent 获取证据的工具节点。
- 分领域检索：按候选科室和 shared 知识检索，减少全库噪声。
- 混合检索：保留 Qdrant 向量检索、BM25 和 RRF 融合。
- Evidence 返回：API 返回 answer、evidence、route_reason 和 debug_info。
- Safety guard：明显红旗输入直接返回线下/急诊提示，不进入普通导诊链路。
- 增强评测：提供 128 条自建评估集，量化路由、检索、安全拦截、回答规范和端到端稳定性。

## 流程

```text
用户输入
  -> safety guard
  -> LLM planner
  -> retrieval tool
  -> answer composer
  -> API response
```

普通门诊导诊需要配置 `DEEPSEEK_API_KEY`。缺少 Key 或 LLM 调用失败时，系统返回清晰错误，不降级到旧规则路由。

## 安装

```bash
pip install -r requirements.txt
```

复制 `.env.example` 为 `.env`，并配置：

```bash
DEEPSEEK_API_KEY=你的 Key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
QDRANT_URL=
QDRANT_API_KEY=
QDRANT_STORAGE_DIR=indexes/qdrant
QDRANT_COLLECTION_PREFIX=patient_triage
```

本地 Demo 可让 `QDRANT_URL` 和 `QDRANT_API_KEY` 为空，使用本地 `indexes/qdrant`。如果使用 Qdrant Cloud，则填写云端 Cluster URL 和 API Key。

## 构建索引

```bash
python scripts/ingest.py
```

`data/raw/emergency` 保留为资料，但不作为普通门诊 RAG 主链路的科室参与。

## 启动服务

```bash
python scripts/run_api.py
python scripts/run_streamlit.py
```

健康检查：

```bash
GET http://127.0.0.1:8000/health
```

调用示例：

```bash
curl -X POST "http://127.0.0.1:8000/chat" ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"最近胃痛反酸，饭后明显\"}"
```

## 增强评测

评测的是普通门诊导诊链路表现，不代表医学诊断准确率。

```bash
python scripts/eval_agentic_rag.py
python scripts/eval_agentic_rag.py --limit 20
python scripts/eval_agentic_rag.py --runs 3
python scripts/eval_agentic_rag.py --case-file data/eval/agentic_rag_cases.jsonl
```

评测输出：

```text
reports/agentic_rag_eval_results.csv
reports/agentic_rag_eval_summary.json
reports/agentic_rag_eval_report.md
```

评测集包含 128 条样本：

- 每个普通科室 12 条。
- Safety 红旗样本 20 条。
- 包含边界样本、混合症状、儿童相关样本、口语化输入和信息不足输入。
- 使用 `acceptable_domains` 记录多个合理科室，避免把真实导诊中的合理分流强行算错。

主要指标：

- `primary_domain_accuracy`：主科室是否等于期望科室。
- `acceptable_domain_accuracy`：主科室是否命中可接受科室。
- `candidate_domain_hit_rate`：候选科室是否包含可接受科室。
- `macro_domain_accuracy` / `per_domain_accuracy`：按科室统计表现。
- `evidence_domain_hit@1/@3/@5`：top-k evidence 是否命中期望证据领域。
- `safety_precision`、`safety_recall`、`safety_f1`：明显红旗拦截能力。
- `success_rate`、`error_rate`、`avg_latency_ms`、`p95_latency_ms`：端到端稳定性。
- `answer_non_empty_rate`、`avoid_diagnosis_rate`、`visit_suggestion_rate`、`evidence_reference_rate`：回答规范。

Markdown 报告只展示总体结论、核心指标表、按科室表现、Safety 指标和少量失败样例；完整逐条结果保存在 CSV。

## 测试

```bash
python -m compileall -q app frontend scripts tests
python -m pytest -q
```

如果本地 Python、pytest 或依赖不可用，需要先修复环境，不能假装测试通过。


