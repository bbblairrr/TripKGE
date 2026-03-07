# TripTailor-GraphRAG

面向个性化旅游规划的多层知识图谱增强检索与生成框架实现。

## 1. 功能对应你的提纲

- 多层图谱构建
1. 层1：城市与交通（`city/transport`）
2. 层2：POI 实体（`attraction/restaurant/hotel`）
3. 层3：偏好与约束（预算桶、强度、兴趣标签）
4. 层4：行程模式（从 `train.json` 的参考行程挖掘 day-slot-activity 骨架）

- Graph-Enhanced RAG
1. 硬约束过滤（城市、预算、餐价、酒店偏好）
2. 向量召回（TF-IDF）
3. 1~2 hop 图扩展
4. 社区检索（连通子图筛选）
5. 融合重排（vector + constraint + graph）

- 摘要层（关键创新）
1. 输入：重排候选 + 图路径证据
2. 输出：`EvidenceSummary`（候选ID、理由、预算风险、按天建议）

- 规划与验证
1. 规划器按候选 ID 生成 `plan`
2. 自动验证器检查：sandbox、预算、餐价、时序、去重、路线
3. 不通过自动 repair 一轮

- 实验
1. Baselines：`direct_llm / naive_rag / kg_only / graphrag_no_summary / graphrag_summary`
2. 指标：
   - `feasibility_pass_rate`
   - `personalization_proxy`（用于近似 Personalized Surpassing）
   - `route_distance_ratio`
   - `faithfulness`
   - `answer_relevancy`
   - `context_precision`
   - `context_recall`
   - `evidence_grounding_rate`（新增）

## 2. 数据

默认读取：
- `data/test.json`
- `data/train.json`
- `data/infomation.json`
- `data/Flight_Schedule.csv`
- `data/Train_Schedule.csv`

说明：为了与 sandbox 验证一致，候选池默认来自 `infomation.json`（按 pid 对应），不全量加载超大 `restaurants/accommodations` 表。

## 3. 快速开始

```bash
python3 -m pip install -e .
```

如果要把图谱写入本地 Neo4j，请安装可选依赖：

```bash
python3 -m pip install -e .[neo4j]
```

### 跑全方法实验（可加 `--limit` 先 smoke）

```bash
python3 scripts/run_experiments.py --limit 50
```

### 跑单条样本

```bash
python3 scripts/run_single.py --pid 1 --method graphrag_summary
```

### 导出知识图谱到本地 Neo4j

```bash
python3 scripts/export_neo4j.py \
  --password '你的neo4j密码' \
  --clear \
  --uri bolt://localhost:7687 \
  --database neo4j
```

只初始化一次（如果库里已有图则跳过）：

```bash
python3 scripts/export_neo4j.py \
  --password '你的neo4j密码' \
  --if-empty \
  --uri bolt://localhost:7687 \
  --database neo4j
```

后续直接从本地 Neo4j 读取图，不再本地重建：

```bash
python3 scripts/run_experiments.py \
  --limit 50 \
  --graph-source neo4j \
  --neo4j-password '你的neo4j密码'
```

单条查询也可直接从 Neo4j 读取：

```bash
python3 scripts/run_single.py \
  --pid 1 \
  --method graphrag_summary \
  --graph-source neo4j \
  --neo4j-password '你的neo4j密码'
```

如果 Neo4j 为空，希望自动“首次构建并写入，再从库里读”：

```bash
python3 scripts/run_experiments.py \
  --limit 50 \
  --graph-source neo4j \
  --neo4j-password '你的neo4j密码' \
  --neo4j-bootstrap
```

### 跑消融实验（组件开关 + 参数网格）

```bash
python3 scripts/run_ablation.py \
  --limit 100 \
  --hops 1,2 \
  --topk-vector 20,30 \
  --topk-final 15,20 \
  --use-graph-expansion 1,0 \
  --use-community 1,0 \
  --use-summary 1,0 \
  --target-metric feasibility_pass_rate
```

常用选项：
- `--max-runs`：限制组合数（先 smoke）  
- `--normalize-weights`：自动归一化 `w-vector/w-constraint/w-graph`  
- `--output-dir`：默认输出到 `outputs/ablation`
- `--graph-source neo4j --neo4j-password ...`：消融时也从 Neo4j 读取图

## 4. 输出

默认写入 `outputs/`：
- `experiment_summary.json`
- `<method>_predictions.jsonl`

## 5. 目录结构

```text
src/triptailor_graphrag/
  config.py
  data_loader.py
  pattern.py
  graph.py
  neo4j_store.py
  vector_index.py
  retrieval.py
  summarizer.py
  planner.py
  validator.py
  metrics.py
  pipeline.py
scripts/
  run_experiments.py
  run_single.py
  run_ablation.py
  create_dev_split.py
  export_neo4j.py
```

## Dev Split Workflow

Use a separate dev split for tuning, but create it from the evaluation set that is already aligned with `infomation.json`.

Create a deterministic `dev_eval/test_final` split from `test.json`:

```powershell
python scripts\create_dev_split.py `
  --data-dir data `
  --seed 42 `
  --dev-ratio 0.2 `
  --source-eval-file test.json `
  --dev-out splits\dev_eval.json `
  --test-out splits\test_final.json
```

Tune on `dev_eval.json`:

```powershell
python scripts\run_experiments.py `
  --data-dir data `
  --eval-file splits\dev_eval.json `
  --methods naive_rag kg_only graphrag_no_summary graphrag_summary `
  --vector-backend faiss `
  --output-dir outputs\dev_baselines
```

After the pipeline is frozen, run the final experiment on `test_final.json`:

```powershell
python scripts\run_experiments.py `
  --data-dir data `
  --eval-file splits\test_final.json `
  --methods naive_rag kg_only graphrag_no_summary graphrag_summary `
  --vector-backend faiss `
  --output-dir outputs\test_baselines
```
## Local LLM Usage

Install local inference dependencies:

```bash
python3 -m pip install -r requirements-local-llm.txt
```

Install FAISS retrieval dependencies only:

```bash
python3 -m pip install -e .[faiss]
```

Run one sample with a local model:

```bash
python3 scripts/run_single.py \
  --pid 1 \
  --method graphrag_summary \
  --llm-backend ollama \
  --llm-model qwen2.5-coder:7b
```

Compare multiple local models:

```bash
python3 scripts/run_experiments.py \
  --methods direct_llm graphrag_summary \
  --limit 20 \
  --llm-backend ollama \
  --llm-models qwen2.5-coder:7b deepseek-r1:8b mistral:latest \
  --judge-llm-backend ollama \
  --judge-llm-model qwen2.5-coder:7b \
  --output-dir outputs/model_compare
```

Notes:

- With `--llm-backend` and `--llm-model`, the planning stage uses the local model to choose a hotel and build each day itinerary.
- With `--judge-llm-backend` and `--judge-llm-model`, the personalization metric is scored by an LLM judge using blind A/B comparison between the generated itinerary and the dataset reference itinerary.
- Without LLM arguments, the project keeps using the existing heuristic planner.
- Multi-model comparison writes each run into `outputs/model_compare/<model_name>/`.

## FAISS Retrieval

The retriever now supports both lexical TF-IDF and dense FAISS backends.

- `vector-backend=auto` is the default. The project tries FAISS first and falls back to TF-IDF if FAISS or the local embedding model is unavailable.
- `vector-backend=faiss` requires a local embedding model and persists the vector index under `.cache/faiss/` by default.
- `vector-backend=tfidf` keeps the previous lexical retrieval behavior.
- The FAISS path is now city-aware and reranks shortlist candidates with a diversity-aware second stage before summary generation.
- Budget, meal range, hotel preference, opening-hours metadata, and route-anchor proximity are injected earlier into retrieval scoring instead of waiting for validator-only feedback.

Example commands:

```powershell
python scripts\run_single.py --pid 1 --method graphrag_summary --vector-backend faiss
python scripts\run_experiments.py --methods graphrag_summary --limit 20 --vector-backend faiss
```

Optional knobs:

- `--vector-cache-dir`: choose where the FAISS index files are stored.
- `--embed-model`: override the local embedding model used for dense retrieval.
- `--embed-batch-size`: control embedding throughput during index build.
- `--rebuild-vector-index`: force rebuilding the FAISS cache instead of reusing it.

## Neo4j Graph Reuse

The project now supports a Neo4j-first graph lifecycle:

- `graph-source=auto` is the default in the run scripts.
- If Neo4j credentials are configured, the project will:
  1. check whether the KG already exists in Neo4j
  2. build the KG once if missing
  3. persist it to Neo4j
  4. reuse the stored graph in later runs
- If Neo4j is not configured, the project falls back to local graph building.

Recommended setup:

```powershell
$env:TRIPTAILOR_NEO4J_URI="bolt://localhost:7687"
$env:TRIPTAILOR_NEO4J_USER="neo4j"
$env:TRIPTAILOR_NEO4J_PASSWORD="your_password"
$env:TRIPTAILOR_NEO4J_DATABASE="neo4j"
```

After that, regular runs will automatically reuse the stored KG:

```powershell
python scripts\run_single.py --pid 1 --method graphrag_summary
python scripts\run_experiments.py --methods graphrag_summary --limit 20
```

If you want to force local graph building for a run:

```powershell
python scripts\run_single.py --pid 1 --method graphrag_summary --graph-source local
```

## Metric Notes

- `feasibility_pass_rate`: hallucination-based feasibility. A plan is infeasible if transport details do not match sandbox options or POIs/hotels cannot be grounded to sandbox candidates.
- `constraint_satisfaction_rate`: the original validator pass signal for budget, meal range, time, deduplication, and route checks.
- `schema_validity_rate`: whether the generated plan structure and required fields are well-formed.
- `entity_grounding_rate`: the share of hotel and POI entries that can be matched back to sandbox candidates.
- `transport_grounding_rate`: the share of outbound/inbound transport details that can be matched back to sandbox transport options.
- `opening_hours_compliance`: the share of sightseeing activities that fit inside the attraction opening hours.
- `stay_duration_feasibility`: the share of activities whose allocated time slot can cover the required stay duration.
- `transport_time_feasibility`: whether outbound arrival and inbound departure times are compatible with first-day and last-day activities.
- `personalization_proxy`: an LLM-judge preference score from blind A/B comparison between the generated itinerary and the dataset reference itinerary for the same user request. The judge only sees `Itinerary A` and `Itinerary B`, with stable randomized order. If no judge model is configured, it falls back to the previous heuristic score.
- `average_route_distance_ratio`: average per-day route distance divided by the ground-truth average per-day route distance.
- `max_single_day_route_km`: the longest single-day route distance in the generated plan.
- Retrieval metrics now include `recall_at_5`, `recall_at_10`, `ndcg_at_5`, and `ndcg_at_10`.
- `answer_relevancy` uses local embedding cosine similarity by default, with lexical fallback if the local embedding model is unavailable.
- Retrieval itself can run on either `TF-IDF` or `FAISS`, depending on `--vector-backend`.

## Output Diagnostics

Per-sample outputs in `run_single` and `<method>_predictions.jsonl` now include:

- `metric_details`: metric-side breakdowns such as heuristic or LLM-judge personalization details.
- `diagnostics.retrieval.trace`: candidate counts, city scope, relaxed-filter notes, and seed counts.
- `diagnostics.retrieval.top_candidates`: top retrieved candidates with raw scores, normalized scores, rerank score, diversity penalty, and constraint notes.
- `diagnostics.retrieval.summary`: chosen shortlist ids, day suggestions, budget risk, and path evidence.
- `diagnostics.validator_failure_reasons` and `diagnostics.validator_checks`.
- `diagnostics.judge`: blind A/B judge details when a judge model is configured, or heuristic fallback details otherwise.
