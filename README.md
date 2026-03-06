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
  export_neo4j.py
```
## Local LLM Usage

Install local inference dependencies:

```bash
python3 -m pip install -r requirements-local-llm.txt
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
  --output-dir outputs/model_compare
```

Notes:

- With `--llm-backend` and `--llm-model`, the planning stage uses the local model to choose a hotel and build each day itinerary.
- Without LLM arguments, the project keeps using the existing heuristic planner.
- Multi-model comparison writes each run into `outputs/model_compare/<model_name>/`.
