# Academic Paper Search Pipeline

基于赛题"科研场景下复杂学术查询的智能论文搜索与推荐"的初始代码实现。

## 目录结构

```
code/
├── config/
│   ├── config.yaml                 # 主配置文件（所有可调参数集中在这里）
│   ├── .env                        # 环境变量（gitignore，需自行创建）
│   └── .env.example                # 环境变量模板
│
├── src/
│   ├── llm/                        # LLM 统一调用层
│   │   ├── __init__.py             #   导出 LLMClient, get_client
│   │   ├── client.py               #   多Provider客户端（自动重试+JSON修复+Token统计）
│   │   └── token_counter.py        #   Token计数 + 累计统计 + 成本估算
│   │
│   ├── prompts/                    # 所有 LLM Prompt 集中管理
│   │   └── __init__.py             #   Step1/3/4 的 System Prompt + User Prompt
│   │
│   ├── sources/                    # 学术 API 客户端
│   │   ├── __init__.py             #   统一导出 s2_client, arxiv_client, openalex_client
│   │   ├── semantic_scholar.py     #   Semantic Scholar API（主力检索+引文）
│   │   ├── arxiv.py                #   arXiv API（最新预印本+全文）
│   │   └── openalex.py             #   OpenAlex API（元数据查询）
│   │
│   ├── retrieval/                  # 检索引擎（零 LLM 调用）
│   │   ├── __init__.py
│   │   ├── bm25.py                 #   BM25 稀疏检索 + 本地重排
│   │   ├── embedding.py            #   向量稠密检索（Qwen-0.6B / MiniLM / TF-IDF）
│   │   ├── fusion.py               #   RRF / CombSUM 多路融合
│   │   └── citation_traversal.py   #   PageRank 引导的引文遍历
│   │
│   ├── evaluation/                 # 评测模块
│   │   ├── __init__.py
│   │   └── metrics.py              #   Precision / Recall / F1 / Recall@K / nDCG
│   │
│   ├── utils/                      # 工具模块
│   │   ├── __init__.py
│   │   ├── config_loader.py        #   YAML 配置 + .env 加载
│   │   ├── logger.py               #   日志设置
│   │   ├── cache.py                #   中间结果缓存（JSON文件）
│   │   └── data_loader.py          #   PaSa & Asta 数据集加载
│   │
│   ├── pipeline.py                 # 端到端流水线编排
│   ├── step1_query_understanding.py  # 模块1：查询理解（1× LLM）
│   ├── step2_hybrid_retrieval.py      # 模块2：混合检索（0 LLM）
│   ├── step3_ranking.py               # 模块3：论文排序（1-2× LLM）
│   └── step4_result_organization.py   # 模块4：结果归纳（0-1× LLM）
│
├── run_pipeline.py                 # 主入口：运行检索流水线
├── run_evaluation.py               # 评测入口：计算指标
├── requirements.txt                # Python 依赖
└── README.md                       # 本文件
```

> **注意**：`data/`（比赛参考数据集）、`papers/`（参考文献 PDF）、`config/.env`（API Key）均已加入 `.gitignore`，不在仓库中。clone 后需自行准备（见下方「数据准备」）。

## 数据准备

仓库不含以下内容，首次运行前需自行准备：

| 目录/文件 | 内容 | 来源 |
|-----------|------|------|
| `data/pasa/test.jsonl` | RealScholarQuery 评测集（50条查询） | 比赛方提供（PaSa 论文数据集） |
| `data/asta/` | PaperFindingBench 评测集（3个json） | 比赛方提供（AstaBench 数据集） |
| `papers/` | 10篇参考文献 PDF | arXiv / ChemRxiv 下载 |
| `config/.env` | API Key + 模型路径 | 从 `.env.example` 复制后填写 |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp config/.env.example config/.env
# 编辑 config/.env，填入 LLM API Key 和本地模型路径
```

需要配置的内容（`config/.env`）：

| 变量 | 用途 | 是否必须 |
|------|------|:--:|
| `DASHSCOPE_API_KEY` | 阿里云 DashScope（Qwen3-32B API） | 三选一 |
| `DEEPSEEK_API_KEY` | DeepSeek Platform（DeepSeek-V3 API） | 三选一 |
| `OPENAI_API_KEY` | OpenAI API（GPT-4o 备选） | 可选 |
| `EMBEDDING_MODEL_PATH` | Qwen3-Embedding-0.6B 本地路径 | 推荐 |
| `EMBEDDING_MODEL_PATH_LIGHT` | all-MiniLM-L6-v2 本地路径 | 推荐 |
| `S2_API_KEY` | Semantic Scholar API Key（提升速率） | 可选 |
| `NCBI_API_KEY` | PubMed Entrez API Key（提升速率） | 可选 |

> Semantic Scholar / arXiv / OpenAlex / DBLP API 免费且无需 Key。

### 3. 调整配置

编辑 `config/config.yaml`，关键参数：

```yaml
dataset:
  name: "pasa"            # "pasa" / "asta"
  query_filter:
    max_queries: 5        # 测试时限制数量，0=全部

pipeline:
  steps: [1, 2, 3, 4]     # 要运行的步骤
  use_cache: true          # 缓存中间结果，调参时避免重复调用API
```

### 4. 运行

```bash
# 完整流程（PaSa 数据集，前5条查询）
python run_pipeline.py --max-queries 5

# 仅运行检索（跳过查询理解）
python run_pipeline.py --steps 2

# Asta 数据集，仅 semantic 查询
python run_pipeline.py --dataset asta --query-type semantic --max-queries 10

# 禁用缓存（全新运行）
python run_pipeline.py --no-cache --max-queries 10

# 增量重跑：只重跑「有问题」的查询，其余保留
python run_pipeline.py --steps 2 --max-queries 0 --merge-existing
```

`--merge-existing` 会读现有结果（`output/stepN_*.json`），判断每条查询是否异常，只重跑异常的：

| 步骤 | 「异常」判定 |
|------|------|
| Step 1 | 子查询 fallback（keyword 只有 1 个且 = 原始查询）或为空 |
| Step 2 | Phase1 召回为空（限流）或 candidates 为空 |

适合跑完全量后发现少量查询因限流/解析失败而不完整，只补跑这几条。

### 5. 评测

```bash
# 评测最新流水线结果
python run_evaluation.py

# 指定结果文件
python run_evaluation.py --results output/step4_final_results.json

# 评测两个数据集
python run_evaluation.py --dataset both
```

## 流水线步骤详解

### Step 1: 查询理解（1× LLM API）

- **输入**: 自然语言查询
- **输出**: 结构化 JSON（`query_type`, `intent`, `domain`, `constraints`, `sub_queries`）
- **Prompt**: `src/prompts/__init__.py` → `QUERY_UNDERSTANDING_PROMPT`
- **LLM 调用**: `src/llm/client.py` → `LLMClient.chat_json()`（自动 JSON 修复重试）
- `specific` / `metadata` 类型查询跳过 LLM，使用规则解析

### Step 2: 混合检索（0 LLM）

| 阶段 | 操作 | 涉及模块 |
|------|------|----------|
| Phase 1 | Semantic Scholar + arXiv + OpenAlex API 检索 | `src/sources/` |
| Phase 2 | 本地 BM25 + Embedding 重排序 + RRF 融合 | `src/retrieval/bm25.py`, `embedding.py`, `fusion.py` |
| Phase 3 | PageRank 引导引文遍历 | `src/retrieval/citation_traversal.py` |

全阶段不消耗 LLM Token。

### Step 3: 排序（1-2× LLM API）

- **Coarse**: 六维自动评分（BM25 / Embedding / 引用量 / 时效性 / Venue / 约束覆盖）
- **Fine**: LLM API 批处理 Top-15 论文细粒度相关性判断
- **Prompt**: `src/prompts/__init__.py` → `FINE_RANK_PROMPT`
- **最终分数**: `0.4 × Coarse + 0.6 × LLM Fine`

### Step 4: 结果归纳（0-1× LLM API）

- 三层分级输出（Highly / Partially / Background）
- 约束满足矩阵
- 方法对比表（LLM API，仅 Top-5，Prompt: `COMPARISON_TABLE_PROMPT`）
- 引用关系图数据
- Markdown 报告生成

## LLM 模块说明 (`src/llm/`)

### LLMClient 特性

| 功能 | 实现 |
|------|------|
| 多 Provider | DashScope (Qwen) / DeepSeek / OpenAI，通过 config.yaml 切换 |
| 自动重试 | 网络/API 错误自动重试，指数退避（2s → 4s → 8s），最多 3 次 |
| JSON 修复 | `chat_json()` 解析失败时自动追加 JSON 格式指令重试 |
| Token 统计 | 每次调用记录 prompt/completion tokens，流水线结束时输出汇总 |
| 成本估算 | 按不同模型的真实费率（$/1M tokens）自动累计估算成本 |
| 调用明细日志 | 可选，记录每次的完整 prompt/response 到 `logs/llm_calls.log` |

### LLM 调用明细日志

在 `config.yaml` 中控制：

```yaml
llm:
  log_prompts: true               # 开启后记录每次调用的 prompt/response
  log_file: "logs/llm_calls.log"  # 单独文件，不污染主日志
  truncate_length: 2000           # 每条内容截断长度（0=不截断）
```

每行一条 JSON：时间戳、model、provider、重试次数、prompt、response。用于调试 Step 1 解析对不对、精排 prompt 长啥样。

### 流水线结束输出示例

```
Pipeline Complete — 50 queries processed
LLM calls: 98 | Retries: 3
Tokens: 152,340 | Est cost: $0.0421
```

### Token 统计 API

```python
from src.llm.token_counter import llm_stats

# 实时查看
stats = llm_stats.snapshot()
# {"total_calls": 42, "total_tokens": 68000, "estimated_cost_usd": 0.0185, ...}

# 重置（新一轮评测前）
llm_stats.reset()
```

## Prompt 管理 (`src/prompts/`)

所有 LLM Prompt 集中在一个文件中，方便检查和修改：

| Prompt 常量 | 用途 | 调用位置 |
|-------------|------|----------|
| `QUERY_UNDERSTANDING_PROMPT` | 查询理解（Step 1） | `step1_query_understanding.py` |
| `FINE_RANK_PROMPT` | 精排相关性判断（Step 3） | `step3_ranking.py` |
| `COMPARISON_TABLE_PROMPT` | 方法对比表提取（Step 4） | `step4_result_organization.py` |

每个 Prompt 配有对应的 `*_SYSTEM` 常量作为 System Prompt。

## Embedding 模型 (`src/retrieval/embedding.py`)

| 模型 | 参数量 | 维度 | 加载方式 |
|------|:-----:|:---:|------|
| Qwen3-Embedding-0.6B（主） | 0.6B | 4096 | `sentence-transformers` 加载本地路径 |
| all-MiniLM-L6-v2（轻量） | 22M | 384 | `sentence-transformers` 加载本地路径 |
| TF-IDF（降级） | — | 1024 | 纯 Python，零依赖 |

模型路径在 `config/.env` 中配置，主模型不可用时自动降级到轻量模型，sentence-transformers 未安装时降级到 TF-IDF。

## 配置参数说明

关键可调参数（`config/config.yaml`）：

| 参数 | 位置 | 说明 |
|------|------|------|
| `max_queries` | `dataset.query_filter` | 测试时限制查询数量 |
| `asta_query_types` | `dataset.query_filter` | Asta 查询类型过滤 |
| `per_source_top_k` | `retrieval` | 每个 API 源返回数量 |
| `candidate_pool_size` | `retrieval` | RRF 融合后保留数量 |
| `citation_traversal.enabled` | `retrieval` | 是否启用引文遍历 |
| `embedding.model` | `retrieval.embedding` | 主 Embedding 模型名称 |
| `embedding.model_light` | `retrieval.embedding` | 轻量降级模型名称 |
| `embedding.device` | `retrieval.embedding` | 推理设备（cpu/cuda） |
| `weights.*` | `ranking.coarse` | 六维粗排权重 |
| `top_k_after_coarse` | `ranking.coarse` | 进入 LLM 精排的论文数 |
| `batch_size` | `ranking.fine` | LLM 精排批处理大小 |
| `coarse_weight` / `llm_weight` | `ranking.fine.final_score` | 粗排/精排分数权重 |
| `relevance_layers.*` | `organization` | 三层分级阈值 |

## 中间结果缓存

运行 `use_cache: true` 时，每一步的结果会缓存到 `cache/` 目录。

调参时可以：
1. 跑完 Step 1 → 缓存查询理解结果
2. 反复调整 Step 2 的检索参数 → 直接从缓存读取 Step 1
3. 反复调整 Step 3 的排序参数 → 直接从缓存读取 Step 2

```bash
rm -rf cache/   # 清除缓存
```

## 适配两个数据集

| 数据集 | CLI 参数 | 查询格式 | 答案格式 |
|--------|----------|----------|----------|
| PaSa (RealScholarQuery) | `--dataset pasa` | `question` + `answer_arxiv_id` | arXiv ID 精确匹配 |
| Asta (PaperFindingBench) | `--dataset asta` | `input.query` + `scorer_criteria` | S2 ID + 加权 criteria |

## 注意事项

1. **API 速率**: Semantic Scholar 免费层 100 req/5min，代码内置了速率控制
2. **LLM 成本**: 每查询约 3-5 次 LLM API 调用，~6,000-8,000 tokens，流水线结束自动打印统计
3. **Embedding 模型**: 本地 Qwen3-Embedding-0.6B 路径在 `.env` 配置，不可用自动降级
4. **Prompt 修改**: 所有 Prompt 在 `src/prompts/__init__.py`，修改后无需改动业务代码
5. **日志**: 主日志输出到 `logs/pipeline.log`，LLM 调用明细（如开启）在 `logs/llm_calls.log`
