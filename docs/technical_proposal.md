# 科研场景智能论文搜索系统 — 初步技术方案

> 基于赛题要求、RealScholarQuery & PaperFindingBench 数据集分析、及相关论文综述
> 版本：v0.2 | 日期：2026-08-05
> 约束：不使用本地LLM、不进行模型训练；可使用向量模型（Embedding）和 BM25

---

## 目录

1. [方案总览](#1-方案总览)
2. [系统架构](#2-系统架构)
3. [模块一：查询理解与分解](#3-模块一查询理解与分解)
4. [模块二：多策略检索执行](#4-模块二多策略检索执行)
5. [模块三：论文综合排序](#5-模块三论文综合排序)
6. [模块四：结果归纳与结构化展示](#6-模块四结果归纳与结构化展示)
7. [技术选型与成本控制](#7-技术选型与成本控制)
8. [创新点总结](#8-创新点总结)
9. [实验计划](#9-实验计划)

---

## 1. 方案总览

### 1.1 设计哲学

三条核心原则，适配新的技术约束：

1. **检索优先，LLM 辅助（Retrieval-First, LLM-Assisted）**：BM25 + 向量相似度承担 80% 的检索和粗筛工作，LLM API 仅用于"必须语义理解"的环节（查询解析、精细排序、结果归纳）
2. **训练无关（Training-Free）**：全部使用预训练 Embedding 模型和 BM25，零训练成本
3. **成本感知（Cost-Aware）**：LLM API 调用次数纳入检索规划，BM25/向量计算零 API 成本 → 大量使用

### 1.2 核心技术路线

```
BM25（稀疏检索） + Embedding 相似度（稠密检索） → 主力检索+粗筛
        ↓
LLM API（Qwen3-32B / DeepSeek-V3） → 仅在查询理解、精排、归纳时调用
```

### 1.3 方案定位

| 维度 | 目标 |
|------|------|
| F1 Score | > 0.35（目标超 SPAR 的 0.30-0.38） |
| 运行效率 | LLM API 调用 < 10 次/查询，Token < 15,000/查询 |
| 结果结构化 | 约束满足矩阵 + 方法对比表 + 引用关系图 |
| 创新性 | Constraint Tree + 混合检索融合 + PageRank 引文遍历 |

---

## 2. 系统架构

```
                         用户自然语言查询
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                MODULE 1: 查询理解与分解 (1× LLM API)           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ 意图分类  │  │ 约束提取  │  │ 查询路由  │  │ 子查询生成   │  │
│  │          │  │ (六维)   │  │(semantic/ │  │ (多角度扩展) │  │
│  │          │  │          │  │specific/  │  │              │  │
│  │          │  │          │  │metadata)  │  │              │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘  │
│                     1 次 LLM API 调用                          │
└──────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│             MODULE 2: 混合检索执行 (0 LLM, 纯检索)             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ 源选择    │  │ BM25     │  │ Dense    │  │ 融合排序     │  │
│  │(arXiv/   │  │ 稀疏检索  │  │ Embedding│  │(CombSUM/    │  │
│  │ S2/      │  │(关键词)  │  │ 稠密检索  │  │ RRF)        │  │
│  │ OpenAlex/│  │          │  │          │  │              │  │
│  │ PubMed/  │  │          │  │          │  │              │  │
│  │ DBLP)   │  │          │  │          │  │              │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  PageRank-引导的引文遍历 (Semantic Scholar Citations API)  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│                  ← 必要时迭代 1-2 轮 →                           │
└──────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│               MODULE 3: 论文排序 (1-2× LLM API)                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ 粗筛     │  │ 精排     │  │ 多维度   │  │ 分层输出     │  │
│  │ BM25+   │  │ LLM API  │  │ 融合     │  │ Highly/      │  │
│  │ Embedding│  │ 细粒度   │  │(相关/权威│  │ Partially    │  │
│  │ 自动评分 │  │ 相关性   │  │ /时效/   │  │ Relevant     │  │
│  │          │  │ 判断     │  │ 多样)    │  │              │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘  │
└──────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│              MODULE 4: 结果归纳 (1-2× LLM API)                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ 主题聚类  │  │ 约束满足  │  │ 方法对比  │  │ 引用关系图   │  │
│  │(Embedding│  │ 矩阵     │  │ 表生成   │  │              │  │
│  │ +KMeans) │  │          │  │          │  │              │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

**LLM API 调用预算**：
- Module 1（查询理解）：**1 次**
- Module 3（精排）：**1-2 次**
- Module 4（结果归纳）：**1-2 次**
- **总计：3-5 次 LLM API 调用/查询，约 10,000-15,000 tokens**

---

## 3. 模块一：查询理解与分解

### 3.1 单次 LLM API 完成全部查询理解

将意图分类、约束提取、查询路由、子查询生成合并为**一次 LLM API 调用**，使用结构化 JSON 输出：

```python
QUERY_UNDERSTANDING_PROMPT = """
You are an academic search expert. Analyze the following research query and output a structured JSON.

Query: {user_query}

Output JSON with these fields:
1. "query_type": one of "semantic" / "specific" / "metadata"
2. "intent": one of "find" / "survey" / "compare"
3. "domain": research domain (e.g., "NLP", "CV", "RL", "Biomedical", etc.)
4. "constraints": array of constraint objects, each with:
   - "dimension": "method" / "task" / "data" / "time" / "venue" / "performance"
   - "value": extracted constraint value
   - "necessity": "must" or "should"
5. "excluded": things to exclude (e.g., ["survey papers"])
6. "sub_queries": array of 3-5 keyword search queries AND 1-2 natural language queries
   - Each keyword query: 3-7 keywords optimized for BM25/search API
   - Each NL query: natural language for embedding similarity search
7. "time_range": {{"start": "YYYY", "end": "YYYY"}} or null
8. "preferred_sources": suggested academic sources

Return ONLY valid JSON, no other text.
"""
```

**设计要点**：

- **一次调用完成全部**：避免多次 LLM 调用的延迟和成本叠加
- **同时生成关键词查询和自然语言查询**：关键词查询 → BM25/Search API；NL 查询 → Embedding 相似度
- **constraints 结构化为后续"约束满足矩阵"提供数据基础**

### 3.2 查询类型路由的后处理

```
LLM 输出 query_type
    │
    ├── "specific" → 精确匹配管道
    │   └── 直接调用 Semantic Scholar title search API
    │
    ├── "metadata" → 结构化转换管道
    │   └── 将 constraints 转换为 OpenAlex API 过滤参数
    │
    └── "semantic" → 完整混合检索管道（默认）
        └── Module 2
```

### 3.3 Constraint Tree 构建（无 LLM 的纯规则方法）

LLM 输出的 constraints 列表可能扁平化，后处理构造树结构：

```python
def build_constraint_tree(constraints, sub_queries):
    """纯规则方法，不需要额外 LLM 调用"""
    tree = {"root": query, "children": []}
    
    for c in constraints:
        if c["necessity"] == "must":
            # must 约束 → 必须覆盖的一级节点
            node = {"constraint": c, "sub_queries": []}
            for sq in sub_queries["keyword"]:
                if constraint_match(sq, c):  # 简单关键词匹配
                    node["sub_queries"].append(sq)
            tree["children"].append(node)
    
    return tree
```

匹配规则基于关键词重叠，不消耗 API 调用。

---

## 4. 模块二：多策略检索执行

### 4.1 多源检索后端

采用 **3 主力 + 2 领域** 免费 API 混合检索，根据查询类型和领域自适应路由：

| 数据源 | 角色 | 优势 | 覆盖范围 | API |
|--------|------|------|----------|-----|
| **Semantic Scholar** | **主力** — 语义搜索 + 引文遍历 | 2亿+论文、引文数据、语义搜索端点 | 全学科 | 免费 REST API，100 req/5min |
| **arXiv** | **主力** — 最新预印本 + 全文 | AI/CS/物理/数学最新论文、全文可解析 | CS, Math, Physics, Stat | 免费 API，速率友好 |
| **OpenAlex** | **主力** — 元数据查询 | 完全开放、作者/机构/venue/引用计数最完整 | 全学科，2.5亿+实体 | 免费 REST API，无速率限制 |
| **PubMed** | 领域增强 — 生物医学 | 生物医学权威、MeSH 术语体系 | 生物医学 | 免费 Entrez API |
| **DBLP** | 领域增强 — CS 会议/期刊 | CS 领域 venue 信息最精确 | CS | 免费 API |

> **设计决策：不使用 Google Scholar**。Google Scholar 无官方 API，第三方爬虫方案（scholarly、SerpAPI 等）存在 IP 封禁风险和合规问题。Semantic Scholar + arXiv 已覆盖 Google Scholar 的核心功能，且提供更好的结构化元数据和引文数据。PaSa 论文中实际使用的是 **Google Search API (Serper.dev) + site:arxiv.org**，等价于直接使用 arXiv API。

#### 源选择路由规则

```python
def route_sources(query_type, domain, intent, constraints):
    sources = []

    # 1. 默认全部使用 Semantic Scholar（通用检索 + 引文数据）
    sources.append("semantic_scholar")

    # 2. AI/CS/数学/物理 → 加 arXiv（最新预印本 + 全文）
    if domain in ["CS", "AI", "ML", "NLP", "CV", "RL", "math", "physics"]:
        sources.append("arxiv")

    # 3. 元数据查询 → OpenAlex 优先（作者/机构/venue 结构化查询）
    if query_type == "metadata":
        sources.insert(0, "openalex")
    elif has_author_or_venue_constraint(constraints):
        sources.append("openalex")

    # 4. 领域路由
    if domain in ["biomedical", "medical", "biology", "chemistry"]:
        sources.append("pubmed")
    if domain in ["CS", "AI", "ML"] and intent in ["survey", "compare"]:
        sources.append("dblp")      # 精确 venue 信息

    # 5. 意图路由
    if intent == "survey":
        # 综述类 → 多源并行，最大化覆盖面
        sources = list(set(sources + ["semantic_scholar", "arxiv"]))
    if intent == "compare":
        # 对比类 → 需要多样性，启用 OpenAlex 补充
        sources = list(set(sources + ["openalex"]))

    return list(dict.fromkeys(sources))
```

#### 各源检索策略

| 源 | 查询方式 | 返回字段 | 成本 |
|------|----------|----------|:--:|
| **Semantic Scholar** | `/paper/search` keyword + `/paper/search` bulk 语义搜索 | paperId, title, abstract, citationCount, year, venue, authors | 免费 |
| **arXiv** | API 关键词搜索 + 分类过滤（cs.AI, cs.CL 等） | arxiv_id, title, abstract, authors, categories, published | 免费 |
| **OpenAlex** | `/works` REST API + 结构化 filter 参数 | id, title, abstract, authorships, cited_by_count, primary_location | 免费 |
| **PubMed** | Entrez `esearch` + `efetch`（MeSH 术语扩展） | title, abstract, authors, pub_date, mesh_terms | 免费 |
| **DBLP** | `/search/publ` API + venue 精确匹配 | title, authors, venue, year | 免费 |

#### 各源结果去重与合并

```python
def merge_cross_source(results_by_source):
    """跨源去重合并 - 基于 arXiv ID + DOI + title 归一化"""
    merged = {}
    for source, papers in results_by_source.items():
        for paper in papers:
            key = paper.get("arxiv_id") or paper.get("doi") or normalize_title(paper["title"])
            if key not in merged:
                merged[key] = {**paper, "sources": [source]}
            else:
                merged[key]["sources"].append(source)
                # 合并多源元数据（取最完整的）
                merged[key] = enrich_metadata(merged[key], paper)
    return list(merged.values())
```

### 4.2 混合检索架构（核心引擎，零 LLM 调用）

这是整个系统的主力模块——**完全基于 BM25 + Embedding 相似度，不调用任何 LLM**。

```
子查询列表
    │
    ├── 关键词子查询 (3-5个) → BM25 检索
    │   └── 调用 Semantic Scholar API (keyword search endpoint)
    │   └── 本地对返回结果做 BM25 重排序
    │
    └── NL子查询 (1-2个) → Dense Embedding 检索
        └── 将查询编码为向量
        └── 与本地论文向量索引做余弦相似度匹配
        └── 或：调用 Semantic Scholar 的语义搜索 API
```

**检索融合**：

两路结果通过 **RRF（Reciprocal Rank Fusion）** 合并：

```python
def rrf_fusion(bm25_results, embedding_results, k=60):
    """Reciprocal Rank Fusion - 零成本融合"""
    scores = {}
    for rank, paper in enumerate(bm25_results):
        scores[paper.id] = scores.get(paper.id, 0) + 1.0 / (k + rank + 1)
    for rank, paper in enumerate(embedding_results):
        scores[paper.id] = scores.get(paper.id, 0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])
```

**为什么 RRF 优于 CombSUM**：
- 不需要对分数做归一化（BM25 和 Cosine Similarity 的分数尺度不同）
- 无超参数、无训练、零计算成本
- 在 TREC 等检索评测中验证有效

### 4.3 向量索引构建（离线预处理）

对可获取全文的论文，预建两套索引：

| 索引 | 内容 | 编码模型 | 用途 |
|------|------|----------|------|
| **标题+摘要索引** | 论文 title + abstract | Qwen3-Embedding-0.6B（主）/ all-MiniLM-L6-v2（轻量） | Dense 检索主索引 |
| **全文 Chunk 索引** | 论文全文按 1500 字符分块 | 同上 | 精细检索（可选，对大语料可跳过） |

**Embedding 模型选择**：

| 模型 | 参数量 | 维度 | 说明 |
|------|:-----:|:---:|------|
| **Qwen3-Embedding-0.6B**（主） | 0.6B | 4096 | 轻量高性能，本地运行 |
| **all-MiniLM-L6-v2**（轻量） | 22M | 384 | 极小模型，快速推理，CPU 友好 |

> 两个模型均为本地部署路径，在 `config/.env` 中配置。不依赖任何 API 或商业服务。

### 4.4 检索执行流程（三阶段，全无 LLM）

```
Phase 1: BROAD RECALL（零 LLM）
├── 每个关键词子查询 → Semantic Scholar keyword search API
│   └── 每子查询返回 top-50，去重合并
├── 每个 NL 子查询 → Embedding 向量相似度 top-50
│   └── 对论文标题+摘要向量索引做余弦相似度
├── RRF 融合两路结果 → Candidate Pool (100-200 篇)
├── 成本：5-10 次 Semantic Scholar API（免费）
└── 目标：Recall > 0.80

Phase 2: SMART FILTER（零 LLM）
├── 对 Candidate Pool 中每篇论文计算多维分数：
│   ├── BM25 Score（查询 vs 标题+摘要）
│   ├── Embedding Cosine Similarity（查询 vs 标题+摘要）
│   ├── Citation Count（归一化到 0-1）
│   └── Recency Score（年份越新越高，指数衰减）
├── 加权融合 → 排序 → 保留 Top-30 → Filtered Pool
├── 成本：0（纯本地计算）
└── 目标：Precision > 0.25, Recall > 0.65

Phase 3: CITATION EXPANSION（零 LLM）
├── 对 Filtered Pool Top-10 构建局部引用图
│   └── Semantic Scholar Citations API 获取引用/被引
├── 计算局部 PageRank → 选 Top-5 高 PR 节点
├── 对每个高 PR 节点，获取其参考文献列表
├── 新增论文去重 → 追加到 Filtered Pool
├── 成本：10-15 次 Citations API（免费）
└── 目标：额外 Recall +0.05~0.10
```

### 4.5 PageRank 引导的引文遍历（创新点）

区别于 SPAR（全遍历）和 PaSa（RL 决定），使用轻量级 PageRank：

```python
def pagerank_guided_traversal(paper_pool, citations_api, top_k=5, max_expand=10):
    """PageRank引导的引文遍历 - 零LLM成本"""
    # 1. 构建局部引用图
    G = nx.DiGraph()
    for paper in paper_pool[:30]:  # Top-30
        G.add_node(paper.id)
        # 获取引用关系（Semantic Scholar API）
        refs = citations_api.get_references(paper.id)
        cites = citations_api.get_citations(paper.id)
        for ref in refs:
            G.add_edge(paper.id, ref.id)
        for cite in cites:
            G.add_edge(cite.id, paper.id)
    
    # 2. 计算 PageRank
    pr = nx.pagerank(G, alpha=0.85)
    
    # 3. 选择 PageRank 最高的 top_k 篇论文
    top_nodes = sorted(pr.items(), key=lambda x: -x[1])[:top_k]
    
    # 4. 遍历高 PR 节点的共同引用
    ref_counter = Counter()
    for node_id, _ in top_nodes:
        refs = citations_api.get_references(node_id)
        for ref in refs:
            ref_counter[ref.id] += 1
    
    # 5. 被多篇 top paper 共同引用的论文优先
    new_papers = [ref_id for ref_id, count in ref_counter.most_common(max_expand)]
    
    return new_papers
```

相比 SPAR 全遍历的优势：
- 只遍历引用图中的**枢纽节点**，而非所有节点
- 共同引用优先 → 高信号论文
- API 调用量降低 ~40%

### 4.6 Metadata 查询的特殊处理（零 LLM）

将 Module 1 提取的 constraints 直接转换为 OpenAlex API 过滤参数：

```python
def metadata_to_openalex_params(constraints):
    """纯规则转换，零 LLM"""
    params = {}
    for c in constraints:
        if c["dimension"] == "author":
            params["filter"] = f'authorships.author.display_name:"{c["value"]}"'
        elif c["dimension"] == "time":
            params["filter"] += f',publication_year:{c["value"]}'
        elif c["dimension"] == "venue":
            params["filter"] += f',primary_location.source.display_name:"{c["value"]}"'
    return params
# 直接调用 OpenAlex REST API → 返回精确匹配结果
```

---

## 5. 模块三：论文综合排序

### 5.1 粗筛阶段（零 LLM）

对 Filtered Pool（来自 Module 2 Phase 2，约 30 篇候选）计算**六维分数**，全部基于可计算指标：

| 维度 | 权重 | 计算方式 | LLM 需求 |
|------|:--:|----------|:--:|
| **BM25 Relevance** | 0.25 | 查询 vs 标题+摘要 BM25 分数（归一化） | ❌ |
| **Embedding Similarity** | 0.25 | 查询向量 vs 标题+摘要向量的余弦相似度 | ❌ |
| **Citation Count** | 0.15 | log(1 + citationCount) / log(1 + maxCitations) | ❌ |
| **Recency** | 0.15 | exp(-(current_year - pub_year) / 3) | ❌ |
| **Venue Prestige** | 0.10 | 顶会/顶刊白名单匹配 → 二元特征 | ❌ |
| **Constraint Coverage** | 0.10 | 论文标题+摘要中覆盖了多少个 must 约束维度（关键词匹配） | ❌ |

粗筛结果 = 加权和 → 保留 Top-15 进入精排。

### 5.2 精排阶段（1-2 次 LLM API）

仅对 Top-15 论文调用 LLM API 做细粒度相关性判断。借鉴 PaperQA2 的 RCS 思路，但针对论文级别（非 Chunk 级别）以降低成本：

```python
RERANK_PROMPT = """
You are an academic paper reviewer. Given a research query and a paper, evaluate its relevance.

Query: {query}
Query Constraints: {constraints_json}

Paper:
  Title: {title}
  Abstract: {abstract}
  Year: {year}
  Venue: {venue}
  Citations: {citations}

For each constraint, rate how well the paper satisfies it (0-1):
{constraint_items}

Then give an overall relevance score (0-100) and a one-sentence evidence summary.

Return JSON:
{{
  "constraint_scores": {{"constraint_name": score, ...}},
  "overall_score": 85,
  "evidence": "This paper proposes a BERT-based method for Chinese sentiment analysis on Weibo data, achieving F1=0.87."
}}
"""

# 批处理：一次 API 调用处理 5 篇论文（减少调用次数）
# 共 15/5 = 3 次 API 调用，或合并为 1-2 次（增大 batch）
```

**精排后的最终分数**：

```
FinalScore = 0.4 × 粗筛分数 + 0.6 × LLM精排分数
```

### 5.3 分层输出

根据最终分数将论文分为三层：

| 层级 | 分数范围 | 标签 |
|------|:------:|------|
| **Highly Relevant** | > 70 | 满足所有 must 约束 |
| **Partially Relevant** | 40 - 70 | 满足部分 must 或仅 should 约束 |
| **Background** | 20 - 40 | 领域相关但不直接匹配 |

**关键**：分层依据来自精排 LLM 返回的 `constraint_scores`，可直接映射到约束满足矩阵（Module 4）。

---

## 6. 模块四：结果归纳与结构化展示

### 6.1 四项输出（全部基于 Embedding + 1-2 次 LLM API）

#### 输出 1：分层论文列表 + 证据摘要

利用精排 LLM 返回的 `evidence` 字段，无需额外 API 调用。

#### 输出 2：约束满足矩阵

利用精排 LLM 返回的 `constraint_scores`，零额外成本：

```
              Method    Task     Data     Time     Perf
              (must)   (must)   (must)   (should) (should)
Paper 1 [87]   0.95     0.90     0.85     1.00     0.70
Paper 2 [79]   0.90     0.85     0.20     0.95     0.90
Paper 3 [72]   0.60     0.80     0.90     1.00     0.30
...
```

#### 输出 3：方法对比表（1 次 LLM API）

对 Top-5 论文生成对比表：

```python
COMPARISON_PROMPT = """
Given the following 5 papers and their abstracts, extract structured information for comparison.

Papers:
{paper_list_json}

Extract for each paper:
- model_base: the main model/architecture used
- dataset: evaluation dataset(s)
- key_metric: primary performance metric and value
- method_category: type of approach (e.g., "fine-tuning", "prompt-tuning", "RL")
- key_innovation: one-sentence novel contribution

Return as JSON array.
"""
# 1 次 API 调用 → 结构化对比表
```

#### 输出 4：引用关系图

利用 Module 2 Phase 3 已获取的引用数据，生成 Mermaid 格式——**零额外 API 调用**：

```python
def build_citation_graph(top_papers, citations_data):
    """纯规则生成 Mermaid 图 - 零 LLM"""
    lines = ["```mermaid", "graph TD"]
    for i, paper in enumerate(top_papers):
        lines.append(f'    P{i}["[{i+1}] {paper["title"][:50]}..."]')
    for i, paper in enumerate(top_papers):
        for cited_id in citations_data.get(paper["id"], []):
            if cited_id in paper_id_to_idx:
                j = paper_id_to_idx[cited_id]
                lines.append(f"    P{i} --> P{j}")
    lines.append("```")
    return "\n".join(lines)
```

### 6.2 主题聚类（纯 Embedding，零 LLM）

```python
from sklearn.cluster import KMeans
import numpy as np

def cluster_papers(paper_embeddings, n_clusters=None):
    """基于 Embedding 的论文聚类 - 零 LLM"""
    if n_clusters is None:
        n_clusters = min(5, len(paper_embeddings) // 3)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(np.array(paper_embeddings))
    
    # 每簇用 TF-IDF 提取关键词作为主题标签（无需 LLM）
    cluster_keywords = {}
    for label in range(n_clusters):
        cluster_texts = [p["title"] + " " + p["abstract"] 
                        for i, p in enumerate(papers) if labels[i] == label]
        # TF-IDF 提取 top-3 关键词
        keywords = extract_top_tfidf_keywords(cluster_texts, top_k=3)
        cluster_keywords[label] = keywords
    
    return labels, cluster_keywords
```

---

## 7. 技术选型与成本控制

### 7.1 技术栈

| 组件 | 选型 | 类型 | 理由 |
|------|------|------|------|
| **主 LLM API** | **Qwen3-32B** 或 **DeepSeek-V3** | LLM API | 开源模型 API，成本低、效果好 |
| **Embedding 模型** | **Qwen3-Embedding-0.6B** + **all-MiniLM-L6-v2** | 向量模型 ✅ | 本地部署，路径在 .env 配置 |
| **BM25 引擎** | **rank-bm25** (Python) | 算法库 | 纯 Python 实现，零依赖 |
| **检索 API** | **Semantic Scholar** + **arXiv** + **OpenAlex** + **PubMed** + **DBLP** | 免费学术 API | 五源互补，覆盖全学科 |
| **向量索引** | **FAISS** (Facebook) | 向量索引库 | 高效余弦相似度搜索 |
| **后端** | Python + FastAPI | — | 快速原型 |

### 7.2 LLM API 调用预算（单次查询）

| 阶段 | 调用次数 | Token 估算 | 说明 |
|------|:------:|:---------:|------|
| Module 1: 查询理解 | **1** | ~800 | 结构化 JSON 输出 |
| Module 2: 混合检索 | **0** | 0 | 纯 BM25 + Embedding + API |
| Module 3: 精排 | **1-2** | ~3,000 | 批处理 15 篇论文 |
| Module 4: 对比表生成 | **1** | ~2,000 | Top-5 结构化提取 |
| **总计** | **3-4** | **~6,000** | |

**成本估算**（以 Qwen3-32B API 计，约 ¥0.001/1K tokens）：
- 每次查询 LLM 成本：**~¥0.006**
- Semantic Scholar / arXiv / OpenAlex / PubMed / DBLP API：**均免费，无需爬虫**
- OpenAlex API：**免费**
- **总成本/查询：< ¥0.01**

对比：
- PaperQA2：$1-3/query（约 ¥7-21）
- 本方案：**低 1000 倍以上**

### 7.3 成本控制策略

1. **BM25 + Embedding 承担 80% 工作量**：检索和粗筛零 LLM 成本
2. **批处理 LLM 调用**：精排阶段一次 API 调用处理多篇论文
3. **结果缓存**：相同/高度相似查询（Embedding 相似度 > 0.95）复用缓存结果
4. **Early-Stop**：候选池过小时跳过精排或引文遍历
5. **Semantic Scholar API 批量化**：合并多个查询为一个 API 请求（如支持）
6. **向量索引预计算**：对高频领域论文预建 FAISS 索引

---

## 8. 创新点总结

### 8.1 方法论创新

| 创新点 | 说明 | 对比现有 |
|--------|------|----------|
| **BM25+Embedding 双路混合检索** | 稀疏+稠密检索互补，RRF 融合 | PaSa 仅关键词搜索；SPAR 仅 LLM 提取关键词 |
| **Constraint Tree + 六维约束** | 结构化的查询约束提取，驱动后续检索和展示 | 所有现有系统均无显式约束建模 |
| **PageRank 引文遍历** | 局部引用图 PageRank 引导选择性遍历 | SPAR 全遍历；PaSa RL 控制；PaperQA2 Overlap 过滤 |
| **全检索管道零/低 LLM** | 检索和粗筛完全基于 BM25+Embedding | PaperQA2 依赖大量 LLM（$1-3/query） |

### 8.2 工程创新

| 创新点 | 说明 |
|--------|------|
| **RRF 融合** | 无需归一化的 BM25 和 Embedding 分数融合 |
| **单次 LLM 完成查询理解** | 意图+约束+子查询，一次 Structed Output |
| **三类型查询路由** | Semantic / Specific / Metadata 统一入口，不同管道 |
| **Embedding 聚类 + TF-IDF 主题标签** | 论文聚类零 LLM 成本 |

### 8.3 展示创新

| 创新点 | 说明 |
|--------|------|
| **约束满足矩阵** | 论文 × 约束维度可视化，与精排 LLM 的 constraint_scores 天然对接 |
| **自动方法对比表** | 结构化提取方法/数据集/指标 |
| **分层输出** | 三层分级（Highly/Partially/Background） |

---

## 9. 实验计划

### 9.1 评测数据集

| 数据集 | 查询数 | 用途 |
|--------|:----:|------|
| RealScholarQuery | 50 | 主评测集（F1/Recall/Precision） |
| PaperFindingBench Val | 66 | 开发调优（三种查询类型） |
| SPARBench | 50 | 跨领域泛化评测 |

### 9.2 消融实验

| 实验 | 变量 | 验证目标 |
|------|------|----------|
| Full System | 全部开启 | 基线 |
| – Embedding Retrieval | 仅 BM25 | Embedding 检索的边际贡献 |
| – BM25 Retrieval | 仅 Embedding | BM25 的边际贡献 |
| – Citation Traversal | 无引文遍历 | PageRank 遍历的 Recall 增益 |
| – LLM Rerank | 仅 BM25+Embedding 排序 | LLM 精排的必要性 |
| – RRF | 用 CombSUM 替代 | RRF 融合的有效性 |

### 9.3 性能目标

| 指标 | 目标值 | 对比 |
|------|:----:|------|
| F1 (RealScholarQuery) | **> 0.35** | SPAR: 0.31, PaSa: 0.28 |
| Recall@50 | **> 0.55** | PaSa-7B: 0.66, Google+GPT4o: 0.26 |
| Precision | **> 0.30** | SPAR: 0.29, PaSa: 0.51 |
| LLM API 调用/查询 | **< 5** | SPAR: ~8-12 |
| Token 消耗/查询 | **< 8,000** | PaperQA2: ~100,000+ |

### 9.4 迭代路线

```
Week 1-2: BM25 + Embedding 检索管道搭建
    → Semantic Scholar + OpenAlex API 对接
    → FAISS 向量索引构建
    → RealScholarQuery Recall 基线

Week 3:   查询理解模块 + Constraint Tree
    → LLM API Structed Output prompt 调试
    → 三类型查询路由

Week 4:   RRF 融合 + PageRank 引文遍历
    → 混合检索参数调优
    → 消融实验

Week 5:   精排 + 结果归纳
    → 约束满足矩阵生成
    → 方法对比表 + 引用图

Week 6:   全系统调优 + 文档
    → 最终评测 + 提交
```

---

## 附录 A：LLM API 调用分布图

```
查询理解 ████ 1次 (800 tokens)
检索执行 ░░░░ 0次 (纯BM25+Embedding)
精排     ██ 1-2次 (3,000 tokens)
结果归纳 ██ 1次 (2,000 tokens)
─────────────────────────────
合计     ████████ 3-4次 (~6,000 tokens, <¥0.01)
```

### LLM 调用基础设施 (`src/llm/`)

所有 LLM 调用通过统一的 `LLMClient` 管理，提供：

| 能力 | 说明 |
|------|------|
| 多 Provider | DashScope (Qwen) / DeepSeek / OpenAI，config.yaml 一键切换 |
| 自动重试 | 指数退避（2s→4s→8s），最多 3 次 |
| JSON 修复 | `chat_json()` 解析失败时自动追加格式指令重试 |
| Token 统计 | 每次调用记录 prompt/completion tokens，流水线结束输出汇总 |
| 成本估算 | 按各模型真实费率（$/1M tokens）自动累计 |

### Prompt 管理 (`src/prompts/`)

所有 LLM Prompt 集中维护在 `src/prompts/__init__.py`：
- `QUERY_UNDERSTANDING_PROMPT` + `_SYSTEM` — Step 1 查询理解
- `FINE_RANK_PROMPT` + `_SYSTEM` — Step 3 精排相关性判断
- `COMPARISON_TABLE_PROMPT` + `_SYSTEM` — Step 4 方法对比表提取

### Embedding 模型

| 模型 | 参数量 | 维度 | 加载方式 |
|------|:-----:|:---:|------|
| **Qwen3-Embedding-0.6B**（主） | 0.6B | 4096 | `sentence-transformers` 加载本地路径 |
| **all-MiniLM-L6-v2**（轻量降级） | 22M | 384 | `sentence-transformers` 加载本地路径 |
| TF-IDF（最终降级） | — | 1024 | 纯 Python，零依赖 |

模型路径在 `config/.env` 中配置，自动降级链：Qwen-0.6B → MiniLM → TF-IDF。

## 附录 B：与现有系统的详细对比

| 能力 | PaSa | SPAR | PaperQA2 | 本方案 |
|------|:---:|:---:|:---:|:---:|
| 训练需求 | RL·8×H100 | 无 | 无 | **无** ✅ |
| 本地模型 | Qwen2.5-7B | 无(API) | GPT-4 API | **无(纯API+向量)** ✅ |
| 检索主力 | Google搜索 | LLM提取关键词+API | LLM生成关键词+API | **BM25 + Embedding** |
| 检索成本 | 高(Google API) | 中(API×N) | 极高($1-3/q) | **极低(<¥0.01/q)** |
| 引文遍历 | 多层RL | 一层全遍历 | 一层Overlap | **PageRank引导** |
| 查询约束 | 隐式 | 三意图分类 | 隐式 | **六维约束树** |
| Metadata查询 | ❌ | ❌ | ❌ | **✅ 结构化转换** |
| 结果归纳 | ❌ | ❌ | ❌ | **✅ 矩阵+表+图** |
| Embedding模型 | ❌ | ❌ | ✅(闭源) | **✅(Qwen-0.6B+MiniLM本地)** |
| BM25 | ❌ | ❌ | ❌ | **✅** |
| LLM Token统计 | ❌ | ❌ | ❌ | **✅(自动累计+成本估算)** |
| Prompt管理 | 内嵌代码 | 内嵌代码 | 内嵌代码 | **✅(集中独立文件)** |

---

*方案版本：v0.3 | 最后更新：2026-08-06*
*技术约束：不使用本地LLM、不训练模型；可使用向量模型和BM25*
