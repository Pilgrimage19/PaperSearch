# AstaBench / PaperFindingBench 数据集结构分析

> 文件路径：`../data/asta/`
> 来源：AstaBench (ICLR 2026) — PaperFindingBench 子集
> 分析日期：2026-08-05

---

## 1. 基本信息

| 属性 | 值 |
|------|-----|
| **数据集名称** | PaperFindingBench (AstaBench 子集) |
| **来源论文** | Feldman et al., "AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite," ICLR 2026 |
| **文件数** | 3 个 JSON 文件 |
| **查询总数** | **333** 条 (test=267, validation=66) |
| **查询类型** | semantic (242), specific/navigational (48), metadata (43) |
| **论文ID格式** | Semantic Scholar 数字ID（5-9位纯数字，如 `13756489`） |

---

## 2. 文件结构

```
asta/
├── normalizer_reference.json    # 535条 — Semantic查询的估计答案集大小（用于计算 estimated-recall）
├── test_2025_05.json            # 267条 — 测试集
└── validation_2025_05.json      # 66条  — 验证集
```

---

## 3. 数据格式

### 3.1 Semantic 查询（242条）

基于内容描述的论文检索，是赛题最核心的查询类型。

```json
{
  "input": {
    "query_id": "semantic_1",
    "query": "Are there any tools or studies that have focused on building a morphological analyzer specifically for handling multiple Arabic dialects?"
  },
  "scorer_criteria": {
    "known_to_be_good": ["10961392", "12160022"],
    "known_to_be_bad": [],
    "relevance_criteria": [
      {
        "name": "Morphological Analyzer",
        "description": "The paper must discuss the development, implementation, or construction of a morphological analyzer, which is a tool or method for analyzing the structure of words, including their roots, affixes, and grammatical features.",
        "weight": 0.4
      },
      {
        "name": "Arabic Dialects",
        "description": "The paper must explicitly address multiple Arabic dialects, not just Modern Standard Arabic (MSA), and should focus on the solutions related to the linguistic diversity among these dialects.",
        "weight": 0.4
      },
      {
        "name": "Integration of Morphological Analysis and Dialects",
        "description": "The paper must establish a clear connection between the morphological analysis and its application or adaptation to multiple Arabic dialects, ensuring that the tool or study is designed for (also) this purpose.",
        "weight": 0.2
      }
    ]
  }
}
```

**字段说明**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `input.query_id` | `string` | 格式 `semantic_{1..N}` |
| `input.query` | `string` | 自然语言学术查询 |
| `scorer_criteria.known_to_be_good` | `string[]` | 已知相关论文的 Semantic Scholar ID（**非穷举**，仅作校准/种子） |
| `scorer_criteria.known_to_be_bad` | `string[]` | 已知不相关论文（本数据集中**全部为空**） |
| `scorer_criteria.relevance_criteria` | `object[]` | 加权的相关性评判标准（供 LLM-Judge 使用） |
| `relevance_criteria[].name` | `string` | 标准名称 |
| `relevance_criteria[].description` | `string` | 详细的评判描述 |
| `relevance_criteria[].weight` | `float` | 权重（所有权重之和 = 1.0） |

### 3.2 Specific/Navigational 查询（48条）

查找一篇（或少数几篇）特定已知论文——"已知论文查找"。

```json
{
  "input": {
    "query_id": "specific_1",
    "query": "Find me the 'attention is all you need' paper"
  },
  "scorer_criteria": {
    "corpus_ids": ["13756489"]
  }
}
```

**特点**：`corpus_ids` 即为完整的正确答案集（精确匹配）。

**示例查询**：
| query_id | 查询 | 答案数 |
|----------|------|:--:|
| specific_1 | "Find me the 'attention is all you need' paper" | 1 |
| specific_5 | "Find me a 2020 paper by Dan Weld about extracting visual content from newspapers" | 1 |
| specific_8 | "the paper that introduced the LEGOBench dataset" | 1 |
| specific_14 | "The paper that introduced the concept of attention in neural networks" | 1 |

### 3.3 Metadata 查询（43条）

基于元数据条件的组合检索——嵌套、否定、多条件。

```json
{
  "input": {
    "query_id": "metadata_3",
    "query": "Journal articles by David Harel from 2020 to 2022"
  },
  "scorer_criteria": {
    "corpus_ids": ["..." , "..."]  // 13个ID — 完整答案集
  }
}
```

**元数据查询类别分布**：

| 约束类型 | 出现频次 | 示例 |
|----------|:------:|------|
| **作者（author）** | 30 | "papers by David Harel" |
| **年份（year）** | 27 | "from 2020 to 2022" |
| **引用（citation）** | 20 | "with at least 200 citations" |
| **发表venue** | 18 | "Journal articles", "NAACL papers" |
| **否定（negation）** | 5 | "not citing the transformers paper" |

**复杂度**：34/43（79%）的 Metadata 查询包含**2种及以上**约束类型。例如：

- `metadata_3`: author + year + venue + citation（4种）
- `metadata_12`: "Journal papers or conference papers authored by at least 3 of the authors of the 'attention is all you need' paper" — 需先解析 specific paper → 提取作者 → 过滤条件

---

## 4. 统计特征

### 4.1 核心统计

| 指标 | Semantic | Specific | Metadata | 总计 |
|------|:----:|:----:|:----:|:----:|
| 查询数 | 242 | 48 | 43 | **333** |
| Test / Val | 194/48 | 38/10 | 35/8 | 267/66 |
| 平均答案数 | — | 1.2 | 24.9 | — |
| 答案范围 | — | 1-5 | 1-172 | — |
| 已知正例总数 | 102 | — | — | — |

### 4.2 Semantic 查询统计

| 指标 | 值 |
|------|-----|
| 平均相关性标准数 | **2.9 个/查询** |
| 标准数范围 | 2-5 个 |
| 平均已知好论文 | 0.42 篇/查询 |
| 已知好论文总数 | 102 篇 |
| 已知坏论文总数 | **0**（全部为空） |
| 平均查询长度 | 18.7 词 |
| 查询长度范围 | 5-57 词 |

**解读**：`known_to_be_good` 平均仅 0.42 篇/查询（且不少查询为 0），说明这些已知正例**仅用作校准种子**，远非完整答案集。真正的评判依赖 **LLM-Judge 根据 relevance_criteria 对系统返回的论文进行评分**。

### 4.3 权重分布

Semantic 查询的相关性标准权重分布：

| 权重 | 出现次数 | 典型含义 |
|:----:|:------:|------|
| **0.40** | 228 | 最核心维度（如"必须使用XX方法"） |
| **0.30** | 209 | 次核心维度 |
| **0.20** | 116 | 辅助维度 |
| **0.50** | 82 | 对半权重（较简单的2标准查询） |
| 0.10 | 18 | 边缘维度 |
| 0.25 | 14 | — |
| 0.60-0.70 | 20 | 主导维度 |

所有权重和 **精确等于 1.0**。

### 4.4 查询长度分布

| 类型 | 平均词数 | 范围 |
|------|:--:|:--:|
| Semantic | 18.7 | 5-57 |
| Specific | 9.5 | 3-23 |
| Metadata | 12.3 | 4-28 |

### 4.5 Normalizer Reference

| 指标 | 值 |
|------|-----|
| 总条目 | 535 |
| 覆盖 Semantic 查询 | **242/242（100%）** |
| 平均估计答案集大小 | 49.4 |
| 中位数 | 32 |
| 范围 | 1 - 199 |

**用途**：在 AstaBench 评测中，Semantic 查询使用 `estimated-recall@K` 指标（而非传统 Recall），因为真实答案集未知。Normalizer 中的值 = 通过多次 PaperFinder 运行 + Union + 乘数因子估算出来的"大致相关论文总数"，用于归一化 Recall 分数。

---

## 5. Semantic 查询主题分布

基于关键词的粗略领域分布（242条 Semantic 查询）：

| 主题领域 | 出现次数 | 占比 |
|----------|:------:|:---:|
| Search/Retrieval（搜索/检索） | 55 | 23% |
| Language Models（语言模型） | 49 | 20% |
| Text Generation（文本生成） | 30 | 12% |
| Datasets/Benchmarks（数据集/基准） | 21 | 9% |
| Recommendation（推荐系统） | 18 | 7% |
| Evaluation（评测方法） | 17 | 7% |
| Agents（智能体） | 14 | 6% |
| Image/Video（图像/视频） | 13+10 | 10% |
| Translation（机器翻译） | 12 | 5% |
| Summarization（摘要） | 10 | 4% |
| Reasoning（推理） | 9 | 4% |

**与 RealScholarQuery 的领域对比**：
- RealScholarQuery：视觉/视频 32%、检索/排序 20%、NLP 16%
- PaperFindingBench：搜索/检索 23%、语言模型 20%、文本生成 12%
- **PaperFindingBench 领域更均衡**，更贴近"通用学术搜索"场景

---

## 6. 与 RealScholarQuery 的关键差异

| 维度 | RealScholarQuery (PaSa) | PaperFindingBench (AstaBench) |
|------|------------------------|------------------------------|
| **查询数** | 50 | **333** |
| **查询类型** | 仅 Semantic（自然语言） | **3种**：Semantic + Specific + Metadata |
| **答案标注** | 论文标题 + arXiv ID（较完整） | Semantic Scholar ID + **加权相关性标准** |
| **答案完整性** | 经多轮聚合，接近完整 | Semantic 仅种子（非完整），Specific/Metadata 完整 |
| **评测方式** | 传统 Precision/Recall/F1 | LLM-Judge 加权 Rubric + estimated-recall |
| **论文ID** | arXiv ID（`2309.04564`） | Semantic Scholar ID（`13756489`，纯数字） |
| **领域覆盖** | 偏 AI/ML（视觉为主） | 更广泛（检索、生成、推荐、医疗、法律等） |
| **难度特征** | 包含复杂约束和多跳推理 | Semantic 含加权多维度评判，Metadata 含嵌套+否定 |

---

## 7. 评测机制分析

### 7.1 Semantic 查询的 LLM-Judge 评分

PaperFindingBench 的核心创新：用 **LLM-Judge + 加权 Rubric** 替代人工穷举标注。

**评分流程**：
```
系统返回论文 + 证据片段
        ↓
LLM-Judge 对每篇论文按 relevance_criteria 逐项评分
        ↓
加权求和 → 0-1 相关性分数
        ↓
estimated-recall = TP / normalizer_value
nDCG = 基于相关性分数的排序质量
        ↓
最终分数 = harmonic_mean(estimated-recall, nDCG)
```

**优势**：
- 无需穷举标注（$304/query → $0）
- 可评判"部分相关"论文
- 多维度评分更细粒度

**风险**：
- LLM-Judge 的偏差（AstaBench 报告与人工评分的 Kendall tau=0.800）
- 对 LLM 自身能力边界内的论文评判可能不准确
- `known_to_be_good` 少 → 无法校准 Judge 的 Recall

### 7.2 Specific/Metadata 查询的精确评测

这两种查询有**完整答案集**（`corpus_ids`），使用传统 F1 评测。

- **Specific（导航类）**：多为 1 篇答案 → 评测系统能否精确找到目标论文
- **Metadata（元数据类）**：平均 25 篇答案 → 评测系统能否正确执行多条件组合过滤

---

## 8. 对赛题评测设计的启示

### 8.1 三种查询类型的必要性

PaperFindingBench 的三种查询类型设计是赛题的重要参考：

| 类型 | 评测能力 | 赛题对应 |
|------|----------|----------|
| **Semantic** | 语义理解 + 检索质量 | ✅ 赛题核心 — 复杂学术查询的语义检索 |
| **Specific** | 精确匹配能力 | 隐含要求 — 用户有时已知论文但需要搜索 |
| **Metadata** | 结构化查询执行 | 赛题查询理解中的"多维约束"解析 |

### 8.2 加权 Relevance Criteria 的借鉴

赛题可以借鉴 AstaBench 的多维度、加权评判方式：

- 查询 → 自动生成 relevance_criteria（如用 LLM 解析约束维度+权重）
- 评测 → 多维度打分而非二元相关/不相关
- 结果展示 → 按维度分组展示（哪些论文满足"方法约束"，哪些满足"数据约束"）

### 8.3 评测指标的选择

PaperFindingBench 对不同类型使用不同指标：

| 查询类型 | 主要指标 | 原因 |
|----------|----------|------|
| Semantic | harmonic_mean(estimated-recall, nDCG) | 答案集不完整，需估计 |
| Specific | F1 | 答案集完整且很小 |
| Metadata | F1 | 答案集完整且确定 |

**对赛题建议**：如果答案集接近完整（类似 RealScholarQuery），可用 Precision/Recall/F1；如果答案集是种子集（类似 PaperFindingBench Semantic），应用 estimated-recall + nDCG。

---

## 9. 代码示例

### 读取与分类

```python
import json

with open('test_2025_05.json', 'r') as f:
    test_data = json.load(f)
with open('validation_2025_05.json', 'r') as f:
    val_data = json.load(f)
with open('normalizer_reference.json', 'r') as f:
    normalizer = json.load(f)

all_data = test_data + val_data

# 按类型分组
semantic = [d for d in all_data if d['input']['query_id'].startswith('semantic')]
specific = [d for d in all_data if d['input']['query_id'].startswith('specific')]
metadata = [d for d in all_data if d['input']['query_id'].startswith('metadata')]
```

### Semantic 查询评测

```python
# 对每个 semantic 查询：
# 1. 系统返回论文列表 + 证据片段
# 2. LLM-Judge 根据 relevance_criteria 加权评分
# 3. 计算 estimated-recall 和 nDCG

query = semantic[0]
criteria = query['scorer_criteria']['relevance_criteria']

# LLM-Judge prompt 示例
judge_prompt = f"""
Query: {query['input']['query']}
Evaluate the following paper against these criteria:
{json.dumps(criteria, indent=2)}
Paper: {{title}} {{abstract}}
Return scores for each criterion (0-1).
"""
```

### Specific/Metadata 评测

```python
# 直接匹配 corpus_ids
def evaluate_f1(predicted_ids: set, gold_ids: set):
    tp = len(predicted_ids & gold_ids)
    precision = tp / len(predicted_ids) if predicted_ids else 0
    recall = tp / len(gold_ids)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return {'precision': precision, 'recall': recall, 'f1': f1}
```

---

*分析完成日期：2026-08-05*
*数据来源：[3] Feldman et al., "AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite," ICLR 2026.*
