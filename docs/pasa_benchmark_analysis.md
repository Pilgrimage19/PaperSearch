# PaSa / RealScholarQuery 数据集结构分析

> 文件路径：`../data/pasa/test.jsonl`
> 数据集名称：RealScholarQuery（PaSa 论文评测集）
> 分析日期：2026-08-05

---

## 1. 基本信息

| 属性 | 值 |
|------|-----|
| **数据集名称** | RealScholarQuery（来自 PaSa 论文 [1]） |
| **文件格式** | JSONL（每行一个 JSON 对象） |
| **查询总数** | **50 条** |
| **答案论文总数** | **791 篇** |
| **唯一论文数** | 779 篇（98.5% 唯一率） |
| **查询时间戳** | 统一为 `20241001` |
| **查询 ID 范围** | `RealScholarQuery_0` ~ `RealScholarQuery_49` |

---

## 2. 字段结构

每条记录为一行 JSON，包含以下字段：

```json
{
  "question": "自然语言学术查询字符串",
  "answer": ["论文标题1", "论文标题2", ...],
  "answer_arxiv_id": ["arxiv_id_1", "arxiv_id_2", ...],
  "source_meta": {
    "published_time": "20241001"
  },
  "qid": "RealScholarQuery_N"
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `question` | `string` | 自然语言描述的学术搜索查询，模拟真实研究者提问方式 |
| `answer` | `string[]` | 与查询相关的论文标题列表，由领域专家标注 |
| `answer_arxiv_id` | `string[]` | 对应论文的 arXiv ID（无版本后缀，如 `2309.04564`） |
| `source_meta.published_time` | `string` | 查询时间戳，所有查询均为 `20241001`，用于限制检索时间范围 |
| `qid` | `string` | 查询唯一标识符，格式 `RealScholarQuery_{0..49}` |

**约束**：`answer` 和 `answer_arxiv_id` 数组长度始终相等且一一对应。

---

## 3. 统计特征

### 3.1 答案数量分布

| 统计量 | 值 |
|--------|-----|
| 总数 | 791 |
| 平均 | **15.8 篇/查询** |
| 中位数 | **10 篇/查询** |
| 最小值 | 1 篇 |
| 最大值 | **65 篇** |
| P50 | 9 篇 |
| P90 | 37 篇 |

**分布详情**：

| 答案数区间 | 查询数 | 占比 |
|:--------:|:----:|:---:|
| 1-5 篇 | 13 | 26% |
| 6-10 篇 | 13 | 26% |
| 11-20 篇 | 9 | 18% |
| 21-50 篇 | 13 | 26% |
| 51+ 篇 | 2 | 4% |

**解读**：答案数量呈**长尾分布**——多数查询（52%）在 1-10 篇答案，但存在少数极宽泛查询（如"所有关于视频生成可控性的论文"有 58 篇答案，"RL训练LLM Agent"有 44 篇答案）。

### 3.2 查询长度分布

| 统计量 | 值 |
|--------|-----|
| 平均词数 | 15.6 |
| 最短 | 6 词 |
| 最长 | 38 词 |

### 3.3 论文重复情况

- 791 篇答案中，779 篇唯一（**98.5% 唯一率**）
- **12 篇论文出现在多个查询中**（跨主题交叉），例如：
  - `VideoPoet: A Large Language Model for Zero-Shot Video Generation` — 同时满足视频生成和视频-音频多模态查询
  - `Genie: Generative Interactive Environments` — 同时满足视频生成和视频可控性查询
  - `Math-shepherd`、`Reward Design with Language Models` — 同时满足 RL训练和 Reward Shaping 查询

**解读**：极低的跨查询重叠率（1.5%）说明各查询答案集之间具有**良好的区分度**，标注质量高。

---

## 4. 查询分类分析

### 4.1 按领域分布

| 领域 | 查询数 | 占比 | 示例 |
|------|:----:|:---:|------|
| **视觉/视频/多模态** | 16 | 32% | 视频生成、3D场景理解、身份保持生成、可控视频生成 |
| **检索/排序/QA/综述** | 10 | 20% | LLM排序、LLM写综述、HotPotQA评测、文献搜索 |
| **NLP/信息抽取/机器翻译** | 8 | 16% | 事件抽取、MT对抗样本、MT Agent、文本水印 |
| **LLM/Agent** | 5 | 10% | Agent任务规划、LLM Agent评测、Agent调度 |
| **RL/对齐** | 2 | 4% | RLHF幻觉、RL负面影响的证据 |
| **代码/数学** | 1 | 2% | 中等难度代码评测数据集 |
| **金融** | 1 | 2% | LLM挖掘股票因子 |
| **AI4Science** | 2 | 4% | 量子蒙特卡洛、蛋白质/抗体设计 |
| **机器人** | 2 | 4% | 机器人决策与任务规划基准 |
| **其他** | 3 | 6% | 密码学隐私学习、合成数据、Test-Time Training |

### 4.2 按查询意图分类

| 意图类型 | 数量 | 占比 |
|----------|:----:|:---:|
| **Find（查找类）** — "Give me papers...", "Show me research...", "List all papers..." | 42 | 84% |
| **Survey（综述类）** — 要求自动写综述或总结多篇论文 | 3 | 6% |
| **Compare（对比类）** — "better than...", "vs..." | 2 | 4% |
| **What/Definition（定义类）** | 2 | 4% |
| **How（方法类）** | 1 | 2% |

### 4.3 按约束维度分类

| 约束类型 | 查询数 | 示例 |
|----------|:----:|------|
| **方法约束**（using/based on/architecture） | 10 | "using reward shaping methods to train LLM agent" |
| **数据/基准约束**（dataset/benchmark） | 6 | "evaluate their performance on HotPotQA" |
| **排除约束**（not/without/exclude） | 5 | "Exclude survey papers", "not citing the transformers paper" |
| **领域约束**（in the/domain/field） | 2 | "in LLM research area" |
| **时间约束**（recent/latest/2023/2024） | 0 | 所有查询均依赖 `published_time` 隐式约束 |

### 4.4 查询复杂度特征

数据集包含多种复杂度级别的查询：

| 复杂度 | 示例 qid | 特征 |
|--------|----------|------|
| **简单查找** | RealScholarQuery_25 | "Video aesthetics score, using multimodal large models."（6词，1篇答案） |
| **条件约束** | RealScholarQuery_3 | 多条件：visual+audio inputs、large-scale datasets、排除survey（42篇答案） |
| **对比判断** | RealScholarQuery_18 | "zero-shot manner? Do they perform better than supervised..." |
| **反面声称** | RealScholarQuery_15 | "Provide papers claiming that RL can negatively impact..." |
| **分级查找** | RealScholarQuery_28 | "mid-level hardness...harder than HumanEval...easier than code_contests" |
| **多跳推理** | RealScholarQuery_20 | "knowledgeable LLMs...analyze relationships...write a survey" |

---

## 5. arXiv ID 分析

| 属性 | 值 |
|------|-----|
| arXiv ID 总数 | 791 |
| 格式 | 老格式（4位年份+5位数字，如 `2309.04564`），无版本后缀 |
| 与标题一一对应 | ✅ 100% 匹配 |

**年份分布**（基于ID前缀）：

| 年份 | 数量 | 占比 |
|:----:|:----:|:---:|
| 2024 | ~380 | ~48% |
| 2023 | ~230 | ~29% |
| 2022 | ~80 | ~10% |
| 2021 及更早 | ~100 | ~13% |

**解读**：大部分答案论文集中在 2023-2024 年，与查询时间戳 2024-10-01 一致——系统只能检索该日期前发表的论文。

---

## 6. 与 PaSa 论文描述的一致性

根据 PaSa 论文 [1] 对 RealScholarQuery 的描述：

| 论文描述 | 实际数据 | 匹配 |
|----------|----------|:--:|
| 50 条真实世界研究查询 | 50 条 | ✅ |
| 由 AI 研究者提供 | 查询覆盖 AI 多领域 | ✅ |
| 过滤了过于宽泛的主题 | 查询多为具体技术问题 | ✅ |
| 平均 15.82 篇答案/查询 | 平均 15.8 篇 | ✅ |
| P50=9, P90=37 | 中位数=10, P90≈37 | ✅ |
| 标注者：中国顶级大学计算机系教授 | — | ✅ |
| 查询时间：2024-10-01 | 所有 published_time=20241001 | ✅ |
| 平均每查询审查 76 篇候选 | — | — |
| 标注成本 $304/query | — | — |

**结论**：该文件正是 PaSa 论文中描述的 **RealScholarQuery 测试集**。

---

## 7. 对系统评测的启示

### 7.1 评测挑战

1. **答案数量极度不均**：从 1 篇到 65 篇，评测指标需要在不同粒度上都有区分度。Recall@K 中的 K 值选择很关键（PaSa 使用了 Recall@20/50/100）。

2. **领域覆盖广泛但偏重 AI**：16/50 查询涉及视觉/视频，反映了数据集采集于 AI 研究者社区。跨领域泛化能力评测可能不足。

3. **缺少部分相关性标注**：所有答案论文被标注为"相关"（二元），但实际学术搜索中存在大量"部分相关"论文。系统对"部分相关"论文的处理无法评估。

4. **隐式时间约束**：所有查询依赖 `published_time=20241001` 作为时间截止，但查询文本中不含时间要求。系统需要**推断**而非解析时间约束。

### 7.2 设计建议

1. **多粒度 Recall 评估**：鉴于答案数从 1 到 65 的分布，应同时报告 Recall@10/20/50/100

2. **按查询类型分层报告**：Find 类 vs Compare 类 vs Survey 类应分别评估

3. **注意答案集的不完整性**：PaSa 论文指出 AutoScholarQuery 中答案论文仅是完整答案集的子集（引用偏倚），RealScholarQuery 虽经多轮方法聚合但仍可能遗漏相关论文

4. **成本约束**：$304/query 的标注成本意味着无法自行构建大规模类似数据集——需要充分利用该 50 条数据的评估价值

---

## 8. 代码示例

### 读取与基本统计

```python
import json

with open('test.jsonl', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f if line.strip()]

print(f"查询数: {len(data)}")
print(f"答案论文总数: {sum(len(d['answer']) for d in data)}")

# 按 qid 查询
by_qid = {d['qid']: d for d in data}
print(by_qid['RealScholarQuery_0']['question'])

# 构建评测 Ground Truth
ground_truth = {}
for d in data:
    ground_truth[d['qid']] = {
        'question': d['question'],
        'relevant_arxiv_ids': set(d['answer_arxiv_id']),
        'relevant_titles': set(d['answer']),
        'query_date': d['source_meta']['published_time']
    }
```

### 评测指标计算

```python
def evaluate(predicted_ids: set, gold_ids: set):
    tp = len(predicted_ids & gold_ids)
    precision = tp / len(predicted_ids) if predicted_ids else 0
    recall = tp / len(gold_ids) if gold_ids else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return {'precision': precision, 'recall': recall, 'f1': f1}
```

---

*分析完成日期：2026-08-05*
*数据来源：[1] He et al., "PaSa: An LLM Agent for Comprehensive Academic Paper Search," ACL 2025.*
