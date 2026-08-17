# 科研场景下复杂学术查询的智能论文搜索与推荐 — 赛题分析与方法综述

> 基于【7.8】2026年中国研究生人工智能大赛--华为赛题 (3).docx 及相关参考文献
> 分析日期：2026-08-05

---

## 目录

1. [赛题概述与核心要求](#1-赛题概述与核心要求)
2. [主流方法综述](#2-主流方法综述)
3. [关键技术对比分析](#3-关键技术对比分析)
4. [现有方法的缺陷与不足](#4-现有方法的缺陷与不足)
5. [改进方向与建议](#5-改进方向与建议)
6. [参考文献](#6-参考文献)

---

## 1. 赛题概述与核心要求

### 1.1 赛题定位

本赛题要求构建一个**端到端的学术论文智能搜索系统**，针对自然语言描述的复杂学术查询，完成从查询理解、多维度多策略检索、论文综合排序到搜索结果归纳整理的全流程自动化。

### 1.2 四大核心功能

| 功能模块 | 具体要求 |
|----------|----------|
| **查询理解与分解** | 解析核心研究意图 → 识别关键实体/方法/数据约束 → 子查询分解 → 查询改写与扩展 |
| **自主搜索策略迭代** | 自主规划搜索词 → 过滤低质量论文 → 根据已找到论文动态调整检索策略 |
| **论文综合排序** | 基于标题/摘要/全文的细粒度相关性评估 → 区分高度相关与部分相关 |
| **搜索结果归纳整理** | 根据查询意图整理结果 → 结构化展示（列表、关系图等） |

### 1.3 评测标准

| 指标 | 权重 | 说明 |
|------|:----:|------|
| **F1 Score** | 70% | 综合衡量精确率与召回率的平衡 |
| **运行效率** | 20% | API调用次数、Token消耗量、端到端延时 |
| **回复结构化** | 10% | 列表、关系图等结构化展示 |

成绩构成：竞赛得分 60%（公开测试集 30% + 隐藏测试集 30%）+ 专家评分 40%（创新性 15% + 方案落地可行性 15% + 算法泛化性 10%）

### 1.4 参考系统性能基线

| 系统 | 核心方法 | 性能参考 |
|------|----------|----------|
| **PaSa-7B** | Crawler+Selector双Agent + RL训练 | RealScholarQuery上Recall@20超Google+GPT-4o 37.78% |
| **SPAR** | RefChain查询分解 + 查询演化 + 多Agent | AutoScholar上F1=0.38，超PaSa 56% |
| **Ai2 Paper Finder** | 多索引语义检索 + LLM改写 + 引文追踪 | PaperFindingBench上得分超ReAct Agent一倍以上 |
| **PaperQA2** | 全文检索 + LLM QA + 引文遍历 | LitQA2上准确率超人类专家 |

---

## 2. 主流方法综述

### 2.1 基于强化学习的双Agent架构：PaSa

**论文**：[1] He et al., "PaSa: An LLM Agent for Comprehensive Academic Paper Search," ACL 2025.

**核心架构**：PaSa 由两个 LLM Agent 组成 —— **Crawler（爬取器）** 和 **Selector（选择器）**。

**Crawler的工作流程**：
1. 接收用户查询，生成多个互补的搜索查询词
2. 调用 [Search] 工具通过 Google (site:arxiv.org) 检索论文
3. 将检索到的论文添加到 Paper Queue
4. 对队列中每篇论文，调用 [Expand] 操作提取引文子节中的参考文献
5. 通过引文网络递归发现更多相关论文
6. 调用 [Stop] 重置上下文，处理队列中的下一篇论文

**Selector的工作流程**：
1. 读取论文队列中的每篇论文（标题+摘要）
2. 生成决策 Token（True/False）判断论文是否满足查询要求
3. 同时生成 Rationale（理由）增强可解释性
4. 在训练阶段作为 Crawler 的辅助奖励模型

**训练策略**：
- 使用 AGILE 框架进行强化学习（PPO算法）
- 创新性的 **Session-level PPO** 解决长轨迹和稀疏奖励问题
- 将完整轨迹切分为多个 Session（以 [Stop] 为界），每个 Session 独立计算 Return
- Selector 作为辅助奖励模型缓解 AutoScholarQuery 标注不完整导致的稀疏奖励

**关键数据**：
- AutoScholarQuery：35k 细粒度学术查询，来自 ICLR/ICML/NeurIPS/ACL/CVPR 顶会论文
- RealScholarQuery：50 条真实世界查询，由教授级专家标注
- PaSa-7B 在 RealScholarQuery 上 Recall@20=0.5798，超 Google+GPT-4o 37.78%

### 2.2 免训练多Agent框架：SPAR

**论文**：[4] Shi et al., "SPAR: Scholar Paper Retrieval with LLM-based Agents for Enhanced Academic Search," 2025.

**核心架构**：SPAR 是一个 **训练无关（training-free）的模块化多Agent框架**，由五个专业Agent组成：

1. **Query Understanding Agent（查询理解）**
   - 意图分类：区分 Survey、Recent Advances、Methodological Comparison
   - 领域识别：将查询锚定到特定学科领域
   - 时间约束检测：解析显式时间限制
   - 源选择：从 {Google, ArXiv, OpenAlex, Semantic Scholar, PubMed} 中智能选择
   - 意图感知的查询重写：根据意图类型生成多角度细化查询

2. **Retrieval Agent（检索）**
   - 源自适应查询：对不同数据源采用不同查询策略（关键词 vs 完整查询）
   - 跨源去重与合并
   - RefChain（引用链）扩展：对相关论文的参考文献进行一层深度遍历

3. **Judgement Agent（判断）**
   - 对检索到的论文进行相关性评估（0-1分）
   - 支持 Brief 和 Complex 两种 Prompt 策略
   - Qwen3-32B + Brief Prompt 达到最佳 F1=0.38

4. **Query Evolver Agent（查询演化）**
   - 基于已检索论文生成三个新查询：方法论替代方案、应用实现、局限与挑战
   - 条件化于检索历史轨迹，避免重复

5. **Reranker Agent（重排序）**
   - 整合三个信号：相关性（原始评分）、发表权威性（会议/期刊声望+作者声誉）、时效性

**关键创新 — RefChain**：
- 模拟研究者追踪引用的行为
- 仅限一层深度（平衡精度与效率）
- 在 AutoScholar 上提升 Recall 7.32%，在 SPARBench 上提升 15.38%

**性能**：
- AutoScholar 上 F1=0.3843（超 PaSa 56.92%）
- SPARBench 上 F1=0.3015，是唯一在所有指标上表现一致的方法

### 2.3 超人类性能的RAG Agent：PaperQA2

**论文**：[5] Skarlinski et al., "Language Agents Achieve Superhuman Synthesis of Scientific Knowledge," NeurIPS 2024.

**核心架构**：PaperQA2 将 RAG 分解为四个可组合的 **Agent 工具**：

1. **Paper Search Tool（论文搜索）**
   - Agent 将用户请求转化为关键词搜索
   - 使用 Semantic Scholar 等 API 检索候选论文
   - 通过 Grobid 解析论文全文（章节、表格、引用）

2. **Gather Evidence Tool（证据收集）**
   - **阶段一：Dense Vector Retrieval** — 使用嵌入向量（text-embedding-large-3 + 稀疏关键词编码）进行 Top-K 排序
   - **阶段二：RCS（Reranking and Contextual Summarization）** — LLM 对每个 Chunk 进行相关性评分和上下文摘要，这是 PaperQA2 最关键的设计

3. **Citation Traversal Tool（引文遍历）**
   - 以高评分摘要对应的论文为起点，向前（未来引用者）和向后（历史参考文献）遍历一层引文
   - 使用 Overlap Filtering 过滤低质量引用（需多篇论文共同引用）
   - 对 DOI Recall 有显著提升

4. **Generate Answer Tool（生成答案）**
   - 将 Top-Ranked 证据摘要注入 LLM 上下文
   - 生成带引用的最终回答

**关键发现**：
- **RCS 是关键**：若无 RCS（直接将 Chunk 注入上下文），性能大幅下降
- **小模型做 RCS 有害**：GPT-3.5-Turbo 和 Llama3-70B 做 RCS 反而不如不做
- **RCS 深度很重要**：Top-K 深度从 1 到 10 有显著提升，10 到 30 收益递减
- **非 Agent 版本性能显著更低**：验证了 Agent 循环（可返回修改搜索）的重要性
- **引文遍历对召回不可或缺**
- **解析质量和 Chunk 大小影响有限**

**性能**：
- LitQA2 上 Precision=85.2%、Accuracy=66.0%
- **超人类表现**：人类专家 Precision=73.8%，PaperQA2=85.2%（p=0.0036）
- WikiCrow 生成的 Wikipedia 风格文章比人类写的 Wikipedia 更准确

### 2.4 检索-LM组合框架：DSP

**论文**：[6] Khattab et al., "Demonstrate-Search-Predict: Composing Retrieval and Language Models for Knowledge-Intensive NLP," 2022.

**核心贡献**：提出了 **DSP（Demonstrate-Search-Predict）** 框架，将冻结的检索模型（RM）和语言模型（LM）通过自然语言文本传递组合为可编程管道——类似于神经网络中的"层"，但传递的是文本而非张量。

**三阶段流程**：

1. **DEMONSTRATE（自动标注）**
   - 使用 `annotate` 原语：对每个训练样本运行完整的 SEARCH+PREDICT 管道
   - 若预测匹配 Ground Truth，则将中间预测（查询、检索段落）缓存为示范
   - 无需手工标注中间步骤（如 hop queries、段落选择）
   - 支持多种采样策略：`sample`（随机 k-shot）、`knn`（最近邻检索示范）、`crossval`（交叉验证 Prompt 选择）

2. **SEARCH（多跳检索）**
   - LM 生成搜索查询 → RM 检索段落 → 结果反馈到下一跳
   - **多跳策略**：每跳 LM 总结上一跳的上下文并生成后续查询；当 LM 输出 "N/A" 时终止
   - **查询融合**（`fused_retrieval`）：每跳生成 n=10 个查询，通过 CombSUM 融合结果
   - **对话式搜索**：组合查询重写（解决对话依赖）与多跳搜索
   - 显式的程序化控制流，避免了 self-ask 的"自分散"问题

3. **PREDICT（生成输出）**
   - **Self-consistency**：对多个 CoT 采样进行 Majority Voting
   - **Pipeline-of-Thoughts（PoT）**：分支整个管道运行（可变跳数/路径），然后在分支间投票——将 Self-consistency 从 Chain-of-Thought 推广到 Pipeline-of-Thought

**关键结果**：使用冻结的 GPT-3.5 + ColBERTv2，在 HotPotQA 上达到 51.4% EM（相比 Vanilla LM 的 28.3%、Retrieve-then-Read 的 36.9%、Self-ask 的 80-290% 提升）。

**与学术搜索的关联**：
- 多跳检索范式可直接应用于学术搜索中的引文遍历和查询分解
- 自动示范标注机制可降低构建训练数据的人力成本
- 其"编程式组合"思想影响了后来的 Agent 框架设计（包括 PaSa 和 SPAR）

### 2.5 LLM Agent强化学习框架：AGILE

**论文**：[7] Feng et al., "AGILE: A Novel Reinforcement Learning Framework of LLM Agents," NeurIPS 2024.

**核心架构**：AGILE 由四个模块组成 —— **LLM（策略模型）、Memory（记忆）、Tools（工具）、Executor（执行器）**。LLM 作为策略模型在 Token 级别的 MDP 中预测动作；Executor 解释函数名 Token 并执行对应的实现，将结果追加到 LLM 上下文。

**Agent 的五种核心能力**：

1. **Reflection（反思）**：当 Agent 通过 `[SeekAdvice]` 获取专家建议后，调用 `[Reflection]` 从中蒸馏通用的可复用知识，写入 Memory
2. **Tool Usage（工具使用）**：生成 SQL 查询进行产品搜索；生成搜索查询进行 Web 检索
3. **Memory（记忆）**：存储历史 QA 对和反思知识，使用嵌入向量相似度检索（all-MiniLM-L6-v2）
4. **Proactive Consultation（主动咨询）**：Agent 学会在置信度低时主动调用 `[SeekAdvice]`——这是**训练出来的决策**（非规则），通过 RL 优化
5. **End-to-End RL Training**：两阶段训练——先模仿学习（SFT），再 PPO 强化学习

**Session-Level PPO 创新**：
- 长轨迹被划分为 Session（独立 QA 轮次）
- State-Advantage Function 建模跨 Session 依赖（早期 Session 积累的知识使后续 Session 受益）
- Proxy Reward = Session Reward + State Advantage Delta

**关键结果**：
- ProductQA：agile-vic13b-ppo（13B）以 85.4% 准确率超越 GPT-4 Agent
- MedMCQA：agile-mek7b-ppo（7B）达到 85.2% 准确率
- HotPotQA：67.5% EM，超越 GPT-4-based ReAct（48.2%）
- **消融实验**：移除工具→25.9% 更多 Consult + 9.3% 分数下降；移除记忆→4.0% 下降；移除反思→1.7% 下降；移除主动咨询→10.7% 准确率下降

**在 PaSa 中的应用**：
- PaSa 直接基于 AGILE 框架实现 Crawler 的 RL 训练
- 将 Crawler 建模为 Token 级别的 MDP，创新性的 Session-level PPO 解决长轨迹挑战
- Selector 作为辅助奖励模型缓解稀疏奖励问题

**对学术搜索系统的启示**：
- Memory 模块可用于累积用户专属知识（如常用搜索领域、偏好venue）
- 学习到的 `[SeekAdvice]` 决策可路由不确定的查询到人工审核
- Session-level RL 可解决学术搜索中多轮交互的长轨迹问题

### 2.6 学术搜索评测基准：LitSearch 与 AstaBench

#### 2.6.1 LitSearch — 首个真实文献检索评测集

**论文**：[2] Ajith et al., "LitSearch: A Retrieval Benchmark for Scientific Literature Search," EMNLP 2024.

**核心贡献**：首个使用**真实研究者查询**的科学文献检索 Benchmark，包含 597 个经过人工审核的自然语言文献搜索问题。

**两种问题来源**：
- **Inline-Citation Questions（351个）**：从 S2ORC 语料库中提取引用段落 → GPT-4 改写为文献检索问题 → 人工筛选。来源论文不限领域，目标论文限定 ACL Anthology。
- **Author-Written Questions（246个）**：由 ACL 2023 和 ICLR 2024 论文作者自己撰写关于其已发表论文的文献检索问题。

**关键发现**：

| 发现 | 具体数据 |
|------|----------|
| **稠密检索远超BM25** | GritLM-7B 在 Broad Questions 上 R@20=70.8，超 BM25 24.8% |
| **LLM重排序有效但有限** | GPT-4o Reranking 在 GritLM 基础上提升 4.4%（R@5），但绝对值天花板仅 ~79.2% |
| **商业搜索引擎表现糟糕** | Google/Google Scholar/Elicit 最高仅 42.8% R@5（vs GritLM 74.8%） |
| **全文编码未必有效** | 加入论文全文（平均6041词）反而**降低**性能——当前Embedding模型上下文长度不足 |
| **In-line问题比作者编写更难** | 作者编写问题与目标论文的词重叠率更高（0.43 vs 0.33），存在标注偏差 |
| **长文档是核心挑战** | 文档平均134词（标题摘要）/ 6041词（全文），远超 MS-MARCO（56词） |

**对学术搜索系统设计的启示**：
- Query风格应与 LitSearch 的详细概念型查询对齐（而非短关键词）
- 需要长上下文 Embedding 模型来处理科学论文的全文编码
- 两阶段管道（稠密检索 + LLM重排序）是最有效的基线架构
- 评估需考虑 Query 类型（Broad/Specific）的差异性

#### 2.6.2 AstaBench — 最全面的科学Agent评测套件

**论文**：[3] Feldman et al., "AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite," ICLR 2026.

**核心贡献**：涵盖 2400+ 问题、11 个 Benchmark、22 个 Agent 类别的全面科学 Agent 评测体系。

**四层任务分类**：
1. **Literature Understanding**：PaperFindingBench（333个查询）、LitQA2-FullText-Search、ScholarQA-CS2（长文本QA）、ArxivDIGESTables-Clean（文献综述表格生成）
2. **Code and Execution**：SUPER-Expert、CORE-Bench-Hard、DS-1000
3. **Data Analysis**：数据分析与统计推理任务
4. **End-to-End Discovery**：完整科学发现流程

**PaperFindingBench 的查询分类**（最相关于赛题）：
- **Semantic Queries（242个）**：基于内容描述的论文查找，含复杂条件、关系、罕见术语、错误术语
- **Navigational Queries（48个）**：从简短引用找到特定已知论文（如 "the alpha-geometry paper"）
- **Metadata Queries（43个）**：嵌套元数据查询（如 "ACL 2024 papers that cite the transformers paper"，含否定条件）

**关键发现**：

| 发现 | 具体数据 |
|------|----------|
| **文献搜索远未解决** | PaperFindingBench 上最佳专用Agent（Asta Paper Finder）仅 ~40%，最强通用Agent（ReAct+GPT-5）仅 26.4% |
| **专用Agent优势明显** | LitQA2-FullText-Search上专用 Agent 达 90.7% Recall，通用 Agent 82.7% |
| **GPT-5相比o3提升有限** | 多数 Benchmark 仅提升 0-5%（绝对值），在4个 Benchmark 上提升 13-25% |
| **更强模型可能降低专用Agent性能** | 专用 Agent（Asta Scholar QA等）配合 GPT-5 反而**更差**——可能因 GPT-5 偏向 ReAct 风格工作流 |
| **开放源Agent表现糟糕** | 最佳开源 Agent（Smolagents Coder + Llama-4-Scout-17B）仅 11.1% |
| **文献综述表生成极弱** | ArxivDIGESTables-Clean 上最佳模型 Recall 仅 ~43% |

**AstaBench 的评测方法论创新**：
- **成本归一化**：使用冻结的 LiteLLM 价格表进行时间不变的美元归一化
- **工具分类**：Standard/Custom Interface/Fully Custom Tools 三级分类
- **开放度分类**：Open-Source Open-Weight/Closed-Weight/Closed-Source API/Closed UI
- **日期截止**：所有文献工具限制在特定日期前的论文，**防止污染**
- **帕累托前沿分析**：Score vs. Cost 的帕累托效率前沿识别最优性价比 Agent

**对赛题评测设计的启示**：
- 应采用 AstaBench 的"成本归一化"方法评测效率
- PaperFindingBench 的 Semantic/Navigational/Metadata 查询分类可指导测试集设计
- LLM-as-Judge with Rubrics 是可行的自动化评分方案（AstaBench 验证了与人类评分的 0.800 Kendall tau 相关性）

### 2.7 其他支撑方法

**GritLM** [8]：提出生成式表示指令微调，统一生成和嵌入到一个模型中。在 LitSearch 评测中，GritLM-7B 是最强的稠密检索基线（R@20=70.8）。对学术搜索中的向量检索环节有直接参考价值。

**Compositionality Gap** [9]：提出并度量了语言模型中的"组合性差距"——模型在处理需要组合多个知识片段的复杂问题时性能显著下降。对理解复杂学术查询的分解策略有理论指导意义。

**QR-RAG** [10]：提出领域对齐的查询重写 RAG 框架（应用于水分解催化领域），将回答准确率从传统 RAG 的 21.3% 提升到 85.6%，同时降低 39% 成本。验证了"结构化知识 + 查询重写"的互补价值。

---

## 3. 关键技术对比分析

### 3.1 查询理解策略对比

| 方法 | 查询理解方式 | 子查询分解 | 意图感知 | 源选择 |
|------|------------|-----------|---------|--------|
| **PaSa** | LLM直接生成多个搜索查询 | 与[Search]耦合，隐式分解 | 无显式意图分类 | 固定 Google (site:arxiv.org) |
| **SPAR** | 专用的Query Understanding Agent | 显式多角度分解（方法/应用/历史/挑战） | **显式意图分类**（Survey/Recent/Method） | **智能多源选择** |
| **PaperQA2** | Agent模型直接生成关键词搜索 | 通过多次Search工具调用隐式分解 | 无显式意图分类 | Semantic Scholar等 |
| **传统方法** | 关键词匹配 | 无分解 | 无 | 单一引擎 |

**分析**：SPAR 的查询理解策略最为系统化，显式的意图分类和源选择对跨领域泛化至关重要。PaSa 和 PaperQA2 依赖 Agent 模型自身的推理能力进行隐式理解。

### 3.2 检索策略对比

| 方法 | 检索后端 | 引文网络利用 | 检索迭代 | 多源融合 |
|------|---------|------------|---------|---------|
| **PaSa** | Google (arXiv限定) | **[Expand]** 操作全文学术网络遍历 | RL训练的策略自主决定何时搜索 | 单一源 |
| **SPAR** | 多源（Google/ArXiv/OpenAlex/PubMed/Semantic Scholar） | **RefChain** 一层引文扩展 | 查询演化驱动的迭代 | **源自适应策略** |
| **PaperQA2** | Semantic Scholar + Crossref | **Citation Traversal** + Overlap过滤 | Agent自主决定 | 单一主源+引文API |
| **Google+LLM** | Google | 无 | 无迭代 | 单一源 |

**分析**：PaSa 和 PaperQA2 的引文网络遍历对召回率提升至关重要（PaSa消融实验中取消[Expand]导致Recall下降22.98%~32.21%；PaperQA2取消Citation Traversal导致DOI Recall显著下降）。SPAR 的多源融合策略在跨领域泛化上更具优势。

### 3.3 相关性判断策略对比

| 方法 | 判断方式 | 判定粒度 | 模型要求 |
|------|---------|---------|---------|
| **PaSa Selector** | 微调的Qwen2.5-7B，直接输出True/False | 论文级别 | 需训练 |
| **SPAR Judgement** | LLM提示（Brief/Complex），0-1评分 | 论文级别 | 无需训练 |
| **PaperQA2 RCS** | LLM对每个Chunk评分+摘要 | **Chunk级别**（更细粒度） | 需强模型（GPT-4-Turbo） |

**分析**：PaperQA2 的 Chunk 级别 RCS 是最精细的判断策略，但计算成本最高（每查询 $1-3）。SPAR 的 Brief Prompt + Qwen3-32B 在成本和效果间取得了最佳平衡。PaSa 的训练方案在小模型上表现优异但泛化受限。

### 3.4 训练/优化策略对比

| 方法 | 是否需要训练 | 优化方式 | 数据需求 |
|------|:----------:|---------|---------|
| **PaSa** | **是** | Session-level PPO + 模仿学习 | 35k 合成数据 + RL环境 |
| **SPAR** | 否 | Static Prompt Engineering | 无训练数据需求 |
| **PaperQA2** | 否 | Prompt Engineering + 参数调优 | 无训练数据需求 |
| **AGILE** | **是** | 端到端 PPO | 标注 Action 数据 |

**分析**：训练 vs 免训练是一个核心权衡。PaSa/AGILE 的训练方案在特定领域（AI/ML）性能卓越，但泛化性和部署成本受限。SPAR 的免训练方案更灵活但性能上限可能不如训练方案。

---

## 4. 现有方法的缺陷与不足

### 4.1 查询理解层面的不足

**（1）意图理解的粒度不足**

现有方法大多采用粗粒度的查询意图分类。SPAR 虽然进行了 Survey/Recent/Method 的分类，但未深入到更细粒度的约束理解（如特定数据集、评估指标、实验条件等）。真实学术查询往往包含多维约束：

> "Find papers that use BERT-based models for sentiment analysis on Chinese social media data, published after 2022, with F1 scores above 0.85"

这种查询需要同时理解：方法约束（BERT-based）、任务约束（sentiment analysis）、数据约束（Chinese social media）、时间约束（after 2022）、性能约束（F1>0.85）。现有系统无一能同时处理所有维度。

**（2）查询分解的质量不可控**

- PaSa 的查询分解与 [Search] 操作耦合，不够透明
- SPAR 的 Query Understanding Agent 虽然进行了多角度分解，但不保证生成的子查询之间有良好的互斥性和完备性
- PaperQA2 完全依赖 Agent 模型的隐式推理，无法审计或优化分解过程

**（3）领域术语理解的缺失**

所有系统在跨领域泛化时面临领域术语理解的挑战。PaSa 仅在 AI/ML 领域训练和测试，SPARBench 虽然包含生物医学查询但查询理解 Agent 的术语知识完全依赖预训练 LLM 的内置知识。

### 4.2 检索策略层面的不足

**（1）多源融合的浅层化**

SPAR 是目前唯一进行显式多源选择的系统，但其融合策略仍较为简单（关键词提取 vs 完整查询的二元选择）。缺乏对不同数据源覆盖范围的动态建模和互补性分析。

**（2）引文遍历的成本-收益失衡**

- PaSa 的 [Expand] 操作依赖 RL 训练决定遍历深度，但在实际部署中仍需硬限制（depth≤3）
- SPAR 仅进行一层 RefChain，可能遗漏深层次的关键引用
- PaperQA2 的 Citation Traversal 在提升召回的同时显著增加了 API 调用成本

**根本问题**：没有系统能**智能化地判断哪些论文值得深入遍历、哪些论文可以跳过**。

**（3）检索覆盖率与精确度的矛盾**

这是赛题明确指出的核心矛盾。现有方法的应对策略各异但都不够理想：
- PaSa 偏向高召回（Crawler Recall=79.31%），牺牲精确度（Precision=14.48%）
- PaperQA2 通过 RCS 深度提升精确度，但成本高昂
- SPAR 通过 RefChain + Query Evolution 平衡，但 F1 仅 0.38（绝对值仍低）

### 4.3 论文排序层面的不足

**（1）排序信号的单一性**

| 方法 | 排序信号 |
|------|---------|
| PaSa | Selector True/False + Token概率（二元） |
| SPAR | 相关性 + 权威性 + 时效性（三元） |
| PaperQA2 | RCS Chunk级评分（细粒度但局部） |

对学术搜索而言，还应考虑：
- **论文影响力**：不仅看发表venue，还应看引用量、领域影响因子
- **方法论严谨性**：样本量、消融实验、可复现性等
- **结果可信度**：是否经过同行评议、是否有重复验证
- **多样性**：避免推荐来自同一作者/机构/范式的过度集中结果

**（2）缺乏与查询意图的对齐**

不同查询意图（Survey vs. 最新方法 vs. 方法对比）应有不同的排序策略，但现有系统鲜有实现：
- Survey 意图应优先推荐高引用量的综述文章
- "最新方法"意图应优先推荐近期高质量论文
- "方法对比"意图应推荐覆盖不同范式的多样化论文

### 4.4 搜索结果归纳层面的不足

这可能是现有系统**最薄弱的环节**：

- **PaSa**：仅返回论文列表（标题+摘要），无归纳功能
- **SPAR**：返回排序后的论文列表，无结构化整理
- **PaperQA2**：可生成带引用的答案（针对QA），但不适用于"找一批论文"的场景

**赛题明确要求的"搜索结果归纳整理"包括**：列表、关系图等结构化展示。但现有系统没有一个实现了以下能力：
- 按子主题/方法/时间的自动聚类
- 论文之间的关系图谱（引用、方法演进、对比等）
- 针对用户查询意图的定制化结果总结

### 4.5 效率与成本层面的不足

| 方法 | Token消耗 | API调用次数 | 端到端延时 | 计算资源 |
|------|----------|------------|----------|---------|
| PaSa-7B | 极高（完整论文全文+RL推理） | 大量[Search] + [Expand] | 高（需遍历论文队列） | 8xH100 GPU训练 |
| SPAR | 中等 | 中等（每个Agent独立调用LLM） | 中高 | 无训练，推理需Qwen3-32B |
| PaperQA2 | **极高**（每查询$1-3） | 大量（多次Search + RCS + Citation） | **极高**（平均14.5篇论文全文处理） | 需GPT-4-Turbo级别模型 |

**赛题的20%效率权重意味着**：不能无限制地使用大模型和大量API调用。现有方法在效率优化方面几乎没有系统性考虑。

### 4.6 评测基准的不足

#### 4.6.1 现有基准对比

| 基准 | 规模 | 领域 | 查询来源 | 关键局限 |
|------|:--:|------|----------|----------|
| **AutoScholarQuery** | 35k | AI/ML only | GPT-4o从Related Work合成 | 合成数据，仅引用子集（稀疏标注），93.7%论文合格率 |
| **RealScholarQuery** | 50 | AI/ML only | 真实用户+教授标注 | 极小规模，$304/query标注成本 |
| **LitSearch** | 597 | CS（NLP为主） | 论文作者撰写+引用改写 | 仅标题摘要可检索，64K固定语料库 |
| **SPARBench** | 50 | CS+生物医学 | GPT-4o生成+专家筛选 | 规模有限，跨领域覆盖有限 |
| **LitQA2** | 248 | 生物医学 | 专家手工编写多选题 | QA任务非检索任务，答案需出现在全文非摘要 |
| **PaperFindingBench** | 333 | 多领域 | PaperFinder/OpenSciLM用户日志 | 评测依赖LLM-Judge（与人工0.800 Kendall tau） |

#### 4.6.2 共性缺陷

**(1) 规模与领域的双重瓶颈**
- 最大基准仅 35k（AutoScholarQuery），且**仅限AI领域**
- 真实查询的基准极小（50-597条），因人工标注成本极高
- 没有任何基准覆盖物理、化学、工程等领域

**(2) "合成数据偏差"问题**
- LitSearch 发现：作者自己写的查询与目标论文词重叠率更高（0.43 vs 0.33），**人为高估检索性能**
- PaSa 的 AutoScholarQuery 仅 93.7% 论文合格率——存在 ~6% 标注噪声
- GPT-4o 生成的查询往往比真实用户查询更规范、更易检索

**(3) 评测指标维度单一**
- 现有指标聚焦 Precision/Recall/F1/nDCG
- **缺乏**：查询理解质量、检索效率、结果多样性、用户满意度、幻觉检测
- **缺乏**："部分相关"的处理——真实场景中大部分论文不是黑白二元的
- LitSearch 的 Stratified Reporting（Broad/Specific 分别报告）是好的实践，但未被广泛采用

**(4) 静态语料库 vs 动态检索的张力**
- LitSearch 使用固定的 64K ACL Anthology 语料库——排除了检索策略和源选择的评测
- AstaBench 的 Date Cutoff 机制是解决"污染"问题的创新，但限制了时效性评测
- "真实学术搜索需要实时检索互联网上不断增长的论文"——与静态Benchmark评测存在根本矛盾

**(5) 人机性能对比的缺失**
- 仅 PaperQA2 进行了系统性的与人类博士/博士后的对比
- 缺乏标准化的"人类专家基线"来衡量系统性能的真实水平
- AstaBench 发现：商业 Agent（OpenAI Deep Research 等）"无法完成 AstaBench 中的全范围任务"

**(6) Agent评测的混杂因素**
- AstaBench 识别了三大混杂因素：**工具差异、模型开放度、推理成本**
- 不同系统使用不同的检索后端（Google vs Semantic Scholar vs OpenAlex），Fair Comparison 困难
- 仅 AstaBench 提供了成本归一化的 Pareto 分析

---

## 5. 改进方向与建议

### 5.1 查询理解增强方向

1. **层次化查询解析**：构建查询约束分类体系（方法约束、任务约束、数据约束、时间约束、性能约束、venue约束），为每类约束设计专门的提取器和验证机制

2. **可审计的查询分解**：将查询分解为独立可追踪的子查询树，每个子节点对应可独立检索的学术问题，支持人工验证和修正

3. **领域感知的术语扩展**：集成学术知识图谱（如 OpenAlex Concepts、Semantic Scholar Fields of Study）进行术语归一化和同义词扩展

4. **多轮对话澄清**：对模糊查询主动向用户提问澄清（如歧义术语、过于宽泛的范围）

### 5.2 检索策略增强方向

1. **自适应多源路由**：根据查询类型（CS理论/CS系统/生物医学等）动态路由到最优数据源组合，而非固定选择

2. **成本感知的检索规划**：将 API 调用成本纳入检索策略规划，根据预算动态调整检索深度和广度

3. **智能引文遍历**：使用引用网络的结构特征（PageRank、社区检测等）筛选高价值遍历节点，而非等概率遍历

4. **混合检索策略**：结合稀疏检索（BM25/Lexical）、稠密检索（Dense Embedding）、基于引文的检索，在不同阶段采用不同策略

### 5.3 排序策略增强方向

1. **多维度排序融合**：整合相关性、权威性、时效性、影响力、方法论质量、多样性等信号

2. **查询意图感知的排序**：根据查询意图动态调整排序权重（如 Survey 意图加权引用量，"最新进展"意图加权时间）

3. **Pairwise/Listwise排序**：借鉴 Learning to Rank 的Pairwise/Listwise方法提升排序质量

4. **部分相关检测**：不仅区分相关/不相关，还识别"部分相关"论文（如方法相关但领域不同、同领域但方法不同）

### 5.4 结果归纳增强方向

1. **自动聚类与主题建模**：对检索结果按子主题自动聚类，生成主题标签

2. **引用关系图构建**：可视化论文之间的引用关系、方法演进路径

3. **结构化摘要生成**：对每个子主题生成简短的文献综述摘要（谁做了什么、用什么方法、达到什么效果）

4. **对比表格生成**：自动提取关键方法和指标，生成方法对比表

### 5.5 效率优化方向

1. **渐进式检索**：先用轻量级方法（标题+摘要）进行初筛，仅对Top-N候选进行全文分析

2. **模型级联**：使用小模型做初筛（如Qwen2.5-7B），大模型做精筛（如Qwen3-32B）

3. **缓存与增量检索**：缓存常见查询的检索结果，对相似查询做增量更新

4. **批处理优化**：将多个子查询的检索请求合并批处理

### 5.6 系统架构建议

综合以上分析，一个理想的端到端学术论文搜索系统应包含以下模块：

```
用户查询
    ↓
┌─────────────────────────────────────┐
│ 1. 查询理解模块                       │
│  • 意图分类 + 约束提取                │
│  • 层次化子查询分解                   │
│  • 领域术语扩展                       │
│  • 多轮对话澄清（可选）                │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 2. 检索规划模块                       │
│  • 自适应源选择（多源）                │
│  • 成本感知的策略规划                  │
│  • 混合检索（稀疏+稠密+引文）          │
└─────────────────────────────────────┘
    ↓  ↑ 迭代
┌─────────────────────────────────────┐
│ 3. 检索执行模块                       │
│  • 第一阶段：标题+摘要快速召回         │
│  • 第二阶段：智能引文遍历             │
│  • 第三阶段：全文精细筛选             │
│  • 查询演化：动态调整检索方向          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 4. 相关性排序模块                     │
│  • 多维度评分（相关/权威/时效/多样）   │
│  • 意图感知的权重调整                 │
│  • 部分相关 vs 高度相关分层           │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 5. 结果归纳模块                       │
│  • 主题聚类 + 关系图                  │
│  • 结构化摘要 + 对比表                │
│  • 分层展示（高度相关/部分相关）       │
└─────────────────────────────────────┘
    ↓
结构化搜索结果
```

---

## 6. 参考文献

[1] He Y, Huang G, Feng P, et al. "PaSa: An LLM Agent for Comprehensive Academic Paper Search." ACL 2025, arXiv:2501.10120.

[2] Ajith A, Xia M, Chevalier A, et al. "LitSearch: A Retrieval Benchmark for Scientific Literature Search." EMNLP 2024, arXiv:2407.18940.

[3] Feldman S, et al. "AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite." arXiv:2510.21652, 2025.

[4] Shi X, Li Y, Kou Q, et al. "SPAR: Scholar Paper Retrieval with LLM-based Agents for Enhanced Academic Search." arXiv:2507.15245, 2025.

[5] Skarlinski M, et al. "Language Agents Achieve Superhuman Synthesis of Scientific Knowledge." NeurIPS 2024, arXiv:2409.13740.

[6] Khattab O, et al. "Demonstrate-Search-Predict: Composing Retrieval and Language Models for Knowledge-Intensive NLP." arXiv:2212.14024, 2022.

[7] Feng P, He Y, Huang G, et al. "AGILE: A Novel Framework of LLM Agents." NeurIPS 2024, arXiv:2405.14751.

[8] Muennighoff N, et al. "GritLM: Generative Representational Instruction Tuning." arXiv:2402.09906, 2024.

[9] Press O, et al. "Measuring and Narrowing the Compositionality Gap in Language Models." arXiv:2210.03350, 2022.

[10] Lee D, et al. "Domain-aligned LLM Framework for Trustworthy Scientific Q/A via Query Reformulation RAG." ChemRxiv, 2025.

---

*分析完成日期：2026-08-05*
