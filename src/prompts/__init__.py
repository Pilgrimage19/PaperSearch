# src/prompts/__init__.py
#
# All LLM prompts in one place for easy inspection and modification.
# Import from here rather than defining inline in pipeline steps.
#

# ---- Step 1: Query Understanding ----

QUERY_UNDERSTANDING_SYSTEM = (
    "You are an expert academic search analyst. "
    "Parse research queries into structured search plans."
)

QUERY_UNDERSTANDING_PROMPT = """Analyze this research query and return a structured JSON search plan.

Query: {user_query}

Return a JSON object with these fields:

1. "query_type": one of "semantic" / "specific" / "metadata"
   - semantic: content-based search (e.g., "papers about X using Y method")
   - specific: finding a known paper (e.g., "the attention is all you need paper")
   - metadata: finding by author/year/venue/citation constraints

2. "intent": one of "find" / "survey" / "compare"
   - find: general search for relevant papers
   - survey: seeking comprehensive overview, review articles
   - compare: seeking comparative analysis between methods

3. "domain": research domain
   One of: "CS", "AI", "ML", "NLP", "CV", "RL", "biomedical", "physics", "math", "other"

4. "constraints": array of constraint objects. Each constraint has:
   - "dimension": one of "method" / "task" / "data" / "time" / "venue" / "performance"
   - "value": the extracted constraint value (specific and concrete)
   - "necessity": "must" (hard requirement) or "should" (soft preference)
   Example: {{"dimension": "method", "value": "BERT-based fine-tuning", "necessity": "must"}}

5. "excluded": array of things to explicitly exclude
   Example: ["survey papers", "non-English papers"]

6. "sub_queries": object with two arrays:
   - "keyword": 3-5 keyword search queries (3-7 technical keywords each, for BM25/API search)
   - "nl": 1-2 natural language queries (for embedding similarity search)
   Example: {{
     "keyword": ["BERT sentiment analysis Chinese social media", "aspect-based sentiment Weibo"],
     "nl": ["How does BERT perform on Chinese sentiment classification tasks?"]
   }}

7. "time_range": {{"start": "2022", "end": "2024"}} or null if no time constraint mentioned

8. "preferred_sources": array of suggested sources
   Options: "semantic_scholar", "arxiv", "openalex", "pubmed", "dblp"

Rules:
- Keyword queries: concise, technical, 3-7 words each
- NL queries: natural sounding, full sentences
- Constraints: must be specific and directly extractable from the query text
- If no time constraint mentioned, use null for time_range
- For "specific" or "metadata" type queries, constraints and sub_queries can be minimal

Return ONLY valid JSON, no other text."""


# ---- Step 3: Fine Ranking ----

FINE_RANK_SYSTEM = (
    "You are an academic paper reviewer. "
    "Evaluate paper relevance to research queries with detailed scoring."
)

FINE_RANK_PROMPT = """Evaluate these papers against a research query with constraint-based scoring.

Query: {query}
Query Constraints: {constraints_json}

Papers to evaluate:
{paper_list}

For EACH paper, return a JSON object with:
{{
  "paper_index": <integer matching the paper number>,
  "constraint_scores": {{"constraint_name": <0.0-1.0>, ...}},
  "overall_score": <0-100 integer, higher = more relevant>,
  "evidence": "<one sentence summarizing the key relevant finding from this paper>",
  "relevance_level": "highly_relevant" / "partially_relevant" / "background"
}}

Scoring guidelines:
- highly_relevant (70-100): satisfies ALL must constraints + most should constraints
- partially_relevant (40-69): satisfies some must constraints or only should constraints
- background (0-39): domain-related but not directly matching query intent
- Use the full 0-100 range; do not cluster scores in the middle
- Evidence should cite specific methods, datasets, or results

Return a JSON array of these objects, one per paper, in the same order as the input.
Return ONLY valid JSON, no other text."""


# ---- Step 4: Comparison Table ----

COMPARISON_TABLE_SYSTEM = (
    "You are a structured data extraction specialist. "
    "Extract key methodological details from academic papers."
)

COMPARISON_TABLE_PROMPT = """Extract structured comparison data from these papers.

Research Query: {query}

Papers:
{paper_list_json}

For each paper, extract these fields:
- "model_base": main model, architecture, or framework used
- "dataset": evaluation dataset(s) used
- "key_metric": primary performance metric name and best value (e.g., "F1=0.87", "Accuracy=92.3%")
- "method_category": approach type (e.g., "fine-tuning", "prompt-tuning", "RL", "ensemble", "distillation")
- "key_innovation": one-sentence summary of what is novel about this work

If a field cannot be determined from the abstract, use "N/A".

Return a JSON array of objects, one per paper, in the same order as the input.
Return ONLY valid JSON, no other text."""


# ---- JSON Repair Instruction ----

JSON_REPAIR_SUFFIX = """
IMPORTANT: Your last response was not valid JSON.
Return ONLY a valid JSON object/array with:
- No markdown code fences (no ```json or ```)
- No trailing commas
- No comments or extra text before/after the JSON
- All strings properly quoted with double quotes"""
