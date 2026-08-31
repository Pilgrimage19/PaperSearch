# ============================================================
# src/step2_critic.py
# Critic（检索自省器）：检查约束覆盖缺口，生成补充查询
# ============================================================

import json
import logging
from typing import List, Dict, Any, Optional
from src.utils.config_loader import config
from src.llm.client import LLMClient

logger = logging.getLogger("paper_search.critic")


CRITIC_SYSTEM = (
    "你是学术检索的自省器。判断当前检索结果是否覆盖了查询的各约束维度，"
    "并针对未覆盖的约束生成补充查询。"
)

CRITIC_PROMPT = """判断当前检索结果是否充分覆盖了查询的各约束维度。

Query: {query}
Constraints: {constraints_json}
当前候选池（前 {top_n} 篇标题）:
{paper_titles}

对每个约束，判断候选池里是否有论文充分覆盖了该维度。

输出 JSON:
{{
  "coverage": [
    {{"constraint": "<dimension>: <value>", "covered": true/false}}
  ],
  "verdict": "complete" / "incomplete",
  "new_queries": ["针对未覆盖约束的补充查询1", "查询2"]
}}

规则:
- "covered" 为 true 表示候选池有充分覆盖该约束的论文
- 只要有任一约束 "covered" 为 false，verdict 就是 "incomplete"
- new_queries 只针对「未覆盖」的约束，用不同措辞/同义词/下位词生成 1-3 个查询
- verdict 为 "complete" 时，new_queries 可以为空数组
- 返回 ONLY valid JSON，不要其他文字。"""


def critique(
    query: str,
    constraints: List[Dict[str, Any]],
    papers: List[Dict[str, Any]],
    client: Optional[LLMClient] = None,
    top_n: int = 20,
) -> Dict[str, Any]:
    """
    Critic 自省：检查候选池是否覆盖了查询的各约束维度。

    Returns:
        {"coverage": [...], "verdict": "complete"/"incomplete", "new_queries": [...]}
    """
    if not constraints:
        return {"verdict": "complete", "new_queries": [], "coverage": []}

    if client is None:
        client = LLMClient(
            model=config.get("ranking", "fine", "model", default="deepseek-v4-flash"),
            provider=config.get("ranking", "fine", "model_provider", default="deepseek"),
            temperature=0.0,
            max_tokens=2000,
        )

    constraints_json = json.dumps(constraints, ensure_ascii=False)
    paper_titles = "\n".join(
        f"- {p.get('title', '')[:80]}" for p in papers[:top_n] if p.get("title")
    )
    if not paper_titles:
        paper_titles = "(空)"

    prompt = CRITIC_PROMPT.format(
        query=query,
        constraints_json=constraints_json,
        top_n=top_n,
        paper_titles=paper_titles,
    )

    try:
        result = client.chat_json(prompt, system=CRITIC_SYSTEM, repair_json=True)
        result.setdefault("verdict", "complete")
        result.setdefault("new_queries", [])
        result.setdefault("coverage", [])
        return result
    except Exception as e:
        logger.warning(f"Critic failed, treat as complete: {e}")
        return {"verdict": "complete", "new_queries": [], "coverage": []}
