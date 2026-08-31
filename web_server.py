# -*- coding: utf-8 -*-
"""
ScholarSeeker Web 服务 —— 提供学术论文搜索接口
启动：python web_server.py
接口：POST /search  {"query": "..."}
"""
import sys
import os

# 确保能 import src 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from src.utils.config_loader import load_config
from src.step1_query_understanding import parse_query
from src.step2_hybrid_retrieval import step2_retrieve_all
from src.step3_ranking import step3_rank_all
from src.step4_result_organization import step4_organize_all

# 加载配置（读取 config/.env 的 API key）
load_config()

app = FastAPI(title="ScholarSeeker", description="约束自省的迭代式学术论文搜索系统")

# 允许跨域（前端页面调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str


class SearchResponse(BaseModel):
    success: bool
    query: str
    constraints: list = []
    sub_queries: dict = {}
    layers: dict = {}
    constraint_matrix: dict = {}
    comparison_table: list = []
    citation_graph: dict = {}
    stats: dict = {}
    error: Optional[str] = None


@app.get("/")
def root():
    return {"service": "ScholarSeeker", "endpoint": "POST /search"}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    """执行一次完整的学术论文搜索（查询理解 → 迭代检索 → 排序 → 结构化）。"""
    query = req.query.strip()
    if not query:
        return SearchResponse(success=False, query=query, error="查询不能为空")

    try:
        # ---- Step 1: 查询理解 ----
        parsed = parse_query(query)

        # ---- Step 2: 迭代检索 ----
        parsed_queries = [{
            "qid": "web_query",
            "query": query,
            "query_type": "semantic",
            "parsed": parsed,
        }]
        retrieval_results = step2_retrieve_all(parsed_queries)
        # 补上 parsed 字段（step2 输出需要）
        for rr in retrieval_results:
            rr["parsed"] = parsed

        # ---- Step 3: 排序 ----
        ranking_results = step3_rank_all(retrieval_results)

        # ---- Step 4: 结构化 ----
        organized = step4_organize_all(ranking_results)

        result = organized[0]
        structured = result.get("structured_output", {})

        # ---- 组装响应 ----
        layers = structured.get("layers", {})
        matrix = structured.get("constraint_matrix", {})
        comp_table = structured.get("comparison_table", [])
        graph = structured.get("citation_graph", {})

        # 统计
        meta = result.get("metadata", {})
        stats = {
            "recalled": meta.get("after_expansion_count", len(result.get("ranked_papers", []))),
            "sources": meta.get("sources_used", []),
        }

        return SearchResponse(
            success=True,
            query=query,
            constraints=parsed.get("constraints", []),
            sub_queries=parsed.get("sub_queries", {}),
            layers=layers,
            constraint_matrix=matrix,
            comparison_table=comp_table,
            citation_graph=graph,
            stats=stats,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return SearchResponse(success=False, query=query, error=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
