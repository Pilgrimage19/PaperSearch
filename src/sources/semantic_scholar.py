# ============================================================
# src/sources/semantic_scholar.py - Semantic Scholar API client
# Docs: https://api.semanticscholar.org/api-docs/
# ============================================================

import time
import logging
import requests
from typing import List, Dict, Any, Optional
from src.utils.config_loader import config

logger = logging.getLogger("paper_search.s2")


BASE_URL = "https://api.semanticscholar.org/graph/v1"

# Common paper fields to request
PAPER_FIELDS = (
    "paperId,title,abstract,year,venue,publicationVenue,authors,"
    "citationCount,referenceCount,externalIds,url,openAccessPdf"
)

SEARCH_FIELDS = "paperId,title,abstract,year,venue,authors,citationCount,externalIds"


class SemanticScholarClient:
    """Semantic Scholar API client with rate limiting."""

    def __init__(self):
        self.api_key = config.get_api_key("S2_API_KEY")
        self.rate = config.get("retrieval", "rate_limit", "semantic_scholar", default=0.3)
        self._last_request = 0.0

    def _rate_limit(self) -> None:
        """Enforce rate limit."""
        elapsed = time.time() - self._last_request
        if elapsed < self.rate:
            time.sleep(self.rate - elapsed)
        self._last_request = time.time()

    def _headers(self) -> Dict[str, str]:
        h = {"Accept": "application/json"}
        if self.api_key:
            h["x-api-key"] = self.api_key
        return h

    def _get(self, url: str, params: Optional[Dict] = None) -> Dict:
        """
        GET request with retry logic.
        429 限流时原地等待窗口滑动（5 分钟），而不是快速失败——因为跳过后面照样限流。
        """
        self._rate_limit()
        for attempt in range(4):
            try:
                resp = requests.get(url, headers=self._headers(), params=params, timeout=30)
                if resp.status_code == 429:
                    wait = 20  # 等 20 秒重试（调用量已降，不会频繁持续限流）
                    logger.warning(
                        f"S2 rate limited (429), waiting {wait}s (attempt {attempt+1}/4)"
                    )
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                if attempt == 3:
                    raise RuntimeError(f"S2 API failed after 4 retries: {e}")
                time.sleep(2 ** attempt)
        return {}

    # ---- Search endpoints ----

    def keyword_search(self, query: str, limit: int = 50,
                       year_start: Optional[int] = None,
                       year_end: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Keyword search on Semantic Scholar.
        Corresponds to BM25 retrieval path.
        """
        url = f"{BASE_URL}/paper/search"
        params = {
            "query": query,
            "limit": min(limit, 100),
            "fields": SEARCH_FIELDS,
        }
        if year_start or year_end:
            params["year"] = f"{year_start or ''}-{year_end or ''}"

        result = self._get(url, params)
        papers = result.get("data", [])

        # Normalize to common schema
        return [self._normalize(p) for p in papers]

    def bulk_search(self, queries: List[str], limit_per_query: int = 20) -> List[Dict[str, Any]]:
        """Bulk keyword search (more efficient for multiple queries)."""
        all_papers = []
        seen_ids = set()
        for q in queries:
            papers = self.keyword_search(q, limit=limit_per_query)
            for p in papers:
                if p["paper_id"] not in seen_ids:
                    seen_ids.add(p["paper_id"])
                    all_papers.append(p)
        return all_papers

    # ---- Paper details ----

    def get_paper(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Get full paper details by S2 paperId."""
        url = f"{BASE_URL}/paper/{paper_id}"
        params = {"fields": PAPER_FIELDS}
        try:
            result = self._get(url, params)
            if "paperId" in result:
                return self._normalize(result)
        except RuntimeError:
            pass
        return None

    def get_papers_batch(self, paper_ids: List[str]) -> List[Dict[str, Any]]:
        """Batch get paper details (max 500 IDs)."""
        papers = []
        for i in range(0, len(paper_ids), 500):
            batch = paper_ids[i:i+500]
            url = f"{BASE_URL}/paper/batch"
            params = {"fields": PAPER_FIELDS}
            self._rate_limit()
            try:
                resp = requests.post(url, headers=self._headers(),
                                     params=params, json={"ids": batch}, timeout=60)
                resp.raise_for_status()
                results = resp.json()
                for r in results:
                    if r and "paperId" in r:
                        papers.append(self._normalize(r))
            except requests.RequestException:
                continue
        return papers

    # ---- Citation endpoints ----

    def get_citations(self, paper_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get papers that cite this paper (forward citations)."""
        url = f"{BASE_URL}/paper/{paper_id}/citations"
        params = {"limit": min(limit, 500), "fields": SEARCH_FIELDS}
        result = self._get(url, params)
        return [self._normalize(c["citingPaper"]) for c in result.get("data", [])
                if c.get("citingPaper")]

    def get_references(self, paper_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get papers cited by this paper (backward references)."""
        url = f"{BASE_URL}/paper/{paper_id}/references"
        params = {"limit": min(limit, 500), "fields": SEARCH_FIELDS}
        result = self._get(url, params)
        return [self._normalize(c["citedPaper"]) for c in result.get("data", [])
                if c.get("citedPaper")]

    # ---- Normalization ----

    def _normalize(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize S2 paper to common schema."""
        authors = paper.get("authors", [])
        return {
            "paper_id": paper.get("paperId") or "",
            "title": paper.get("title") or "",
            "abstract": paper.get("abstract") or "",
            "year": paper.get("year"),
            "venue": self._extract_venue(paper),
            "authors": [a.get("name") or "" for a in authors] if authors else [],
            "citation_count": paper.get("citationCount", 0),
            "reference_count": paper.get("referenceCount", 0),
            "arxiv_id": (paper.get("externalIds") or {}).get("ArXiv") or "",
            "doi": (paper.get("externalIds") or {}).get("DOI") or "",
            "url": paper.get("url") or "",
            "source": "semantic_scholar",
        }

    @staticmethod
    def _extract_venue(paper: Dict) -> Optional[str]:
        """Extract venue name from S2 paper metadata."""
        venue = paper.get("venue")
        # /paper/search 端点 venue 是字符串；某些端点可能是对象
        if isinstance(venue, dict):
            return venue.get("name") or venue.get("publicationVenue")
        if venue:
            return venue
        # 回退到 publicationVenue 对象
        pub_venue = paper.get("publicationVenue")
        if isinstance(pub_venue, dict):
            return pub_venue.get("name")
        return None


# Singleton
s2_client = SemanticScholarClient()
