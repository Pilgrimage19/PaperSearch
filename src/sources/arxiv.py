# ============================================================
# src/sources/arxiv.py - arXiv API client
# Docs: https://info.arxiv.org/help/api/
# ============================================================

import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
from src.utils.config_loader import config


BASE_URL = "http://export.arxiv.org/api/query"

# arXiv category mapping for AI/CS subfields
CATEGORY_MAP = {
    "CS": "cs", "AI": "cs.AI", "ML": "cs.LG", "NLP": "cs.CL",
    "CV": "cs.CV", "RL": "cs.LG", "math": "math", "physics": "physics",
    "statistics": "stat", "biology": "q-bio",
}

NAMESPACES = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


class ArxivClient:
    """arXiv API client with rate limiting."""

    def __init__(self):
        self.rate = config.get("retrieval", "rate_limit", "arxiv", default=1.0)
        self._last_request = 0.0

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request
        if elapsed < self.rate:
            time.sleep(self.rate - elapsed)
        self._last_request = time.time()

    def search(self, query: str, max_results: int = 50,
               categories: Optional[List[str]] = None,
               year_start: Optional[int] = None,
               year_end: Optional[int] = None,
               sort_by: str = "relevance") -> List[Dict[str, Any]]:
        """
        Search arXiv by keywords.
        Corresponds to the PaSa-style site:arxiv.org search.
        """
        # Build query with category filters
        search_query = query
        if categories:
            cat_str = " OR ".join(f"cat:{c}" for c in categories)
            search_query = f"({search_query}) AND ({cat_str})"

        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": min(max_results, 100),
            "sortBy": sort_by,
            "sortOrder": "descending" if sort_by == "submittedDate" else "descending",
        }

        url = f"{BASE_URL}?{urllib.parse.urlencode(params)}"
        papers = self._fetch_and_parse(url)

        # Client-side year filter (arXiv API doesn't support it natively)
        if year_start:
            papers = [p for p in papers if p.get("year") and p["year"] >= year_start]
        if year_end:
            papers = [p for p in papers if p.get("year") and p["year"] <= year_end]

        return papers

    def search_by_id(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """Look up a specific paper by arXiv ID."""
        params = {
            "id_list": arxiv_id,
            "max_results": 1,
        }
        url = f"{BASE_URL}?{urllib.parse.urlencode(params)}"
        papers = self._fetch_and_parse(url)
        return papers[0] if papers else None

    def search_by_ids(self, arxiv_ids: List[str]) -> List[Dict[str, Any]]:
        """Batch lookup papers by arXiv IDs."""
        all_papers = []
        for i in range(0, len(arxiv_ids), 100):
            batch = arxiv_ids[i:i+100]
            params = {
                "id_list": ",".join(batch),
                "max_results": len(batch),
            }
            url = f"{BASE_URL}?{urllib.parse.urlencode(params)}"
            all_papers.extend(self._fetch_and_parse(url))
        return all_papers

    # ---- Internal ----

    def _fetch_and_parse(self, url: str) -> List[Dict[str, Any]]:
        """Fetch XML from arXiv API and parse into paper dicts."""
        self._rate_limit()
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "PaperSearch/0.1"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    xml_data = resp.read().decode("utf-8")
                return self._parse_response(xml_data)
            except Exception as e:
                if attempt == 2:
                    print(f"[WARN] arXiv API failed after 3 retries: {e}")
                    return []
                time.sleep(2 ** attempt)
        return []

    def _parse_response(self, xml_data: str) -> List[Dict[str, Any]]:
        """Parse arXiv Atom XML response."""
        root = ET.fromstring(xml_data)
        papers = []
        for entry in root.findall("atom:entry", NAMESPACES):
            papers.append(self._parse_entry(entry))
        return papers

    def _parse_entry(self, entry: ET.Element) -> Dict[str, Any]:
        """Parse a single arXiv Atom entry."""
        title_el = entry.find("atom:title", NAMESPACES)
        title = (title_el.text or "").strip().replace("\n", " ") if title_el is not None else ""

        summary_el = entry.find("atom:summary", NAMESPACES)
        abstract = (summary_el.text or "").strip().replace("\n", " ") if summary_el is not None else ""

        # Extract arXiv ID from the <id> tag
        id_el = entry.find("atom:id", NAMESPACES)
        arxiv_id = ""
        if id_el is not None and id_el.text:
            arxiv_id = id_el.text.split("/abs/")[-1]

        # Published date
        published_el = entry.find("atom:published", NAMESPACES)
        year = None
        if published_el is not None and published_el.text:
            year = int(published_el.text[:4])

        # Authors
        authors = []
        for author_el in entry.findall("atom:author", NAMESPACES):
            name_el = author_el.find("atom:name", NAMESPACES)
            if name_el is not None and name_el.text:
                authors.append(name_el.text.strip())

        # Categories
        categories = []
        for cat_el in entry.findall("arxiv:primary_category", NAMESPACES):
            if cat_el.get("term"):
                categories.append(cat_el.get("term"))
        for cat_el in entry.findall("atom:category", NAMESPACES):
            if cat_el.get("term") and cat_el.get("term") not in categories:
                categories.append(cat_el.get("term"))

        return {
            "paper_id": arxiv_id,
            "title": title,
            "abstract": abstract,
            "year": year,
            "venue": None,          # arXiv doesn't provide venue
            "authors": authors,
            "citation_count": 0,    # arXiv doesn't provide citations
            "reference_count": 0,
            "arxiv_id": arxiv_id,
            "doi": "",
            "url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
            "categories": categories,
            "source": "arxiv",
        }


# Singleton
arxiv_client = ArxivClient()
