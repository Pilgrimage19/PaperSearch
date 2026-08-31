# ============================================================
# src/sources/openalex.py - OpenAlex API client
# Docs: https://docs.openalex.org/
# ============================================================

import re
import time
import requests
from typing import List, Dict, Any, Optional
from src.utils.config_loader import config


BASE_URL = "https://api.openalex.org"


class OpenAlexClient:
    """OpenAlex API client - used primarily for metadata queries."""

    def __init__(self):
        self.rate = config.get("retrieval", "rate_limit", "openalex", default=5.0)
        self.email = config.get_api_key("OPENALEX_EMAIL") or "research@example.com"
        self._last_request = 0.0

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request
        if elapsed < (1.0 / self.rate):
            time.sleep(1.0 / self.rate - elapsed)
        self._last_request = time.time()

    def _get(self, url: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """GET with mailto param + 429 retry."""
        params = dict(params or {})
        params["mailto"] = self.email
        self._rate_limit()
        for attempt in range(3):
            try:
                resp = requests.get(url, params=params, timeout=30)
                if resp.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                if attempt == 2:
                    print(f"[WARN] OpenAlex API error: {e}")
                    return None
                time.sleep(2 ** attempt)
        return None

    def search_works(self, query: str, limit: int = 50,
                     filters: Optional[Dict[str, Any]] = None,
                     year_start: Optional[int] = None,
                     year_end: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search works (papers) on OpenAlex.
        Supports structured filters for metadata queries.
        """
        params = {
            "search": query,
            "per_page": min(limit, 200),
        }

        # Build filter string
        filter_parts = []
        if year_start:
            filter_parts.append(f"publication_year:>{year_start - 1}")
        if year_end:
            filter_parts.append(f"publication_year:<{year_end + 1}")
        if filters:
            for key, value in filters.items():
                if isinstance(value, list):
                    filter_parts.append(f"{key}:{'|'.join(str(v) for v in value)}")
                else:
                    filter_parts.append(f"{key}:{value}")

        if filter_parts:
            params["filter"] = ",".join(filter_parts)

        data = self._get(f"{BASE_URL}/works", params=params)
        if data is None:
            return []
        return [self._normalize(w) for w in data.get("results", [])]

    def search_by_author(self, author_name: str, year_start: Optional[int] = None,
                         year_end: Optional[int] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Search papers by author name - for metadata queries."""
        return self.search_works(author_name, limit=limit,
                                 year_start=year_start, year_end=year_end)

    def get_work(self, openalex_id: str) -> Optional[Dict[str, Any]]:
        """Get a single work by OpenAlex ID."""
        data = self._get(f"{BASE_URL}/works/{openalex_id}")
        if data is None:
            return None
        return self._normalize(data)

    def get_citations(self, openalex_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get works that cite this work."""
        data = self._get(f"{BASE_URL}/works/{openalex_id}/cited_by",
                         params={"per_page": min(limit, 200)})
        if data is None:
            return []
        return [self._normalize(w) for w in data.get("results", [])]

    # ---- Normalization ----

    def _normalize(self, work: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize OpenAlex work to common schema."""
        authors = []
        for a in work.get("authorships", []):
            author_info = a.get("author", {})
            authors.append(author_info.get("display_name", ""))

        # Extract venue
        venue = None
        primary_loc = work.get("primary_location")
        if primary_loc:
            source = primary_loc.get("source") or {}
            venue = source.get("display_name")

        # Extract IDs
        ids = work.get("ids", {})
        arxiv_id = self._extract_arxiv_id(work)
        doi = ids.get("doi", "")
        if doi:
            doi = doi.replace("https://doi.org/", "")

        return {
            "paper_id": (work.get("id") or "").replace("https://openalex.org/", ""),
            "title": work.get("title") or "",
            "abstract": self._extract_abstract(work),
            "year": work.get("publication_year"),
            "venue": venue,
            "authors": authors,
            "citation_count": work.get("cited_by_count") or 0,
            "reference_count": work.get("referenced_works_count") or 0,
            "arxiv_id": arxiv_id,
            "doi": doi,
            "url": ids.get("openalex") or "",
            "source": "openalex",
        }

    @staticmethod
    def _extract_arxiv_id(work: Dict) -> str:
        """Extract arXiv ID from OpenAlex location URLs (strip version suffix)."""
        locs = [work.get("primary_location")] + (work.get("locations") or [])
        for loc in locs:
            if not isinstance(loc, dict):
                continue
            for key in ("landing_page_url", "pdf_url"):
                url = loc.get(key) or ""
                m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", url)
                if m:
                    return m.group(1)
        return ""

    @staticmethod
    def _extract_abstract(work: Dict) -> str:
        """OpenAlex stores abstract in an inverted index."""
        idx = work.get("abstract_inverted_index")
        if not idx:
            return ""
        try:
            # Reconstruct from inverted index
            word_positions = []
            for word, positions in idx.items():
                for pos in positions:
                    word_positions.append((pos, word))
            word_positions.sort()
            return " ".join(w for _, w in word_positions)
        except Exception:
            return ""


# Singleton
openalex_client = OpenAlexClient()
