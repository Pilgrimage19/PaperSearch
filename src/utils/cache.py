# ============================================================
# src/utils/cache.py - Simple JSON-based result caching
# ============================================================

import json
import hashlib
import os
from pathlib import Path
from typing import Any, Optional


class ResultCache:
    """File-based cache for intermediate pipeline results."""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

    def _key(self, step: str, params: dict) -> str:
        """Generate a deterministic cache key from step name + parameters."""
        canonical = json.dumps(params, sort_keys=True, ensure_ascii=False)
        digest = hashlib.md5(canonical.encode()).hexdigest()[:12]
        return f"{step}_{digest}.json"

    def get(self, step: str, params: dict) -> Optional[Any]:
        """Retrieve cached result, returns None if not found."""
        key = self._key(step, params)
        path = self.cache_dir / key
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def set(self, step: str, params: dict, data: Any) -> None:
        """Save result to cache."""
        key = self._key(step, params)
        path = self.cache_dir / key
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
