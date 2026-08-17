# ============================================================
# src/utils/config_loader.py - Configuration and API key loading
# ============================================================

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """Singleton configuration loader from YAML + .env files."""

    _instance: Optional["Config"] = None
    _config: Dict[str, Any] = {}
    _api_keys: Dict[str, str] = {}

    def __new__(cls) -> "Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self, config_path: str = "config/config.yaml") -> "Config":
        """Load YAML config and .env API keys."""
        project_root = Path(__file__).resolve().parent.parent.parent
        self.project_root = project_root

        # --- Load YAML config ---
        yaml_path = project_root / config_path
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

        # --- Load config from .env file ---
        env_path = project_root / "config" / ".env"
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        self._api_keys[key.strip()] = value.strip()
        else:
            print(f"[WARN] .env file not found: {env_path}")
            print(f"       Copy config/.env.example to config/.env and fill in your values")

        # --- Resolve relative paths to absolute ---
        self._resolve_paths(project_root)

        return self

    def _resolve_paths(self, project_root: Path) -> None:
        """Convert relative paths in config to absolute paths."""
        dataset = self._config.get("dataset", {})
        for key in ["pasa_data_path", "asta_data_path", "asta_val_data_path",
                     "asta_normalizer_path"]:
            if key in dataset and not os.path.isabs(dataset[key]):
                dataset[key] = str(project_root / dataset[key])

        cache_dir = self._config.get("pipeline", {}).get("cache_dir", "cache")
        if not os.path.isabs(cache_dir):
            self._config["pipeline"]["cache_dir"] = str(project_root / cache_dir)

        log_file = self._config.get("logging", {}).get("file", "logs/pipeline.log")
        if not os.path.isabs(log_file):
            self._config["logging"]["file"] = str(project_root / log_file)

    # ---- Accessors ----

    @property
    def pipeline(self) -> Dict[str, Any]:
        return self._config.get("pipeline", {})

    @property
    def dataset(self) -> Dict[str, Any]:
        return self._config.get("dataset", {})

    @property
    def query_understanding(self) -> Dict[str, Any]:
        return self._config.get("query_understanding", {})

    @property
    def retrieval(self) -> Dict[str, Any]:
        return self._config.get("retrieval", {})

    @property
    def ranking(self) -> Dict[str, Any]:
        return self._config.get("ranking", {})

    @property
    def organization(self) -> Dict[str, Any]:
        return self._config.get("organization", {})

    @property
    def evaluation(self) -> Dict[str, Any]:
        return self._config.get("evaluation", {})

    def get_api_key(self, key_name: str) -> Optional[str]:
        """Get an API key, checking .env first then environment variables."""
        return self._api_keys.get(key_name) or os.environ.get(key_name)

    def get(self, *keys: str, default: Any = None) -> Any:
        """Deep get from config dict, e.g. config.get('retrieval', 'bm25', 'k1')."""
        value = self._config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
            if value is None:
                return default
        return value


# ---- Global singleton ----

config = Config()


def load_config(config_path: str = "config/config.yaml") -> Config:
    """Load configuration (call once at startup)."""
    return config.load(config_path)
