# ============================================================
# src/llm/client.py - Unified LLM API client
#
# Features:
#   - Multi-provider: DashScope (Qwen), DeepSeek, OpenAI
#   - Automatic retry with exponential backoff
#   - Token usage tracking
#   - Structured JSON output with repair
# ============================================================

import json
import os
import time
import re
import logging
from typing import List, Dict, Optional, Any
from src.utils.config_loader import config
from src.llm.token_counter import token_counter, llm_stats

logger = logging.getLogger("paper_search.llm")


# Module-level state for LLM call logging
_llm_log_path = None  # None = not initialized, False = disabled, str = file path


def _write_llm_log(entry: Dict[str, Any]) -> None:
    """Append one LLM call record to the dedicated log file."""
    global _llm_log_path
    if _llm_log_path is None:
        if not config.get("llm", "log_prompts", default=False):
            _llm_log_path = False  # disabled
            return
        log_file = config.get("llm", "log_file", default="logs/llm_calls.log")
        project_root = getattr(config, "project_root", None)
        if project_root is None:
            _llm_log_path = False
            return
        path = project_root / log_file
        os.makedirs(path.parent, exist_ok=True)
        _llm_log_path = str(path)

    if _llm_log_path is False:
        return

    with open(_llm_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


class LLMClient:
    """
    Unified LLM client with:
    - Multi-provider support (dashscope / deepseek / openai)
    - Automatic retry on failure (max_retries, exponential backoff)
    - Token usage tracking via llm_stats
    - JSON output with automatic repair on parse failure
    """

    def __init__(
        self,
        model: str = "deepseek-v4-flash",
        provider: str = "dashscope",
        temperature: float = 0.0,
        max_tokens: int = 2000,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    # === Public API ===

    def chat(
        self,
        prompt: str,
        system: str = "",
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Single-turn chat. Returns raw response text.
        Auto-retries on network/API errors.
        """
        if messages is None:
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": prompt})
        else:
            msgs = messages

        return self._call_with_retry(msgs)

    def chat_json(
        self,
        prompt: str,
        system: str = "",
        repair_json: bool = True,
    ) -> Dict[str, Any]:
        """
        Chat and return parsed JSON.
        If repair_json=True, retries with JSON-fix prompt on parse failure.
        """
        for attempt in range(self.max_retries + 1):
            raw = self.chat(prompt, system=system)

            # Try to parse JSON
            try:
                return self._extract_json(raw)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"JSON parse failed (attempt {attempt+1}): {e}")
                if not repair_json or attempt >= self.max_retries:
                    raise
                # Retry with stricter JSON instruction
                prompt = (
                    prompt
                    + "\n\nIMPORTANT: Your last response was not valid JSON. "
                    + "Return ONLY a valid JSON object with no markdown fences, "
                    + "no trailing commas, and no extra text."
                )

        raise RuntimeError("Failed to get valid JSON after all retries")

    # === Provider implementations ===

    def _call_with_retry(self, messages: List[Dict[str, str]]) -> str:
        """Call LLM API with retry logic."""
        retries = 0
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                result = self._call_provider(messages)
                self._log_call(messages, result, retries=retries)
                return result
            except Exception as e:
                retries = attempt
                last_error = e
                logger.warning(f"LLM call failed (attempt {attempt+1}/{self.max_retries+1}): {e}")
                if attempt < self.max_retries:
                    wait = self.retry_delay * (2 ** attempt)
                    logger.info(f"  Retrying in {wait:.1f}s...")
                    time.sleep(wait)

        llm_stats.record(0, 0, model=self.model, retries=retries)
        raise RuntimeError(f"LLM call failed after {retries} retries: {last_error}")

    def _log_call(self, messages: List[Dict[str, str]], result: str, retries: int = 0) -> None:
        """Log one LLM call (prompt + response) to the dedicated file."""
        try:
            truncate = config.get("llm", "truncate_length", default=2000) or 0

            def _trunc(s: str) -> str:
                if truncate and len(s) > truncate:
                    return s[:truncate] + f"...[truncated {len(s) - truncate} chars]"
                return s

            entry = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": self.model,
                "provider": self.provider,
                "retries": retries,
                "prompt": _trunc(json.dumps(messages, ensure_ascii=False)),
                "response": _trunc(result),
            }
            _write_llm_log(entry)
        except Exception as e:
            logger.warning(f"Failed to write LLM call log: {e}")

    def _call_provider(self, messages: List[Dict[str, str]]) -> str:
        """Dispatch to the correct provider."""
        if self.provider == "dashscope":
            return self._call_dashscope(messages)
        elif self.provider == "deepseek":
            return self._call_deepseek(messages)
        elif self.provider == "openai":
            return self._call_openai(messages)
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def _call_dashscope(self, messages: List[Dict[str, str]]) -> str:
        """Alibaba DashScope (Qwen) API."""
        import requests
        api_key = config.get_api_key("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY not set in config/.env")

        resp = requests.post(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        # Track token usage
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", token_counter.count_messages(messages))
        completion_tokens = usage.get("completion_tokens", 0)
        llm_stats.record(prompt_tokens, completion_tokens, model=self.model)

        return data["choices"][0]["message"]["content"]

    def _call_deepseek(self, messages: List[Dict[str, str]]) -> str:
        """DeepSeek API."""
        import requests
        api_key = config.get_api_key("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY not set in config/.env")

        resp = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        usage = data.get("usage", {})
        llm_stats.record(
            usage.get("prompt_tokens", token_counter.count_messages(messages)),
            usage.get("completion_tokens", 0),
            model=self.model,
        )
        return data["choices"][0]["message"]["content"]

    def _call_openai(self, messages: List[Dict[str, str]]) -> str:
        """OpenAI API."""
        from openai import OpenAI
        api_key = config.get_api_key("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in config/.env")
        client = OpenAI(api_key=api_key)

        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        usage = resp.usage
        llm_stats.record(
            usage.prompt_tokens if usage else token_counter.count_messages(messages),
            usage.completion_tokens if usage else 0,
            model=self.model,
        )
        return resp.choices[0].message.content

    # === JSON extraction utilities ===

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response, handling markdown fences and common errors."""
        text = text.strip()

        # Remove markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON block between { and }
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse JSON from response: {text[:200]}...")


# === Global client factory ===

_default_client: Optional[LLMClient] = None


def get_client(
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> LLMClient:
    """
    Get or create the default LLM client.
    Uses config settings if model/provider not specified.
    """
    global _default_client

    m = model or config.get("query_understanding", "model", default="deepseek-v4-flash")
    p = provider or config.get("query_understanding", "model_provider", default="deepseek")

    if _default_client is None or _default_client.model != m or _default_client.provider != p:
        _default_client = LLMClient(model=m, provider=p)

    return _default_client


