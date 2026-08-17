# ============================================================
# src/llm/token_counter.py - Token counting utility
# ============================================================

from typing import List, Dict


class TokenCounter:
    """Simple token counter with optional tiktoken support."""

    def __init__(self):
        self._encoder = None
        try:
            import tiktoken
            self._encoder = tiktoken.get_encoding("cl100k_base")
        except (ImportError, Exception):
            self._encoder = None

    def count(self, text: str) -> int:
        """Count tokens in a text string."""
        if self._encoder is not None:
            try:
                return len(self._encoder.encode(text))
            except Exception:
                pass
        # Fallback: ~4 chars per token for English text
        return max(1, len(text) // 4)

    def count_messages(self, messages: List[Dict[str, str]]) -> int:
        """Count total tokens across a message list."""
        total = 0
        for msg in messages:
            for _, value in msg.items():
                total += self.count(str(value))
        # Add overhead for message formatting (~4 tokens per message)
        total += len(messages) * 4
        return total


# Global singleton
token_counter = TokenCounter()


class TokenStats:
    """Accumulate token usage statistics across LLM calls."""

    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0
        self.total_retries = 0
        self.total_cost_estimate = 0.0

    def record(self, prompt_tokens: int, completion_tokens: int,
               model: str = "", retries: int = 0) -> None:
        """Record token usage from one LLM call."""
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_calls += 1
        self.total_retries += retries

        # Rough cost estimate (per 1M tokens)
        costs = {
            "deepseek-v4-flash": (0.27, 1.10),  # DeepSeek pricing
            "qwen3-32b":         (0.5, 1.0),    # $0.5/1M input, $1/1M output
            "gpt-4o":            (2.50, 10.0),  # OpenAI pricing
        }
        input_cost, output_cost = costs.get(model, (0.5, 1.0))
        self.total_cost_estimate += (
            prompt_tokens / 1_000_000 * input_cost +
            completion_tokens / 1_000_000 * output_cost
        )

    def snapshot(self) -> Dict:
        """Return current statistics dict."""
        return {
            "total_calls": self.total_calls,
            "total_retries": self.total_retries,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "estimated_cost_usd": round(self.total_cost_estimate, 6),
        }

    def reset(self) -> None:
        """Reset all statistics."""
        self.__init__()


# Global stats tracker
llm_stats = TokenStats()


def get_token_stats() -> Dict:
    """Get accumulated token usage statistics."""
    return llm_stats.snapshot()


def reset_token_stats() -> None:
    """Reset token statistics (call at start of pipeline run)."""
    llm_stats.reset()
