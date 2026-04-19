"""Context window management with sliding window and summarization."""

from __future__ import annotations

import logging
from typing import Any

import tiktoken

logger = logging.getLogger(__name__)

# Gemma uses a different tokenizer, but tiktoken's cl100k_base gives a
# reasonable approximation for context budget tracking.
_ENCODER = None


def _get_encoder():
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER


def count_tokens(text: str) -> int:
    """Approximate token count for a string."""
    return len(_get_encoder().encode(text))


def count_message_tokens(messages: list[dict[str, Any]]) -> int:
    """Approximate token count for a list of chat messages."""
    total = 0
    for msg in messages:
        content = msg.get("content") or ""
        if isinstance(content, str):
            total += count_tokens(content) + 4  # role + formatting overhead
        # Tool call messages
        if "tool_calls" in msg:
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                total += count_tokens(fn.get("name", ""))
                args = fn.get("arguments", "")
                if isinstance(args, str):
                    total += count_tokens(args)
                else:
                    import json
                    total += count_tokens(json.dumps(args))
    return total


class ContextManager:
    """Manages a sliding window of messages with summarization."""

    def __init__(
        self,
        window_size: int = 60,
        max_tokens: int = 100_000,
        result_truncation: int = 8000,
    ):
        self.window_size = window_size
        self.max_tokens = max_tokens
        self.result_truncation = result_truncation
        self._summary: str = ""

    @property
    def summary(self) -> str:
        return self._summary

    @summary.setter
    def summary(self, value: str):
        self._summary = value

    def truncate_tool_result(self, text: str) -> str:
        """Truncate a tool result if it exceeds the limit."""
        if len(text) <= self.result_truncation:
            return text
        half = self.result_truncation // 2
        return (
            text[:half]
            + f"\n\n[... truncated {len(text) - self.result_truncation} chars ...]\n\n"
            + text[-half:]
        )

    def trim_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Trim messages to fit within the window and token budget.

        Returns a new list. The first message (system prompt) is always kept.
        Older messages beyond the window are dropped (their content should
        already have been captured by periodic summarization).
        """
        if len(messages) <= 1:
            return list(messages)

        # Always keep system prompt
        system = [messages[0]] if messages[0].get("role") == "system" else []
        rest = messages[len(system):]

        # Sliding window: keep last N messages
        if len(rest) > self.window_size:
            rest = rest[-self.window_size:]

        trimmed = system + rest

        # Check token budget; drop oldest non-system messages if over
        while count_message_tokens(trimmed) > self.max_tokens and len(trimmed) > 2:
            trimmed.pop(1)  # drop oldest after system prompt

        return trimmed

    def build_summary_prompt(
        self, dropped_messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Build a prompt to summarize dropped messages.

        Call this with the messages that fell off the window, then send to
        the LLM to get a condensed summary. Append the result to self._summary.
        """
        content_parts = []
        for msg in dropped_messages:
            role = msg.get("role", "?")
            text = msg.get("content", "")
            if text:
                content_parts.append(f"[{role}] {text[:500]}")

        transcript = "\n".join(content_parts)

        return [
            {
                "role": "system",
                "content": (
                    "Summarize this conversation excerpt in 2-3 sentences. "
                    "Focus on what was attempted, what tools were used, and "
                    "what the outcomes were. Be concise."
                ),
            },
            {"role": "user", "content": transcript},
        ]

    def coalesce_user_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Merge consecutive user messages and drop leading orphan assistants.

        Stricter chat templates (e.g. Mistral's) enforce user/assistant
        alternation starting with user after the system prompt, and return a
        400 on violations. Two situations can produce violations in the
        tester's history:

        1. Consecutive user messages — the tester turn appends several
           ephemeral user messages per round (goal status, coverage, variety
           hint, tool transcript). Merge them.
        2. Leading assistant after system — `_strip_stale_ephemeral` drops
           the ephemeral user messages that led to a past assistant reply,
           orphaning that assistant at the head of history. Drop leading
           assistants until the first non-system message is user (or tool,
           which is template-valid after tool_calls).

        Applied at the LLM boundary so stored history is unaffected.
        """
        # Drop leading orphan assistant messages after the system prompt
        result: list[dict[str, Any]] = []
        i = 0
        if messages and messages[0].get("role") == "system":
            result.append(messages[0])
            i = 1
        while i < len(messages) and messages[i].get("role") == "assistant":
            i += 1

        for msg in messages[i:]:
            prev = result[-1] if result else None
            if (
                prev is not None
                and msg.get("role") == "user"
                and prev.get("role") == "user"
                and isinstance(msg.get("content"), str)
                and isinstance(prev.get("content"), str)
            ):
                merged = dict(prev)
                merged["content"] = prev["content"] + "\n\n" + msg["content"]
                result[-1] = merged
            else:
                result.append(msg)
        return result

    def inject_summary(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Inject the running summary into the system message."""
        if not self._summary or not messages:
            return messages

        result = list(messages)
        if result[0].get("role") == "system":
            result[0] = dict(result[0])
            result[0]["content"] = (
                result[0]["content"]
                + "\n\n## Conversation Summary (older context)\n"
                + self._summary
            )
        return result
