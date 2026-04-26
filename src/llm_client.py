"""Thin wrapper around the OpenAI SDK for LM Studio inference."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageToolCall,
)

logger = logging.getLogger(__name__)


def _sanitize_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Coerce `content: None` to `""` on every message.

    The OpenAI API accepts `content: null` on assistant messages with
    `tool_calls`, but several local-LLM Jinja chat templates apply
    `{{ message.content | string }}` and fail with "Cannot apply filter
    'string' to type: NullValue". An empty string renders identically for
    every template we've seen, so the coercion is safe.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.get("content") is None:
            m = dict(m)
            m["content"] = ""
        out.append(m)
    return out


@dataclass
class LLMConfig:
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "lm-studio"
    model: str = "gemma-4-e4b"
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout_seconds: int = 120
    max_retries: int = 3
    retry_delay_seconds: float = 5.0
    reasoning_effort: str | None = None  # "low" | "medium" | "high" (gpt-oss)
    system_prompt_suffix: str | None = None  # e.g. "/no_think" for Qwen3
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    min_p: float | None = None  # LM Studio extension, routed via extra_body
    degeneration_char_threshold: int = 40  # 0 disables detection


@dataclass
class LLMResponse:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: ChatCompletion | None = None


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


class LLMClient:
    """Async client for LM Studio via OpenAI-compatible API."""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        self._client = AsyncOpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout=self.config.timeout_seconds,
        )
        t = self.config.degeneration_char_threshold
        self._degen_re: re.Pattern[str] | None = (
            re.compile(rf"(.)\1{{{t - 1},}}", re.DOTALL) if t > 0 else None
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Send a chat completion request with optional tool definitions.

        Retries with exponential backoff on transient failures.
        Falls back to text-parsing for malformed tool calls.
        """
        last_error: Exception | None = None
        sanitized_messages = _sanitize_messages(messages)
        for attempt in range(self.config.max_retries):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.config.model,
                    "messages": sanitized_messages,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                }
                if self.config.top_p is not None:
                    kwargs["top_p"] = self.config.top_p
                if self.config.frequency_penalty is not None:
                    kwargs["frequency_penalty"] = self.config.frequency_penalty
                if self.config.presence_penalty is not None:
                    kwargs["presence_penalty"] = self.config.presence_penalty
                if self.config.min_p is not None:
                    kwargs["extra_body"] = {"min_p": self.config.min_p}
                if self.config.reasoning_effort:
                    kwargs["reasoning_effort"] = self.config.reasoning_effort
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"

                completion = await self._client.chat.completions.create(**kwargs)
                response = self._parse_response(completion, has_tools=bool(tools))
                self._check_degeneration(response.content)
                return response

            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay_seconds * (2**attempt)
                    logger.warning(
                        "LLM call failed (attempt %d/%d): %s. Retrying in %.1fs",
                        attempt + 1,
                        self.config.max_retries,
                        e,
                        delay,
                    )
                    await asyncio.sleep(delay)

        raise RuntimeError(
            f"LLM call failed after {self.config.max_retries} attempts"
        ) from last_error

    def _check_degeneration(self, content: str | None) -> None:
        """Raise RuntimeError if the model emitted a long repeated-char run.

        Local models (especially gpt-oss MLX) sometimes collapse into emitting
        the same token repeatedly (e.g. "@@@@..."). A clean 200 OK makes this
        invisible to the transport layer — we catch it here and raise so the
        chat() retry loop + orchestrator's compact-and-retry path kick in.
        """
        if not self._degen_re or not content:
            return
        match = self._degen_re.search(content)
        if not match:
            return
        run_len = match.end() - match.start()
        char = match.group(1)
        raise RuntimeError(
            f"LLM output degenerated: {run_len} consecutive {char!r} chars"
        )

    def _parse_response(self, completion: ChatCompletion, has_tools: bool = False) -> LLMResponse:
        """Parse a completion into an LLMResponse, with fallback parsing."""
        choice = completion.choices[0]
        message = choice.message
        content = message.content
        tool_calls: list[ToolCall] = []

        # Try structured tool_calls first
        if message.tool_calls:
            for tc in message.tool_calls:
                parsed = self._parse_tool_call(tc)
                if parsed:
                    tool_calls.append(parsed)

        # Fallback: if no structured tool calls, try to extract from text
        if not tool_calls and content and has_tools:
            tool_calls = self._extract_tool_calls_from_text(content)

        return LLMResponse(content=content, tool_calls=tool_calls, raw=completion)

    def _parse_tool_call(
        self, tc: ChatCompletionMessageToolCall
    ) -> ToolCall | None:
        """Parse a single structured tool call."""
        try:
            args = tc.function.arguments
            if isinstance(args, str):
                args = json.loads(args)
            return ToolCall(id=tc.id, name=tc.function.name, arguments=args)
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning("Failed to parse tool call %s: %s", tc.id, e)
            return None

    def _extract_tool_calls_from_text(self, text: str) -> list[ToolCall]:
        """Fallback: extract tool calls from response text.

        Handles cases where smaller models embed tool calls as JSON in
        their text response instead of using the structured format.
        """
        tool_calls: list[ToolCall] = []

        # Look for JSON blocks that look like function calls
        # Pattern: {"name": "...", "arguments": {...}}
        pattern = r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]*\}[^}]*\}'
        matches = re.findall(pattern, text, re.DOTALL)

        for i, match in enumerate(matches):
            try:
                parsed = json.loads(match)
                name = parsed.get("name", "")
                arguments = parsed.get("arguments", {})
                if name:
                    tool_calls.append(
                        ToolCall(
                            id=f"fallback_{i}",
                            name=name,
                            arguments=arguments,
                        )
                    )
            except json.JSONDecodeError:
                continue

        if tool_calls:
            logger.info(
                "Extracted %d tool call(s) from text via fallback parsing",
                len(tool_calls),
            )
        return tool_calls

    async def close(self):
        await self._client.close()
