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
        for attempt in range(self.config.max_retries):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.config.model,
                    "messages": messages,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                }
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"

                completion = await self._client.chat.completions.create(**kwargs)
                return self._parse_response(completion, has_tools=bool(tools))

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
