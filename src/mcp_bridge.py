"""Bridge between the MCP server and OpenAI-format tool definitions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.client.sse import sse_client

logger = logging.getLogger(__name__)


@dataclass
class MCPConfig:
    transport: str = "streamable_http"  # "streamable_http" or "sse"
    url: str = "http://localhost:8080/mcp"
    timeout: float = 30.0
    oauth: bool = False
    oauth_callback_port: int = 8100
    oauth_scopes: str = ""
    oauth_token_file: str = "state/oauth_tokens.json"


class MCPBridge:
    """Connects to an MCP server and exposes tools in OpenAI format."""

    def __init__(self, config: MCPConfig | None = None):
        self.config = config or MCPConfig()
        self._session: ClientSession | None = None
        self._transport_ctx = None
        self._session_ctx = None
        self._tools_cache: list[dict[str, Any]] | None = None

    async def connect(self):
        """Establish connection to the MCP server."""
        # Build httpx client with OAuth if configured
        httpx_client = None
        if self.config.oauth:
            from .auth import create_oauth_provider
            from mcp.shared._httpx_utils import create_mcp_http_client

            oauth = create_oauth_provider(
                server_url=self.config.url,
                callback_port=self.config.oauth_callback_port,
                token_file=self.config.oauth_token_file,
                scopes=self.config.oauth_scopes,
            )
            httpx_client = create_mcp_http_client(auth=oauth)
            logger.info("OAuth authentication enabled")

        if self.config.transport == "streamable_http":
            kwargs = {"url": self.config.url}
            if httpx_client:
                kwargs["http_client"] = httpx_client
            self._transport_ctx = streamable_http_client(**kwargs)
        elif self.config.transport == "sse":
            kwargs = {
                "url": self.config.url,
                "sse_read_timeout": self.config.timeout,
            }
            if httpx_client:
                kwargs["httpx_client_factory"] = lambda **_: httpx_client
            self._transport_ctx = sse_client(**kwargs)
        else:
            raise ValueError(f"Unknown transport: {self.config.transport}")

        transport_result = await self._transport_ctx.__aenter__()
        # streamable_http_client yields (read, write, get_session_id);
        # sse_client yields (read, write)
        read_stream, write_stream = transport_result[0], transport_result[1]
        self._session_ctx = ClientSession(read_stream, write_stream)
        self._session = await self._session_ctx.__aenter__()
        await self._session.initialize()
        logger.info("Connected to MCP server at %s", self.config.url)

    async def disconnect(self):
        """Close the MCP connection."""
        if self._session_ctx:
            await self._session_ctx.__aexit__(None, None, None)
            self._session = None
        if self._transport_ctx:
            await self._transport_ctx.__aexit__(None, None, None)
        logger.info("Disconnected from MCP server")

    async def list_tools(self) -> list[dict[str, Any]]:
        """Discover tools and return them in OpenAI function-calling format."""
        if not self._session:
            raise RuntimeError("Not connected to MCP server")

        result = await self._session.list_tools()
        openai_tools = []
        for tool in result.tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema,
                },
            }
            openai_tools.append(openai_tool)

        self._tools_cache = openai_tools
        logger.info("Discovered %d MCP tools", len(openai_tools))
        return openai_tools

    async def call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> str:
        """Execute an MCP tool and return the result as text."""
        if not self._session:
            raise RuntimeError("Not connected to MCP server")

        logger.info("Calling tool: %s(%s)", name, arguments)
        try:
            result = await self._session.call_tool(name, arguments)

            # Extract text from result content
            texts = []
            for content in result.content:
                if hasattr(content, "text"):
                    texts.append(content.text)
                else:
                    texts.append(str(content))

            output = "\n".join(texts)

            if result.isError:
                logger.warning("Tool %s returned error: %s", name, output[:200])
                return f"[TOOL ERROR] {output}"

            return output

        except Exception as e:
            logger.error("Tool call failed: %s(%s) -> %s", name, arguments, e)
            return f"[TOOL ERROR] {type(e).__name__}: {e}"

    def get_cached_tools(self) -> list[dict[str, Any]]:
        """Return previously discovered tools without re-querying."""
        return self._tools_cache or []
