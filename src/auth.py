"""OAuth authentication for MCP server connections.

Provides:
- File-based token storage (persists tokens across restarts)
- Browser-based redirect handler (opens auth URL in default browser)
- Local callback server (receives the OAuth redirect with auth code)
"""

from __future__ import annotations

import asyncio
import json
import logging
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.parse import parse_qs, urlparse

from pydantic import AnyUrl

from mcp.client.auth import OAuthClientProvider
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken

logger = logging.getLogger(__name__)

# Default callback port — must match the redirect_uri registered with the server
CALLBACK_PORT = 8100
CALLBACK_PATH = "/callback"


class FileTokenStorage:
    """Persists OAuth tokens and client info to a JSON file."""

    def __init__(self, path: str | Path = "state/oauth_tokens.json"):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> dict[str, Any]:
        if self._path.exists():
            return json.loads(self._path.read_text(encoding="utf-8"))
        return {}

    def _write(self, data: dict[str, Any]):
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        tmp.replace(self._path)

    async def get_tokens(self) -> OAuthToken | None:
        data = self._read()
        raw = data.get("tokens")
        if raw:
            return OAuthToken(**raw)
        return None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        data = self._read()
        data["tokens"] = tokens.model_dump()
        self._write(data)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        data = self._read()
        raw = data.get("client_info")
        if raw:
            return OAuthClientInformationFull(**raw)
        return None

    async def set_client_info(
        self, client_info: OAuthClientInformationFull
    ) -> None:
        data = self._read()
        data["client_info"] = client_info.model_dump()
        self._write(data)


class _CallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler that captures the OAuth callback."""

    auth_code: str | None = None
    state: str | None = None
    _event: asyncio.Event | None = None
    _loop: asyncio.AbstractEventLoop | None = None

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path != CALLBACK_PATH:
            self.send_response(404)
            self.end_headers()
            return

        params = parse_qs(parsed.query)
        _CallbackHandler.auth_code = params.get("code", [None])[0]
        _CallbackHandler.state = params.get("state", [None])[0]

        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(
            b"<html><body><h2>Authentication successful!</h2>"
            b"<p>You can close this tab and return to the terminal.</p>"
            b"</body></html>"
        )

        # Signal the async waiter — must use call_soon_threadsafe since
        # this handler runs in the HTTP server thread, not the event loop thread
        if _CallbackHandler._event and _CallbackHandler._loop:
            _CallbackHandler._loop.call_soon_threadsafe(
                _CallbackHandler._event.set
            )

    def log_message(self, format, *args):
        logger.debug("OAuth callback server: %s", format % args)


async def _open_browser(url: str) -> None:
    """Open the authorization URL in the default browser."""
    logger.info("Opening browser for OAuth: %s", url)
    webbrowser.open(url)


async def _wait_for_callback() -> tuple[str, str | None]:
    """Start a local HTTP server and wait for the OAuth callback.

    Returns (authorization_code, state).
    """
    _CallbackHandler.auth_code = None
    _CallbackHandler.state = None
    _CallbackHandler._event = asyncio.Event()
    _CallbackHandler._loop = asyncio.get_running_loop()

    server = HTTPServer(("127.0.0.1", CALLBACK_PORT), _CallbackHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    logger.info(
        "Waiting for OAuth callback on http://127.0.0.1:%d%s",
        CALLBACK_PORT,
        CALLBACK_PATH,
    )

    try:
        await _CallbackHandler._event.wait()
    finally:
        server.shutdown()
        thread.join(timeout=2)

    code = _CallbackHandler.auth_code
    state = _CallbackHandler._event = None
    state_val = _CallbackHandler.state

    if not code:
        raise RuntimeError("OAuth callback did not include an authorization code")

    return (code, state_val)


def create_oauth_provider(
    server_url: str,
    callback_port: int = CALLBACK_PORT,
    token_file: str = "state/oauth_tokens.json",
    scopes: str = "",
) -> OAuthClientProvider:
    """Create an OAuthClientProvider configured for browser-based auth.

    Args:
        server_url: The MCP server URL (used for metadata discovery).
        callback_port: Port for the local callback server.
        token_file: Path to persist tokens across restarts.
        scopes: Space-separated OAuth scopes to request.
    """
    global CALLBACK_PORT
    CALLBACK_PORT = callback_port

    redirect_uri = f"http://127.0.0.1:{callback_port}{CALLBACK_PATH}"

    client_metadata = OAuthClientMetadata(
        redirect_uris=[AnyUrl(redirect_uri)],
        token_endpoint_auth_method="none",  # public client
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        scope=scopes or None,
        client_name="MCP Usability Test Harness",
    )

    return OAuthClientProvider(
        server_url=server_url,
        client_metadata=client_metadata,
        storage=FileTokenStorage(token_file),
        redirect_handler=_open_browser,
        callback_handler=_wait_for_callback,
    )
