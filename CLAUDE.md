# CLAUDE.md

## Project overview

This is a dual-LLM test harness for MCP server usability testing. Two LLM agents (both running locally via LM Studio) converse: one ("User") uses MCP tools as a retail marketing analyst, the other ("Tester") observes and records structured usability observations. The system under test is the MCP server's tool interface — not functionality, but UX.

## Key architecture decisions

- **Custom over AutoGen/CrewAI** — Only two agents with strict turn-taking and asymmetric context views. Frameworks add dependency complexity and quadratic token replay. The custom orchestrator is ~300 lines.
- **Asymmetric context** — The User sees tool definitions and can call tools. The Tester sees tool call transcripts but never calls tools. The Tester's `[OBSERVATION]` blocks are stripped before showing its messages to the User.
- **MCP transport is HTTP/SSE** (not stdio) — The MCP server runs independently; the harness connects via `streamable_http_client` or `sse_client` from the MCP SDK.
- **OAuth via MCP SDK** — The SDK provides `OAuthClientProvider` (PKCE, token refresh, metadata discovery, dynamic client registration). We provide `FileTokenStorage`, a `webbrowser.open()` redirect handler, and a local HTTP callback server in `src/auth.py`.
- **Model: Gemma 4 E4B-it** — 4.5B effective params, 128K context, native function-calling. Context management is tuned for this (60-message window, 100K token budget, 8K result truncation).

## Module responsibilities

- `orchestrator.py` — Main loop. Three-phase rounds: tester turn, user tool loop, observation checkpoint. Wires all other modules together.
- `llm_client.py` — Thin `openai.AsyncOpenAI` wrapper pointing at LM Studio. Exponential backoff retries. Fallback regex parser for malformed tool calls (common with smaller models).
- `mcp_bridge.py` — Connects to MCP server, discovers tools, converts MCP tool schemas to OpenAI function-calling format (JSON Schema is shared), executes tool calls. Holds a long-lived `ClientSession`.
- `auth.py` — OAuth browser flow. `FileTokenStorage` persists to `state/oauth_tokens.json`. Callback server on port 8100. Import path for `create_mcp_http_client` is `mcp.shared._httpx_utils` (internal, may change across SDK versions).
- `context.py` — Sliding window + LLM-based summarization. Token counting via `tiktoken` cl100k_base (approximation — Gemma uses a different tokenizer but this is close enough for budget tracking).
- `observations.py` — Regex extraction of `[OBSERVATION]...[/OBSERVATION]` blocks. Writes timestamped markdown files.
- `state.py` — Session persistence to `state/session.json` (atomic write via tmp file). Append-only transcript to `state/transcript.jsonl`.
- `display.py` — Rich panels: cyan=tester, green=user, yellow=tool calls, gray=results, magenta=observations.
- `prompts.py` — System prompts for both agents. User prompt includes dynamically injected tool definitions. Tester prompt includes previous observations to prevent repetition.

## Important internal details

- The `mcp.shared._httpx_utils.create_mcp_http_client` import path is a private module. If the MCP SDK changes this, check `mcp_bridge.py` connect method.
- The SSE client `httpx_client_factory` parameter expects a callable; when passing a pre-built OAuth httpx client, it's wrapped in a lambda. This may need adjustment if the SSE client API changes.
- Tool result truncation happens in `context.py` (`ContextManager.truncate_tool_result`), not in the MCP bridge — the bridge returns full results.
- The orchestrator's inner tool loop has a configurable max iterations (`max_tool_iterations`) to prevent runaway tool-calling.
- Observation extraction uses `strip_observations()` to remove `[OBSERVATION]` blocks from the Tester's response before passing the remainder to the User as instructions.

## Running tests / verification

No test suite yet. Manual verification:
1. `python3 -c "from src.main import load_config, build_orchestrator_config; c = build_orchestrator_config(load_config('config.yaml')); print(f'{len(c.scenarios)} scenarios loaded')"` — verifies config loading
2. Start LM Studio + MCP server, then `python -m src.main` for a live run
3. Check `observations/` for output, `state/transcript.jsonl` for full logs

## Files to be cautious with

- `config.yaml` — User-specific connection details (URLs, ports). Not secret but machine-specific.
- `state/oauth_tokens.json` — Contains OAuth access/refresh tokens. Gitignored.
- `state/session.json` — May contain conversation content. Gitignored.
