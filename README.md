# MCP Usability Testing Harness

A dual-LLM test harness that evaluates the usability of an MCP (Model Context Protocol) server by having two locally-running LLM agents converse with each other. One agent plays the role of a "user" interacting with the MCP tools, while the other acts as a "UX tester" observing the experience and recording structured observations.

Think of it as a crossover cable connecting two LLMs — one uses your MCP server's tools while the other watches, directs, and takes notes on what works and what doesn't.

## Why

Traditional testing checks whether tools *work*. This checks whether they're *usable* — are tool names intuitive? Are parameters clear? Do error messages help recovery? Are workflows efficient? The output is a prioritized list of UX improvements for your MCP server's tool interface.

## Architecture

```
┌──────────────┐                              ┌──────────────┐
│ Tester Agent │◄────── Orchestrator ────────►│  User Agent  │
│ (LM Studio)  │       (Python, async)        │ (LM Studio)  │
└──────────────┘              │               └──────────────┘
                              │ MCP (HTTP/SSE)
                         ┌────▼────┐
                         │   MCP   │
                         │ Server  │
                         └─────────┘
                              │
                    ┌─────────▼──────────┐
                    │ observations/*.md  │
                    │ state/session.json │
                    └────────────────────┘
```

**Asymmetric views:** The User agent sees MCP tool definitions and can call them. The Tester sees the User's tool calls and results as a transcript but never calls tools itself. The Tester's internal observations are never shown to the User.

### Conversation flow (per round)

1. **Tester turn** — Given scenario goals and the recent tool interaction transcript, the Tester issues a task to the User and optionally records observations.
2. **User turn (inner loop)** — The User works toward the goal, calling MCP tools as needed. The loop continues until no more tool calls or a max iteration limit is reached.
3. **Observation checkpoint** (every N rounds) — The Tester gets a dedicated prompt to review the full recent transcript and write detailed observations.

### Termination

Scenarios end when one of these conditions is met:
- **Goal completion** — If `eval_goals` are defined, the scenario ends when all goals are marked done by the Tester.
- **Session end signal** — The Tester signals wrap-up (e.g., "Session Concluded") and the orchestrator detects it.
- **Max rounds** — Safety cap to prevent runaway sessions (configurable per scenario).

## Requirements

- **Python 3.11+**
- **LM Studio** with a model loaded and the developer server enabled (port 1234 by default)
- An **MCP server** running and accessible over HTTP/SSE
- A model with function/tool-calling support (tested with Gemma 4 E4B-it, 128K context)

## Setup

```bash
# Clone and install
git clone <repo-url>
cd mcp-usability-test
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Configuration

Edit `config.yaml`:

```yaml
llm:
  base_url: "http://localhost:1234/v1"   # LM Studio endpoint
  api_key: "lm-studio"                   # LM Studio ignores this, but SDK requires it
  model: "gemma-4-e4b"                   # Model name as shown in LM Studio
  temperature: 0.7
  max_tokens: 2048                        # Max response tokens per LLM call
  timeout_seconds: 120                    # Local inference can be slow
  max_retries: 3
  retry_delay_seconds: 5.0

mcp_server:
  transport: "streamable_http"            # "streamable_http" or "sse"
  url: "http://localhost:8080/mcp"        # Your MCP server URL
  timeout: 30.0
  oauth:
    enabled: true                         # Set to false if no auth required
    callback_port: 8100                   # Local port for OAuth redirect
    scopes: ""                            # Space-separated scopes (empty = server default)
    token_file: "state/oauth_tokens.json" # Persisted across restarts

orchestrator:
  max_tool_iterations: 10    # Max tool calls per user turn
  observation_interval: 3    # Tester deep-observes every N rounds
  context_window_size: 60    # Messages kept in sliding window
  max_context_tokens: 100000 # Token budget per LLM call (128K model)
  result_truncation: 8000    # Max chars per tool result

scenarios:
  - scenarios/ab_test_analysis.yaml
  - scenarios/basic_exploration.yaml
  - scenarios/store_comparison.yaml
  - scenarios/error_recovery.yaml

output:
  observations_dir: "observations"
  state_dir: "state"
```

### OAuth authentication

If your MCP server requires OAuth (like Claude App connectors do), the harness handles the full authorization code flow with PKCE:

1. On first run, it opens your browser to the authorization URL.
2. You log in and approve access.
3. The browser redirects to `http://127.0.0.1:8100/callback`, where a local server captures the authorization code.
4. Tokens are exchanged and saved to `state/oauth_tokens.json`.
5. On subsequent runs, cached tokens are reused. The SDK auto-refreshes expired tokens if the server provides refresh tokens.

Set `oauth.enabled: false` to skip this for servers that don't require authentication.

## Running

```bash
# Start fresh
python -m src.main

# Resume a previous session
python -m src.main --resume

# Clear saved state and start over
python -m src.main --fresh

# Free exploration mode (ignore configured scenarios, systematically test all tools)
python -m src.main --explore

# Combine flags
python -m src.main --explore --fresh

# Verbose logging (also written to harness.log)
python -m src.main -v

# Custom config file
python -m src.main -c my_config.yaml
```

Press `Ctrl+C` at any time — state is saved after every round. Use `--resume` to pick up where you left off.

### Console output

The terminal shows a live color-coded conversation:

- **Cyan** — Tester messages (tasks and guidance)
- **Green** — User messages (reasoning and responses)
- **Yellow** — Tool calls (name and parameters)
- **Gray** — Tool results (truncated for display)
- **Magenta** — Observations as they're captured

## Scenarios

Scenarios are YAML files that define what the Tester directs the User to do:

```yaml
name: "A/B Test Result Analysis"
description: "Evaluate whether the tools effectively support A/B test analysis"
persona: >
  You are a marketing analyst at a regional grocery chain. Your manager wants
  a summary of recent in-store test performance for the quarterly review.

# General exploration goals (used for broad guidance)
goals:
  - "Discover the available analytics tools and understand the data model"
  - "Retrieve and interpret A/B test results"

# Evaluation goals with concrete tasks and completion tracking
eval_goals:
  - id: "find_active_tests"
    task: "Find all marketing tests currently running or recently completed"
    success_hint: "User discovers tests via search_tests or get_site_tests"
  - id: "test_summary"
    task: "Get detailed results for a specific test including sales lift"
    success_hint: "User retrieves test-level results with lift metrics"
  - id: "store_level_results"
    task: "Find per-store performance for a multi-store test"
    success_hint: "User gets store-level breakdowns within a test"

tester_focus:
  - "Watch for: Are test result metrics clearly labeled and interpretable?"
  - "Watch for: Can the user navigate from clients to sites to tests to results?"

max_rounds: 35
```

### Evaluation goals

When `eval_goals` are defined, the Tester tracks them explicitly:

- Each goal has an `id`, a `task` description, and an optional `success_hint`
- The Tester assigns goals one at a time and marks them done with `GOAL DONE: <goal_id>`
- The scenario ends automatically when all goals are complete
- `max_rounds` serves as a safety cap; if hit, incomplete goals are reported
- Goal status (pending/completed) is injected into the Tester's context each round

If `eval_goals` is omitted, the scenario runs for `max_rounds` using the general `goals` for guidance (original behavior).

### Included scenarios

- `ab_test_analysis.yaml` — Concrete A/B test analysis tasks with eval_goals
- `basic_exploration.yaml` — Tool discovery and first-use experience
- `store_comparison.yaml` — Multi-tool workflows and data comparison
- `error_recovery.yaml` — Error handling, typos, and edge cases

Add your own by creating a new YAML file and adding it to the `scenarios` list in `config.yaml`.

### Free exploration mode

When no scenarios are configured or the `--explore` flag is passed, the harness runs in **free exploration mode**. The Tester systematically walks the User through every available tool:

1. Lists all available tools and asks the User to describe each one
2. Tests each tool individually with reasonable inputs
3. Probes whether parameters are clear, results are interpretable, and errors are helpful
4. Attempts multi-tool workflows to test tool composability
5. Continues until all tools have been exercised

This is useful for initial assessment of an unfamiliar MCP server.

## Output

### Observations (`observations/`)

Structured markdown files with timestamped observations. The harness supports multiple observation formats for reliability with smaller LLMs:

**Single-line format** (preferred — most reliable with small models):
```
OBS: [major] [tool-naming] The search_tests tool name suggests text search but requires integer IDs
```

**Block format** (more detailed):
```
[OBSERVATION]
category: tool-naming
severity: major
tool: search_tests
description: The tool name suggests text search but requires integer IDs
[/OBSERVATION]
```

**Numbered list format** (used in observation checkpoints):
```
1. [major] The search_tests tool requires integer IDs but users try string names.
2. [minor] Error messages don't suggest alternative inputs.
```

All formats are automatically parsed. Categories: `tool-naming`, `parameter-clarity`, `error-messages`, `workflow-efficiency`, `missing-capability`, `data-format`, `documentation`, `discoverability`. Severities: `critical`, `major`, `minor`, `suggestion`.

### Transcript (`state/transcript.jsonl`)

Append-only JSONL file with every message, tool call, and result — timestamped for post-hoc analysis.

### Session state (`state/session.json`)

Current session state for resumability. Includes message windows, summaries, scenario progress, and completed goal tracking.

## Context window management

The harness is designed for long-running sessions. With Gemma 4 E4B's 128K context:

- **Sliding window**: Last 60 messages kept in full detail
- **Summarization**: Older messages condensed via LLM call when approaching 100K tokens
- **Full transcript**: Always persisted to `state/transcript.jsonl` regardless of window

## Project structure

```
mcp-usability-test/
├── config.yaml              # Runtime configuration
├── pyproject.toml            # Python project and dependencies
├── src/
│   ├── main.py               # CLI entry point
│   ├── orchestrator.py        # Main conversation loop
│   ├── llm_client.py          # LM Studio API wrapper (retries, fallback parsing)
│   ├── mcp_bridge.py          # MCP client, tool discovery, schema conversion
│   ├── auth.py                # OAuth flow (browser + callback server + token storage)
│   ├── observations.py        # Observation parser and markdown writer
│   ├── context.py             # Sliding window and summarization
│   ├── state.py               # Session persistence
│   ├── display.py             # Rich terminal UI
│   └── prompts.py             # System prompts for both agents
├── scenarios/                 # Test scenario definitions
├── observations/              # Output (generated, gitignored)
└── state/                     # Session state (generated, gitignored)
```
