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

1. **Context injection** — Before the Tester speaks, the orchestrator injects:
   - current goal status (eval-goal mode) **or** a coverage-status block showing called/uncalled tools, current phase, and current dimension (exploration mode)
   - the recent tool-interaction transcript
   - a **variety hint** rotating through the configured `exploration_dimensions` (scenario mode only)
2. **Tester turn** — Given all that context, the Tester issues a task to the User, records observations, and marks goals `GOAL DONE: <id>` as appropriate. In exploration mode a pleasantry guard re-prompts the Tester if it tries to write closing remarks.
3. **User turn (inner loop)** — The User works toward the goal, calling MCP tools as needed. The loop continues until no more tool calls or `max_tool_iterations` is reached. Tool names are recorded into the coverage tracker.
4. **Observation checkpoint** (every `observation_interval` rounds) — The Tester gets a dedicated prompt to review the full recent transcript and write detailed observations.
5. **Progress check** — Goal completion, stall-breaker chain (scenario mode), or phase/dimension counter advancement (exploration mode). See [Termination](#termination).

### Termination

Scenarios end when one of these conditions is met:
- **Goal completion** — If `eval_goals` are defined, the scenario ends when all goals are marked done by the Tester via `GOAL DONE: <goal_id>` markers.
- **Stall auto-advance** — If a goal sits stuck for too long, the orchestrator nudges the Tester at 3 rounds, injects a stronger "move on" directive at 6 rounds, and at 9 rounds force-marks the stalled goal DONE itself and records a `stalled-goal` observation describing the blocker. The philosophy: a blocked task is a *usability finding*, not a failure — record it and advance.
- **Max rounds** — Safety cap to prevent runaway sessions (configurable per scenario).

Exploration mode (`--explore`) is **unbounded** — it never terminates on its own. The scenario-level `max_rounds` is set to a high safety value (100,000) and the only clean stop is `Ctrl+C`. See [Free exploration mode](#free-exploration-mode) for details.

> **Note:** Earlier versions of the harness tried to detect session-end intent by fuzzy-matching phrases like "wrap up" or "all tasks complete" in the Tester's output. That heuristic fired on natural mid-scenario phrasing and ended sessions with goals still incomplete — it has been removed. Scenarios now terminate only on `_all_goals_complete` or `max_rounds`. The Tester is also prompted never to write closing remarks or session summaries.

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

Copy the example config, then edit your local copy (which is gitignored):

```bash
cp config.example.yaml config.yaml
```

`config.yaml`:

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

# Optional — semantic axes of your MCP server's domain to rotate through in
# both exploration mode (as probe subjects) and scenario mode (as per-round
# variety hints). Accepts an inline list or a path to an external YAML file.
# See the "Exploration dimensions" section for details.
exploration_dimensions:
  - name: Clients
    description: Probe tools that filter, scope, or group by client.
  - name: Time
    description: Try varied time ranges, granularities, comparisons, and boundaries.
  - name: Tests
    description: Exercise A/B test discovery, summary, and analysis tools.
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
# Start fresh — runs every scenario in config.yaml sequentially
python -m src.main

# Resume a previous session (picks up at the exact scenario + round it left off)
python -m src.main --resume

# Clear saved state and start over
python -m src.main --fresh

# Free exploration mode — ignore configured scenarios, run forever, and
# systematically probe every tool across all configured dimensions
python -m src.main --explore

# Override exploration dimensions from a file without editing config.yaml
python -m src.main --explore --dimensions custom_dimensions.yaml

# Combine flags
python -m src.main --explore --fresh

# Verbose logging (also written to harness.log)
python -m src.main -v

# Custom config file
python -m src.main -c my_config.yaml
```

### Flags

| Flag | Purpose |
| --- | --- |
| `-c, --config PATH` | Use a different config file (default `config.yaml`) |
| `--resume` | Resume from the saved session state in `state/session.json` |
| `--fresh` | Delete saved state before starting |
| `--explore` | Ignore configured scenarios and run free exploration mode forever |
| `--dimensions PATH` | Load exploration dimensions from an external YAML file, overriding `config.yaml`'s `exploration_dimensions` key |
| `-v, --verbose` | Write debug logs to `harness.log` and stderr |

Press `Ctrl+C` at any time — state is saved after every round. Use `--resume` to pick up where you left off. Exploration mode runs until you stop it; scenario mode stops automatically when all scenarios complete or each scenario hits its termination condition.

### Console output

The terminal shows a live color-coded conversation:

- **Cyan** — Tester messages (tasks and guidance)
- **Green** — User messages (reasoning and responses)
- **Yellow** — Tool calls (name and parameters)
- **Gray** — Tool results (truncated for display)
- **Magenta** — Observations as they're captured

## Scenarios

Scenarios are YAML files that define what the Tester directs the User to do. Each scenario encodes a persona, a set of goals, a list of things the Tester should watch for, and a safety cap on how long it can run. The orchestrator loads every scenario in the `scenarios:` list from `config.yaml` and runs them sequentially.

### Anatomy of a scenario file

```yaml
name: "A/B Test Result Analysis"
description: "Evaluate whether the tools effectively support A/B test analysis"

# The in-character briefing for the User agent. Written as a persona so the
# User behaves like a specific kind of analyst, not a generic assistant.
persona: >
  You are a marketing analyst at a regional grocery chain. Your manager wants
  a summary of recent in-store test performance for the quarterly review.

# Optional broad guidance for the Tester. Used when eval_goals is absent, or
# as supplemental context when both are set.
goals:
  - "Discover the available analytics tools and understand the data model"
  - "Retrieve and interpret A/B test results"

# The main concrete task list. Each entry has a stable id, a task description,
# and an optional success_hint that helps the Tester decide when to mark it
# done. The Tester assigns these one at a time.
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

# Things the Tester should actively pay attention to when writing observations.
# Think of these as "reviewer instructions" — not things the User does.
tester_focus:
  - "Watch for: Are test result metrics clearly labeled and interpretable?"
  - "Watch for: Can the user navigate from clients to sites to tests to results?"

# Safety cap. Most scenarios converge well before this under normal operation,
# but if a model gets stuck the stall-breaker force-advances goals first and
# this limit is only hit if something is really wrong.
max_rounds: 35
```

### Evaluation goals (the main mode)

When `eval_goals` are defined, the Tester tracks them explicitly:

- Each goal has an `id`, a `task` description, and an optional `success_hint`.
- The Tester assigns goals one at a time and marks them done with `GOAL DONE: <goal_id>` on its own line.
- The orchestrator parses those markers and updates pending/completed state, which is re-injected into the Tester's context every round.
- The scenario ends automatically when **all** `eval_goals` are marked done.
- **Stall-breaker chain** — if no new goal is marked complete for several rounds in a row, the orchestrator intervenes:
  - **3 rounds** — soft nudge reminding the Tester of the `GOAL DONE: <id>` marker format.
  - **6 rounds** — stronger "stall warning" directive naming the specific stalled goal and telling the Tester that a blocked task is a valid finding.
  - **9 rounds** — the orchestrator force-marks the stalled goal `DONE` itself and writes a `stalled-goal` observation describing the blocker. This guarantees forward progress.
- `max_rounds` is a safety cap; if hit, any incomplete goals are reported in the console but not re-tried.

If `eval_goals` is omitted, the scenario runs for `max_rounds` using the general `goals` for guidance only (no completion tracking, no stall-breaker). Most of the built-in scenarios use `eval_goals`.

### Authoring a new scenario — step by step

1. **Copy an existing scenario as a template.** `scenarios/ab_test_analysis.yaml` and `scenarios/lift_drivers.yaml` are good starting points for eval-goal-based scenarios.
2. **Write the persona.** Two or three sentences. Name a role, a workplace, and a concrete business need — this steers the User's language and the tools it reaches for first.
3. **Draft `eval_goals` as *outcomes*, not *tool calls*.** Say "Find a recently completed test with positive sales lift" rather than "Call `search_tests` with status=completed." Abstract goals reveal how discoverable the tools actually are. If you prescribe the tool name, you're testing obedience, not usability.
4. **Pick goal IDs that read well in logs.** They show up in `GOAL DONE: find_active_tests` lines in the transcript and in observation files — prefer lowercase snake_case that names the outcome.
5. **Write `success_hint`s as guidance for the Tester, not hard rules.** They help the Tester decide when to mark a goal done, but the Tester is free to mark a goal done even if the success hint was missed — as long as the User attempted the task. A blocked task counts as "attempted."
6. **Set `max_rounds` generously.** A typical well-scoped scenario with 3–6 `eval_goals` converges in 8–20 rounds. Set `max_rounds` to roughly 3× the number of eval_goals plus some headroom — 20 to 40 is a reasonable range. The stall-breaker usually kicks in well before this.
7. **Add the new file to `config.yaml`.**

   ```yaml
   scenarios:
     - scenarios/ab_test_analysis.yaml
     - scenarios/my_new_scenario.yaml   # <-- add here
   ```

8. **Sanity-check the scenario file loads** before running against an LLM:

   ```bash
   .venv/bin/python -c "
   from src.main import load_config, build_orchestrator_config
   c = build_orchestrator_config(load_config('config.yaml'))
   print(f'{len(c.scenarios)} scenarios loaded')
   for s in c.scenarios:
       print(f'  - {s.name}: {len(s.eval_goals)} eval_goals, max_rounds={s.max_rounds}')
   "
   ```

### Iterating on a scenario

Running the full config every time you tweak a scenario is slow. Two tricks:

**Run one scenario at a time.** Create a throwaway config that points at just the file you're iterating on, and pass it with `-c`:

```bash
# one-scenario config for fast iteration
cat > /tmp/debug.yaml <<'EOF'
llm: {base_url: "http://localhost:1234/v1", api_key: "lm-studio", model: "gemma-4-e4b-it"}
mcp_server:
  transport: "streamable_http"
  url: "https://your-mcp-server/mcp/http"
  oauth: {enabled: true, callback_port: 8100, token_file: "state/oauth_tokens.json"}
scenarios:
  - scenarios/my_new_scenario.yaml
output: {observations_dir: "observations", state_dir: "state"}
EOF

python -m src.main -c /tmp/debug.yaml --fresh
```

**Watch the transcript in another pane** while the run is active:

```bash
tail -f state/transcript.jsonl | python3 -c "
import json, sys
for line in sys.stdin:
    e = json.loads(line)
    agent = e.get('agent', '?')
    rnd = e.get('round', '?')
    if 'tool_call' in e:
        print(f'[r{rnd} user] {e[\"tool_call\"]}({e[\"arguments\"]})')
    elif 'content' in e:
        snippet = e['content'][:120].replace(chr(10), ' ')
        print(f'[r{rnd} {agent}] {snippet}')
"
```

Look for `GOAL DONE:` markers, stall nudges, force-advance events, and the `Variety Hint` blocks (see next section). If a goal never gets marked done, your `task` text is probably too vague or the tools genuinely can't support it — both are valid findings, but the second one is what you're hunting for.

**Re-run just the scenarios you changed** by renaming the bad `state/session.json` and running with `--fresh`. The previous session's observations stay in `observations/` so you can compare runs.

### Variety hints — rotating dimensions inside a scenario

A common failure mode of small models: once the User discovers *some* client ID, test ID, or product category, both agents latch onto it and reuse the same entity for every subsequent task. Scenarios end up exercising the same narrow slice of data across every run.

To break that, the orchestrator injects a **variety hint** into the Tester's context at the start of every round in scenario mode. The hint names one dimension from your configured `exploration_dimensions` list (see [Exploration dimensions](#exploration-dimensions)) and tells the Tester to prefer picking entities related to that dimension for this round's task — and specifically to avoid reusing entities the User has already worked with.

- **Rotation is deterministic within a scenario** — dimensions rotate by round index, so every dimension gets a turn within a single run.
- **A random offset is picked at each scenario's start** — so different runs of the same scenario file start on different dimensions, broadening coverage across repeated runs.
- **Variety hints only run in scenario mode** — exploration mode has its own dimension cycling (see below).
- **No scenario file changes needed** — variety hints use the `exploration_dimensions` defined at the top level of `config.yaml`, so one config applies to every scenario.

If `exploration_dimensions` is empty, no variety hint is injected and scenarios behave as before.

Example of what the Tester sees each round:

```
## Variety Hint
When this round's task requires picking specific entities (client, test,
product category, date range, site tag, etc.), prefer examples related to
this dimension: **Time** — Try varied time ranges, granularities,
comparisons, and boundaries. Do NOT reuse entities the User has already
worked with this session if you can avoid it — pick a different one to
broaden coverage. Other dimensions available for later rounds: Clients,
Products/Categories, Site tags, Tests, Custom charts, Lift explorers,
Rollout analyses.
```

### Included scenarios

- `ab_test_analysis.yaml` — Concrete A/B test analysis tasks with eval_goals
- `lift_drivers.yaml` — Category/brand lift drill-down and the "lifts don't add up" edge case
- `basic_exploration.yaml` — Tool discovery and first-use experience
- `store_comparison.yaml` — Multi-tool workflows and data comparison
- `error_recovery.yaml` — Error handling, typos, and edge cases
- `result_interpretation.yaml` — Interpreting test result metrics
- `rollout_planning.yaml` / `rollout_analysis.yaml` — Rollout workflows
- `store_availability.yaml` — Store-level data availability checks
- `stakeholder_report.yaml` — End-to-end reporting workflows
- `portfolio_monitoring.yaml` — Multi-test portfolio monitoring

Add your own by creating a new YAML file and adding it to the `scenarios:` list in `config.yaml`.

## Exploration dimensions

**Dimensions** are the semantic axes of your MCP server's domain — the "things worth varying" when you want broad coverage. For a retail analytics server, they might be *Clients*, *Products/Categories*, *Site tags*, *Time*, *Tests*, *Custom charts*, *Lift explorers*, and *Rollout analyses*. For a CI/CD MCP server they might be *Pipelines*, *Jobs*, *Deployments*, *Environments*, etc. For a calendar server, *Accounts*, *Calendars*, *Events*, *Attendees*, *Time zones*.

Dimensions are used in two places:

1. **Exploration mode** — a dimension is the *subject* of each probe. The phase state machine (Coverage / Combinations / EdgeCases / Depth) controls the *type* of probe; the dimension controls *what the probe is about*. Dimensions rotate every 4 rounds.
2. **Scenario mode** — a dimension is injected as a per-round **variety hint** that steers the Tester toward different clients/tests/categories across rounds and across repeated runs. See the previous section.

### Configuring dimensions

You have three options — pick whichever fits your workflow.

**Option 1: inline list in `config.yaml`** (simplest — edit one file):

```yaml
exploration_dimensions:
  - name: Clients
    description: Probe tools that filter, scope, or group by client.
  - name: Products/Categories
    description: Exercise product- and category-level filters and rollups.
  - name: Time
    description: Try varied time ranges, granularities, comparisons, and boundaries.
  - name: Tests
    description: Exercise A/B test discovery, summary, and analysis tools.
```

**Option 2: external file referenced from `config.yaml`** (keep `config.yaml` compact, share dimension files across projects):

```yaml
# config.yaml
exploration_dimensions: dimensions/retail_analytics.yaml
```

```yaml
# dimensions/retail_analytics.yaml
dimensions:
  - name: Clients
    description: Probe tools that filter, scope, or group by client.
  - name: Time
    description: Try varied time ranges and granularities.
```

The external file may contain either a bare list or a mapping with a `dimensions:` key — both forms are accepted.

**Option 3: CLI override** (test multiple dimension sets without editing anything):

```bash
python -m src.main --explore --dimensions dimensions/edge_cases_only.yaml
```

The `--dimensions` flag overrides whatever's in `config.yaml` for this run only.

### Dimension file formats

All three of these are valid:

```yaml
# Minimal — bare strings
- Clients
- Time
- Tests
```

```yaml
# Rich — name + description pairs (preferred)
- name: Clients
  description: Probe tools that filter, scope, or group by client.
- name: Time
  description: Try varied time ranges and granularities.
```

```yaml
# External file with a 'dimensions:' key wrapper
dimensions:
  - name: Clients
    description: ...
  - name: Time
    description: ...
```

Descriptions are optional but strongly recommended — they become the phrasing the Tester reads when deciding how to interpret the dimension, so a one-liner that names *what to try* beats a bare label every time.

### Good dimension design

- **Keep the list short (5–10 items).** Dimensions rotate — you want every one to get enough air time to matter.
- **Pick dimensions that name *things in the server's domain*, not *things the Tester should do*.** "Clients", "Tests", "Time" — yes. "Try edge cases", "Probe errors" — no, those are phases, not dimensions.
- **Make descriptions concrete.** "Try varied time ranges, granularities, comparisons, and boundaries" gives the Tester a menu to pick from; "Time-related stuff" doesn't.
- **Don't duplicate phases.** The phase state machine already covers Coverage/Combinations/EdgeCases/Depth — your dimensions shouldn't mirror that axis.
- **Iterate.** If a dimension never seems to produce distinctive Tester behavior in the transcript, rewrite its description to be more specific, or drop it.

## Free exploration mode

When no scenarios are configured or the `--explore` flag is passed, the harness runs in **free exploration mode**. This mode is designed to run indefinitely and systematically build coverage of your MCP server's entire tool surface — use it for initial assessment of an unfamiliar server, for overnight soak runs, or for broad regression testing after a server change.

```bash
python -m src.main --explore
```

Let it run. It will not terminate on its own — `Ctrl+C` is the only clean stop.

### How it works

Exploration mode is built on two orthogonal concepts:

- **Phases** — the *type* of probe (`Coverage` → `Combinations` → `EdgeCases` → `Depth` → back to `Combinations`, repeating forever)
- **Dimensions** — the *subject* of the probe, rotating every 4 rounds through your configured list

Each round, the orchestrator injects a **Coverage Status** block into the Tester's context containing:

- The list of tools called so far and the list of tools never called (ground truth — not the Tester's memory)
- The current phase
- The current dimension (name + description)
- A reminder that the session does not end and the Tester must never write closing remarks

The Tester uses that status as its source of truth and picks a concrete task for the User. The User calls tools. Phase and dimension counters advance. Repeat forever.

### Phase state machine

- **Coverage** — asks the User to invoke each untried tool at least once. Advances to Combinations only when every discovered tool has been called at least once.
- **Combinations** — chain two or more tools to answer a business question. Runs for 15 rounds.
- **EdgeCases** — invalid inputs, missing required params, boundary values, nonsense arguments. Runs for 15 rounds.
- **Depth** — re-call already-tried tools with varied arguments to see how responses change. Runs for 15 rounds.
- After Depth, the cycle wraps back to Combinations (Coverage is skipped on wrap-around since every tool has already been called once).

### Pleasantry guard

Small models love to wrap up. If the Tester's output contains any closing-remark phrase (`"wrap up"`, `"that concludes"`, `"thank you for"`, etc.), the orchestrator:

1. Discards the Tester's pleasantry response.
2. Injects a forced directive naming the current phase, the current dimension, and 5 specific untried tools to pick from.
3. Re-prompts the Tester for a concrete task.
4. Retries up to 2 times per round.

This breaks the "both agents politely try to end the session forever" loop that was the biggest failure mode of earlier versions.

### Command examples

```bash
# Overnight soak — runs until morning (or Ctrl+C)
python -m src.main --explore --fresh

# Resume yesterday's exploration session exactly where it stopped
python -m src.main --explore --resume

# Point at a dimensions file designed for edge-case hunting
python -m src.main --explore --dimensions dimensions/edge_cases.yaml

# Different MCP server, different dimensions — one-off run
python -m src.main -c configs/calendar_mcp.yaml --explore --dimensions dimensions/calendar.yaml

# Verbose mode — writes everything to harness.log for post-run analysis
python -m src.main --explore -v
```

### Tips for exploration runs

- **Configure dimensions before running `--explore`.** Without dimensions, the Tester still gets phase-based guidance but has no subject rotation, so it's much more likely to latch onto whatever it explored first.
- **Let it run long.** The interesting observations about combinations, edge cases, and depth come after the Coverage phase finishes — typically 20–40 rounds in for a server with ~15 tools. A one-hour run will surface more unique findings than ten ten-minute runs.
- **Watch the `Observations` count.** Each phase transition and dimension rotation shifts the Tester's focus, which usually produces a fresh batch of observations. If the observation rate drops to zero for 10+ rounds, your Tester model may be stuck — check `harness.log`.
- **Use `--fresh` for apples-to-apples comparisons.** If you're comparing two versions of an MCP server, run each with `--fresh` and identical dimensions so the phase/dimension state starts from the same place.
- **Scenarios and exploration are complementary.** Run the configured scenarios first to get targeted findings on known workflows, then switch to exploration for long-tail discovery. The observations pile up in the same `observations/` directory.

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

All formats are automatically parsed. Categories: `tool-naming`, `parameter-clarity`, `error-messages`, `workflow-efficiency`, `missing-capability`, `data-format`, `documentation`, `discoverability`, `tool-error`. Severities: `critical`, `major`, `minor`, `suggestion`.

### Auto-captured tool errors

When an MCP tool call actually fails (the bridge prepends `[TOOL ERROR]` to any response where `result.isError` is true or an exception escapes), the orchestrator automatically writes a structured `tool-error` observation containing:

- the tool name
- the full input arguments (as formatted JSON)
- the complete raw error response, captured **before** any result truncation for the LLM's context window — so even if the error message is larger than `result_truncation`, the observation file still has the whole thing (capped at 4000 chars with a "truncated" marker to keep the observations file sane)

This runs in parallel with the Tester's own observations — it guarantees the raw forensic data is always captured even when the Tester LLM forgets to comment, and it shows up as a magenta panel on the console in real time so you see failures as they happen. If you want to grep the observations file for every MCP-server bug the run surfaced: `grep -c 'Category: tool-error' observations/session_*.md`.

### Transcript (`state/transcript.jsonl`)

Append-only JSONL file with every message, tool call, and result — timestamped for post-hoc analysis.

### Session state (`state/session.json`)

Current session state for resumability. Includes message windows, summaries, scenario progress, completed goal tracking, tool-coverage set, exploration phase state (`exploration_phase`, `exploration_phase_round`), dimension rotation state (`current_dimension_index`, `dimension_round`), and the per-scenario `variety_offset` used to randomize the starting dimension of scenario-mode variety hints.

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
