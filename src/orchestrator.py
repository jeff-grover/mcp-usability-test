"""Main conversation loop — the 'crossover cable' between two LLM agents."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from typing import Any

from .llm_client import LLMClient, LLMConfig
from .mcp_bridge import MCPBridge, MCPConfig
from .observations import (
    ObservationWriter,
    parse_observations,
    strip_observations,
    parse_goal_completions,
    strip_goal_markers,
)
from .context import ContextManager, count_message_tokens
from .state import StateManager, SessionState
from .display import Display
from .prompts import (
    build_user_system_prompt,
    build_tester_system_prompt,
    build_observation_prompt,
)

logger = logging.getLogger(__name__)


# Prefixes identifying ephemeral per-round injections into the Tester's
# context. Every round we regenerate these messages, so prior copies are
# stale and should be dropped before appending the new ones — otherwise
# context bloats round-over-round (especially the tool transcript, which
# already contains the last N entries so old copies are fully redundant).
_STALE_TESTER_PREFIXES = (
    "## Coverage Status",
    "## Goal Status",
    "## Variety Hint",
    "Here is the User's recent tool interaction transcript:",
)


def _halve_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop the oldest half of non-system messages — last-resort recovery
    when prompt processing keeps timing out on a bloated context.

    Also drops orphan `role: "tool"` messages at the head of the kept tail,
    since the OpenAI-format API rejects tool replies whose paired
    tool_calls assistant message is no longer present.
    """
    if len(messages) <= 2:
        return list(messages)
    system = [messages[0]] if messages[0].get("role") == "system" else []
    rest = messages[len(system):]
    keep = max(4, len(rest) // 2)
    tail = rest[-keep:]
    i = 0
    while i < len(tail) and tail[i].get("role") == "tool":
        i += 1
    return system + tail[i:]


def _strip_stale_ephemeral(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove stale ephemeral user messages (coverage/goal/variety/transcript)."""
    return [
        m for m in messages
        if not (
            m.get("role") == "user"
            and isinstance(m.get("content"), str)
            and m["content"].lstrip().startswith(_STALE_TESTER_PREFIXES)
        )
    ]


@dataclass
class Scenario:
    name: str = ""
    description: str = ""
    persona: str = ""
    goals: list[str] = field(default_factory=list)
    eval_goals: list[dict[str, str]] = field(default_factory=list)
    tester_focus: list[str] = field(default_factory=list)
    max_rounds: int = 20


@dataclass
class OrchestratorConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    scenarios: list[Scenario] = field(default_factory=list)
    max_tool_iterations: int = 10
    observation_interval: int = 3
    context_window_size: int = 60
    max_context_tokens: int = 100_000
    result_truncation: int = 8000
    observations_dir: str = "observations"
    state_dir: str = "state"
    exploration_dimensions: list[dict[str, str]] = field(default_factory=list)


class Orchestrator:
    """Manages the dual-agent conversation loop."""

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.llm = LLMClient(config.llm)
        self.mcp = MCPBridge(config.mcp)
        self.obs_writer = ObservationWriter(config.observations_dir)
        self.state_mgr = StateManager(config.state_dir)
        self.display = Display()

        self.user_ctx = ContextManager(
            window_size=config.context_window_size,
            max_tokens=config.max_context_tokens,
            result_truncation=config.result_truncation,
        )
        self.tester_ctx = ContextManager(
            window_size=config.context_window_size,
            max_tokens=config.max_context_tokens,
            result_truncation=config.result_truncation,
        )

        self._tools: list[dict[str, Any]] = []
        self._tool_transcript: list[str] = []
        self._state = SessionState()

    async def run(self, resume: bool = False):
        """Run the full test session across all scenarios."""
        # Connect to MCP server
        self.display.info("Connecting to MCP server...")
        await self.mcp.connect()
        self._tools = await self.mcp.list_tools()
        self.display.info(f"Discovered {len(self._tools)} tools")

        try:
            if resume and self.state_mgr.has_saved_state():
                self._state = self.state_mgr.load()
                # Compact resumed state: older sessions accumulated stale
                # ephemeral messages each round. Strip them so prompt
                # processing doesn't blow up on resume.
                before = count_message_tokens(self._state.tester_messages)
                self._state.tester_messages = _strip_stale_ephemeral(
                    self._state.tester_messages
                )
                after = count_message_tokens(self._state.tester_messages)
                if before != after:
                    logger.info(
                        "Compacted tester context on resume: %d -> %d tokens",
                        before, after,
                    )
                self.obs_writer.set_session_id(self._state.session_id)
                self.user_ctx.summary = self._state.user_summary
                self.tester_ctx.summary = self._state.tester_summary
                self.display.info(
                    f"Resumed session {self._state.session_id} "
                    f"at scenario {self._state.scenario_index}, "
                    f"round {self._state.round_num}"
                )
            else:
                self._state = SessionState(
                    session_id=self.obs_writer.session_id,
                )
                from datetime import datetime
                self._state.started_at = datetime.now().isoformat()
                self.obs_writer.write_header({
                    "session_id": self._state.session_id,
                    "model": self.config.llm.model,
                    "mcp_url": self.config.mcp.url,
                    "started_at": self._state.started_at,
                })

            # Run scenarios
            scenarios = self.config.scenarios
            for i in range(self._state.scenario_index, len(scenarios)):
                self._state.scenario_index = i
                scenario = scenarios[i]
                await self._run_scenario(scenario)

            self.display.info("All scenarios complete!")

        finally:
            await self.mcp.disconnect()
            await self.llm.close()

    async def _run_scenario(self, scenario: Scenario):
        """Run a single scenario."""
        self.display.banner(scenario.name)
        self.display.status(scenario.description)

        # Build system prompts
        user_system = build_user_system_prompt(
            persona=scenario.persona,
            tools_json=self._tools,
        )
        tester_system = build_tester_system_prompt(
            scenario_name=scenario.name,
            goals=scenario.goals,
            tester_focus=scenario.tester_focus,
            previous_observations=self.obs_writer.get_previous_summaries(),
            eval_goals=scenario.eval_goals or None,
            completed_goal_ids=self._state.completed_goals,
            exploration_dimensions=self.config.exploration_dimensions or None,
        )

        # Initialize message histories (or use resumed state)
        if not self._state.user_messages:
            self._state.user_messages = [
                {"role": "system", "content": user_system}
            ]
        if not self._state.tester_messages:
            self._state.tester_messages = [
                {"role": "system", "content": tester_system}
            ]

        self._tool_transcript = []
        round_num = self._state.round_num
        rounds_without_completion = 0
        is_exploration = not scenario.eval_goals and not scenario.goals
        reached_max = True  # flipped False if we break early

        # Pick a fresh random variety offset at the start of each scenario so
        # the per-round dimension rotation starts on a different dimension
        # every run. Only re-seed on a fresh scenario (round 0), not a resume.
        if (
            not is_exploration
            and round_num == 0
            and self.config.exploration_dimensions
        ):
            self._state.variety_offset = random.randint(0, 10_000)

        while round_num < scenario.max_rounds:
            self._state.round_num = round_num
            self.display.banner(scenario.name, round_num)

            if is_exploration:
                self._update_exploration_phase()

            self.display.progress(
                self._build_progress_line(scenario, is_exploration)
            )

            # Snapshot goal-completion count BEFORE the tester turn,
            # since _tester_turn is where GOAL DONE markers get parsed.
            completed_count_before = len(self._state.completed_goals)

            # Phase 1: Tester gives direction (with pleasantry guard for exploration)
            tester_response = await self._tester_turn_with_guard(
                scenario, is_exploration
            )
            if tester_response is None:
                self.display.error("Tester produced no response, ending scenario")
                reached_max = False
                break

            # Phase 2: User works with tools
            await self._user_turn(tester_response, scenario)

            # Phase 3: Periodic deep observation
            if (round_num + 1) % self.config.observation_interval == 0:
                await self._observation_checkpoint(scenario)

            # --- Scenario-mode: goal tracking and stall-breaker ---
            if scenario.eval_goals:
                new_completions = (
                    len(self._state.completed_goals) - completed_count_before
                )

                if new_completions > 0:
                    rounds_without_completion = 0
                else:
                    rounds_without_completion += 1

                if self._all_goals_complete(scenario):
                    self.display.info("All evaluation goals completed!")
                    self._save_state()
                    reached_max = False
                    break

                # Stall-breaker chain
                if rounds_without_completion == 3:
                    self._inject_goal_reminder()
                elif rounds_without_completion == 6:
                    self._inject_force_advance(scenario)
                elif rounds_without_completion >= 9:
                    self._auto_force_advance(scenario)
                    rounds_without_completion = 0

            # --- Exploration mode: advance dimension + phase round counter ---
            if is_exploration:
                self._advance_exploration_counters()

            self._save_state()
            round_num += 1

        if reached_max and scenario.eval_goals:
            incomplete = self._incomplete_goals(scenario)
            if incomplete:
                self.display.info(
                    f"Reached max rounds. Incomplete goals: {', '.join(incomplete)}"
                )

        # Reset for next scenario
        self._state.round_num = 0
        self._state.user_messages = []
        self._state.tester_messages = []
        self._state.completed_goals = []
        self._state.exploration_phase = "Coverage"
        self._state.exploration_phase_round = 0
        self._state.current_dimension_index = 0
        self._state.dimension_round = 0
        self._state.variety_offset = 0
        self._save_state()

    async def _chat_with_recovery(
        self,
        history: list[dict[str, Any]],
        ctx: ContextManager,
        tools: list[dict[str, Any]] | None = None,
        is_tester: bool = False,
    ):
        """Call the LLM; on timeout/failure, aggressively compact and retry.

        Mutates `history` in place when compaction runs so subsequent turns
        don't immediately re-bloat. Recovery strategy: (1) strip stale
        ephemeral messages (tester only) and halve history, (2) halve again.
        """
        def _build_messages():
            trimmed = ctx.trim_messages(history)
            return ctx.inject_summary(trimmed)

        try:
            return await self.llm.chat(_build_messages(), tools=tools)
        except RuntimeError as e:
            logger.warning("LLM call failed (%s) — compacting and retrying", e)
            self.display.status("LLM timed out — compacting context and retrying...")

        # First recovery attempt: strip ephemeral + halve
        if is_tester:
            history[:] = _strip_stale_ephemeral(history)
        history[:] = _halve_history(history)
        try:
            return await self.llm.chat(_build_messages(), tools=tools)
        except RuntimeError as e:
            logger.warning("Retry after compaction still failed (%s) — halving again", e)

        # Second recovery attempt: halve once more
        history[:] = _halve_history(history)
        return await self.llm.chat(_build_messages(), tools=tools)

    async def _tester_turn(self, scenario: Scenario) -> str | None:
        """Have the Tester agent produce a task for the User."""
        is_exploration = not scenario.eval_goals and not scenario.goals

        # Drop prior ephemeral injections so we don't pile up stale copies
        # of the coverage/goal/variety/transcript messages each round.
        self._state.tester_messages = _strip_stale_ephemeral(
            self._state.tester_messages
        )

        # Inject goal status before transcript if eval_goals are active
        if scenario.eval_goals:
            self._state.tester_messages.append({
                "role": "user",
                "content": self._build_goal_status(scenario),
            })

        # In exploration mode, inject live coverage status
        if is_exploration:
            self._state.tester_messages.append({
                "role": "user",
                "content": self._build_coverage_status(),
            })

        # In scenario mode (eval_goals or plain goals), inject a rotating
        # variety hint drawn from the configured exploration_dimensions list,
        # so the Tester picks different clients/tests/categories each round
        # instead of latching onto whatever the User found first.
        if not is_exploration and self.config.exploration_dimensions:
            hint = self._build_variety_hint()
            if hint:
                self._state.tester_messages.append({
                    "role": "user",
                    "content": hint,
                })

        # Add tool transcript to tester's context
        if self._tool_transcript:
            transcript_text = "\n".join(self._tool_transcript[-20:])
            self._state.tester_messages.append({
                "role": "user",
                "content": (
                    "Here is the User's recent tool interaction transcript:\n\n"
                    + transcript_text
                    + "\n\nBased on this, provide the next task or follow-up."
                ),
            })

        response = await self._chat_with_recovery(
            self._state.tester_messages,
            self.tester_ctx,
            is_tester=True,
        )
        if not response.content:
            return None

        full_text = response.content

        # Extract observations
        observations = parse_observations(full_text)
        for obs in observations:
            self.obs_writer.write_observation(
                obs,
                scenario=scenario.name,
                round_num=self._state.round_num,
            )
            self.display.observation(obs.category, obs.severity, obs.description)
            self._state.observation_count += 1

        # Extract goal completions
        completed = parse_goal_completions(full_text)
        for goal_id in completed:
            if goal_id not in self._state.completed_goals:
                self._state.completed_goals.append(goal_id)
                self.display.info(f"Goal completed: {goal_id}")

        # Get user-facing content (without observations and goal markers)
        user_facing = strip_goal_markers(strip_observations(full_text))
        self.display.tester_message(user_facing)

        # Record in tester's history
        self._state.tester_messages.append({
            "role": "assistant",
            "content": full_text,
        })

        # Log to transcript
        self.state_mgr.append_transcript({
            "agent": "tester",
            "round": self._state.round_num,
            "content": full_text,
        })

        return user_facing

    async def _user_turn(self, tester_instruction: str, scenario: Scenario):
        """Have the User agent work toward the task, calling tools as needed."""
        self._state.user_messages.append({
            "role": "user",
            "content": tester_instruction,
        })

        for iteration in range(self.config.max_tool_iterations):
            response = await self._chat_with_recovery(
                self._state.user_messages,
                self.user_ctx,
                tools=self._tools,
            )

            # Handle tool calls
            if response.tool_calls:
                # Build the assistant message with tool_calls
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in response.tool_calls
                    ],
                }
                self._state.user_messages.append(assistant_msg)

                if response.content:
                    self.display.user_message(response.content)

                # Execute each tool call
                for tc in response.tool_calls:
                    self.display.tool_call(tc.name, tc.arguments)

                    if tc.name not in self._state.tools_called:
                        self._state.tools_called.append(tc.name)

                    raw_result = await self.mcp.call_tool(tc.name, tc.arguments)

                    # Auto-capture tool failures as observations, before any
                    # truncation, so post-run triage has the full args + error.
                    if raw_result.startswith("[TOOL ERROR]"):
                        self._record_tool_error(
                            tc.name, tc.arguments, raw_result, scenario
                        )

                    result = self.user_ctx.truncate_tool_result(raw_result)

                    self.display.tool_result(tc.name, result)

                    # Add tool result to messages
                    self._state.user_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

                    # Record in transcript for Tester
                    transcript_entry = (
                        f"TOOL CALL: {tc.name}({json.dumps(tc.arguments)})\n"
                        f"RESULT: {result[:2000]}"
                    )
                    self._tool_transcript.append(transcript_entry)

                    self.state_mgr.append_transcript({
                        "agent": "user",
                        "round": self._state.round_num,
                        "iteration": iteration,
                        "tool_call": tc.name,
                        "arguments": tc.arguments,
                        "result": result[:2000],
                    })

            else:
                # No tool calls — User is done with this step
                if response.content:
                    self.display.user_message(response.content)
                    self._state.user_messages.append({
                        "role": "assistant",
                        "content": response.content,
                    })
                    self._tool_transcript.append(
                        f"USER RESPONSE: {response.content[:1000]}"
                    )
                    self.state_mgr.append_transcript({
                        "agent": "user",
                        "round": self._state.round_num,
                        "iteration": iteration,
                        "content": response.content,
                    })
                break

        # Feed User's final response back to Tester
        if self._state.user_messages:
            last_user = self._state.user_messages[-1]
            if last_user.get("role") == "assistant" and last_user.get("content"):
                self._state.tester_messages.append({
                    "role": "user",
                    "content": f"The User responded: {last_user['content']}",
                })

    async def _observation_checkpoint(self, scenario: Scenario):
        """Dedicated observation pass — Tester reviews the full transcript."""
        if not self._tool_transcript:
            return

        self.display.status("Running observation checkpoint...")

        transcript = "\n\n".join(self._tool_transcript[-30:])
        prompt = build_observation_prompt(
            tool_transcript=transcript,
            scenario_name=scenario.name,
            previous_observations=self.obs_writer.get_previous_summaries(),
        )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Analyze the transcript and write observations."},
        ]

        response = await self.llm.chat(messages)
        if response.content:
            observations = parse_observations(response.content)
            for obs in observations:
                self.obs_writer.write_observation(
                    obs,
                    scenario=scenario.name,
                    round_num=self._state.round_num,
                )
                self.display.observation(
                    obs.category, obs.severity, obs.description
                )
                self._state.observation_count += 1

            self.display.status(
                f"Checkpoint: {len(observations)} new observation(s)"
            )

    def _all_goals_complete(self, scenario: Scenario) -> bool:
        """Check if all eval_goals have been marked complete."""
        required = {g["id"] for g in scenario.eval_goals}
        return required.issubset(set(self._state.completed_goals))

    def _incomplete_goals(self, scenario: Scenario) -> list[str]:
        """Return IDs of eval_goals not yet completed."""
        completed = set(self._state.completed_goals)
        return [g["id"] for g in scenario.eval_goals if g["id"] not in completed]

    def _build_goal_status(self, scenario: Scenario) -> str:
        """Build a goal status message to inject into tester context."""
        completed = set(self._state.completed_goals)
        pending = []
        done = []
        for g in scenario.eval_goals:
            if g["id"] in completed:
                done.append(g["id"])
            else:
                pending.append(f"[{g['id']}] {g['task']}")

        lines = ["## Goal Status"]
        if done:
            lines.append(f"Completed: {', '.join(done)}")
        if pending:
            lines.append("Pending tasks:")
            for p in pending:
                lines.append(f"  - {p}")
        else:
            lines.append(
                "All evaluation tasks are marked DONE. "
                "The orchestrator will end the scenario automatically — "
                "do not write closing remarks."
            )
        return "\n".join(lines)

    def _build_progress_line(self, scenario: Scenario, is_exploration: bool) -> str:
        """Compact per-round status line printed under the banner.

        Scenario mode:   [scenario 3/11 | goals 2/5 | round 7/35]
        Exploration:     [phase Combinations 4/15 | tools 12/15 | dim Tests 2/4 | round 42]
        """
        parts: list[str] = []

        if is_exploration:
            phase = self._state.exploration_phase
            if phase == "Coverage":
                parts.append(f"phase {phase}")
            else:
                phase_pos = self._state.exploration_phase_round + 1
                parts.append(f"phase {phase} {phase_pos}/{self._PHASE_ROUND_BUDGET}")

            total_tools = len(self._tools)
            called_tools = len(
                [t for t in self._state.tools_called if t]
            )
            parts.append(f"tools {called_tools}/{total_tools}")

            dim = self._current_dimension()
            if dim:
                dim_pos = self._state.dimension_round + 1
                parts.append(
                    f"dim {dim['name']} {dim_pos}/{self._DIMENSION_ROTATION_ROUNDS}"
                )

            parts.append(f"round {self._state.round_num + 1}")
        else:
            total_scenarios = len(self.config.scenarios)
            scen_pos = self._state.scenario_index + 1
            parts.append(f"scenario {scen_pos}/{total_scenarios}")

            if scenario.eval_goals:
                total_goals = len(scenario.eval_goals)
                done_goals = len(
                    [g for g in scenario.eval_goals
                     if g["id"] in self._state.completed_goals]
                )
                parts.append(f"goals {done_goals}/{total_goals}")

            parts.append(
                f"round {self._state.round_num + 1}/{scenario.max_rounds}"
            )

        return "[" + " | ".join(parts) + "]"

    def _build_variety_hint(self) -> str | None:
        """Pick a dimension for this round to steer the Tester toward variety.

        Uses a round-based rotation with a random per-scenario offset so that
        (a) different runs of the same scenario start on different dimensions
        and (b) every dimension gets touched within a single scenario.
        """
        dims = self.config.exploration_dimensions
        if not dims:
            return None
        idx = (self._state.round_num + self._state.variety_offset) % len(dims)
        active = dims[idx]
        name = active["name"]
        desc = (active.get("description") or "").strip().rstrip(".")
        dim_line = f"**{name}**"
        if desc:
            dim_line += f" — {desc}"

        other_names = [d["name"] for i, d in enumerate(dims) if i != idx]
        return (
            "## Variety Hint\n"
            f"When this round's task requires picking specific entities "
            f"(client, test, product category, date range, site tag, etc.), "
            f"prefer examples related to this dimension: {dim_line}. "
            "Do NOT reuse entities the User has already worked with this "
            "session if you can avoid it — pick a different one to broaden "
            "coverage. Other dimensions available for later rounds: "
            f"{', '.join(other_names)}."
        )

    def _inject_goal_reminder(self):
        """Nudge the tester to mark goals as done if it hasn't recently."""
        self._state.tester_messages.append({
            "role": "user",
            "content": (
                "REMINDER: When a task is complete, you must write "
                "GOAL DONE: <goal_id> on its own line. "
                "Check if any pending tasks have already been accomplished."
            ),
        })

    def _inject_force_advance(self, scenario: Scenario):
        """Stronger nudge: the current pending task has stalled for 6 rounds."""
        pending = self._incomplete_goals(scenario)
        if not pending:
            return
        target = pending[0]
        self._state.tester_messages.append({
            "role": "user",
            "content": (
                f"STALL WARNING: Task '{target}' has been attempted for 6 rounds "
                "without completion. If it cannot be accomplished with the "
                "available tools, immediately write GOAL DONE: " + target + " "
                "on its own line and move to the next task. Record the blocker "
                "as an OBS describing what was missing."
            ),
        })
        self.display.info(f"Stall nudge: forcing advance warning on '{target}'")

    _TOOL_ERROR_MAX_CHARS = 4000

    def _record_tool_error(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        raw_result: str,
        scenario: Scenario,
    ):
        """Auto-write a tool-error observation when an MCP call fails.

        The bridge prepends '[TOOL ERROR]' to every failed call (both
        result.isError and exceptions), which is the sentinel we trigger on.
        We capture the *pre-truncation* raw result so post-run triage always
        has the full error response, even if the LLM-facing context was
        truncated for token budget reasons.
        """
        from .observations import Observation

        err_text = raw_result
        if len(err_text) > self._TOOL_ERROR_MAX_CHARS:
            err_text = (
                err_text[: self._TOOL_ERROR_MAX_CHARS]
                + f"\n… (truncated at {self._TOOL_ERROR_MAX_CHARS} chars)"
            )

        try:
            args_json = json.dumps(arguments, indent=2, default=str)
        except (TypeError, ValueError):
            args_json = repr(arguments)

        description = (
            "Tool call FAILED — auto-captured by the orchestrator. "
            "Full input arguments and raw error response recorded below "
            "for post-run triage.\n\n"
            f"**Arguments:**\n```json\n{args_json}\n```\n\n"
            f"**Error response:**\n```\n{err_text}\n```"
        )

        obs = Observation(
            category="tool-error",
            severity="major",
            tool=tool_name,
            description=description,
        )
        self.obs_writer.write_observation(
            obs,
            scenario=scenario.name,
            round_num=self._state.round_num,
        )
        self._state.observation_count += 1

        # Show a compact line on the console so the failure is visible in
        # real-time; the full details live in the observation markdown file.
        summary = raw_result.replace("\n", " ")
        if len(summary) > 200:
            summary = summary[:200] + "…"
        self.display.observation(
            "tool-error",
            "major",
            f"{tool_name} failed: {summary}",
        )

    def _auto_force_advance(self, scenario: Scenario):
        """Hard force-advance: orchestrator itself marks the stalled goal DONE."""
        from .observations import Observation
        pending = self._incomplete_goals(scenario)
        if not pending:
            return
        target = pending[0]
        self._state.completed_goals.append(target)
        self.display.info(f"Auto force-advance: marked '{target}' DONE after 9 stalled rounds")

        # Record as a blocker finding
        obs = Observation(
            category="stalled-goal",
            severity="major",
            description=(
                f"Goal '{target}' was auto-advanced after 9 rounds without "
                "completion. The Tester and User could not find a path to "
                "accomplish this task with the available tools. This is a "
                "usability finding: either the task is ambiguous or the tool "
                "surface does not expose the required capability."
            ),
        )
        self.obs_writer.write_observation(
            obs,
            scenario=scenario.name,
            round_num=self._state.round_num,
        )
        self._state.observation_count += 1

        # Let the tester know the orchestrator advanced it
        self._state.tester_messages.append({
            "role": "user",
            "content": (
                f"The orchestrator has auto-advanced task '{target}' and recorded "
                "a stalled-goal observation. Move immediately to the next pending "
                "task. Do not discuss this advancement with the User."
            ),
        })

    # ------------------------------------------------------------------
    # Exploration-mode helpers
    # ------------------------------------------------------------------

    _PHASE_ORDER = ("Coverage", "Combinations", "EdgeCases", "Depth")
    _PHASE_ROUND_BUDGET = 15
    _DIMENSION_ROTATION_ROUNDS = 4
    _PLEASANTRY_PHRASES = (
        "wrap up", "thank you for", "session conclud", "testing conclud",
        "concludes our", "that concludes", "nothing more", "no further",
        "farewell", "this concludes",
    )

    def _update_exploration_phase(self):
        """Advance exploration phase state based on current coverage."""
        all_tool_names = {t.get("function", {}).get("name", "") for t in self._tools}
        all_tool_names.discard("")
        covered = set(self._state.tools_called)

        phase = self._state.exploration_phase

        if phase == "Coverage":
            if all_tool_names and all_tool_names.issubset(covered):
                self._state.exploration_phase = "Combinations"
                self._state.exploration_phase_round = 0
                self.display.info("Exploration phase: Coverage → Combinations")
        else:
            if self._state.exploration_phase_round >= self._PHASE_ROUND_BUDGET:
                idx = self._PHASE_ORDER.index(phase)
                # Skip Coverage on wrap-around since all tools are hit
                next_idx = idx + 1
                if next_idx >= len(self._PHASE_ORDER):
                    next_idx = 1  # back to Combinations
                self._state.exploration_phase = self._PHASE_ORDER[next_idx]
                self._state.exploration_phase_round = 0
                self.display.info(
                    f"Exploration phase: {phase} → {self._state.exploration_phase}"
                )

    def _advance_exploration_counters(self):
        """Increment per-round counters after a completed exploration round."""
        if self._state.exploration_phase != "Coverage":
            self._state.exploration_phase_round += 1

        if self.config.exploration_dimensions:
            self._state.dimension_round += 1
            if self._state.dimension_round >= self._DIMENSION_ROTATION_ROUNDS:
                self._state.dimension_round = 0
                self._state.current_dimension_index = (
                    (self._state.current_dimension_index + 1)
                    % len(self.config.exploration_dimensions)
                )
                active = self.config.exploration_dimensions[
                    self._state.current_dimension_index
                ]
                self.display.info(f"Dimension rotation → {active['name']}")

    def _current_dimension(self) -> dict[str, str] | None:
        dims = self.config.exploration_dimensions
        if not dims:
            return None
        idx = self._state.current_dimension_index % len(dims)
        return dims[idx]

    def _build_coverage_status(self) -> str:
        """Build the Coverage Status message injected each exploration round."""
        all_tool_names = sorted(
            t.get("function", {}).get("name", "") for t in self._tools
        )
        all_tool_names = [n for n in all_tool_names if n]
        covered = set(self._state.tools_called)
        uncovered = [n for n in all_tool_names if n not in covered]
        called = [n for n in all_tool_names if n in covered]

        lines = ["## Coverage Status"]
        lines.append(
            f"Tools called so far ({len(called)}/{len(all_tool_names)}): "
            + (", ".join(called) if called else "(none yet)")
        )
        if uncovered:
            lines.append(f"Tools never called: {', '.join(uncovered)}")
        else:
            lines.append("Tools never called: (all tools have been called at least once)")
        lines.append(f"Current phase: {self._state.exploration_phase}")

        dim = self._current_dimension()
        if dim:
            desc = dim.get("description", "").strip()
            dim_line = f"Current dimension: {dim['name']}"
            if desc:
                dim_line += f" — {desc}"
            lines.append(dim_line)

        lines.append(
            "REMINDER: This session does not end. If you run out of ideas, "
            "rotate dimensions or start a new phase. Never write closing remarks."
        )
        return "\n".join(lines)

    def _looks_like_pleasantry(self, text: str) -> bool:
        if not text:
            return False
        lower = text.lower()
        return any(phrase in lower for phrase in self._PLEASANTRY_PHRASES)

    async def _tester_turn_with_guard(
        self, scenario: Scenario, is_exploration: bool
    ) -> str | None:
        """_tester_turn wrapper that detects pleasantries in exploration mode
        and re-prompts the Tester with a concrete next-action directive.
        """
        response = await self._tester_turn(scenario)
        if not is_exploration or response is None:
            return response

        for _ in range(2):
            if not self._looks_like_pleasantry(response):
                return response

            self.display.info("Pleasantry detected — injecting forced next-action directive")
            dim = self._current_dimension()
            dim_text = f" Current dimension: {dim['name']}." if dim else ""
            uncovered = [
                t.get("function", {}).get("name", "")
                for t in self._tools
                if t.get("function", {}).get("name", "") not in set(self._state.tools_called)
            ]
            uncovered = [n for n in uncovered if n][:5]
            if uncovered:
                next_action = (
                    "Ask the User to call one of these untried tools with "
                    "realistic inputs: " + ", ".join(uncovered) + "."
                )
            else:
                next_action = (
                    "Pick one tool the User has already called and probe a new "
                    "edge case, invalid input, or varied argument."
                )
            self._state.tester_messages.append({
                "role": "user",
                "content": (
                    "Do not write closing remarks, farewells, or session summaries. "
                    "The session continues. Current phase: "
                    f"{self._state.exploration_phase}.{dim_text} {next_action} "
                    "Reply with a concrete task for the User, nothing else."
                ),
            })
            response = await self._tester_turn(scenario)
            if response is None:
                return None
        return response

    def _save_state(self):
        """Persist current state."""
        self._state.user_summary = self.user_ctx.summary
        self._state.tester_summary = self.tester_ctx.summary
        self.state_mgr.save(self._state)
