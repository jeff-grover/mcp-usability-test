"""Main conversation loop — the 'crossover cable' between two LLM agents."""

from __future__ import annotations

import json
import logging
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
        rounds_since_goal_completion = 0

        while round_num < scenario.max_rounds:
            self._state.round_num = round_num
            self.display.banner(scenario.name, round_num)

            # Phase 1: Tester gives direction
            tester_response = await self._tester_turn(scenario)
            if tester_response is None:
                self.display.error("Tester produced no response, ending scenario")
                break

            # Detect tester signaling session end (e.g. "Session Concluded")
            if self._is_session_end_signal(tester_response):
                self.display.info("Tester signaled session end, wrapping up scenario")
                self._save_state()
                break

            # Phase 2: User works with tools
            await self._user_turn(tester_response)

            # Phase 3: Periodic deep observation
            if (round_num + 1) % self.config.observation_interval == 0:
                await self._observation_checkpoint(scenario)

            # Check goal-based termination
            if scenario.eval_goals and self._all_goals_complete(scenario):
                self.display.info("All evaluation goals completed!")
                self._save_state()
                break

            # Nudge tester if no goals marked done after several rounds
            if scenario.eval_goals:
                rounds_since_goal_completion += 1
                if rounds_since_goal_completion >= 3:
                    self._inject_goal_reminder()
                    rounds_since_goal_completion = 0

            # Save state after each round
            self._save_state()
            round_num += 1
        else:
            if scenario.eval_goals:
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
        self._save_state()

    async def _tester_turn(self, scenario: Scenario) -> str | None:
        """Have the Tester agent produce a task for the User."""
        # Inject goal status before transcript if eval_goals are active
        if scenario.eval_goals:
            self._state.tester_messages.append({
                "role": "user",
                "content": self._build_goal_status(scenario),
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

        # Trim and add summary
        self._state.tester_messages = self.tester_ctx.trim_messages(
            self._state.tester_messages
        )
        messages = self.tester_ctx.inject_summary(self._state.tester_messages)

        response = await self.llm.chat(messages)
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

    async def _user_turn(self, tester_instruction: str):
        """Have the User agent work toward the task, calling tools as needed."""
        self._state.user_messages.append({
            "role": "user",
            "content": tester_instruction,
        })

        for iteration in range(self.config.max_tool_iterations):
            # Trim and add summary
            self._state.user_messages = self.user_ctx.trim_messages(
                self._state.user_messages
            )
            messages = self.user_ctx.inject_summary(self._state.user_messages)

            response = await self.llm.chat(messages, tools=self._tools)

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

                    result = await self.mcp.call_tool(tc.name, tc.arguments)
                    result = self.user_ctx.truncate_tool_result(result)

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
            lines.append("All tasks complete. You may wrap up the session.")
        return "\n".join(lines)

    @staticmethod
    def _is_session_end_signal(tester_response: str) -> bool:
        """Detect when the tester is signaling the session is over."""
        stripped = tester_response.strip().strip("()")
        # Short responses that are just wrap-up phrases
        if len(stripped) < 80:
            end_phrases = [
                "session concluded", "session complete", "session ended",
                "testing complete", "testing concluded",
                "all tasks complete", "all goals complete",
                "end of session", "wrap up",
            ]
            lower = stripped.lower()
            if any(phrase in lower for phrase in end_phrases):
                return True
        return False

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

    def _save_state(self):
        """Persist current state."""
        self._state.user_summary = self.user_ctx.summary
        self._state.tester_summary = self.tester_ctx.summary
        self.state_mgr.save(self._state)
