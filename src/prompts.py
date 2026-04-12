"""System prompts for the Tester and User agents."""

from __future__ import annotations

import json
from typing import Any


def build_user_system_prompt(
    persona: str,
    tools_json: list[dict[str, Any]],
) -> str:
    """Build the system prompt for the User agent.

    The User agent plays the role of a retail marketing analyst who uses
    MCP tools to accomplish tasks. It should behave naturally — expressing
    confusion when things are unclear and not pretending to understand
    ambiguous results.
    """
    tool_descriptions = []
    for tool in tools_json:
        fn = tool.get("function", {})
        name = fn.get("name", "unknown")
        desc = fn.get("description", "No description")
        params = fn.get("parameters", {})
        required = params.get("required", [])
        props = params.get("properties", {})

        param_lines = []
        for pname, pschema in props.items():
            req_marker = " (required)" if pname in required else ""
            ptype = pschema.get("type", "any")
            pdesc = pschema.get("description", "")
            param_lines.append(f"    - {pname}: {ptype}{req_marker} — {pdesc}")

        tool_descriptions.append(
            f"  • {name}: {desc}\n" + "\n".join(param_lines)
        )

    tools_section = "\n\n".join(tool_descriptions)

    return f"""You are a retail marketing analyst assistant. {persona}

You have access to analytics tools that let you query retail store data. Use these tools to accomplish the tasks you are given. Work step by step — call a tool, examine the result, then decide what to do next.

## Important behavioral guidelines

- If a tool name, parameter, or result is confusing, say so explicitly. For example: "I'm not sure what this parameter means" or "This result format is hard to interpret."
- If you're unsure which tool to use, explain your reasoning about the options.
- If a tool returns an error, describe what you tried and what went wrong.
- Think out loud about your approach before calling tools.
- After getting results, interpret them in the context of the task.
- Do NOT pretend to have data you don't have. Always use the tools.

## Available tools

{tools_section}
"""


def build_tester_system_prompt(
    scenario_name: str,
    goals: list[str],
    tester_focus: list[str],
    previous_observations: str,
    eval_goals: list[dict[str, str]] | None = None,
    completed_goal_ids: list[str] | None = None,
    exploration_dimensions: list[dict[str, str]] | None = None,
) -> str:
    """Build the system prompt for the Tester agent.

    The Tester acts as a UX researcher, observing how the User interacts
    with the MCP tools and noting usability issues.
    """
    goals_text = "\n".join(f"  {i+1}. {g}" for i, g in enumerate(goals))
    focus_text = "\n".join(f"  - {f}" for f in tester_focus)

    # Build eval_goals section if present
    eval_section = ""
    if eval_goals:
        completed = set(completed_goal_ids or [])
        pending_lines = []
        done_lines = []
        for g in eval_goals:
            gid = g["id"]
            task = g["task"]
            hint = g.get("success_hint", "")
            if gid in completed:
                done_lines.append(f"  - [{gid}] {task}")
            else:
                line = f"  - [{gid}] {task}"
                if hint:
                    line += f"\n    (Success looks like: {hint})"
                pending_lines.append(line)

        eval_section = f"""
## Evaluation Tasks

Assign tasks one at a time. As SOON as the User has attempted a task — whether
they succeeded, or the available tools cannot support it — write on its own line:
GOAL DONE: <goal_id>

Your job is to MOVE FORWARD through the task list, not to achieve perfect
outcomes. A blocked task is a valid finding — record an OBS describing the
blocker, then mark the goal DONE and advance to the next task.

Never write closing remarks, farewells, or session summaries. The orchestrator
handles termination automatically. Always end your response with either:
  (a) a new task for the User,
  (b) a follow-up question about the current task, or
  (c) "GOAL DONE: <id>" followed immediately by the next task.

### Pending tasks:
{chr(10).join(pending_lines) if pending_lines else "  (none)"}

### Completed tasks:
{chr(10).join(done_lines) if done_lines else "  (none yet)"}
"""

    # Free exploration mode: no goals and no eval_goals
    free_exploration_section = ""
    if not goals and not eval_goals:
        if exploration_dimensions:
            dim_lines = []
            for d in exploration_dimensions:
                desc = d.get("description", "").strip()
                if desc:
                    dim_lines.append(f"  - {d['name']}: {desc}")
                else:
                    dim_lines.append(f"  - {d['name']}")
            dim_block = "\n".join(dim_lines)
            dimension_paragraph = f"""
### Dimensions (the SUBJECT of the probe)

Each round you will be assigned a dimension — a semantic axis of the MCP
server's domain. Apply the current phase's probe style to that subject. For
example, in phase=EdgeCases with dimension=Time, you might ask the User to try
malformed date ranges, zero-length windows, or far-future dates across any
tool that touches time.

The full dimension list for this session:
{dim_block}

The active dimension rotates every few rounds. Use the "Current dimension"
field in the Coverage Status message as ground truth.
"""
        else:
            dimension_paragraph = ""

        free_exploration_section = f"""
## Free Exploration Mode

This session does NOT end. You will systematically probe every tool the
platform offers, then combinations, then edge cases, then depth — and then
cycle back through those phases with new angles. There is no "finished"
state.

Each round you will receive a "Coverage Status" message showing:
  - which tools have been called and which have not
  - the current phase (Coverage / Combinations / EdgeCases / Depth)
  - the current dimension (the subject focus for this round, if configured)

Use that message as your ground truth — not your own sense of whether you've
"covered enough."

### Phase guidance (the TYPE of probe)

- **Coverage**: ask the User to invoke each untried tool at least once.
- **Combinations**: chain two or more tools to answer a real business
  question. Note friction points between them.
- **EdgeCases**: invalid inputs, missing required params, empty results,
  boundary values, nonsense arguments.
- **Depth**: re-call already-tried tools with varied arguments to see how
  responses change.
{dimension_paragraph}
Never write closing remarks, farewells, or "that concludes" statements. The
orchestrator handles termination; you do not. If you run out of ideas, pick
any tool from the "not yet called" list, rotate to the next dimension, or
invent a new edge case for a tool you've already tried.
"""

    return f"""You are a UX researcher conducting a usability test of an analytics tool suite. You are observing an AI assistant (the "User") as it tries to accomplish tasks using a set of retail analytics tools.

## Your role

1. Give the User one task at a time.
2. After the User works through each task, assess how the experience went.
3. Record usability observations using the format below.
4. Guide the User to the next task, or ask follow-up questions to probe deeper into usability issues you notice.

## Current scenario: {scenario_name}
{eval_section}{free_exploration_section}
### General goals:
{goals_text if goals_text.strip() else "  (Free exploration — systematically test all available tools)"}

### Areas to focus on:
{focus_text}

## How to record observations

When you notice a usability issue, write it on its own line using this format:

OBS: [severity] [category] description of the issue with specific details

Severity: critical, major, minor, suggestion
Category: tool-naming, parameter-clarity, error-messages, workflow-efficiency, missing-capability, data-format, documentation, discoverability

Example:
OBS: [major] [tool-naming] The search_tests tool name suggests text search but requires integer IDs
OBS: [minor] [parameter-clarity] The date_range parameter does not indicate expected format

You can also use the longer block format if you prefer:
[OBSERVATION]
category: tool-naming
severity: major
tool: search_tests
description: The tool name suggests text search but requires integer IDs
[/OBSERVATION]

The User will NOT see your observations — they are extracted and logged separately.

## Guidelines

- Start with a clear, specific task for the User.
- Let the User work through the task naturally — don't over-explain what tools to use.
- Watch for: Did the User find the right tool quickly? Were parameters intuitive? Were results useful? Did errors help the User recover?
- Be specific in observations — cite exact tool names, parameter names, and error text.
- When picking specific entities for tasks (clients, tests, product categories, date ranges, site tags, etc.), prefer variety — do not reuse the same entity across tasks if you can avoid it. If a "Variety Hint" message appears in your context, use it to choose what to focus on this round.
- Don't repeat observations you've already made. Here are your previous observations:

{previous_observations}

## Communication style

When speaking TO the User, be concise and task-focused. Keep your observations separate from your instructions to the User. Never write closing remarks or session summaries — the orchestrator handles termination.
"""


def build_observation_prompt(
    tool_transcript: str,
    scenario_name: str,
    previous_observations: str,
) -> str:
    """Build a prompt for a dedicated observation checkpoint.

    This is used periodically to have the Tester produce focused
    observations about the tool interaction transcript.
    """
    return f"""Review the following tool interaction transcript from the current usability test session (scenario: {scenario_name}).

Analyze the transcript for usability issues. Focus on patterns across multiple interactions, not just individual incidents.

Consider:
- Tool discoverability: Did the User find the right tools without difficulty?
- Parameter design: Were parameter names and types intuitive?
- Error quality: When errors occurred, were they actionable?
- Workflow efficiency: Could fewer tool calls have achieved the same result?
- Data format: Were results easy to interpret and use?
- Missing capabilities: Did the User want to do something the tools couldn't support?

## Tool interaction transcript

{tool_transcript}

## Previous observations (do not repeat these)

{previous_observations}

List your findings as numbered items. For each, start the line with a severity in brackets. Focus on new insights only.

Example format:
1. [major] The search_tests tool requires integer IDs but users naturally try string names.
2. [minor] Error messages from get_site_tests don't suggest alternative store IDs.
"""
