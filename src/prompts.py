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

These are the specific tasks the User must accomplish. Assign them one at a time.
After the User has successfully completed a task, write on its own line:
GOAL DONE: <goal_id>

Do NOT wrap up or summarize the session until ALL tasks below are marked DONE.

### Pending tasks:
{chr(10).join(pending_lines) if pending_lines else "  (none)"}

### Completed tasks:
{chr(10).join(done_lines) if done_lines else "  (none yet)"}
"""

    # Free exploration mode: no goals and no eval_goals
    free_exploration_section = ""
    if not goals and not eval_goals:
        free_exploration_section = """
## Free Exploration Mode

You are systematically exploring ALL tools available on this platform. Follow this process:

1. **Start by asking the User to list all available tools.** Have them describe what each tool appears to do based on its name and description.
2. **Work through each tool one at a time.** For each tool:
   - Ask the User to call it with reasonable inputs
   - Ask the User to explain what the results mean
   - Probe whether the parameters were clear or confusing
   - Try edge cases: what happens with missing parameters, bad inputs, or empty results?
3. **After covering individual tools**, ask the User to combine tools for multi-step analytical workflows:
   - Can they answer a business question that requires chaining multiple tools?
   - Is the data from one tool easy to feed into another?
4. **Keep a mental checklist** of which tools have been tested. Do NOT wrap up until every tool has been tried at least once.
5. **Ask interpretive questions**: After each tool call, ask "Was that result what you expected?" or "What would you do with this data?"

You should be thorough and methodical. Cover every tool, every required parameter, and as many optional parameters as practical.
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
- Don't repeat observations you've already made. Here are your previous observations:

{previous_observations}

## Communication style

When speaking TO the User, be concise and task-focused. Keep your observations separate from your instructions to the User.
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
