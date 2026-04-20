"""CLI entry point for the MCP usability testing harness."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import yaml

from typing import Any

from .llm_client import LLMConfig
from .mcp_bridge import MCPConfig
from .orchestrator import Orchestrator, OrchestratorConfig, Scenario


def load_config(path: str) -> dict:
    """Load and return the YAML config file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_scenario(path: str) -> Scenario:
    """Load a scenario from a YAML file."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Scenario(
        name=data.get("name", Path(path).stem),
        description=data.get("description", ""),
        persona=data.get("persona", ""),
        goals=data.get("goals", []),
        eval_goals=data.get("eval_goals", []),
        tester_focus=data.get("tester_focus", []),
        max_rounds=data.get("max_rounds", 20),
    )


def _normalize_dimensions(raw: Any) -> list[dict[str, str]]:
    """Accept a list of strings or {name, description} dicts; return list of dicts."""
    if not raw:
        return []
    if not isinstance(raw, list):
        raise ValueError(
            f"exploration_dimensions must be a list, got {type(raw).__name__}"
        )
    out: list[dict[str, str]] = []
    for item in raw:
        if isinstance(item, str):
            out.append({"name": item, "description": ""})
        elif isinstance(item, dict) and "name" in item:
            out.append({
                "name": str(item["name"]),
                "description": str(item.get("description", "")),
            })
        else:
            raise ValueError(
                f"exploration_dimensions items must be strings or "
                f"{{name, description}} dicts, got: {item!r}"
            )
    return out


def load_exploration_dimensions(spec: Any) -> list[dict[str, str]]:
    """Load dimensions from either an inline list or a path to a YAML file.

    A YAML file may contain a bare list or a mapping with a 'dimensions' key.
    """
    if spec is None:
        return []
    if isinstance(spec, str):
        with open(spec, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict) and "dimensions" in data:
            data = data["dimensions"]
        return _normalize_dimensions(data)
    return _normalize_dimensions(spec)


def free_exploration_scenario() -> Scenario:
    """Generate a built-in scenario for systematic tool exploration."""
    return Scenario(
        name="Free Exploration",
        description="Systematic exploration of all available MCP tools",
        persona=(
            "You are an analyst who has just been given access to a new "
            "analytics platform. You have never used these tools before. "
            "Explore every tool available to understand what the platform "
            "can do."
        ),
        goals=[],  # empty — triggers free exploration prompt in prompts.py
        eval_goals=[],
        tester_focus=[
            "Watch for: Are tool names self-explanatory?",
            "Watch for: Are parameters intuitive without reading documentation?",
            "Watch for: Are results easy to interpret?",
            "Watch for: Do error messages help the user recover?",
            "Watch for: Can tools be combined for multi-step analysis?",
        ],
        max_rounds=100_000,
    )


def _resolve_model_profile(
    model: str, profiles: dict[str, dict]
) -> dict:
    """Find the profile whose key is a substring of the model name.

    Longest matching key wins so specific entries (e.g. "gpt-oss-120b")
    beat generic ones ("gpt-oss"). Returns an empty dict when none match.
    """
    if not profiles:
        return {}
    matches = [(k, v) for k, v in profiles.items() if k.lower() in model.lower()]
    if not matches:
        return {}
    return max(matches, key=lambda kv: len(kv[0]))[1]


def build_orchestrator_config(raw: dict) -> OrchestratorConfig:
    """Convert raw YAML config into typed config objects."""
    llm_raw = raw.get("llm", {})
    model = llm_raw.get("model", "gemma-4-e4b")
    profile = _resolve_model_profile(model, raw.get("model_profiles", {}))

    def _pick(key: str, default):
        """Top-level llm.<key> wins; otherwise profile; otherwise default."""
        if key in llm_raw:
            return llm_raw[key]
        if key in profile:
            return profile[key]
        return default

    llm = LLMConfig(
        base_url=llm_raw.get("base_url", "http://localhost:1234/v1"),
        api_key=llm_raw.get("api_key", "lm-studio"),
        model=model,
        temperature=_pick("temperature", 0.7),
        max_tokens=_pick("max_tokens", 2048),
        timeout_seconds=llm_raw.get("timeout_seconds", 120),
        max_retries=llm_raw.get("max_retries", 3),
        retry_delay_seconds=llm_raw.get("retry_delay_seconds", 5.0),
        reasoning_effort=_pick("reasoning_effort", None),
        system_prompt_suffix=_pick("system_prompt_suffix", None),
        top_p=_pick("top_p", None),
        frequency_penalty=_pick("frequency_penalty", None),
        presence_penalty=_pick("presence_penalty", None),
        min_p=_pick("min_p", None),
        degeneration_char_threshold=llm_raw.get(
            "degeneration_char_threshold", 40
        ),
    )

    mcp_raw = raw.get("mcp_server", {})
    oauth_raw = mcp_raw.get("oauth", {})
    mcp = MCPConfig(
        transport=mcp_raw.get("transport", "streamable_http"),
        url=mcp_raw.get("url", "http://localhost:8080/mcp"),
        timeout=mcp_raw.get("timeout", 30.0),
        oauth=oauth_raw.get("enabled", False),
        oauth_callback_port=oauth_raw.get("callback_port", 8100),
        oauth_scopes=oauth_raw.get("scopes", ""),
        oauth_token_file=oauth_raw.get("token_file", "state/oauth_tokens.json"),
    )

    orch_raw = raw.get("orchestrator", {})
    output_raw = raw.get("output", {})

    # Load scenarios
    scenario_paths = raw.get("scenarios", [])
    scenarios = []
    for sp in scenario_paths:
        try:
            scenarios.append(load_scenario(sp))
        except FileNotFoundError:
            print(f"Warning: scenario file not found: {sp}", file=sys.stderr)

    dimensions = load_exploration_dimensions(raw.get("exploration_dimensions"))

    return OrchestratorConfig(
        llm=llm,
        mcp=mcp,
        scenarios=scenarios,
        max_tool_iterations=orch_raw.get("max_tool_iterations", 10),
        observation_interval=orch_raw.get("observation_interval", 3),
        context_window_size=orch_raw.get("context_window_size", 60),
        max_context_tokens=orch_raw.get("max_context_tokens", 100_000),
        result_truncation=orch_raw.get("result_truncation", 8000),
        observations_dir=output_raw.get("observations_dir", "observations"),
        state_dir=output_raw.get("state_dir", "state"),
        exploration_dimensions=dimensions,
    )


def main():
    parser = argparse.ArgumentParser(
        description="MCP Usability Testing Harness — dual-LLM test runner"
    )
    parser.add_argument(
        "-c", "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from saved session state",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Clear saved state and start fresh",
    )
    parser.add_argument(
        "--explore",
        action="store_true",
        help="Ignore configured scenarios and run in free exploration mode",
    )
    parser.add_argument(
        "--dimensions",
        default=None,
        help=(
            "Path to a YAML file listing exploration dimensions "
            "(overrides config.yaml's exploration_dimensions)"
        ),
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler("harness.log"),
            logging.StreamHandler() if args.verbose else logging.NullHandler(),
        ],
    )

    # Load config
    try:
        raw_config = load_config(args.config)
    except FileNotFoundError:
        print(f"Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    config = build_orchestrator_config(raw_config)

    if args.dimensions:
        config.exploration_dimensions = load_exploration_dimensions(args.dimensions)

    if args.explore or not config.scenarios:
        config.scenarios = [free_exploration_scenario()]
        print("Running in free exploration mode.", file=sys.stderr)
        if config.exploration_dimensions:
            names = ", ".join(d["name"] for d in config.exploration_dimensions)
            print(f"Exploration dimensions: {names}", file=sys.stderr)

    orch = Orchestrator(config)

    if args.fresh:
        orch.state_mgr.clear()

    # Run
    try:
        asyncio.run(orch.run(resume=args.resume))
    except KeyboardInterrupt:
        print("\nInterrupted. State has been saved — use --resume to continue.")


if __name__ == "__main__":
    main()
