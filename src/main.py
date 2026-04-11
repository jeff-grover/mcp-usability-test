"""CLI entry point for the MCP usability testing harness."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import yaml

from .llm_client import LLMConfig
from .mcp_bridge import MCPConfig
from .orchestrator import Orchestrator, OrchestratorConfig, Scenario


def load_config(path: str) -> dict:
    """Load and return the YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_scenario(path: str) -> Scenario:
    """Load a scenario from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return Scenario(
        name=data.get("name", Path(path).stem),
        description=data.get("description", ""),
        persona=data.get("persona", ""),
        goals=data.get("goals", []),
        tester_focus=data.get("tester_focus", []),
        max_rounds=data.get("max_rounds", 20),
    )


def build_orchestrator_config(raw: dict) -> OrchestratorConfig:
    """Convert raw YAML config into typed config objects."""
    llm_raw = raw.get("llm", {})
    llm = LLMConfig(
        base_url=llm_raw.get("base_url", "http://localhost:1234/v1"),
        api_key=llm_raw.get("api_key", "lm-studio"),
        model=llm_raw.get("model", "gemma-4-e4b"),
        temperature=llm_raw.get("temperature", 0.7),
        max_tokens=llm_raw.get("max_tokens", 2048),
        timeout_seconds=llm_raw.get("timeout_seconds", 120),
        max_retries=llm_raw.get("max_retries", 3),
        retry_delay_seconds=llm_raw.get("retry_delay_seconds", 5.0),
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

    if not config.scenarios:
        print("No scenarios configured. Add scenario files to config.yaml.", file=sys.stderr)
        sys.exit(1)

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
