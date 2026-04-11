"""Parse [OBSERVATION] blocks from Tester responses and write markdown."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class Observation:
    category: str = ""
    severity: str = ""
    tool: str = ""
    description: str = ""
    timestamp: str = ""
    scenario: str = ""
    round_num: int = 0


OBSERVATION_PATTERN = re.compile(
    r"\[OBSERVATION\](.*?)\[/OBSERVATION\]", re.DOTALL
)


def parse_observations(text: str) -> list[Observation]:
    """Extract structured observations from text."""
    observations: list[Observation] = []
    for match in OBSERVATION_PATTERN.finditer(text):
        block = match.group(1).strip()
        obs = Observation()
        for line in block.splitlines():
            line = line.strip()
            if ":" in line:
                key, _, value = line.partition(":")
                key = key.strip().lower()
                value = value.strip()
                if key == "category":
                    obs.category = value
                elif key == "severity":
                    obs.severity = value
                elif key == "tool":
                    obs.tool = value
                elif key == "description":
                    obs.description = value
            elif obs.description:
                # Continuation of description
                obs.description += " " + line
        observations.append(obs)
    return observations


def strip_observations(text: str) -> str:
    """Remove [OBSERVATION] blocks from text, returning user-facing content."""
    return OBSERVATION_PATTERN.sub("", text).strip()


class ObservationWriter:
    """Writes observations to timestamped markdown files."""

    def __init__(self, output_dir: str | Path = "observations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._file_path = self.output_dir / f"session_{self._session_id}.md"
        self._count = 0
        self._written: list[Observation] = []

    @property
    def session_id(self) -> str:
        return self._session_id

    def set_session_id(self, session_id: str):
        """Set session ID (for resuming)."""
        self._session_id = session_id
        self._file_path = self.output_dir / f"session_{session_id}.md"

    def write_header(self, metadata: dict[str, Any]):
        """Write the file header with session metadata."""
        with open(self._file_path, "w") as f:
            f.write("---\n")
            for k, v in metadata.items():
                f.write(f"{k}: {v}\n")
            f.write("---\n\n")
            f.write("# MCP Usability Test Observations\n\n")

    def write_observation(
        self,
        obs: Observation,
        scenario: str = "",
        round_num: int = 0,
    ):
        """Append a single observation to the markdown file."""
        obs.timestamp = datetime.now().isoformat()
        obs.scenario = scenario
        obs.round_num = round_num
        self._count += 1
        self._written.append(obs)

        with open(self._file_path, "a") as f:
            f.write(f"## Observation #{self._count}\n\n")
            f.write(f"- **Time**: {obs.timestamp}\n")
            f.write(f"- **Scenario**: {obs.scenario}\n")
            f.write(f"- **Round**: {obs.round_num}\n")
            f.write(f"- **Category**: {obs.category}\n")
            f.write(f"- **Severity**: {obs.severity}\n")
            if obs.tool:
                f.write(f"- **Tool**: {obs.tool}\n")
            f.write(f"\n{obs.description}\n\n---\n\n")

    def get_previous_summaries(self, last_n: int = 10) -> str:
        """Return a summary of recent observations for the Tester's context."""
        recent = self._written[-last_n:]
        if not recent:
            return "No observations recorded yet."

        lines = []
        for obs in recent:
            lines.append(
                f"- [{obs.severity}] {obs.category}"
                + (f" ({obs.tool})" if obs.tool else "")
                + f": {obs.description[:120]}"
            )
        return "\n".join(lines)
