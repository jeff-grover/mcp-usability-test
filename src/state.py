"""Session persistence — save and load harness state for resumability."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SessionState:
    session_id: str = ""
    scenario_index: int = 0
    scenario_name: str = ""
    round_num: int = 0
    tester_messages: list[dict[str, Any]] = field(default_factory=list)
    user_messages: list[dict[str, Any]] = field(default_factory=list)
    tester_summary: str = ""
    user_summary: str = ""
    observation_count: int = 0
    completed_goals: list[str] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)
    exploration_phase: str = "Coverage"
    exploration_phase_round: int = 0
    current_dimension_index: int = 0
    dimension_round: int = 0
    variety_offset: int = 0
    started_at: str = ""
    updated_at: str = ""


class StateManager:
    """Handles saving/loading session state and the append-only transcript."""

    def __init__(self, state_dir: str | Path = "state"):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._state_file = self.state_dir / "session.json"
        self._transcript_file = self.state_dir / "transcript.jsonl"

    def has_saved_state(self) -> bool:
        return self._state_file.exists()

    def save(self, state: SessionState):
        """Save current session state to disk."""
        state.updated_at = datetime.now().isoformat()
        data = asdict(state)
        # Write atomically via tmp file
        tmp = self._state_file.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp.replace(self._state_file)
        logger.debug("Session state saved (round %d)", state.round_num)

    def load(self) -> SessionState:
        """Load session state from disk."""
        with open(self._state_file, encoding="utf-8") as f:
            data = json.load(f)
        state = SessionState(**data)
        logger.info(
            "Resumed session %s at round %d", state.session_id, state.round_num
        )
        return state

    def clear(self):
        """Remove saved state (start fresh)."""
        if self._state_file.exists():
            self._state_file.unlink()
        if self._transcript_file.exists():
            self._transcript_file.unlink()
        logger.info("Session state cleared")

    def append_transcript(self, entry: dict[str, Any]):
        """Append an entry to the append-only transcript log."""
        entry["_ts"] = datetime.now().isoformat()
        with open(self._transcript_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
