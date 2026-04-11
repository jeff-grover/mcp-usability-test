"""Rich terminal UI for live conversation display."""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table


class Display:
    """Color-coded terminal output for the dual-agent conversation."""

    def __init__(self):
        self.console = Console()

    def banner(self, scenario: str, round_num: int = 0):
        """Show scenario/round banner."""
        self.console.rule(
            f"[bold white] Scenario: {scenario} | Round: {round_num} ",
            style="bright_blue",
        )

    def tester_message(self, text: str):
        self.console.print(
            Panel(
                Text(text),
                title="[bold cyan]TESTER[/]",
                border_style="cyan",
                padding=(0, 1),
            )
        )

    def user_message(self, text: str):
        self.console.print(
            Panel(
                Text(text),
                title="[bold green]USER[/]",
                border_style="green",
                padding=(0, 1),
            )
        )

    def tool_call(self, name: str, arguments: dict[str, Any]):
        args_str = json.dumps(arguments, indent=2)
        if len(args_str) > 500:
            args_str = args_str[:500] + "\n..."
        self.console.print(
            Panel(
                f"[bold]{name}[/]\n{args_str}",
                title="[bold yellow]TOOL CALL[/]",
                border_style="yellow",
                padding=(0, 1),
            )
        )

    def tool_result(self, name: str, result: str):
        display_result = result
        if len(display_result) > 800:
            display_result = display_result[:800] + "\n... truncated for display"
        self.console.print(
            Panel(
                Text(display_result),
                title=f"[bold bright_black]RESULT: {name}[/]",
                border_style="bright_black",
                padding=(0, 1),
            )
        )

    def observation(self, category: str, severity: str, description: str):
        label = f"{severity.upper()} | {category}"
        self.console.print(
            Panel(
                Text(description),
                title=f"[bold magenta]OBSERVATION: {label}[/]",
                border_style="magenta",
                padding=(0, 1),
            )
        )

    def status(self, message: str):
        self.console.print(f"[dim]  ▸ {message}[/dim]")

    def error(self, message: str):
        self.console.print(f"[bold red]  ✗ {message}[/]")

    def info(self, message: str):
        self.console.print(f"[bold blue]  ℹ {message}[/]")

    def separator(self):
        self.console.rule(style="dim")
