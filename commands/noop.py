from __future__ import annotations

from commands.base import Command, CommandContext


class NoOpCommand(Command):
    def execute(self, ctx: CommandContext) -> None:
        return
