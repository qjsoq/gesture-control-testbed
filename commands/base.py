from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol


class VerticalServoLike(Protocol):
    def move_vertical(self, direction: str) -> None: ...


class HorizontalServoLike(Protocol):
    def move_horizontal(self, direction: str) -> None: ...


@dataclass(frozen=True, slots=True)
class CommandContext:
    """Лише дані події жесту — жодних пристроїв (Д4: «dataType»).

    Привід (receiver) тримає сама команда з моменту створення (патерн GoF
    «Команда»), тож контекст несе тільки нормалізовану позицію руки.
    """

    hand_x_norm: float | None = None
    hand_y_norm: float | None = None


class Command(ABC):
    @abstractmethod
    def execute(self, ctx: CommandContext) -> None: ...
