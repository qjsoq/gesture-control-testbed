from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True, slots=True)
class GestureEvent:
    label: str
    x_norm: float | None = None
    y_norm: float | None = None


class GestureSource(ABC):
    @abstractmethod
    def __iter__(self) -> Iterator[GestureEvent]: ...
