from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Features:
    data: Any
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GesturePrediction:
    label: str
    confidence: float
    hand_x: float | None = None
    hand_y: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)
