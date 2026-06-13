import logging
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel

from gesture_control.recognizers.gesture import Features, GesturePrediction

logger = logging.getLogger(__name__)


_DEFAULT_RULES: dict[int, str] = {
    0: "fist",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "open",
}


class RuleBasedClassifierConfig(BaseModel):
    type: Literal["rule_based"]
    rules_path: Path | None = None


class RuleBasedClassifier:
    def __init__(self, cfg: RuleBasedClassifierConfig) -> None:
        self._rules = self._load_rules(cfg.rules_path)
        logger.info("rule-based classifier loaded rules=%s", self._rules)

    @staticmethod
    def _load_rules(path: Path | None) -> dict[int, str]:
        if path is None:
            return dict(_DEFAULT_RULES)
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return {int(k): str(v) for k, v in data["fingers_to_gesture"].items()}

    def classify(self, features: Features) -> GesturePrediction | None:
        landmarks = features.data
        if not isinstance(landmarks, np.ndarray) or landmarks.shape != (21, 3):
            return None
        n = self._count_extended_fingers(landmarks)
        label = self._rules.get(n, "unknown")
        center = landmarks[:, :2].mean(axis=0)
        return GesturePrediction(
            label=label,
            confidence=1.0 if label != "unknown" else 0.0,
            hand_x=float(center[0]),
            hand_y=float(center[1]),
            raw={"extended_fingers": n},
        )

    @staticmethod
    def _count_extended_fingers(lm: np.ndarray) -> int:
        # Index, middle, ring, pinky: tip y above PIP joint y means extended
        # (image y grows downward, so smaller y = higher on the image).
        extended = 0
        for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            if lm[tip, 1] < lm[pip, 1]:
                extended += 1
        # Thumb: lateral check — tip x further from index MCP than IP joint x
        if abs(lm[4, 0] - lm[5, 0]) > abs(lm[3, 0] - lm[5, 0]):
            extended += 1
        return extended
