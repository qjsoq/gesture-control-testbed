import logging
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel

from gesture_control.recognizers.gesture import Features

logger = logging.getLogger(__name__)


class MediaPipeHandsConfig(BaseModel):
    type: Literal["mediapipe_hands"]
    max_hands: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


class MediaPipeHandsExtractor:
    def __init__(self, cfg: MediaPipeHandsConfig) -> None:
        import mediapipe as mp

        self._cfg = cfg
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=cfg.max_hands,
            min_detection_confidence=cfg.min_detection_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        logger.info("mediapipe hands initialized max=%d", cfg.max_hands)

    def extract(self, frame: np.ndarray) -> Features | None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None
        lm = result.multi_hand_landmarks[0].landmark
        landmarks = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
        handedness = None
        if result.multi_handedness:
            handedness = result.multi_handedness[0].classification[0].label
        return Features(
            data=landmarks,
            raw={
                "num_hands": len(result.multi_hand_landmarks),
                "handedness": handedness,
            },
        )

    def __del__(self) -> None:
        try:
            self._hands.close()
        except Exception:
            pass
