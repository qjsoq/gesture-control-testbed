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
    # Шлях до моделі hand_landmarker.task (Tasks API). Потрібен — старий
    # mp.solutions у mediapipe >=0.10.3x вилучено, тож використовуємо Tasks.
    model_path: str = ""


class MediaPipeHandsExtractor:
    """21-точковий екстрактор кисті через MediaPipe **Tasks** `HandLandmarker`
    (а не застарілий `mp.solutions.hands`, якого нема в mediapipe 0.10.3x).

    Видає `Features(data=np.ndarray(21,3))` — той самий формат, що очікує
    `RuleBasedClassifier`, тож решта конвеєра без змін.
    """

    def __init__(self, cfg: MediaPipeHandsConfig) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            HandLandmarker,
            HandLandmarkerOptions,
            RunningMode,
        )

        if not cfg.model_path:
            raise ValueError(
                "mediapipe_hands: вкажи `model_path` до hand_landmarker.task"
            )

        self._cfg = cfg
        self._mp = mp
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=cfg.model_path),
            running_mode=RunningMode.IMAGE,
            num_hands=cfg.max_hands,
            min_hand_detection_confidence=cfg.min_detection_confidence,
            min_hand_presence_confidence=cfg.min_detection_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        logger.info(
            "mediapipe HandLandmarker (Tasks) initialized max=%d model=%s",
            cfg.max_hands, cfg.model_path,
        )

    def extract(self, frame: np.ndarray) -> Features | None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        if not result.hand_landmarks:
            return None
        lm = result.hand_landmarks[0]
        landmarks = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
        handedness = None
        if result.handedness:
            handedness = result.handedness[0][0].category_name
        return Features(
            data=landmarks,
            raw={
                "num_hands": len(result.hand_landmarks),
                "handedness": handedness,
            },
        )

    def __del__(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            pass
