import logging
import time
from types import TracebackType
from typing import Iterator, Literal

import cv2
from pydantic import BaseModel, Field
import numpy as np

from gesture_control.sources.generic_source_config import SourceConfig
logger = logging.getLogger(__name__)


class WebcamConfig(SourceConfig):
    type: str = Field(default="webcam", frozen=True)
    device_index: int = 0
    resolution: tuple[int, int] | None = None
    fps: float | None = None


class WebcamSource:
    def __init__(self, cfg: WebcamConfig) -> None:
        self._cfg = cfg
        self._cap: cv2.VideoCapture | None = None
        self._index = 0

    def __enter__(self) -> "WebcamSource":
        self._cap = cv2.VideoCapture(self._cfg.device_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open webcam at index {self._cfg.device_index}")
        if self._cfg.resolution is not None:
            w, h = self._cfg.resolution
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        if self._cfg.fps is not None:
            self._cap.set(cv2.CAP_PROP_FPS, self._cfg.fps)
        logger.info("webcam opened device=%s", self._cfg.device_index)
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("webcam released")

    def __iter__(self) -> Iterator[np.ndarray]:
        if self._cap is None:
            raise RuntimeError("WebcamSource used outside its context manager")
        while True:
            ok, image = self._cap.read()
            if not ok:
                logger.warning("webcam returned no frame, stopping")
                return
            yield image
            self._index += 1
