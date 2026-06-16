from email.policy import default
import logging
import time
from pathlib import Path
from types import TracebackType
from typing import Iterator, Literal
from gesture_control.sources.generic_source_config import SourceConfig
from gesture_control.sources.registry import register_source

import cv2
from pydantic import BaseModel, Field

import numpy as np
logger = logging.getLogger(__name__)


class FileSourceConfig(SourceConfig):
    type: str = Field(default="file", frozen=True)
    path: Path
    loop: bool = False
    realtime: bool = True


class FileSource:
    def __init__(self, cfg: FileSourceConfig) -> None:
        self._cfg = cfg
        self._cap: cv2.VideoCapture | None = None
        self._index = 0
        self._fps: float = 0.0

    def __enter__(self) -> "FileSource":
        path = self._cfg.path
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {path}")
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        logger.info("file opened path=%s fps=%.2f", path, self._fps)
        return self

    def __exit__(self,  exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __iter__(self) -> Iterator[np.ndarray]:
        if self._cap is None:
            raise RuntimeError("FileSource used outside its context manager")
        period = 1.0 / self._fps if (self._cfg.realtime and self._fps > 0) else 0.0
        while True:
            t0 = time.monotonic()
            ok, image = self._cap.read()
            if not ok:
                if self._cfg.loop:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                return
            yield image
            if period > 0:
                dt = time.monotonic() - t0
                if dt < period:
                    time.sleep(period - dt)


# FileSourceConfig визначено до FileSource, тож реєструємо явно тут.
register_source("file", FileSource)(FileSourceConfig)
