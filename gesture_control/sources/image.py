from __future__ import annotations
import logging
import time
from pathlib import Path
from types import TracebackType
from typing import Iterator, Literal

import cv2
from pydantic import BaseModel, Field
import numpy as np
from gesture_control.sources.registry import register_source
from gesture_control.sources.generic_source_config import SourceConfig
logger = logging.getLogger(__name__)

class ImageSource:
    def __init__(self, cfg: ImageSourceConfig) -> None:
        self._cfg = cfg
        self._image = None
        self._index = 0

    def __enter__(self) -> "ImageSource":
        if not self._cfg.path.exists():
            raise FileNotFoundError(f"Image not found: {self._cfg.path}")
        img = cv2.imread(str(self._cfg.path))
        if img is None:
            raise RuntimeError(f"Cannot decode image: {self._cfg.path}")
        self._image = img
        logger.info("image opened path=%s shape=%s", self._cfg.path, img.shape)
        return self

    def __exit__(self,  exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None) -> None:
        self._image = None

    def __iter__(self) -> Iterator[np.ndarray]:
        if self._image is None:
            raise RuntimeError("ImageSource used outside its context manager")
        period = 1.0 / self._cfg.fps if self._cfg.fps > 0 else 0.0
        while True:
            yield self._image
            if not self._cfg.hold:
                return
            if period > 0:
                time.sleep(period)

@register_source("image", ImageSource)
class ImageSourceConfig(SourceConfig):
    type: str = Field(default="image", frozen=True)
    path: Path
    hold: bool = True  # keep yielding the same frame so the viewer stays interactive
    fps: float = 30.0  # yield rate when hold=True
