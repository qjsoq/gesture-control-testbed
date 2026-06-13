import logging
import threading
from typing import Iterator

import numpy as np
from gesture_control.metrics.metrics import FpsMeter
from gesture_control.recognizers.gesture import GesturePrediction
from gesture_control.recognizers.generic_recognizer import GestureRecognizer
from gesture_control.sources.video_source import VideoSource

logger = logging.getLogger(__name__)


def run_with_frames(
    source: VideoSource,
    recognizer: GestureRecognizer,
    *,
    stop: threading.Event | None = None,
    log_fps_every: int = 100,
) -> Iterator[tuple[np.ndarray, GesturePrediction | None]]:
    fps = FpsMeter()
    n = 0
    with source as s:
        for frame in s:
            if stop is not None and stop.is_set():
                logger.info("stop requested, exiting pipeline")
                return
            try:
                pred = recognizer.predict(frame)
            except Exception:
                logger.exception("recognizer failed for frame index=%d", n)
                pred = None
            if pred:
                logger.info(f"gesture predicted {pred.label}")
            fps.tick()
            n += 1
            if log_fps_every > 0 and n % log_fps_every == 0:
                logger.info("pipeline fps=%.1f frames=%d", fps.fps, n)
            yield frame, pred


def run(
    source: VideoSource,
    recognizer: GestureRecognizer,
    *,
    stop: threading.Event | None = None,
    log_fps_every: int = 100,
) -> Iterator[GesturePrediction]:
    for _frame, pred in run_with_frames(
        source, recognizer, stop=stop, log_fps_every=log_fps_every
    ):
        if pred is not None:
            yield pred
