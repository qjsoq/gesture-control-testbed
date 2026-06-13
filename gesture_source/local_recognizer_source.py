from __future__ import annotations

import logging
import threading
from typing import Iterator

from gesture_control.pipeline import run as run_pipeline
from gesture_control.recognizers.generic_recognizer import GestureRecognizer
from gesture_control.sources.video_source import VideoSource

from gesture_source.base import GestureEvent, GestureSource

logger = logging.getLogger(__name__)


class LocalRecognizerSource(GestureSource):
    """Drives a gesture_control pipeline locally (e.g. on the Pi) and adapts
    each GesturePrediction into a transport-agnostic GestureEvent. No TCP.
    """

    def __init__(
        self,
        video_source: VideoSource,
        recognizer: GestureRecognizer,
        *,
        stop: threading.Event | None = None,
    ) -> None:
        self._video_source = video_source
        self._recognizer = recognizer
        self._stop = stop

    def __iter__(self) -> Iterator[GestureEvent]:
        for pred in run_pipeline(self._video_source, self._recognizer, stop=self._stop):
            yield GestureEvent(
                label=pred.label,
                x_norm=pred.hand_x,
                y_norm=pred.hand_y,
            )
