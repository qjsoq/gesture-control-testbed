import logging
import threading
from pathlib import Path

import cv2
import numpy as np

from gesture_control.metrics import FpsMeter
from gesture_control.pipeline import run_with_frames
from gesture_control.recognizers.gesture import GesturePrediction
from gesture_control.recognizers.generic_recognizer import GestureRecognizer
from gesture_control.sources.video_source import VideoSource

logger = logging.getLogger(__name__)

_BOX_COLOR = (0, 200, 0)
_TEXT_COLOR = (255, 255, 255)
_BG_COLOR = (0, 200, 0)


def _draw_detections(image: np.ndarray, prediction: GesturePrediction | None) -> None:
    if prediction is None:
        return
    detections = prediction.raw.get("detections")
    if detections:
        for det in detections:
            x1, y1, x2, y2 = (int(v) for v in det["bbox"])
            label = f"{det['label']} {det['confidence']:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), _BOX_COLOR, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw + 4, y1), _BG_COLOR, -1)
            cv2.putText(
                image,
                label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                _TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )
    else:
        # Fallback for recognizers without bboxes (e.g. two-stage MediaPipe)
        cv2.putText(
            image,
            f"{prediction.label} {prediction.confidence:.2f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            _BOX_COLOR,
            2,
            cv2.LINE_AA,
        )


def _draw_hud(image: np.ndarray, fps: float) -> None:
    cv2.putText(
        image,
        f"FPS: {fps:.1f}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def show(
    source: VideoSource,
    recognizer: GestureRecognizer,
    *,
    stop: threading.Event | None = None,
    window_name: str = "gesture_control",
    log_fps_every: int = 0,
) -> None:
    """Run the pipeline and render annotated frames in a cv2 window. Press q/Esc to quit."""
    fps = FpsMeter()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        for frame, pred in run_with_frames(
            source, recognizer, stop=stop, log_fps_every=log_fps_every
        ):
            fps.tick()
            image = frame.copy()
            _draw_detections(image, pred)
            _draw_hud(image, fps.fps)
            cv2.imshow(window_name, image)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                logger.info("quit key pressed, stopping viewer")
                if stop is not None:
                    stop.set()
                return
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                logger.info("window closed, stopping viewer")
                if stop is not None:
                    stop.set()
                return
    finally:
        cv2.destroyWindow(window_name)


def save_annotated(
    source: VideoSource,
    recognizer: GestureRecognizer,
    output_path: Path,
    *,
    stop: threading.Event | None = None,
    log_fps_every: int = 0,
    max_frames: int | None = 1,
) -> None:
    """Run pipeline and write annotated frames to disk. Headless-safe (no cv2.imshow)."""
    fps = FpsMeter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    for frame, pred in run_with_frames(
        source, recognizer, stop=stop, log_fps_every=log_fps_every
    ):
        fps.tick()
        image = frame.copy()
        _draw_detections(image, pred)
        _draw_hud(image, fps.fps)
        if max_frames == 1:
            target = output_path
        else:
            target = output_path.with_name(f"{output_path.stem}_{n:05d}{output_path.suffix}")
        if not cv2.imwrite(str(target), image):
            raise RuntimeError(f"failed to write image: {target}")
        logger.info(
            "saved annotated frame path=%s prediction=%s",
            target,
            None if pred is None else f"{pred.label} {pred.confidence:.2f}",
        )
        n += 1
        if max_frames is not None and n >= max_frames:
            if stop is not None:
                stop.set()
            return
