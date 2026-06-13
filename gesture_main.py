"""Pi-local entry point.

Wires:
  webcam frames  ->  gesture_control pipeline (two-stage: mediapipe + rule-based)
  ->  CommandDispatcher  ->  Command.execute(ctx)
  ->  servoexecutor.VerticalServo / HorizontalServo

Renders an annotated preview window (gesture label, hand center, 3-zone grid,
FPS, dispatcher busy state). Disable with SHOW=0 for headless mode.

Run on Pi:
    cd ~/green-thing
    PYTHONPATH=. python3 gesture_main.py
"""
from __future__ import annotations

import logging
import os
import signal
import sys
import threading

SERVO_PATH_DEFAULT = "/home/pi/test-toggle/half_duplex_transmit/servos"
SERVO_PATH = os.environ.get("SERVO_PATH", SERVO_PATH_DEFAULT)
sys.path.insert(0, SERVO_PATH)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from servoexecutor import HorizontalServo, VerticalServo  # noqa: E402

from commands import CommandContext, CommandDispatcher, build_default_registry  # noqa: E402
from commands.follow_palm import (  # noqa: E402
    BOTTOM_THRESHOLD,
    LEFT_THRESHOLD,
    RIGHT_THRESHOLD,
    TOP_THRESHOLD,
)
from gesture_control.metrics import FpsMeter  # noqa: E402
from gesture_control.pipeline import run_with_frames  # noqa: E402
from gesture_control.recognizers.classifiers.rule_based import (  # noqa: E402
    RuleBasedClassifier,
    RuleBasedClassifierConfig,
)
from gesture_control.recognizers.extractors.mediapipe_hands import (  # noqa: E402
    MediaPipeHandsConfig,
    MediaPipeHandsExtractor,
)
from gesture_control.recognizers.gesture import GesturePrediction  # noqa: E402
from gesture_control.recognizers.two_stage import TwoStageRecognizer  # noqa: E402
from gesture_control.sources.webcam import WebcamConfig, WebcamSource  # noqa: E402

logger = logging.getLogger(__name__)

WINDOW_NAME = "gesture_control"
ZONE_COLOR = (60, 60, 60)
LABEL_COLOR = (0, 200, 0)
CENTER_COLOR = (0, 255, 255)
BUSY_COLOR = (0, 0, 255)
HUD_COLOR = (0, 255, 255)
BOX_COLOR = (0, 200, 0)
BOX_TEXT_BG = (0, 200, 0)
BOX_TEXT_FG = (255, 255, 255)


def build_recognizer() -> TwoStageRecognizer:
    extractor = MediaPipeHandsExtractor(
        MediaPipeHandsConfig(type="mediapipe_hands", max_hands=1)
    )
    classifier = RuleBasedClassifier(RuleBasedClassifierConfig(type="rule_based"))
    return TwoStageRecognizer(extractor=extractor, classifier=classifier)


def draw_overlay(
    image: np.ndarray,
    prediction: GesturePrediction | None,
    *,
    fps: float,
    busy: bool,
) -> None:
    h, w = image.shape[:2]

    # 3-zone grid (matches FollowPalmCommand thresholds)
    for tx in (LEFT_THRESHOLD, RIGHT_THRESHOLD):
        x = int(tx * w)
        cv2.line(image, (x, 0), (x, h), ZONE_COLOR, 1)
    for ty in (TOP_THRESHOLD, BOTTOM_THRESHOLD):
        y = int(ty * h)
        cv2.line(image, (0, y), (w, y), ZONE_COLOR, 1)

    if prediction is not None:
        detections = prediction.raw.get("detections")
        if detections:
            for det in detections:
                x1, y1, x2, y2 = (int(v) for v in det["bbox"])
                cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, 2)
                tag = f"{det['label']} {det['confidence']:.2f}"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw + 4, y1), BOX_TEXT_BG, -1)
                cv2.putText(
                    image, tag, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, BOX_TEXT_FG, 1, cv2.LINE_AA,
                )
        else:
            cv2.putText(
                image,
                f"{prediction.label} {prediction.confidence:.2f}",
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, LABEL_COLOR, 2, cv2.LINE_AA,
            )

        if prediction.hand_x is not None and prediction.hand_y is not None:
            cx = int(prediction.hand_x * w)
            cy = int(prediction.hand_y * h)
            cv2.drawMarker(
                image, (cx, cy), CENTER_COLOR,
                markerType=cv2.MARKER_CROSS, markerSize=24, thickness=2,
            )

    cv2.putText(
        image, f"FPS: {fps:.1f}", (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, HUD_COLOR, 2, cv2.LINE_AA,
    )
    if busy:
        cv2.putText(
            image, "BUSY", (w - 90, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, BUSY_COLOR, 2, cv2.LINE_AA,
        )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    show = os.environ.get("SHOW", "1") != "0"

    v_servo = VerticalServo()
    h_servo = HorizontalServo()
    logger.info("servos initialized")

    stop = threading.Event()

    def _on_signal(signum, frame):  # noqa: ARG001
        logger.info("signal %s received, stopping", signum)
        stop.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    def make_ctx(x: float | None, y: float | None) -> CommandContext:
        return CommandContext(
            v_servo=v_servo,
            h_servo=h_servo,
            hand_x_norm=x,
            hand_y_norm=y,
        )

    dispatcher = CommandDispatcher(
        registry=build_default_registry(),
        context_factory=make_ctx,
    )

    webcam = WebcamSource(WebcamConfig(device_index=0))
    recognizer = build_recognizer()

    if show:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    fps = FpsMeter()

    logger.info("pipeline starting (show=%s)", show)
    try:
        for frame, pred in run_with_frames(webcam, recognizer, stop=stop):
            fps.tick()
            if pred is not None:
                dispatcher.handle(pred.label, pred.hand_x, pred.hand_y)

            if show:
                image = frame.copy()
                draw_overlay(image, pred, fps=fps.fps, busy=dispatcher.is_busy)
                cv2.imshow(WINDOW_NAME, image)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    logger.info("quit key pressed")
                    stop.set()
                    break
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    logger.info("window closed")
                    stop.set()
                    break
    finally:
        dispatcher.stop()
        if show:
            cv2.destroyAllWindows()
        logger.info("pipeline stopped")


if __name__ == "__main__":
    main()
