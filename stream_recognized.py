"""MJPEG HTTP stream with ONNX gesture recognition + servo command dispatch.

Pi camera (rpicam-vid) → MJPEG → ONNX YOLO inference → CommandDispatcher → servos,
with annotated frames published over HTTP for browser viewing.

Run on Pi:
    cd ~/green-thing
    PYTHONPATH=. python3 stream_recognized.py \
        --config gesture_control/config_onnx_stream.yaml

Open in browser (from laptop):
    http://192.168.0.107:8000/

Disable servos (dev mode, e.g. running on PC for stream-only test):
    NO_SERVOS=1 PYTHONPATH=. python3 stream_recognized.py --config ...
"""
from __future__ import annotations

import argparse
import logging
import os
import socketserver
import subprocess
import sys
import time
from http import server
from pathlib import Path
from threading import Condition, Lock, Thread
from typing import Literal

import cv2
import numpy as np
import yaml
from pydantic import BaseModel

os.environ.setdefault("OMP_NUM_THREADS", "4")

SERVO_PATH_DEFAULT = "/home/pi/test-toggle/half_duplex_transmit/servos"
sys.path.insert(0, os.environ.get("SERVO_PATH", SERVO_PATH_DEFAULT))

from gesture_control.recognizers.onnx_yolo import (  # noqa: E402
    OnnxYoloRecognizer,
    OnnxYoloRecognizerConfig,
)

from commands import CommandContext, CommandDispatcher, build_default_registry  # noqa: E402
from commands.follow_palm import (  # noqa: E402
    BOTTOM_THRESHOLD,
    LEFT_THRESHOLD,
    RIGHT_THRESHOLD,
    TOP_THRESHOLD,
)

logger = logging.getLogger(__name__)

BOX_COLOR = (0, 200, 0)
TEXT_COLOR = (255, 255, 255)
ZONE_COLOR = (60, 60, 60)
CENTER_COLOR = (0, 255, 255)
BUSY_COLOR = (0, 0, 255)
HUD_COLOR = (0, 255, 255)


class CameraConfig(BaseModel):
    width: int = 640
    height: int = 480
    framerate: int = 15
    codec: Literal["mjpeg"] = "mjpeg"


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    jpeg_quality: int = 75


class StreamConfig(BaseModel):
    recognizer: OnnxYoloRecognizerConfig
    camera: CameraConfig = CameraConfig()
    server: ServerConfig = ServerConfig()


class StreamingOutput:
    def __init__(self) -> None:
        self.frame: bytes | None = None
        self.condition = Condition()

    def write(self, buf: bytes) -> None:
        with self.condition:
            self.frame = buf
            self.condition.notify_all()


class LatestFrame:
    def __init__(self) -> None:
        self._jpg: bytes | None = None
        self._lock = Lock()
        self._condition = Condition(self._lock)

    def put(self, jpg: bytes) -> None:
        with self._condition:
            self._jpg = jpg
            self._condition.notify()

    def take(self, timeout: float = 1.0) -> bytes | None:
        with self._condition:
            if self._jpg is None:
                self._condition.wait(timeout=timeout)
            jpg, self._jpg = self._jpg, None
            return jpg


output = StreamingOutput()
latest = LatestFrame()


def draw_zone_grid(image: np.ndarray) -> None:
    h, w = image.shape[:2]
    for tx in (LEFT_THRESHOLD, RIGHT_THRESHOLD):
        x = int(tx * w)
        cv2.line(image, (x, 0), (x, h), ZONE_COLOR, 1)
    for ty in (TOP_THRESHOLD, BOTTOM_THRESHOLD):
        y = int(ty * h)
        cv2.line(image, (0, y), (w, y), ZONE_COLOR, 1)


def draw_detections(image: np.ndarray, detections: list[dict]) -> None:
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det["bbox"])
        label = f"{det['label']} {det['confidence']:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw + 4, y1), BOX_COLOR, -1)
        cv2.putText(
            image, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1, cv2.LINE_AA,
        )


def draw_hand_center(image: np.ndarray, x_norm: float | None, y_norm: float | None) -> None:
    if x_norm is None or y_norm is None:
        return
    h, w = image.shape[:2]
    cv2.drawMarker(
        image, (int(x_norm * w), int(y_norm * h)), CENTER_COLOR,
        markerType=cv2.MARKER_CROSS, markerSize=24, thickness=2,
    )


def draw_hud(image: np.ndarray, fps: float, inference_ms: float, busy: bool) -> None:
    cv2.putText(
        image,
        f"FPS {fps:4.1f}  inf {inference_ms:5.0f}ms",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, HUD_COLOR, 2, cv2.LINE_AA,
    )
    if busy:
        h, w = image.shape[:2]
        cv2.putText(
            image, "BUSY", (w - 90, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, BUSY_COLOR, 2, cv2.LINE_AA,
        )


def make_streaming_handler(jpeg_quality: int) -> type[server.BaseHTTPRequestHandler]:
    class Handler(server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:  # noqa: A002
            return

        def do_GET(self) -> None:
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(
                    b'<html><head><title>Pi Cam + ONNX</title></head>'
                    b'<body style="margin:0;background:#000;">'
                    b'<img src="/stream.mjpg" style="width:100%;height:100vh;object-fit:contain;"/>'
                    b'</body></html>'
                )
            elif self.path == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header(
                    "Content-Type", "multipart/x-mixed-replace; boundary=FRAME"
                )
                self.end_headers()
                try:
                    while True:
                        with output.condition:
                            output.condition.wait()
                            frame = output.frame
                        if frame is None:
                            continue
                        self.wfile.write(b"--FRAME\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(frame)))
                        self.end_headers()
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

    Handler._jpeg_quality = jpeg_quality
    return Handler


def capture_loop(proc: subprocess.Popen) -> None:
    buf = b""
    while True:
        chunk = proc.stdout.read(4096)
        if not chunk:
            break
        buf += chunk
        while True:
            soi = buf.find(b"\xff\xd8")
            if soi == -1:
                buf = b""
                break
            eoi = buf.find(b"\xff\xd9", soi + 2)
            if eoi == -1:
                buf = buf[soi:]
                break
            latest.put(buf[soi : eoi + 2])
            buf = buf[eoi + 2 :]


def inference_loop(
    recognizer: OnnxYoloRecognizer,
    jpeg_quality: int,
    dispatcher: CommandDispatcher | None,
) -> None:
    last_log = time.monotonic()
    processed = 0
    fps = 0.0
    inf_ms = 0.0
    while True:
        jpg = latest.take(timeout=1.0)
        if jpg is None:
            continue

        arr = np.frombuffer(jpg, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            continue

        t0 = time.monotonic()
        try:
            pred = recognizer.predict(img)
        except Exception:
            logger.exception("inference error")
            pred = None
        inf_ms = (time.monotonic() - t0) * 1000.0

        draw_zone_grid(img)
        if pred is not None:
            draw_detections(img, pred.raw.get("detections", []))
            draw_hand_center(img, pred.hand_x, pred.hand_y)
            if dispatcher is not None:
                dispatcher.handle(pred.label, pred.hand_x, pred.hand_y)
        draw_hud(img, fps, inf_ms, dispatcher.is_busy if dispatcher else False)

        ok, enc = cv2.imencode(
            ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        )
        if ok:
            output.write(enc.tobytes())

        processed += 1
        now = time.monotonic()
        dt = now - last_log
        if dt >= 2.0:
            fps = processed / dt
            logger.info("inference fps=%.2f last_inf=%.0fms", fps, inf_ms)
            processed = 0
            last_log = now


class ThreadingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def load_config(path: Path) -> StreamConfig:
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return StreamConfig.model_validate(raw)


def build_dispatcher() -> CommandDispatcher | None:
    if os.environ.get("NO_SERVOS") == "1":
        logger.info("NO_SERVOS=1 — running stream-only, no servo dispatch")
        return None
    try:
        from servoexecutor import HorizontalServo, VerticalServo
    except Exception:
        logger.exception("servoexecutor import failed — running stream-only")
        return None
    v_servo = VerticalServo()
    h_servo = HorizontalServo()
    logger.info("servos initialized")

    def make_ctx(x: float | None, y: float | None) -> CommandContext:
        return CommandContext(
            v_servo=v_servo, h_servo=h_servo,
            hand_x_norm=x, hand_y_norm=y,
        )

    return CommandDispatcher(
        registry=build_default_registry(),
        context_factory=make_ctx,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "gesture_control" / "config_onnx_stream.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not cfg.recognizer.model_path.exists():
        raise SystemExit(f"Model not found: {cfg.recognizer.model_path}")

    recognizer = OnnxYoloRecognizer(cfg.recognizer)
    dispatcher = build_dispatcher()

    proc = subprocess.Popen(
        [
            "rpicam-vid",
            "-t", "0",
            "--width", str(cfg.camera.width),
            "--height", str(cfg.camera.height),
            "--framerate", str(cfg.camera.framerate),
            "--codec", cfg.camera.codec,
            "--inline",
            "--nopreview",
            "-o", "-",
        ],
        stdout=subprocess.PIPE,
    )
    Thread(target=capture_loop, args=(proc,), daemon=True).start()
    Thread(
        target=inference_loop,
        args=(recognizer, cfg.server.jpeg_quality, dispatcher),
        daemon=True,
    ).start()

    handler = make_streaming_handler(cfg.server.jpeg_quality)
    logger.info("streaming on http://<pi-ip>:%d/", cfg.server.port)
    try:
        ThreadingServer((cfg.server.host, cfg.server.port), handler).serve_forever()
    finally:
        proc.terminate()
        if dispatcher is not None:
            dispatcher.stop()


if __name__ == "__main__":
    main()
