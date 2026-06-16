"""Єдина точка входу (Д4: клас `Main`).

`Main` залежить лише від абстракцій та фабрик (DIP):

    RecognizerFactory  -> GestureRecognizer
    ServoFactory       -> VerticalServoLike / HorizontalServoLike
    CommandRegistry    -> dict[str, Command]
    GestureSource      -> LocalRecognizerSource | TcpLabelSource

Три режими, обрані через YAML-поле `mode` (або `--mode`):

  stream  Pi-камера (rpicam-vid) -> ONNX -> диспетчер -> сервоприводи,
          анотований MJPEG на :8000 (capture_loop + інференс-цикл).
  local   веб-камера -> LocalRecognizerSource -> диспетчер -> сервоприводи.
  tcp     PC шле мітки по TCP -> TcpLabelSource -> диспетчер -> сервоприводи.

Запуск:
    PYTHONPATH=. python3 main.py --config gesture_control/config_onnx_stream.yaml
    NO_SERVOS=1 PYTHONPATH=. python3 main.py --config ...   # без заліза
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import socketserver
import subprocess
import threading
import time
from http import server
from pathlib import Path
from threading import Condition, Lock, Thread
from typing import Literal

import cv2
import numpy as np
import yaml
from pydantic import BaseModel, model_validator

os.environ.setdefault("OMP_NUM_THREADS", "4")

from commands import CommandDispatcher, CommandRegistry  # noqa: E402
from commands.follow_palm import (  # noqa: E402
    BOTTOM_THRESHOLD,
    LEFT_THRESHOLD,
    RIGHT_THRESHOLD,
    TOP_THRESHOLD,
)
from gesture_control.recognizers import RecognizerConfig, RecognizerFactory  # noqa: E402
from gesture_control.recognizers.generic_recognizer import GestureRecognizer  # noqa: E402
from gesture_control.sources import SourceConfig, build_source, get_source_config  # noqa: E402
from gesture_source import GestureSource, LocalRecognizerSource, TcpLabelSource  # noqa: E402
from servo_factory import ServoFactory  # noqa: E402

logger = logging.getLogger(__name__)

BOX_COLOR = (0, 200, 0)
TEXT_COLOR = (255, 255, 255)
ZONE_COLOR = (60, 60, 60)
CENTER_COLOR = (0, 255, 255)
BUSY_COLOR = (0, 0, 255)
HUD_COLOR = (0, 255, 255)


# --------------------------------------------------------------------------- #
# Конфігурація
# --------------------------------------------------------------------------- #
class CameraConfig(BaseModel):
    width: int = 640
    height: int = 480
    framerate: int = 15
    codec: Literal["mjpeg"] = "mjpeg"


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    jpeg_quality: int = 75


class TcpConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 15482


class MainConfig(BaseModel):
    mode: Literal["stream", "local", "tcp"] = "stream"
    recognizer: RecognizerConfig | None = None
    source: SourceConfig | None = None
    camera: CameraConfig = CameraConfig()
    server: ServerConfig = ServerConfig()
    tcp: TcpConfig = TcpConfig()

    @model_validator(mode="before")
    @classmethod
    def _resolve_source(cls, raw_data: dict) -> dict:
        # Як у gesture_control.app.AppConfig: підставити конкретний клас конфіга
        # відеоджерела за полем `source.type`.
        if isinstance(raw_data, dict) and isinstance(raw_data.get("source"), dict):
            specific_type = raw_data["source"].get("type", "")
            config_class = get_source_config(specific_type)
            if config_class:
                raw_data["source"] = config_class.model_validate(raw_data["source"])
        return raw_data


def load_config(path: Path) -> MainConfig:
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return MainConfig.model_validate(raw)


# --------------------------------------------------------------------------- #
# MJPEG-інфраструктура (режим stream)
# --------------------------------------------------------------------------- #
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


class ThreadingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


# --------------------------------------------------------------------------- #
# Main (Д4)
# --------------------------------------------------------------------------- #
class Main:
    """Шар збирання (composition root): будує абстракції через фабрики й
    проганяє обраний режим. Поля типізовані абстракціями, як на креслянику Д2/Д4.
    """

    def __init__(self, cfg: MainConfig) -> None:
        self._cfg = cfg
        self._stop = threading.Event()

        self.recognizer: GestureRecognizer | None = (
            RecognizerFactory.build_recognizer(cfg.recognizer) if cfg.recognizer else None
        )
        self.v_servo = ServoFactory.create_vertical()
        self.h_servo = ServoFactory.create_horizontal()
        self.dispatcher = CommandDispatcher(
            registry=CommandRegistry.build_default_registry(self.v_servo, self.h_servo),
        )
        self.source: GestureSource | None = None

    # --- публічний запуск ------------------------------------------------- #
    def run(self) -> None:
        if self._cfg.mode == "stream":
            self._run_stream()
        elif self._cfg.mode == "local":
            self._run_local()
        elif self._cfg.mode == "tcp":
            self._run_tcp()
        else:  # pragma: no cover — pydantic уже валідує
            raise SystemExit(f"unknown mode: {self._cfg.mode}")

    # --- режим stream (Pi-камера АБО відеофайл + MJPEG) ------------------- #
    def _run_stream(self) -> None:
        if self.recognizer is None:
            raise SystemExit("stream mode requires a `recognizer` in the config")

        cfg = self._cfg
        src_cfg = cfg.source
        use_file = src_cfg is not None and getattr(src_cfg, "type", None) == "file"

        proc: subprocess.Popen | None = None
        if use_file:
            # Відеофайл як джерело (тест без камери): кадри -> latest -> інференс.
            Thread(
                target=self._file_capture_loop,
                args=(cfg.server.jpeg_quality,),
                daemon=True,
            ).start()
        else:
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
            Thread(target=self.capture_loop, args=(proc,), daemon=True).start()

        Thread(
            target=self._inference_loop,
            args=(cfg.server.jpeg_quality,),
            daemon=True,
        ).start()

        handler = make_streaming_handler(cfg.server.jpeg_quality)
        src_desc = "file" if use_file else "rpicam"
        logger.info("streaming (%s) on http://<pi-ip>:%d/", src_desc, cfg.server.port)
        try:
            ThreadingServer((cfg.server.host, cfg.server.port), handler).serve_forever()
        finally:
            if proc is not None:
                proc.terminate()
            self.dispatcher.stop()

    def _file_capture_loop(self, jpeg_quality: int) -> None:
        """Читає кадри з відеофайлу (FileSource) і штовхає їх у `latest`,
        той самий конвеєр що й камера. realtime/loop задаються в конфізі джерела."""
        source = build_source(self._cfg.source)
        with source as frames:
            for img in frames:
                ok, enc = cv2.imencode(
                    ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
                )
                if ok:
                    latest.put(enc.tobytes())
        logger.info("video file exhausted (set source.loop=true to repeat)")

    def capture_loop(self, proc: subprocess.Popen) -> None:
        """Розрізає потік rpicam-vid (MJPEG) на окремі JPEG-кадри."""
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

    def _inference_loop(self, jpeg_quality: int) -> None:
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
                pred = self.recognizer.predict(img)
            except Exception:
                logger.exception("inference error")
                pred = None
            inf_ms = (time.monotonic() - t0) * 1000.0

            draw_zone_grid(img)
            if pred is not None:
                draw_detections(img, pred.raw.get("detections", []))
                draw_hand_center(img, pred.hand_x, pred.hand_y)
                self.dispatcher.handle(pred.label, pred.hand_x, pred.hand_y)
            draw_hud(img, fps, inf_ms, self.dispatcher.is_busy)

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

    # --- режим local (веб-камера через GestureSource) --------------------- #
    def _run_local(self) -> None:
        if self.recognizer is None or self._cfg.source is None:
            raise SystemExit("local mode requires `recognizer` and `source` in the config")
        self._install_signal_handlers()
        self.source = LocalRecognizerSource(
            build_source(self._cfg.source), self.recognizer, stop=self._stop
        )
        logger.info("local source starting")
        try:
            self._consume(self.source)
        finally:
            self.dispatcher.stop()
            logger.info("local source stopped")

    # --- режим tcp (мітки з PC) ------------------------------------------- #
    def _run_tcp(self) -> None:
        self._install_signal_handlers()
        logger.info("tcp label sink on %s:%d", self._cfg.tcp.host, self._cfg.tcp.port)
        try:
            while not self._stop.is_set():
                self.source = TcpLabelSource(
                    host=self._cfg.tcp.host, port=self._cfg.tcp.port, stop=self._stop
                )
                self._consume(self.source)
                logger.info("client disconnected, waiting for a new connection")
        except KeyboardInterrupt:
            logger.info("interrupted, shutting down")
        finally:
            self.dispatcher.stop()

    # --- спільне -------------------------------------------------------- #
    def _consume(self, source: GestureSource) -> None:
        for ev in source:
            logger.debug("label=%s x=%s y=%s", ev.label, ev.x_norm, ev.y_norm)
            self.dispatcher.handle(ev.label, ev.x_norm, ev.y_norm)

    def _install_signal_handlers(self) -> None:
        def _handler(signum, _frame):  # noqa: ANN001
            logger.info("signal %s received, stopping", signum)
            self._stop.set()

        signal.signal(signal.SIGINT, _handler)
        try:
            signal.signal(signal.SIGTERM, _handler)
        except (AttributeError, ValueError):
            pass


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Gesture control — єдина точка входу")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "gesture_control" / "config_onnx_stream.yaml",
    )
    parser.add_argument(
        "--mode",
        choices=["stream", "local", "tcp"],
        default=None,
        help="override the config's mode",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.mode is not None:
        cfg = cfg.model_copy(update={"mode": args.mode})

    Main(cfg).run()


if __name__ == "__main__":
    main()
