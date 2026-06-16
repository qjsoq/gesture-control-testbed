"""PC-сторона віддаленого режиму.

Слухає TCP, приймає JPEG-кадри з Pi (`RemoteInferenceSource`), проганяє
обраний у YAML розпізнавач (напр. MediaPipe two-stage) і повертає мітку жесту.
За бажанням віддає анотований MJPEG у браузер (--view-port, типово 8000).

    PC:  python pc_inference_server.py --config gesture_control/config_mediapipe_server.yaml --port 15483
    Pi:  PYTHONPATH=. python3 main.py --config gesture_control/config_remote_pi.yaml
    Браузер: http://localhost:8000/

Протокол (симетричний із `gesture_source.RemoteInferenceSource`):
    Pi -> PC : b"<jpeg_len>\\n" + <jpeg>
    PC -> Pi : b"<json_len>\\n" + json{"label","x","y"}
"""
from __future__ import annotations

import argparse
import json
import logging
import socketserver
import threading
from http import server as http_server
from pathlib import Path
from threading import Condition

import cv2
import numpy as np
import socket
import yaml
from pydantic import BaseModel

from gesture_control.recognizers import RecognizerConfig, RecognizerFactory

logger = logging.getLogger(__name__)


class _ServerConfig(BaseModel):
    recognizer: RecognizerConfig


def _load_recognizer(path: Path):
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = _ServerConfig.model_validate(raw)
    return RecognizerFactory.build_recognizer(cfg.recognizer)


# --------------------------------------------------------------------------- #
# MJPEG-стрім у браузер (анотовані кадри)
# --------------------------------------------------------------------------- #
class _StreamingOutput:
    def __init__(self) -> None:
        self.frame: bytes | None = None
        self.condition = Condition()

    def write(self, buf: bytes) -> None:
        with self.condition:
            self.frame = buf
            self.condition.notify_all()


def _make_handler(output: _StreamingOutput):
    class Handler(http_server.BaseHTTPRequestHandler):
        def log_message(self, *_a) -> None:
            return

        def do_GET(self) -> None:
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(
                    b'<html><head><title>MediaPipe (remote)</title></head>'
                    b'<body style="margin:0;background:#000;">'
                    b'<img src="/stream.mjpg" style="width:100%;height:100vh;object-fit:contain;"/>'
                    b'</body></html>'
                )
            elif self.path == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Cache-Control", "no-cache, private")
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

    return Handler


class _ThreadingServer(socketserver.ThreadingMixIn, http_server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


# Стандартні зв'язки 21-точкового скелета кисті MediaPipe.
_HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),            # великий палець
    (0, 5), (5, 6), (6, 7), (7, 8),            # вказівний
    (5, 9), (9, 10), (10, 11), (11, 12),       # середній
    (9, 13), (13, 14), (14, 15), (15, 16),     # безіменний
    (13, 17), (17, 18), (18, 19), (19, 20),    # мізинець
    (0, 17),                                   # основа долоні
)


def _draw_landmarks(img: np.ndarray, landmarks: np.ndarray) -> None:
    h, w = img.shape[:2]
    pts = [(int(p[0] * w), int(p[1] * h)) for p in landmarks]
    for a, b in _HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], (255, 200, 0), 2, cv2.LINE_AA)
    for px, py in pts:
        cv2.circle(img, (px, py), 4, (0, 0, 255), -1, cv2.LINE_AA)


def _annotate(
    img: np.ndarray,
    label: str,
    x: float | None,
    y: float | None,
    landmarks: np.ndarray | None = None,
) -> None:
    if landmarks is not None and getattr(landmarks, "shape", None) == (21, 3):
        _draw_landmarks(img, landmarks)
    cv2.putText(
        img, f"mediapipe: {label}", (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA,
    )
    if x is not None and y is not None:
        h, w = img.shape[:2]
        cv2.drawMarker(
            img, (int(x * w), int(y * h)), (0, 255, 0),
            markerType=cv2.MARKER_CROSS, markerSize=28, thickness=2,
        )


# --------------------------------------------------------------------------- #
# TCP-приймач кадрів
# --------------------------------------------------------------------------- #
def _recv_frame(conn: socket.socket, buffer: bytes) -> tuple[bytes | None, bytes]:
    while b"\n" not in buffer:
        chunk = conn.recv(65536)
        if not chunk:
            return None, buffer
        buffer += chunk
    prefix, _, rest = buffer.partition(b"\n")
    try:
        size = int(prefix)
    except ValueError:
        logger.warning("bad length prefix %r, dropping connection", prefix)
        return None, buffer
    while len(rest) < size:
        chunk = conn.recv(size - len(rest))
        if not chunk:
            return None, buffer
        rest += chunk
    return rest[:size], rest[size:]


def _serve_one(conn: socket.socket, recognizer, output: _StreamingOutput | None) -> None:
    buffer = b""
    frames = 0
    while True:
        jpg, buffer = _recv_frame(conn, buffer)
        if jpg is None:
            return
        img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        label, x, y = "no_gesture", None, None
        landmarks = None
        if img is not None:
            try:
                pred = recognizer.predict(img)
            except Exception:
                logger.exception("inference error")
                pred = None
            if pred is not None:
                label, x, y = pred.label, pred.hand_x, pred.hand_y
                landmarks = pred.raw.get("landmarks")
        payload = json.dumps({"label": label, "x": x, "y": y}).encode("utf-8")
        conn.sendall(f"{len(payload)}\n".encode("ascii") + payload)

        if output is not None and img is not None:
            _annotate(img, label, x, y, landmarks)
            ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                output.write(enc.tobytes())

        frames += 1
        if frames % 30 == 0:
            logger.info("served %d frames, last label=%s", frames, label)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="PC inference server (remote mode)")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=15483)
    parser.add_argument(
        "--view-port", type=int, default=8000,
        help="MJPEG-перегляд анотованих кадрів у браузері; 0 = вимкнути",
    )
    args = parser.parse_args(argv)

    recognizer = _load_recognizer(args.config)
    logger.info("recognizer ready: %s", type(recognizer).__name__)

    output: _StreamingOutput | None = None
    if args.view_port > 0:
        output = _StreamingOutput()
        handler = _make_handler(output)
        httpd = _ThreadingServer((args.host, args.view_port), handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        logger.info("MJPEG view on http://localhost:%d/", args.view_port)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((args.host, args.port))
        srv.listen(1)
        logger.info("listening on %s:%d", args.host, args.port)
        try:
            while True:
                conn, addr = srv.accept()
                logger.info("Pi connected from %s", addr)
                with conn:
                    _serve_one(conn, recognizer, output)
                logger.info("Pi disconnected, waiting for a new connection")
        except KeyboardInterrupt:
            logger.info("interrupted, shutting down")


if __name__ == "__main__":
    main()
