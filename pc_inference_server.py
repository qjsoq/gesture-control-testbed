"""PC-сторона віддаленого режиму.

Слухає TCP, приймає JPEG-кадри з Pi (`RemoteInferenceSource`), проганяє
обраний у YAML розпізнавач (напр. MediaPipe two-stage) і повертає мітку жесту.

    PC:  python pc_inference_server.py --config gesture_control/config_mediapipe_server.yaml --port 15483
    Pi:  PYTHONPATH=. python3 main.py --config gesture_control/config_remote_pi.yaml

Протокол (симетричний із `gesture_source.RemoteInferenceSource`):
    Pi -> PC : b"<jpeg_len>\\n" + <jpeg>
    PC -> Pi : b"<json_len>\\n" + json{"label","x","y"}
"""
from __future__ import annotations

import argparse
import json
import logging
import socket
from pathlib import Path

import cv2
import numpy as np
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


def _recv_frame(conn: socket.socket, buffer: bytes) -> tuple[bytes | None, bytes]:
    """Читає одне повідомлення `<len>\\n<bytes>`; повертає (jpeg_bytes, buffer)
    або (None, buffer) на закритті/помилці."""
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


def _serve_one(conn: socket.socket, recognizer) -> None:
    buffer = b""
    frames = 0
    while True:
        jpg, buffer = _recv_frame(conn, buffer)
        if jpg is None:
            return
        img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        label, x, y = "no_gesture", None, None
        if img is not None:
            try:
                pred = recognizer.predict(img)
            except Exception:
                logger.exception("inference error")
                pred = None
            if pred is not None:
                label, x, y = pred.label, pred.hand_x, pred.hand_y
        payload = json.dumps({"label": label, "x": x, "y": y}).encode("utf-8")
        conn.sendall(f"{len(payload)}\n".encode("ascii") + payload)
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
    args = parser.parse_args(argv)

    recognizer = _load_recognizer(args.config)
    logger.info("recognizer ready: %s", type(recognizer).__name__)

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
                    _serve_one(conn, recognizer)
                logger.info("Pi disconnected, waiting for a new connection")
        except KeyboardInterrupt:
            logger.info("interrupted, shutting down")


if __name__ == "__main__":
    main()
