from __future__ import annotations

import json
import logging
import socket
import threading
from typing import Iterator

import cv2
import numpy as np

from gesture_control.sources.video_source import VideoSource
from gesture_source.base import GestureEvent, GestureSource

logger = logging.getLogger(__name__)


class RemoteInferenceSource(GestureSource):
    """Pi-сторона віддаленого режиму: бере кадри з БУДЬ-ЯКОГО `VideoSource`
    (камера/файл/…), шле кожен кадр JPEG на PC і чекає мітку жесту у відповідь,
    яку віддає як `GestureEvent`. Інференс — на PC; Pi лишається I/O.

    Не залежить від джерела кадрів — `video_source` впорскується ззовні
    (`build_source(cfg.source)`), тож камеру можна замінити файлом без змін тут.

    Протокол (симетричне кадрування, як у `TcpLabelSource`):
        Pi -> PC : b"<jpeg_len>\\n" + <jpeg-байти>
        PC -> Pi : b"<json_len>\\n" + json{"label", "x", "y"}
    PC виступає TCP-сервером; Pi під'єднується клієнтом.
    """

    def __init__(
        self,
        video_source: VideoSource,
        host: str,
        port: int = 15483,
        *,
        jpeg_quality: int = 75,
        stop: threading.Event | None = None,
    ) -> None:
        self._video_source = video_source
        self._host = host
        self._port = port
        self._jpeg_quality = jpeg_quality
        self._stop = stop

    def __iter__(self) -> Iterator[GestureEvent]:
        logger.info("remote_inference: connecting to PC %s:%d", self._host, self._port)
        with socket.create_connection((self._host, self._port)) as sock, \
                self._video_source as frames:
            logger.info("remote_inference: connected, streaming frames")
            recv_buf = b""
            for frame in frames:
                if self._stop is not None and self._stop.is_set():
                    return
                ok, enc = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
                )
                if not ok:
                    continue
                payload = enc.tobytes()
                sock.sendall(f"{len(payload)}\n".encode("ascii") + payload)

                msg, recv_buf = _recv_framed(sock, recv_buf)
                if msg is None:
                    logger.info("remote_inference: PC closed connection")
                    return
                yield GestureEvent(
                    label=str(msg.get("label", "no_gesture")),
                    x_norm=_optional_float(msg.get("x")),
                    y_norm=_optional_float(msg.get("y")),
                )


def _recv_framed(sock: socket.socket, buffer: bytes) -> tuple[dict | None, bytes]:
    """Читає одне повідомлення `<len>\\n<json>` зі сокета. Повертає (msg, buffer)
    або (None, buffer) якщо з'єднання закрите / повідомлення зіпсоване."""
    while b"\n" not in buffer:
        chunk = sock.recv(4096)
        if not chunk:
            return None, buffer
        buffer += chunk
    prefix, _, rest = buffer.partition(b"\n")
    try:
        size = int(prefix)
    except ValueError:
        logger.warning("remote_inference: bad length prefix %r", prefix)
        return None, buffer
    while len(rest) < size:
        chunk = sock.recv(size - len(rest))
        if not chunk:
            return None, buffer
        rest += chunk
    payload, buffer = rest[:size], rest[size:]
    try:
        return json.loads(payload.decode("utf-8")), buffer
    except (UnicodeDecodeError, json.JSONDecodeError):
        logger.warning("remote_inference: bad JSON payload")
        return None, buffer


def _optional_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
