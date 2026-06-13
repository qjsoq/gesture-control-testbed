from __future__ import annotations

import json
import logging
import socket
import threading
from typing import Iterator

from gesture_source.base import GestureEvent, GestureSource

logger = logging.getLogger(__name__)


class TcpLabelSource(GestureSource):

    def __iter__(self) -> Iterator[GestureEvent]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((self._host, self._port))
            srv.listen(1)
            logger.info("tcp_label_source listening on %s:%d", self._host, self._port)
            conn, addr = srv.accept()
            logger.info("tcp_label_source client connected from %s", addr)
            with conn:
                yield from self._read_events(conn)
                
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 15482,
        *,
        stop: threading.Event | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._stop = stop



    def _read_events(self, conn: socket.socket) -> Iterator[GestureEvent]:
        buffer = b""
        while not (self._stop is not None and self._stop.is_set()):
            while b"\n" not in buffer:
                chunk = conn.recv(4096)
                if not chunk:
                    return
                buffer += chunk
            prefix, _, rest = buffer.partition(b"\n")
            try:
                size = int(prefix)
            except ValueError:
                logger.warning("malformed length prefix %r, dropping connection", prefix)
                return
            while len(rest) < size:
                chunk = conn.recv(size - len(rest))
                if not chunk:
                    return
                rest += chunk
            payload, buffer = rest[:size], rest[size:]
            try:
                msg = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                logger.warning("malformed JSON, dropping connection")
                return
            yield GestureEvent(
                label=str(msg.get("label", "no_gesture")),
                x_norm=_optional_float(msg.get("x")),
                y_norm=_optional_float(msg.get("y")),
            )


def _optional_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
