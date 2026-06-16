from __future__ import annotations

import logging
import threading

from commands.base import Command, CommandContext
from commands.noop import NoOpCommand

logger = logging.getLogger(__name__)


class CommandDispatcher:
    """Routes labeled gesture events to Commands on a single background worker.

    - Non-blocking `handle(label, ctx)`: callers (the gesture source loop)
      never wait on servo motion.
    - 1-slot mailbox: if a new label arrives while another command is still
      running, the dispatcher records it as the *pending* slot, overwriting
      any previous pending. When the running command finishes, the worker
      consumes that slot. This guarantees we react to recent gestures but
      never queue up a stale backlog.
    - `busy` flag: while a command is executing, the dispatcher exposes
      `is_busy` for observability.
    """

    def __init__(
        self,
        registry: dict[str, Command],
        *,
        default: Command | None = None,
    ) -> None:
        self._registry = registry
        self._default = default if default is not None else NoOpCommand()

        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._pending: tuple[str, float | None, float | None] | None = None
        self._busy = False

        self._worker = threading.Thread(
            target=self._run, name="command-dispatcher", daemon=True
        )
        self._worker.start()

    @property
    def is_busy(self) -> bool:
        return self._busy

    def handle(self, label: str, x_norm: float | None, y_norm: float | None) -> None:
        with self._lock:
            if self._busy:
                logger.debug("dispatcher busy, dropping label=%s", label)
                return
            self._pending = (label, x_norm, y_norm)
        self._wake.set()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        self._worker.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait()
            self._wake.clear()
            if self._stop.is_set():
                return
            with self._lock:
                item = self._pending
                self._pending = None
                if item is None:
                    continue
                self._busy = True
            try:
                label, x, y = item
                cmd = self._registry.get(label, self._default)
                ctx = CommandContext(hand_x_norm=x, hand_y_norm=y)
                logger.info("dispatch label=%s -> %s", label, type(cmd).__name__)
                cmd.execute(ctx)
            except Exception:
                logger.exception("command failed for label=%s", item[0])
            finally:
                with self._lock:
                    self._busy = False
