"""Єдина точка створення приводів (Д4: «factory» ServoFactory).

Повертає типи-протоколи (`VerticalServoLike` / `HorizontalServoLike`), не
конкретні класи — точки збирання (Main) не знають про `servoexecutor`.

- `SERVO_PATH` `sys.path`-хак та `import servoexecutor` живуть лише тут.
- `NO_SERVOS=1` повертає заглушку `_NoOpServo`, тож диспетчер працює завжди
  (без гілки `if dispatcher is not None`).
- Майбутній привід (інша плата, ROS, імітатор) — нова гілка у фабриці,
  Main без змін.
"""
from __future__ import annotations

import logging
import os
import sys

from commands.base import HorizontalServoLike, VerticalServoLike

logger = logging.getLogger(__name__)

SERVO_PATH_DEFAULT = "/home/pi/test-toggle/half_duplex_transmit/servos"


class _NoOpServo:
    """Заглушка для розробки без заліза (NO_SERVOS=1) — реалізує обидва протоколи."""

    def move_vertical(self, direction: str) -> None:
        logger.debug("noop servo: move_vertical(%s)", direction)

    def move_horizontal(self, direction: str) -> None:
        logger.debug("noop servo: move_horizontal(%s)", direction)


def _servoexecutor():
    """Лінивий імпорт `servoexecutor` із Pi-дерева (лише на Pi)."""
    sys.path.insert(0, os.environ.get("SERVO_PATH", SERVO_PATH_DEFAULT))
    import servoexecutor  # noqa: PLC0415 — імпорт навмисно відкладено сюди

    return servoexecutor


class ServoFactory:
    """Створює приводи. Кожен `create_*` повертає протокол, не конкретний клас."""

    @staticmethod
    def create_vertical() -> VerticalServoLike:
        if os.environ.get("NO_SERVOS") == "1":
            logger.info("NO_SERVOS=1 — vertical servo stubbed")
            return _NoOpServo()
        v_servo = _servoexecutor().VerticalServo()
        logger.info("vertical servo initialized")
        return v_servo

    @staticmethod
    def create_horizontal() -> HorizontalServoLike:
        if os.environ.get("NO_SERVOS") == "1":
            logger.info("NO_SERVOS=1 — horizontal servo stubbed")
            return _NoOpServo()
        h_servo = _servoexecutor().HorizontalServo()
        logger.info("horizontal servo initialized")
        return h_servo
