from __future__ import annotations

import logging
import time

from commands.base import Command, CommandContext, VerticalServoLike

logger = logging.getLogger(__name__)

STEPS = 3
INTER_STEP_SLEEP = 0.05
HOLD_SLEEP = 1.0


class ThumbsUpCommand(Command):
    """Tilt up by STEPS * 5 degrees, hold, then return by the same amount.
    Uses the existing 5-degree-per-call API on VerticalServo. Total ~1.5 s.

    Використовує лише вертикальний привід (Д4: `_v`).
    """

    def __init__(self, v_servo: VerticalServoLike) -> None:
        self._v = v_servo

    def execute(self, ctx: CommandContext) -> None:
        logger.info("thumbs_up: tilting up %d steps", STEPS)
        for _ in range(STEPS):
            self._v.move_vertical("up")
            time.sleep(INTER_STEP_SLEEP)
        time.sleep(HOLD_SLEEP)
        logger.info("thumbs_up: returning %d steps", STEPS)
        for _ in range(STEPS):
            self._v.move_vertical("down")
            time.sleep(INTER_STEP_SLEEP)
