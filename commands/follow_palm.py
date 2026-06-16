from __future__ import annotations

import logging

from commands.base import (
    Command,
    CommandContext,
    HorizontalServoLike,
    VerticalServoLike,
)

logger = logging.getLogger(__name__)

LEFT_THRESHOLD = 0.33
RIGHT_THRESHOLD = 0.67
TOP_THRESHOLD = 0.33
BOTTOM_THRESHOLD = 0.67


class FollowPalmCommand(Command):
    """Pulse servos toward the palm position. One frame yields at most one
    horizontal pulse + one vertical step. Center deadzone is the middle third
    along each axis.

    Приводи (receivers) тримаються в самій команді (Д4: `_v`, `_h`).
    """

    def __init__(self, v_servo: VerticalServoLike, h_servo: HorizontalServoLike) -> None:
        self._v = v_servo
        self._h = h_servo

    def execute(self, ctx: CommandContext) -> None:
        x, y = ctx.hand_x_norm, ctx.hand_y_norm
        if x is None and y is None:
            return

        if x is not None:
            if x < LEFT_THRESHOLD:
                logger.debug("follow_palm: horizontal left (x=%.3f)", x)
                self._h.move_horizontal("left")
            elif x > RIGHT_THRESHOLD:
                logger.debug("follow_palm: horizontal right (x=%.3f)", x)
                self._h.move_horizontal("right")

        if y is not None:
            if y < TOP_THRESHOLD:
                logger.debug("follow_palm: vertical up (y=%.3f)", y)
                self._v.move_vertical("up")
            elif y > BOTTOM_THRESHOLD:
                logger.debug("follow_palm: vertical down (y=%.3f)", y)
                self._v.move_vertical("down")
