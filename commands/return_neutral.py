from __future__ import annotations

import logging
import time

from commands.base import Command, CommandContext, VerticalServoLike
from commands.nod_state import VerticalNodState

logger = logging.getLogger(__name__)

INTER_STEP_SLEEP = 0.05


class ReturnNeutralCommand(Command):
    """Реакція на `no_gesture` (тобто `like` зник з кадру): повернути
    вертикальний привід у нейтраль і зняти `engaged`, щоб наступна поява
    `like` знову нодла.

    Нейтраль = чистий зсув `state.offset` зведений до 0. Якщо нод був
    симетричним (вгору=вниз), зсув уже 0 і руху не буде — лише re-arm.
    """

    def __init__(self, v_servo: VerticalServoLike, state: VerticalNodState) -> None:
        self._v = v_servo
        self._state = state

    def execute(self, ctx: CommandContext) -> None:
        if not self._state.engaged and self._state.offset == 0:
            return  # нема що робити — вже в нейтралі й не зведено

        if self._state.offset != 0:
            direction = "down" if self._state.offset > 0 else "up"
            steps = abs(self._state.offset)
            logger.info("return_neutral: recentering %d steps %s", steps, direction)
            for _ in range(steps):
                self._v.move_vertical(direction)
                time.sleep(INTER_STEP_SLEEP)
            self._state.offset = 0

        if self._state.engaged:
            logger.info("return_neutral: like gone — re-arming")
            self._state.engaged = False
