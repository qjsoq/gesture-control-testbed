from __future__ import annotations

import logging

from commands.base import Command, CommandContext, VerticalServoLike
from commands.nod_state import VerticalNodState, drive_to

logger = logging.getLogger(__name__)


class ReturnNeutralCommand(Command):
    """Реакція на `no_gesture` (тобто `like` зник з кадру): повернути
    вертикальний привід у нейтраль (0°) і зняти `engaged`, щоб наступна поява
    `like` знову відпрацювала цикл.
    """

    def __init__(self, v_servo: VerticalServoLike, state: VerticalNodState) -> None:
        self._v = v_servo
        self._state = state

    def execute(self, ctx: CommandContext) -> None:
        if not self._state.engaged and self._state.angle == VerticalNodState.NEUTRAL:
            return  # вже в нейтралі й нічого не тримаємо

        logger.info("return_neutral: back to neutral (%d)", VerticalNodState.NEUTRAL)
        drive_to(self._v, self._state, VerticalNodState.NEUTRAL)
        self._state.engaged = False
