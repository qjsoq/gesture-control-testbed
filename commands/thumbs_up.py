from __future__ import annotations

import logging

from commands.base import Command, CommandContext, VerticalServoLike
from commands.nod_state import VerticalNodState, drive_to

logger = logging.getLogger(__name__)


class ThumbsUpCommand(Command):
    """Реакція на мітку `like` (фронт): підняти вертикальний привід ВГОРУ до
    максимуму, потім опустити ВНИЗ до мінімуму і ТРИМАТИ це положення поки
    `like` лишається в кадрі.

    Edge-trigger через спільний `VerticalNodState`: поки `engaged=True` повторні
    кадри з `like` нічого не роблять (привід стоїть унизу). Прапорець скидає
    `ReturnNeutralCommand` коли `like` зникає (мітка `no_gesture`) — привід
    повертається в нейтраль, наступна поява `like` повторює цикл.

    Використовує лише вертикальний привід (Д4: `_v`).
    """

    def __init__(self, v_servo: VerticalServoLike, state: VerticalNodState) -> None:
        self._v = v_servo
        self._state = state

    def execute(self, ctx: CommandContext) -> None:
        if self._state.engaged:
            return  # уже відпрацьовано — тримаємось, поки `like` у кадрі

        logger.info("thumbs_up: sweep UP to max (%d)", VerticalNodState.ANGLE_MAX)
        drive_to(self._v, self._state, VerticalNodState.ANGLE_MAX)
        logger.info("thumbs_up: tilt DOWN to min (%d) and hold", VerticalNodState.ANGLE_MIN)
        drive_to(self._v, self._state, VerticalNodState.ANGLE_MIN)

        self._state.engaged = True
