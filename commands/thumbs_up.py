from __future__ import annotations

import logging
import time

from commands.base import Command, CommandContext, VerticalServoLike
from commands.nod_state import VerticalNodState

logger = logging.getLogger(__name__)

STEPS = 3
INTER_STEP_SLEEP = 0.05


class ThumbsUpCommand(Command):
    """Реакція на мітку `like`: нахилити вертикальний привід ВНИЗ на STEPS кроків
    і ТРИМАТИ це положення поки `like` лишається в кадрі.

    Edge-trigger через спільний `VerticalNodState`: поки `engaged=True` повторні
    кадри з `like` нічого не роблять (привід стоїть нахилений). Прапорець скидає
    `ReturnNeutralCommand` коли `like` зникає (мітка `no_gesture`) — привід
    повертається в нейтраль, а наступна поява `like` знову нахилить вниз.

    Використовує лише вертикальний привід (Д4: `_v`) та крокову 5°-API.
    """

    def __init__(self, v_servo: VerticalServoLike, state: VerticalNodState) -> None:
        self._v = v_servo
        self._state = state

    def execute(self, ctx: CommandContext) -> None:
        if self._state.engaged:
            return  # уже нахилено вниз — тримаємось, поки `like` у кадрі

        logger.info("thumbs_up: tilting down %d steps and holding", STEPS)
        for _ in range(STEPS):
            self._v.move_vertical("down")
            self._state.offset -= 1
            time.sleep(INTER_STEP_SLEEP)

        self._state.engaged = True
