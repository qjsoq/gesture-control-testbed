from __future__ import annotations

import logging
import time

from commands.base import Command, CommandContext, VerticalServoLike
from commands.nod_state import VerticalNodState

logger = logging.getLogger(__name__)

STEPS = 3
INTER_STEP_SLEEP = 0.05
HOLD_SLEEP = 1.0


class ThumbsUpCommand(Command):
    """Реакція на мітку `like`: один нод (вгору STEPS кроків, пауза, вниз STEPS),
    далі привід ТРИМАЄТЬСЯ поки `like` лишається в кадрі.

    Edge-trigger через спільний `VerticalNodState`: поки `engaged=True` повторні
    кадри з `like` нічого не роблять. Прапорець скидає `ReturnNeutralCommand`
    коли `like` зникає (мітка `no_gesture`), тож наступна поява знову нодне.

    Використовує лише вертикальний привід (Д4: `_v`) та крокову 5°-API.
    """

    def __init__(self, v_servo: VerticalServoLike, state: VerticalNodState) -> None:
        self._v = v_servo
        self._state = state

    def execute(self, ctx: CommandContext) -> None:
        if self._state.engaged:
            return  # нод уже зроблено — тримаємось, поки `like` у кадрі

        logger.info("thumbs_up: nod up %d steps", STEPS)
        for _ in range(STEPS):
            self._v.move_vertical("up")
            self._state.offset += 1
            time.sleep(INTER_STEP_SLEEP)
        time.sleep(HOLD_SLEEP)
        logger.info("thumbs_up: nod down %d steps", STEPS)
        for _ in range(STEPS):
            self._v.move_vertical("down")
            self._state.offset -= 1
            time.sleep(INTER_STEP_SLEEP)

        self._state.engaged = True
