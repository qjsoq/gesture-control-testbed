from __future__ import annotations

from commands.base import Command, HorizontalServoLike, VerticalServoLike
from commands.follow_palm import FollowPalmCommand
from commands.noop import NoOpCommand
from commands.thumbs_up import ThumbsUpCommand


class CommandRegistry:
    """Д4: «factory» — зв'язує HaGRID-мітки з командами та їх приводами.

    Це єдине місце, де приводи передаються в команди. Додати нову прив'язку =
    новий рядок у `build_default_registry`.
    """

    @staticmethod
    def build_default_registry(
        v_servo: VerticalServoLike, h_servo: HorizontalServoLike
    ) -> dict[str, Command]:
        return {
            "palm": FollowPalmCommand(v_servo, h_servo),
            "like": ThumbsUpCommand(v_servo),
            "no_gesture": NoOpCommand(),
        }


# Модульний аліас — зворотна сумісність та зручний імпорт.
build_default_registry = CommandRegistry.build_default_registry
