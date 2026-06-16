from __future__ import annotations

from commands.base import Command, HorizontalServoLike, VerticalServoLike
from commands.follow_palm import FollowPalmCommand
from commands.nod_state import VerticalNodState
from commands.return_neutral import ReturnNeutralCommand
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
        # Спільний стан ноду: `like` нодне раз і тримає, `no_gesture` повертає
        # в нейтраль і знову озброює (re-arm).
        nod_state = VerticalNodState()
        follow_palm = FollowPalmCommand(v_servo, h_servo)
        return {
            "palm": follow_palm,
            "open": follow_palm,   # MediaPipe rule_based: розкрита долоня = стеження
            "like": ThumbsUpCommand(v_servo, nod_state),
            "no_gesture": ReturnNeutralCommand(v_servo, nod_state),
        }


# Модульний аліас — зворотна сумісність та зручний імпорт.
build_default_registry = CommandRegistry.build_default_registry
