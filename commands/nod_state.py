from __future__ import annotations

import time


class VerticalNodState:
    """Спільний стан вертикального приводу між `ThumbsUpCommand` (мітка `like`)
    та `ReturnNeutralCommand` (мітка `no_gesture`).

    Привід (`servoexecutor.VerticalServo`) має лише крокову API
    `move_vertical("up"/"down")` (±5°) і клемпить кут у [`ANGLE_MIN`,`ANGLE_MAX`];
    абсолютної установки нема. Тож тут ДЗЕРКАЛИМО кут (`angle`, старт 0 = нейтраль)
    тією ж арифметикою, щоб знати позицію і доводити привід до потрібного кута.

    - `engaged` — `like` уже відпрацьовано, привід тримається (поки `like` у кадрі).
    - `angle`   — поточний дзеркалений кут приводу.

    Один екземпляр створюється у `CommandRegistry` і впорскується в обидві команди.
    """

    ANGLE_MIN = -27   # межі з servoexecutor.VerticalServo
    ANGLE_MAX = 28
    NEUTRAL = 0
    STEP = 5

    __slots__ = ("engaged", "angle")

    def __init__(self) -> None:
        self.engaged: bool = False
        self.angle: int = 0


def _clamp(a: int) -> int:
    return max(VerticalNodState.ANGLE_MIN, min(VerticalNodState.ANGLE_MAX, a))


def drive_to(servo, state: VerticalNodState, target: int, *, step_sleep: float = 0.05) -> int:
    """Покроково веде вертикальний привід до `target`° дзеркаленою арифметикою.

    Крок 5°, але межі клемпа (−27/+28) не кратні 5 — тому зупиняємось на
    досяжному куті, найближчому до цілі (без осциляції навколо нейтралі).
    """
    target = _clamp(target)
    guard = 0
    while state.angle != target and guard < 40:
        guard += 1
        if state.angle < target:
            nxt = _clamp(state.angle + VerticalNodState.STEP)
            direction = "up"
        else:
            nxt = _clamp(state.angle - VerticalNodState.STEP)
            direction = "down"
        if nxt == state.angle:
            break  # уперлись у межу — далі не зрушити
        if abs(nxt - target) >= abs(state.angle - target):
            break  # наступний крок перестрибує ціль — поточний найближчий
        servo.move_vertical(direction)
        state.angle = nxt
        time.sleep(step_sleep)
    return state.angle
