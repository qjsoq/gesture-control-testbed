from __future__ import annotations


class VerticalNodState:
    """Спільний стан між `ThumbsUpCommand` (мітка `like`) та
    `ReturnNeutralCommand` (мітка `no_gesture`).

    - `engaged`  — нод уже виконано і привід «тримається» (поки `like` у кадрі).
    - `offset`   — чистий зсув вертикального приводу від нейтралі у кроках
                   (+ вгору / − вниз), щоб гарантовано повернути в нейтраль
                   коли `like` зникає, навіть якщо нод завершився не по центру.

    Один екземпляр створюється у `CommandRegistry` і впорскується в обидві
    команди — так вони бачать той самий стан без зв'язку через диспетчер.
    """

    __slots__ = ("engaged", "offset")

    def __init__(self) -> None:
        self.engaged: bool = False
        self.offset: int = 0
