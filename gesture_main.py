"""Сумісний шим: режим `local` (веб-камера) єдиної точки входу `main.Main`.

Уся логіка тепер у `main.py` (клас `Main`). Конфіг за замовчуванням —
`config_local_webcam.yaml` (MediaPipe + правила). Залишено для сумісності:

    PYTHONPATH=. python3 gesture_main.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from main import main

_DEFAULT = Path(__file__).parent / "gesture_control" / "config_local_webcam.yaml"


def _has_config(argv: list[str]) -> bool:
    return any(a == "--config" or a.startswith("--config=") for a in argv)


if __name__ == "__main__":
    argv = sys.argv[1:]
    if not _has_config(argv):
        argv = ["--config", str(_DEFAULT), *argv]
    main(argv)
