"""Сумісний шим: режим `stream` єдиної точки входу `main.Main`.

Уся логіка тепер у `main.py` (клас `Main`). Цей файл лишено, щоб не ламати
наявні команди з RUN.md:

    PYTHONPATH=. python3 stream_recognized.py --config gesture_control/config_onnx_stream.yaml
    NO_SERVOS=1 PYTHONPATH=. python3 stream_recognized.py --config ...
"""
from __future__ import annotations

import sys
from pathlib import Path

from main import main

_DEFAULT = Path(__file__).parent / "gesture_control" / "config_onnx_stream.yaml"


def _has_config(argv: list[str]) -> bool:
    return any(a == "--config" or a.startswith("--config=") for a in argv)


if __name__ == "__main__":
    argv = sys.argv[1:]
    if not _has_config(argv):
        argv = ["--config", str(_DEFAULT), *argv]
    main(argv)
