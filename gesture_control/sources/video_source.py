from __future__ import annotations

from types import TracebackType
from typing import Iterator, Protocol

import numpy as np

class VideoSource(Protocol):
    def __enter__(self) -> VideoSource: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...
    def __iter__(self) -> Iterator[np.ndarray]: ...
