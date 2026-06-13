from typing import Callable

from gesture_control.recognizers.generic_recognizer import FeatureExtractor
from gesture_control.recognizers.extractors.mediapipe_hands import (
    MediaPipeHandsConfig,
    MediaPipeHandsExtractor,
)

# When a second extractor is added, switch this alias to a discriminated union:
#   ExtractorConfig = Annotated[
#       Union[MediaPipeHandsConfig, OpenPoseConfig],
#       Field(discriminator="type"),
#   ]
ExtractorConfig = MediaPipeHandsConfig

_BUILDERS: dict[type, Callable[[object], FeatureExtractor]] = {
    MediaPipeHandsConfig: lambda c: MediaPipeHandsExtractor(c),
}


def build_extractor(cfg: object) -> FeatureExtractor:
    builder = _BUILDERS.get(type(cfg))
    if builder is None:
        raise ValueError(f"No extractor builder registered for {type(cfg).__name__}")
    return builder(cfg)


__all__ = ["ExtractorConfig", "MediaPipeHandsConfig", "build_extractor"]
