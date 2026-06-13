from typing import Annotated, Callable, Union, Type

from pydantic import BaseModel, Field

from gesture_control.sources.video_source import VideoSource
from gesture_control.sources.file import FileSource, FileSourceConfig
from gesture_control.sources.image import ImageSource, ImageSourceConfig
from gesture_control.sources.webcam import WebcamConfig, WebcamSource
from gesture_control.sources.generic_source_config import SourceConfig
from gesture_control.sources.registry import SOURCE_BUILDERS, SOURCE_CONFIGS
from . import webcam, file, image

def build_source(cfg: SourceConfig) -> VideoSource:
    builder: Callable[..., VideoSource] | None = SOURCE_BUILDERS.get(cfg.type)
    if builder is None:
        raise ValueError(f"No video-source builder registered for {type(cfg).__name__}")
    return builder(cfg)

def get_source_config(type: str)-> Type[SourceConfig] | None:
    return SOURCE_CONFIGS.get(type)

__all__ = [
    "SourceConfig",
    "build_source"
]
