from pydantic import BaseModel
from typing import Callable, ClassVar, Any, Dict, Type

from gesture_control.sources.video_source import VideoSource

class SourceConfig(BaseModel):
    type: str  
