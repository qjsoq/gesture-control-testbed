from typing import Callable, Dict, TypeVar, Type
from pydantic import BaseModel

from gesture_control.sources.generic_source_config import SourceConfig
T = TypeVar("T", bound=SourceConfig)
SOURCE_CONFIGS: Dict[str, Type[SourceConfig]] = {}
SOURCE_BUILDERS: Dict[str, Callable] = {}

def register_source(source_type: str, builder: Callable):
    def wrapper(config_class: Type[SourceConfig]):
        SOURCE_CONFIGS[source_type] = config_class
        SOURCE_BUILDERS[source_type] = builder
        return config_class
    return wrapper