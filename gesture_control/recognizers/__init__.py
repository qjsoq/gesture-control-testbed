from typing import Annotated, Callable, Union

from pydantic import Field

from gesture_control.recognizers.generic_recognizer import GestureRecognizer
from gesture_control.recognizers.classifiers import ClassifierConfig, build_classifier
from gesture_control.recognizers.extractors import ExtractorConfig, build_extractor
from gesture_control.recognizers.two_stage import (
    TwoStageRecognizer,
    TwoStageRecognizerConfig,
)
from gesture_control.recognizers.yolo import YoloRecognizer, YoloRecognizerConfig
from gesture_control.recognizers.onnx_yolo import (
    OnnxYoloRecognizer,
    OnnxYoloRecognizerConfig,
)

RecognizerConfig = Annotated[
    Union[TwoStageRecognizerConfig, YoloRecognizerConfig, OnnxYoloRecognizerConfig],
    Field(discriminator="type"),
]

class RecognizerFactory:
    """Д4: «factory» — обирає клас розпізнавача за типом Pydantic-конфіга.

    Кожен конкретний розпізнавач реєструється у `_BUILDERS` через свій
    `from_config`. Додати новий розпізнавач = новий рядок тут, точки збирання
    (Main) без змін.
    """

    _BUILDERS: dict[type, Callable[[object], GestureRecognizer]] = {
        TwoStageRecognizerConfig: lambda c: TwoStageRecognizer.from_config(c),
        YoloRecognizerConfig: lambda c: YoloRecognizer.from_config(c),
        OnnxYoloRecognizerConfig: lambda c: OnnxYoloRecognizer.from_config(c),
    }

    @staticmethod
    def build_recognizer(cfg: object) -> GestureRecognizer:
        builder = RecognizerFactory._BUILDERS.get(type(cfg))
        if builder is None:
            raise ValueError(
                f"No recognizer builder registered for {type(cfg).__name__}"
            )
        return builder(cfg)


# Модульний аліас — зворотна сумісність із наявними викликами (app.py тощо).
build_recognizer = RecognizerFactory.build_recognizer


__all__ = [
    "ClassifierConfig",
    "ExtractorConfig",
    "OnnxYoloRecognizerConfig",
    "RecognizerConfig",
    "RecognizerFactory",
    "TwoStageRecognizerConfig",
    "YoloRecognizerConfig",
    "build_classifier",
    "build_extractor",
    "build_recognizer",
]
