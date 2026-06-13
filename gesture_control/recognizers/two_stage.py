from typing import Literal

from pydantic import BaseModel

import numpy as np
from gesture_control.recognizers import ClassifierConfig, build_classifier, ExtractorConfig, build_extractor
from gesture_control.recognizers.gesture import GesturePrediction
from gesture_control.recognizers.generic_recognizer import FeatureExtractor, GestureClassifier


class TwoStageRecognizerConfig(BaseModel):
    type: Literal["two_stage"]
    extractor: ExtractorConfig
    classifier: ClassifierConfig
    


class TwoStageRecognizer:
    def __init__(self, extractor: FeatureExtractor, classifier: GestureClassifier) -> None:
        self._extractor = extractor
        self._classifier = classifier

    @classmethod
    def from_config(cls, cfg: TwoStageRecognizerConfig) -> "TwoStageRecognizer":
        return cls(
            extractor=build_extractor(cfg.extractor),
            classifier=build_classifier(cfg.classifier),
        )

    def predict(self, frame: np.ndarray) -> GesturePrediction | None:
        features = self._extractor.extract(frame)
        if features is None:
            return None
        return self._classifier.classify(features)
