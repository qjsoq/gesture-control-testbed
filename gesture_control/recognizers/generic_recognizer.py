from typing import Protocol

from gesture_control.recognizers.gesture import Features, GesturePrediction
import numpy as np

class GestureRecognizer(Protocol):
    def predict(self, frame: np.ndarray) -> GesturePrediction | None: ...


class FeatureExtractor(Protocol):
    def extract(self, frame: np.ndarray) -> Features | None: ...


class GestureClassifier(Protocol):
    def classify(self, features: Features) -> GesturePrediction | None: ...
