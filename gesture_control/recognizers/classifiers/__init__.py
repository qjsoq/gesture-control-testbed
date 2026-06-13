from typing import Callable

from gesture_control.recognizers.generic_recognizer import GestureClassifier
from gesture_control.recognizers.classifiers.rule_based import (
    RuleBasedClassifier,
    RuleBasedClassifierConfig,
)

# When a second classifier is added, switch this alias to a discriminated union:
#   ClassifierConfig = Annotated[
#       Union[RuleBasedClassifierConfig, MlpConfig],
#       Field(discriminator="type"),
#   ]
ClassifierConfig = RuleBasedClassifierConfig

_BUILDERS: dict[type, Callable[[object], GestureClassifier]] = {
    RuleBasedClassifierConfig: lambda c: RuleBasedClassifier(c),
}


def build_classifier(cfg: object) -> GestureClassifier:
    builder = _BUILDERS.get(type(cfg))
    if builder is None:
        raise ValueError(f"No classifier builder registered for {type(cfg).__name__}")
    return builder(cfg)


__all__ = ["ClassifierConfig", "RuleBasedClassifierConfig", "build_classifier"]
