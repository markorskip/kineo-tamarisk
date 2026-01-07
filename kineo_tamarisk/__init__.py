"""kineo_tamarisk: Image-based salt cedar (tamarisk) identification."""

from kineo_tamarisk.model.classifier import (
    TamariskClassifier,
    default_transforms,
    load_classifier,
)

__all__ = ["TamariskClassifier", "default_transforms", "load_classifier"]
__version__ = "0.1.0"
