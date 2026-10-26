from typing import Any

from defer.base import DeferModel, DeferTrainer
from defer.estimate import (
  EstimateClf01Defer,
  EstimateDeferTrainer,
  EstimateDiff01Defer,
  EstimateDiffMProbDefer,
  EstimateDiffProbDefer,
  EstimateMaxProbDefer,
)

from .basic.confidence import ConfidenceDefer, ConfidenceDeferTrainer
from .basic.random import RandomDefer, RandomDeferTrainer

defer_model_train_class_dict: dict[
  str,
  tuple[type[DeferModel], type[DeferTrainer], dict[str, Any]],
] = {
  "conf": (
    ConfidenceDefer,
    ConfidenceDeferTrainer,
    {},
  ),
  "rand": (
    RandomDefer,
    RandomDeferTrainer,
    {},
  ),
  ###################### Estimate Family ######################
  "estimate-diff-01": (
    EstimateDiff01Defer,
    EstimateDeferTrainer,
    {},
  ),
  "estimate-diff-prob": (
    EstimateDiffProbDefer,
    EstimateDeferTrainer,
    {},
  ),
  "estimate-max-prob": (
    EstimateMaxProbDefer,
    EstimateDeferTrainer,
    {},
  ),
  "estimate-diff-mprob": (
    EstimateDiffMProbDefer,
    EstimateDeferTrainer,
    {},
  ),
  "estimate-clf-01": (
    EstimateClf01Defer,
    EstimateDeferTrainer,
    {},
  ),
}
