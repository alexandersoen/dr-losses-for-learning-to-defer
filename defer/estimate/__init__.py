from defer.estimate.base import EstimateDeferModel, EstimateDeferTrainer
from defer.estimate.objectives import (
  ESTIMATE_CLF_01,
  ESTIMATE_DIFF_01,
  ESTIMATE_DIFF_MPROB,
  ESTIMATE_DIFF_PROB,
  ESTIMATE_MAX_PROB,
  EstimateObjective,
)


class EstimateDiff01Defer(EstimateDeferModel):
  objective = ESTIMATE_DIFF_01


class EstimateDiffProbDefer(EstimateDeferModel):
  objective = ESTIMATE_DIFF_PROB


class EstimateMaxProbDefer(EstimateDeferModel):
  objective = ESTIMATE_MAX_PROB


class EstimateDiffMProbDefer(EstimateDeferModel):
  objective = ESTIMATE_DIFF_MPROB


class EstimateClf01Defer(EstimateDeferModel):
  objective = ESTIMATE_CLF_01


__all__ = [
  "EstimateClf01Defer",
  "EstimateDeferModel",
  "EstimateDeferTrainer",
  "EstimateDiff01Defer",
  "EstimateDiffMProbDefer",
  "EstimateDiffProbDefer",
  "EstimateMaxProbDefer",
  "EstimateObjective",
  "ESTIMATE_CLF_01",
  "ESTIMATE_DIFF_01",
  "ESTIMATE_DIFF_MPROB",
  "ESTIMATE_DIFF_PROB",
  "ESTIMATE_MAX_PROB",
]
