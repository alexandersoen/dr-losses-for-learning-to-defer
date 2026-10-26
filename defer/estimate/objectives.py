from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.loss as L

from train.utils import to_probs

LabelFn = Callable[
  [torch.Tensor, torch.Tensor, nn.Module, nn.Module, float, torch.Tensor | None],
  torch.Tensor,
]
InferenceFn = Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]


def _identity_inference(
  defer_out: torch.Tensor,
  small_out: torch.Tensor,
  temp: float,
) -> torch.Tensor:
  del small_out, temp
  return defer_out


@dataclass(frozen=True)
class EstimateObjective:
  name: str
  criterion_factory: Callable[[], L._Loss]
  label_fn: LabelFn
  inference_fn: InferenceFn = _identity_inference

  def build_criterion(self) -> L._Loss:
    return self.criterion_factory()

  def generate_label(
    self,
    data: torch.Tensor,
    target: torch.Tensor,
    small_model: nn.Module,
    large_model: nn.Module,
    temp: float,
    small_out: torch.Tensor | None = None,
  ) -> torch.Tensor:
    return self.label_fn(data, target, small_model, large_model, temp, small_out)

  def inference_score(
    self,
    defer_out: torch.Tensor,
    small_out: torch.Tensor,
    temp: float,
  ) -> torch.Tensor:
    return self.inference_fn(defer_out, small_out, temp)


def _get_small_out(
  data: torch.Tensor,
  small_model: nn.Module,
  small_out: torch.Tensor | None,
) -> torch.Tensor:
  if small_out is None:
    return small_model(data)
  return small_out


def diff_01_label(
  data: torch.Tensor,
  target: torch.Tensor,
  small_model: nn.Module,
  large_model: nn.Module,
  temp: float,
  small_out: torch.Tensor | None = None,
) -> torch.Tensor:
  del temp
  small_logits = _get_small_out(data, small_model, small_out)
  large_out = large_model(data)

  small_ind = (torch.argmax(small_logits, dim=1) == target).float()
  large_ind = (torch.argmax(large_out, dim=1) == target).float()
  return large_ind - small_ind


def clf_01_label(
  data: torch.Tensor,
  target: torch.Tensor,
  small_model: nn.Module,
  large_model: nn.Module,
  temp: float,
  small_out: torch.Tensor | None = None,
) -> torch.Tensor:
  del temp
  small_logits = _get_small_out(data, small_model, small_out)
  large_out = large_model(data)

  small_err_ind = torch.argmax(small_logits, dim=1) != target
  large_cor_ind = torch.argmax(large_out, dim=1) == target
  return (small_err_ind * large_cor_ind).float()


def clf_01_inference(
  defer_out: torch.Tensor,
  small_out: torch.Tensor,
  temp: float,
) -> torch.Tensor:
  del small_out, temp
  return F.sigmoid(defer_out)


def diff_prob_label(
  data: torch.Tensor,
  target: torch.Tensor,
  small_model: nn.Module,
  large_model: nn.Module,
  temp: float,
  small_out: torch.Tensor | None = None,
) -> torch.Tensor:
  small_logits = _get_small_out(data, small_model, small_out)
  large_out = large_model(data)

  index = target.unsqueeze(1)
  small_probs = to_probs(small_logits, temp)
  large_probs = to_probs(large_out, temp)
  return torch.take_along_dim(large_probs - small_probs, index, dim=1).squeeze()


def max_prob_label(
  data: torch.Tensor,
  target: torch.Tensor,
  small_model: nn.Module,
  large_model: nn.Module,
  temp: float,
  small_out: torch.Tensor | None = None,
) -> torch.Tensor:
  del target, small_model, small_out

  large_out = large_model(data)
  large_probs = to_probs(large_out, temp)
  return torch.max(large_probs, dim=1).values


def max_prob_inference(
  defer_out: torch.Tensor,
  small_out: torch.Tensor,
  temp: float,
) -> torch.Tensor:
  small_probs = to_probs(small_out, temp)
  max_small_prob = torch.max(small_probs, dim=1).values
  return defer_out - max_small_prob


def diff_mprob_label(
  data: torch.Tensor,
  target: torch.Tensor,
  small_model: nn.Module,
  large_model: nn.Module,
  temp: float,
  small_out: torch.Tensor | None = None,
) -> torch.Tensor:
  del target
  small_logits = _get_small_out(data, small_model, small_out)
  large_out = large_model(data)

  small_probs = to_probs(small_logits, temp)
  large_probs = to_probs(large_out, temp)
  small_mprobs = small_probs.max(dim=1).values
  large_mprobs = large_probs.max(dim=1).values
  return large_mprobs - small_mprobs


ESTIMATE_DIFF_01 = EstimateObjective(
  name="diff-01",
  criterion_factory=nn.MSELoss,
  label_fn=diff_01_label,
)
ESTIMATE_DIFF_PROB = EstimateObjective(
  name="diff-prob",
  criterion_factory=nn.L1Loss,
  label_fn=diff_prob_label,
)
ESTIMATE_MAX_PROB = EstimateObjective(
  name="max-prob",
  criterion_factory=nn.MSELoss,
  label_fn=max_prob_label,
  inference_fn=max_prob_inference,
)
ESTIMATE_DIFF_MPROB = EstimateObjective(
  name="diff-mprob",
  criterion_factory=nn.MSELoss,
  label_fn=diff_mprob_label,
)
ESTIMATE_CLF_01 = EstimateObjective(
  name="clf-01",
  criterion_factory=nn.BCEWithLogitsLoss,
  label_fn=clf_01_label,
  inference_fn=clf_01_inference,
)
