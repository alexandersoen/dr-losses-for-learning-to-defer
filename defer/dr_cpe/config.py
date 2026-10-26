from dataclasses import dataclass
from enum import Enum
from typing import Callable

import torch.nn as nn
import torch.nn.modules.loss as L

from loss.loss import from_str_to_criterion
from loss.partial import BinaryPartialLoss, LogisticLoss, from_str_to_partial


class Criterion(Enum):
  CROSS_ENTROPY = "cross-entropy"
  MATSUSHITA = "matsushita"
  MAX_PROB_CROSS_ENTROPY = "max-prob-cross-entropy"
  MAX_NEG_PROB = "max-neg-prob"

  # Zero-one stuff
  ZERO_ONE = "zero-one"
  PROB_TARGET = "prob-target"

  # "Robust Losses"
  GCE = "gce"
  FOCAL = "focal"

  # Top K stuff
  TOP_K_PROB_TARGET = "top-k-prob-target"
  TOP_K_ZERO_ONE = "top-k-zero-one"


def to_torch_criterion(criterion: Criterion | str) -> L._Loss:
  criterion_str = criterion.value if isinstance(criterion, Criterion) else criterion
  return from_str_to_criterion(criterion_str)


class _DRCPEConfigMixin:
  criterion: Criterion
  partial_loss_factory: Callable[[], BinaryPartialLoss]

  def build_criterion(self) -> nn.Module:
    return to_torch_criterion(self.criterion)

  def build_partial_loss(self) -> BinaryPartialLoss:
    return self.partial_loss_factory()


@dataclass(frozen=True)
class DRCPEConfig(_DRCPEConfigMixin):
  criterion: Criterion = Criterion.CROSS_ENTROPY
  partial_loss_factory: Callable[[], BinaryPartialLoss] = LogisticLoss
  num_classes: int = 100
  temp: float = 1.0
  small_gamma: float = 1.0
  large_gamma: float = 1.0
  large_posthoc: bool = False
  threshold_target: str = "scorer"

  @classmethod
  def from_strings(
    cls,
    *,
    criterion: str,
    partial_loss: str,
    num_classes: int,
    temp: float,
    small_gamma: float,
    large_gamma: float,
    large_posthoc: bool,
    threshold_target: str = "scorer",
  ) -> "DRCPEConfig":
    return cls(
      criterion=Criterion(criterion),
      partial_loss_factory=lambda: from_str_to_partial(partial_loss),
      num_classes=num_classes,
      temp=temp,
      small_gamma=small_gamma,
      large_gamma=large_gamma,
      large_posthoc=large_posthoc,
      threshold_target=threshold_target,
    )
