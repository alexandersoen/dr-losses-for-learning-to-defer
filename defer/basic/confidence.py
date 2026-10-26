import torch

from configs.base import DeferTrainConfig
from defer.base import DeferModel, DeferTrainer
from train.meters import DeferMeter
from train.utils import to_probs


class ConfidenceDefer(DeferModel):
  """Defer based on the small model's confidence."""

  def __init__(self, num_classes: int = 100, temp: float = 1.0) -> None:
    super().__init__(num_classes)

    self.temp = temp

  def defer_value(self, x: torch.Tensor, small_out: torch.Tensor) -> torch.Tensor:
    """This should return a bool tensor"""
    del x

    logits = small_out
    probs = to_probs(logits, self.temp, dim=1)
    max_softmax_prob = torch.max(probs, dim=1).values
    return max_softmax_prob


class ConfidenceDeferTrainer(DeferTrainer[ConfidenceDefer]):
  @property
  def no_loop(self) -> bool:
    return True

  def _train_batch_step(
    self,
    data: torch.Tensor,
    target: torch.Tensor,
    config: DeferTrainConfig,
    metrics: DeferMeter,
    batch_idx: int,
    epoch: int,
  ) -> None:
    del data, target, config, metrics, batch_idx, epoch
    return

  def _test_batch_step(
    self,
    data: torch.Tensor,
    target: torch.Tensor,
    config: DeferTrainConfig,
    metrics: DeferMeter,
    batch_idx: int,
    epoch: int,
  ) -> None:
    del data, target, config, metrics, batch_idx, epoch
    return
