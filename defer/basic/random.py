import torch

from configs.base import DeferTrainConfig
from defer.base import DeferModel, DeferTrainer
from train.meters import DeferMeter


class RandomDefer(DeferModel):
  """Completely random defer mechanism.
  This should just be a linear interpolation of the performance between model and
  expert.
  """

  def __init__(self, num_classes: int = 100) -> None:
    super().__init__(num_classes)

  def defer_value(self, x: torch.Tensor, small_out: torch.Tensor) -> torch.Tensor:
    """This should return a bool tensor."""
    del small_out

    rand_v = torch.empty(x.shape[0]).uniform_().to(x.device)
    return rand_v


class RandomDeferTrainer(DeferTrainer[RandomDefer]):
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
