from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
from absl import logging

from configs.base import DeferTrainConfig
from train.meters import DeferMeter


class DeferModel(ABC, nn.Module):
  def __init__(self, num_classes: int = 100) -> None:
    super().__init__()
    self.num_classes = num_classes

  @abstractmethod
  def model_to_ddp(self, rank: int) -> None:
    pass

  @abstractmethod
  def defer_value(self, x: torch.Tensor, small_out: torch.Tensor) -> torch.Tensor:
    """This should return a bool tensor.
    `small_out` is assumed to return logits of a model."""
    pass

  def defer(
    self, x: torch.Tensor, small_out: torch.Tensor, rej_cost: float
  ) -> torch.Tensor:
    """This should return a bool tensor."""
    return self.defer_value(x, small_out) <= rej_cost

  def forward(
    self, x: torch.Tensor, small_out: torch.Tensor, rej_cost: float
  ) -> torch.Tensor:
    return self.defer(x, small_out, rej_cost)

  def __call__(
    self, x: torch.Tensor, small_out: torch.Tensor, rej_cost: float
  ) -> torch.Tensor:
    return self.defer(x, small_out, rej_cost)


TDeferModel = TypeVar("TDeferModel", bound=DeferModel)


class DeferTrainer(ABC, Generic[TDeferModel]):
  def __init__(
    self,
    small_model: nn.Module,
    large_model: nn.Module,
    defer_model: TDeferModel,
    eval_tradeoff: bool = True,
  ) -> None:
    self.small_model = small_model
    self.large_model = large_model
    self.defer_model = defer_model

    # Eval setting
    self.eval_tradeoff = eval_tradeoff

    self.optimizer: optim.Optimizer | None = None
    self.scheduler: optim.lr_scheduler.LRScheduler | None = None

  @abstractmethod
  def _train_batch_step(
    self,
    data: torch.Tensor,
    target: torch.Tensor,
    config: DeferTrainConfig,
    metrics: DeferMeter,
    batch_idx: int,
    epoch: int,
  ) -> torch.Tensor | None:
    pass

  def train_epoch_step(
    self,
    config: DeferTrainConfig,
    metrics: DeferMeter,
    device: torch.device,
    train_sampler: utils.Sampler[Any],
    train_loader: utils.DataLoader[Any],
    epoch: int,
    rank: int,
    use_multi_gpu: bool,
  ) -> None:
    # Set up sampler for DDP
    if use_multi_gpu and isinstance(train_sampler, utils.DistributedSampler):
      train_sampler.set_epoch(epoch)

    # Set train
    self.defer_model.train()

    # Set up metric
    metrics.reset()
    for batch_idx, (data, target) in enumerate(train_loader):  # noqa: F402
      metrics.timer.mark_batch_start()

      data = data.to(device, non_blocking=True)
      target = target.to(device, non_blocking=True)

      defer_loss = self._train_batch_step(
        data,
        target,
        config,
        metrics,
        batch_idx,
        epoch,
      )
      metrics.timer.update_train_time()

      # Update metrics for defer loss
      if defer_loss is not None:
        metrics.defer_loss.update(defer_loss.item(), target.size(0))

      # Evaluate trade-offs
      if self.eval_tradeoff:
        small_output = self.small_model(data)
        large_output = self.large_model(data)
        defer_values = self.defer_model.defer_value(data, small_output)
        metrics.trade_off.update(defer_values, small_output, large_output, target)

      # Time batch & reset
      metrics.timer.update_eval_time()

      # print info
      if rank == 0 and batch_idx % config.log_every_steps == 0:
        prefix_log = f"Epoch:\t[{epoch}][{batch_idx}/{len(train_loader)}]\t"
        metric_log = metrics.log_str()
        logging.info(prefix_log + metric_log)

    if self.scheduler is not None:
      self.scheduler.step()

  @abstractmethod
  def _test_batch_step(
    self,
    data: torch.Tensor,
    target: torch.Tensor,
    config: DeferTrainConfig,
    metrics: DeferMeter,
    batch_idx: int,
    epoch: int,
  ) -> torch.Tensor | None:
    pass

  def test_epoch_step(
    self,
    config: DeferTrainConfig,
    metrics: DeferMeter,
    device: torch.device,
    test_sampler: utils.Sampler[Any],
    test_loader: utils.DataLoader[Any],
    epoch: int,
    rank: int,
    use_multi_gpu: bool,
  ) -> None:
    # Set up sampler for DDP
    if use_multi_gpu and isinstance(test_sampler, utils.DistributedSampler):
      test_sampler.set_epoch(epoch)

    # Set eval mode
    self.defer_model.eval()

    # Set up metric
    metrics.reset()
    with torch.no_grad():
      for batch_idx, (data, target) in enumerate(test_loader):  # noqa: F402
        metrics.timer.mark_batch_start()

        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        defer_loss = self._test_batch_step(
          data,
          target,
          config,
          metrics,
          batch_idx,
          epoch,
        )

        # Update metrics for defer loss
        if defer_loss is not None:
          metrics.defer_loss.update(defer_loss.item(), target.size(0))

        # Evaluate trade-offs
        if self.eval_tradeoff:
          small_output = self.small_model(data)
          large_output = self.large_model(data)
          defer_values = self.defer_model.defer_value(data, small_output)
          metrics.trade_off.update(defer_values, small_output, large_output, target)

        # Time batch & reset
        metrics.timer.update_eval_time()

        # print info
        if rank == 0 and batch_idx % config.log_every_steps == 0:
          prefix_log = f"Test:\t[{batch_idx}/{len(test_loader)}]\t"
          metric_log = metrics.log_str(no_train_time=True)
          logging.info(prefix_log + metric_log)
