from abc import ABC
from typing import cast

import torch
import torch.nn as nn
import torch.optim as optim

from configs.base import DeferTrainConfig
from defer.base import DeferModel, DeferTrainer
from defer.estimate.objectives import EstimateObjective
from model.posthoc import PostHocMLP
from train.meters import DeferMeter
from train.utils import model_to_ddp


class EstimateDeferModel(DeferModel, ABC):
  objective: EstimateObjective

  def __init__(self, num_classes: int = 100, temp: float = 1.0) -> None:
    super().__init__(num_classes)

    self.posthoc_mlp = PostHocMLP(self.num_classes, temp=temp)
    self.temp = temp

  def model_to_ddp(self, rank: int) -> None:
    self.posthoc_mlp = model_to_ddp(self.posthoc_mlp, rank)

  def defer_value(self, x: torch.Tensor, small_out: torch.Tensor) -> torch.Tensor:
    defer_out = self.posthoc_mlp((x, small_out)).squeeze()
    defer_val = self.objective.inference_score(defer_out, small_out, self.temp)
    return -1 * defer_val


class EstimateDeferTrainer(DeferTrainer[EstimateDeferModel]):
  def __init__(
    self,
    small_model: nn.Module,
    large_model: nn.Module,
    defer_model: EstimateDeferModel,
    eval_tradeoff: bool = True,
  ) -> None:
    super().__init__(small_model, large_model, defer_model, eval_tradeoff)

    self.criterion = defer_model.objective.build_criterion()
    self.optimizer = optim.Adam(
      self.defer_model.posthoc_mlp.parameters(),
      lr=0.0007,
      weight_decay=0.001,
    )

  def generate_label(self, data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return self.defer_model.objective.generate_label(
      data=data,
      target=target,
      small_model=self.small_model,
      large_model=self.large_model,
      temp=self.defer_model.temp,
    )

  def _train_batch_step(
    self,
    data: torch.Tensor,
    target: torch.Tensor,
    config: DeferTrainConfig,
    metrics: DeferMeter,
    batch_idx: int,
    epoch: int,
  ) -> torch.Tensor | None:
    del config, metrics, batch_idx, epoch

    small_out = self.small_model(data)
    label = self.defer_model.objective.generate_label(
      data=data,
      target=target,
      small_model=self.small_model,
      large_model=self.large_model,
      temp=self.defer_model.temp,
      small_out=small_out,
    )
    pred = self.defer_model.posthoc_mlp((data, small_out))
    loss = self.criterion(pred, label)

    self.optimizer = cast(optim.Adam, self.optimizer)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return loss

  def _test_batch_step(
    self,
    data: torch.Tensor,
    target: torch.Tensor,
    config: DeferTrainConfig,
    metrics: DeferMeter,
    batch_idx: int,
    epoch: int,
  ) -> torch.Tensor | None:
    del config, metrics, batch_idx, epoch

    small_out = self.small_model(data)
    label = self.defer_model.objective.generate_label(
      data=data,
      target=target,
      small_model=self.small_model,
      large_model=self.large_model,
      temp=self.defer_model.temp,
      small_out=small_out,
    )
    pred = self.defer_model.posthoc_mlp((data, small_out))
    loss = self.criterion(pred, label)

    return loss
