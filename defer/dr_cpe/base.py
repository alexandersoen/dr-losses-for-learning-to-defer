from abc import abstractmethod
from typing import cast

import torch
import torch.nn as nn
import torch.optim as optim

from configs.base import DeferTrainConfig
from defer.base import DeferModel, DeferTrainer
from defer.dr_cpe.config import DRCPEConfig
from loss.partial import EPS, BinaryPartialLoss, scorer_constraint_module
from model.posthoc import PostHocLargeMLP, PostHocMLP
from train.meters import DeferMeter
from train.utils import model_to_ddp

WEIGHT_EPS = 1e-3
MIN_LOG_WEIGHT = -20.0
MAX_LOG_WEIGHT = 20.0


class _DRCPEDeferBase(DeferModel):
  partial_loss: BinaryPartialLoss

  def __init__(self, config: DRCPEConfig) -> None:
    super().__init__(config.num_classes)
    self.config = config
    self.partial_loss = config.build_partial_loss()

    # Select which architecture to use
    if config.large_posthoc:
      posthoc_mlp = PostHocLargeMLP(config.num_classes)
    else:
      posthoc_mlp = PostHocMLP(config.num_classes, temp=config.temp)

    scorer_constraint = scorer_constraint_module(self.partial_loss)
    if scorer_constraint is not None:
      self.posthoc_mlp = nn.Sequential(
        posthoc_mlp,
        scorer_constraint,
      )
    else:
      self.posthoc_mlp = posthoc_mlp

    # Defer Rule
    self.temp = config.temp

  def model_to_ddp(self, rank: int) -> None:
    self.posthoc_mlp = model_to_ddp(self.posthoc_mlp, rank)

  def scorer_value(self, x: torch.Tensor, small_out: torch.Tensor) -> torch.Tensor:
    return self.posthoc_mlp((x, small_out)).reshape(-1)

  def density_ratio_value(
    self, x: torch.Tensor, small_out: torch.Tensor
  ) -> torch.Tensor:
    v = self.scorer_value(x, small_out)
    p = torch.clamp(self.partial_loss.inv_link(v), min=EPS, max=1 - EPS)
    return p / (1 - p)

  def defer_value(self, x: torch.Tensor, small_out: torch.Tensor) -> torch.Tensor:
    # By Lemma 8, thresholding the density ratio is equivalent to thresholding
    # the learned scorer under the corresponding link-transformed threshold.
    if self.config.threshold_target == "scorer":
      return self.scorer_value(x, small_out)
    if self.config.threshold_target == "density-ratio":
      return self.density_ratio_value(x, small_out) - 1.0
    raise ValueError(f"Unknown threshold_target: {self.config.threshold_target}")


class DRCPEDefer(_DRCPEDeferBase):
  config: DRCPEConfig

  def __init__(self, config: DRCPEConfig) -> None:
    super().__init__(config)
    self.small_gamma = config.small_gamma
    self.large_gamma = config.large_gamma


class _DRCPEDeferTrainer(DeferTrainer[_DRCPEDeferBase]):
  criterion: nn.Module

  def __init__(
    self,
    small_model: nn.Module,
    large_model: nn.Module,
    defer_model: _DRCPEDeferBase,
    eval_tradeoff: bool = True,
  ) -> None:
    super().__init__(small_model, large_model, defer_model, eval_tradeoff)
    self.criterion = defer_model.config.build_criterion()

    self.optimizer = optim.Adam(
      self.defer_model.posthoc_mlp.parameters(),
      lr=0.0007,
      weight_decay=0.001,
    )

  def weights(
    self,
    loss: torch.Tensor,
    gamma: float,
  ) -> torch.Tensor:
    log_w = torch.clamp(-loss / gamma, min=MIN_LOG_WEIGHT)
    w = torch.exp(log_w) + WEIGHT_EPS
    return w

  @property
  @abstractmethod
  def small_gamma(self) -> float:
    pass

  @property
  @abstractmethod
  def large_gamma(self) -> float:
    pass

  def batch_loss(self, data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    small_out = self.small_model(data)
    large_out = self.large_model(data)
    pred_out = self.defer_model.posthoc_mlp((data, small_out))

    # Calculate loss
    loss_small = self.criterion(small_out, target)
    loss_large = self.criterion(large_out, target)

    # Partials
    pos_loss = self.defer_model.partial_loss.pos_loss(pred_out)
    neg_loss = self.defer_model.partial_loss.neg_loss(pred_out)

    # Weights
    pos_w = self.weights(
      loss_small,
      self.small_gamma,
    )
    neg_w = self.weights(
      loss_large,
      self.large_gamma,
    )

    # Double ideal distribution reweighting
    total_loss = (pos_w * pos_loss).mean() + (neg_w * neg_loss).mean()

    return total_loss

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

    # Using density ratio loss
    loss = self.batch_loss(data, target)

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

    loss = self.batch_loss(data, target)
    return loss


class DRCPEDeferTrainer(_DRCPEDeferTrainer):
  defer_model: DRCPEDefer

  def __init__(
    self,
    small_model: nn.Module,
    large_model: nn.Module,
    defer_model: DRCPEDefer,
    eval_tradeoff: bool = True,
  ) -> None:
    super().__init__(small_model, large_model, defer_model, eval_tradeoff)

  @property
  def small_gamma(self) -> float:
    return self.defer_model.small_gamma

  @property
  def large_gamma(self) -> float:
    return self.defer_model.large_gamma
