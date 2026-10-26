from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

import torch
import torch.nn as nn

EPS = 1e-6


class BinaryPartialLoss(ABC):
  @abstractmethod
  def pos_loss(self, v: torch.Tensor) -> torch.Tensor:
    pass

  @abstractmethod
  def neg_loss(self, v: torch.Tensor) -> torch.Tensor:
    pass

  @abstractmethod
  def inv_link(self, v: torch.Tensor) -> torch.Tensor:
    pass


###############################################################################


def from_str_to_partial(partial_str: str) -> BinaryPartialLoss:
  match partial_str:
    case "kliep":
      partial = KLIEPLoss()
    case "lsif":
      partial = LSIFLoss()
    case "square":
      partial = SquareLoss()
    case "log":
      partial = LogisticLoss()
    case "exp":
      partial = ExponentialLoss()
    case _:
      raise ValueError(f"Unknown partial str: {partial_str}")

  return partial


def from_str_to_loss_transform(
  loss_transform_str: str,
) -> Callable[[BinaryPartialLoss, Dict[str, Any]], BinaryPartialLoss]:
  match loss_transform_str:
    case "none":
      return lambda loss, _: loss
    case "fully_capped":
      return lambda loss, kwargs: FullyCappedLoss(loss, cap=kwargs["cap"])
    case "partial_capped":
      return lambda loss, kwargs: PartialCappedLoss(loss, cap=kwargs["cap"])
    case _:
      raise ValueError(f"Unknown loss transform str: {loss_transform_str}")


###############################################################################


class KLIEPLoss(BinaryPartialLoss):
  def pos_loss(self, v: torch.Tensor) -> torch.Tensor:
    return -torch.log(v)

  def neg_loss(self, v: torch.Tensor) -> torch.Tensor:
    return v

  def inv_link(self, v: torch.Tensor) -> torch.Tensor:
    return v / (1 + v)


class LSIFLoss(BinaryPartialLoss):
  def pos_loss(self, v: torch.Tensor) -> torch.Tensor:
    return -v

  def neg_loss(self, v: torch.Tensor) -> torch.Tensor:
    return 0.5 * v.square()

  def inv_link(self, v: torch.Tensor) -> torch.Tensor:
    return v / (1 + v)


class AlphaPowerLoss(BinaryPartialLoss):
  def __init__(self, alpha: float) -> None:
    super().__init__()
    self.alpha = alpha

    assert alpha > 1.0

  def pos_loss(self, v: torch.Tensor) -> torch.Tensor:
    return (1 - v.pow(self.alpha)) / self.alpha

  def neg_loss(self, v: torch.Tensor) -> torch.Tensor:
    return (v.pow(1 + self.alpha) - 1) / (1 + self.alpha)

  def inv_link(self, v: torch.Tensor) -> torch.Tensor:
    return v / (1 + v)


class SquareLoss(BinaryPartialLoss):
  def pos_loss(self, v: torch.Tensor) -> torch.Tensor:
    return (1 - v).square()

  def neg_loss(self, v: torch.Tensor) -> torch.Tensor:
    return (1 + v).square()

  def inv_link(self, v: torch.Tensor) -> torch.Tensor:
    return 2 * v - 1


class LogisticLoss(BinaryPartialLoss):
  def __init__(self) -> None:
    super().__init__()
    self.sp = nn.Softplus()
    self.sigmoid = nn.Sigmoid()

  def pos_loss(self, v: torch.Tensor) -> torch.Tensor:
    return self.sp(-v)

  def neg_loss(self, v: torch.Tensor) -> torch.Tensor:
    return self.sp(v)

  def inv_link(self, v: torch.Tensor) -> torch.Tensor:
    return self.sigmoid(v)


class ExponentialLoss(BinaryPartialLoss):
  def __init__(self) -> None:
    super().__init__()
    self.sigmoid = nn.Sigmoid()

  def pos_loss(self, v: torch.Tensor) -> torch.Tensor:
    return torch.exp(-v)

  def neg_loss(self, v: torch.Tensor) -> torch.Tensor:
    return torch.exp(v)

  def inv_link(self, v: torch.Tensor) -> torch.Tensor:
    return self.sigmoid(2 * v)


class FullyCappedLoss(BinaryPartialLoss):
  def __init__(self, base_loss: BinaryPartialLoss, cap: float) -> None:
    super().__init__()
    self.base_loss = base_loss
    self.cap = cap

  def pos_loss(self, v: torch.Tensor) -> torch.Tensor:
    capped_v = torch.clamp(v, max=self.cap)
    return self.base_loss.pos_loss(capped_v)

  def neg_loss(self, v: torch.Tensor) -> torch.Tensor:
    capped_v = torch.clamp(v, max=self.cap)
    return self.base_loss.neg_loss(capped_v)

  def inv_link(self, v: torch.Tensor) -> torch.Tensor:
    return self.base_loss.inv_link(v)


class PartialCappedLoss(BinaryPartialLoss):
  def __init__(self, base_loss: BinaryPartialLoss, cap: float) -> None:
    super().__init__()
    self.base_loss = base_loss
    self.cap = cap

  def pos_loss(self, v: torch.Tensor) -> torch.Tensor:
    return self.base_loss.pos_loss(v)

  def neg_loss(self, v: torch.Tensor) -> torch.Tensor:
    capped_v = torch.clamp(v, max=self.cap)
    return self.base_loss.neg_loss(capped_v)

  def inv_link(self, v: torch.Tensor) -> torch.Tensor:
    return self.base_loss.inv_link(v)


def scorer_constraint_module(partial_loss: BinaryPartialLoss) -> nn.Module | None:
  if isinstance(partial_loss, (KLIEPLoss, LSIFLoss, AlphaPowerLoss)):
    return nn.Softplus()
  return None
