import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.loss as L

DEFAULT_TOP_K = 5


def from_str_to_criterion(criterion_str: str) -> L._Loss:
  match criterion_str:
    case "cross-entropy":
      torch_criterion = nn.CrossEntropyLoss(reduction="none")
    case "gce":
      torch_criterion = GeneralizedCrossEntropyLoss(reduction="none", q=0.7)
    case "focal":
      torch_criterion = FocalLoss(reduction="none", gamma=2.0)
    case "max-prob-cross-entropy":
      torch_criterion = MaxProbCrossEntropy(reduction="none")
    case "max-neg-prob":
      torch_criterion = MaxNegProb(reduction="none")
    case "matsushita":
      torch_criterion = MatsushitaLoss(reduction="none")
    case "zero-one":
      torch_criterion = ZeroOneLoss(reduction="none")
    case "prob-target" | "zero-one-prob":
      torch_criterion = OneHotL2Loss(reduction="none", metric="l2")
    case "top-k-zero-one":
      torch_criterion = TopKZeroOneLoss(reduction="none", k=DEFAULT_TOP_K)
    case "top-k-prob-target":
      torch_criterion = TopKOneHotL2Loss(
        reduction="none",
        metric="l2",
        k=DEFAULT_TOP_K,
      )
    case _:
      raise ValueError(f"Unknown critersion str: {criterion_str}")

  return torch_criterion


class GeneralizedCrossEntropyLoss(L._Loss):
  def __init__(
    self,
    size_average: bool | None = None,
    reduce: bool | None = None,
    reduction: str = "mean",
    q: float = 0.7,
    eps: float = 1e-12,
  ) -> None:
    super().__init__(size_average, reduce, reduction)
    if not (0.0 < q <= 1.0):
      raise ValueError(f"q must be in (0, 1], got {q}")
    self.q = q
    self.eps = eps

  def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if input.shape == target.shape:
      raise NotImplementedError

    probs = F.softmax(input, dim=-1)
    pt = probs.gather(dim=1, index=target.unsqueeze(1)).squeeze(1)
    pt = pt.clamp_min(self.eps)
    loss = (1 - pt.pow(self.q)) / self.q

    if self.reduction == "mean":
      return loss.mean()
    if self.reduction == "sum":
      return loss.sum()
    if self.reduction == "none":
      return loss
    raise ValueError(f"Invalid reduction: {self.reduction}")


class FocalLoss(L._Loss):
  def __init__(
    self,
    size_average: bool | None = None,
    reduce: bool | None = None,
    reduction: str = "mean",
    gamma: float = 2.0,
    eps: float = 1e-12,
  ) -> None:
    super().__init__(size_average, reduce, reduction)
    if gamma < 0:
      raise ValueError(f"gamma must be >= 0, got {gamma}")
    self.gamma = gamma
    self.eps = eps

  def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if input.shape == target.shape:
      raise NotImplementedError

    log_probs = F.log_softmax(input, dim=-1)
    log_pt = log_probs.gather(dim=1, index=target.unsqueeze(1)).squeeze(1)
    pt = log_pt.exp().clamp_min(self.eps)
    loss = -torch.pow(1 - pt, self.gamma) * log_pt

    if self.reduction == "mean":
      return loss.mean()
    if self.reduction == "sum":
      return loss.sum()
    if self.reduction == "none":
      return loss
    raise ValueError(f"Invalid reduction: {self.reduction}")


class ZeroOneLoss(L._Loss):
  def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # probs = F.softmax(input, dim=-1)
    preds = torch.argmax(input, dim=-1)
    if input.shape != target.shape:
      error = (preds != target).to(torch.float)
    else:
      raise NotImplementedError

    if self.reduction == "mean":
      error = error.mean()
    elif self.reduction == "sum":
      error = error.sum()

    return error


class TopKZeroOneLoss(L._Loss):
  def __init__(
    self,
    size_average: bool | None = None,
    reduce: bool | None = None,
    reduction: str = "mean",
    k: int = DEFAULT_TOP_K,
  ) -> None:
    super().__init__(size_average, reduce, reduction)
    if k <= 0:
      raise ValueError(f"k must be positive, got {k}")
    self.k = k

  def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if input.shape == target.shape:
      raise NotImplementedError

    k = min(self.k, input.shape[-1])
    topk_indices = input.topk(k, dim=-1).indices
    correct = topk_indices.eq(target.unsqueeze(1)).any(dim=1)
    error = (~correct).to(torch.float)

    if self.reduction == "mean":
      return error.mean()
    if self.reduction == "sum":
      return error.sum()
    if self.reduction == "none":
      return error
    raise ValueError(f"Invalid reduction: {self.reduction}")


class OneHotL2Loss(L._Loss):
  def __init__(
    self,
    size_average: bool | None = None,
    reduce: bool | None = None,
    reduction: str = "mean",
    metric: str = "l1",
  ) -> None:
    super().__init__(size_average, reduce, reduction)

    if metric not in {"l1", "l2"}:
      raise ValueError(f"Unknown metric: {metric}")

    self.metric = metric

  def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    n_classes = input.shape[-1]
    probs = F.softmax(input, dim=-1)

    if input.shape != target.shape:
      target = F.one_hot(target, num_classes=n_classes)
    else:
      target = target.to(dtype=torch.long)

    target = target.to(probs.dtype)

    diff = probs - target
    if self.metric == "l1":
      error = torch.abs(diff).sum(dim=-1)
    else:
      error = torch.square(diff).sum(dim=-1)

    if self.reduction == "mean":
      error = error.mean()
    elif self.reduction == "sum":
      error = error.sum()

    return error


class TopKOneHotL2Loss(OneHotL2Loss):
  def __init__(
    self,
    size_average: bool | None = None,
    reduce: bool | None = None,
    reduction: str = "mean",
    metric: str = "l2",
    k: int = DEFAULT_TOP_K,
  ) -> None:
    super().__init__(size_average, reduce, reduction, metric)
    if k <= 0:
      raise ValueError(f"k must be positive, got {k}")
    self.k = k

  def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    n_classes = input.shape[-1]
    probs = F.softmax(input, dim=-1)
    k = min(self.k, n_classes)

    if input.shape != target.shape:
      target = F.one_hot(target, num_classes=n_classes)
    else:
      target = target.to(dtype=torch.long)

    target = target.to(probs.dtype)

    topk_indices = probs.topk(k, dim=-1).indices
    topk_mask = torch.zeros_like(probs).scatter_(1, topk_indices, 1.0)
    topk_probs = probs * topk_mask

    diff = topk_probs - target
    if self.metric == "l1":
      error = torch.abs(diff).sum(dim=-1)
    else:
      error = torch.square(diff).sum(dim=-1)

    if self.reduction == "mean":
      return error.mean()
    if self.reduction == "sum":
      return error.sum()
    if self.reduction == "none":
      return error
    raise ValueError(f"Invalid reduction: {self.reduction}")


class MaxProbCrossEntropy(L._Loss):
  def __init__(
    self,
    size_average: bool | None = None,
    reduce: bool | None = None,
    reduction: str = "mean",
  ) -> None:
    super().__init__(size_average, reduce, reduction)

    self.ce = L.CrossEntropyLoss(
      size_average=size_average, reduce=reduce, reduction=reduction
    )

  def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    del target

    preds = torch.argmax(input, dim=-1)
    return self.ce(input, preds)


class MaxNegProb(L._Loss):
  def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    del target

    probs = F.softmax(input, dim=-1)
    preds = torch.argmax(input, dim=-1)

    largest_prob = torch.gather(probs, dim=1, index=preds.unsqueeze(-1)).squeeze()

    neg_prob = -largest_prob

    if self.reduction == "mean":
      neg_prob = neg_prob.mean()
    elif self.reduction == "sum":
      neg_prob = neg_prob.sum()

    return neg_prob


def _log1mexp(x: torch.Tensor) -> torch.Tensor:
  """
  Stable computation of log(1 - exp(x)) for x <= 0.
  """
  log2 = math.log(2.0)
  return torch.where(
    x < -log2,
    torch.log1p(-torch.exp(x)),
    torch.log(-torch.expm1(x)),
  )


class MatsushitaLoss(L._Loss):
  def __init__(self, reduction: str = "mean", eps: float = 1e-12):
    super().__init__(reduction=reduction)
    self.eps = eps

  def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Stable log probabilities
    log_probs = F.log_softmax(input, dim=-1)

    if input.shape != target.shape:
      # Select only the true-class log probability: shape [N]
      log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)

      # Matsushita partial:
      # sqrt((1 - p_t) / p_t) = exp(0.5 * (log(1 - p_t) - log p_t))
      log1m_pt = _log1mexp(log_pt)
      log_loss = 0.5 * (log1m_pt - log_pt)

      # Prevent overflow in exp for extremely small p_t
      max_log = math.log(torch.finfo(input.dtype).max)
      log_loss = log_loss.clamp(max=max_log)

      loss = torch.exp(log_loss)
    else:
      raise NotImplementedError

    if self.reduction == "mean":
      return loss.mean()
    if self.reduction == "sum":
      return loss.sum()
    if self.reduction == "none":
      return loss

    raise ValueError(f"Invalid reduction: {self.reduction}")
