import time
from abc import abstractmethod
from typing import Any, Iterator

import torch
import torch.distributed as dist


class AbstractMeter:
  def __init__(self) -> None:
    self.reset()

  @abstractmethod
  def reset(self) -> None:
    pass

  @abstractmethod
  def update(self, *_: Any, **__: Any) -> None:
    pass

  @abstractmethod
  def reduce_all(self, device: torch.device) -> None:
    """To handle DDP aggregation / sync."""
    pass


class AbstractMultiMeter(AbstractMeter):
  @abstractmethod
  def to_dict(self, *_: Any, **__: Any) -> dict[str, float]:
    pass

  def log_str(self, *_: Any, **__: Any) -> str:
    return ""


class AverageMeter(AbstractMeter):
  """Computes and stores the average and current value"""

  def reset(self) -> None:
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1) -> None:
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

  def reduce_all(self, device: torch.device) -> None:
    total = torch.tensor([self.sum, self.count], device=device)
    dist.all_reduce(total, op=dist.ReduceOp.SUM, async_op=False)

    self.sum, self.count = total.tolist()
    self.avg = 0 if self.count == 0 else self.sum / self.count


def accuracy(output, target, topk=(1,)):
  """Computes the accuracy over the k top predictions for the specified values of k"""
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
      correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
    return res


class TimerMeter(AbstractMeter):
  def __init__(self) -> None:
    self.train_time = AverageMeter()
    self.eval_time = AverageMeter()

    super().__init__()

  def reset(self) -> None:
    self.train_time.reset()
    self.eval_time.reset()

    self.batch_start_time = -1.0

  def mark_batch_start(self) -> None:
    self.batch_start_time = time.time()

  def update_eval_time(self) -> None:
    self.eval_time.update(time.time() - self.batch_start_time)

  def update_train_time(self) -> None:
    self.train_time.update(time.time() - self.batch_start_time)

  def reduce_all(self, device: torch.device) -> None:
    self.train_time.reduce_all(device)
    self.eval_time.reduce_all(device)

  def to_dict(self, no_train_time: bool = False) -> dict[str, float]:
    res = {}
    res["eval_time"] = self.eval_time.avg
    if not no_train_time:
      res["train_time"] = self.train_time.avg

    return res

  def log_str(self, no_train_time: bool = False) -> str:
    if no_train_time:
      return "Time {eval_time.val:.3f} ({eval_time.avg:.3f})".format(
        eval_time=self.eval_time,
      )
    else:
      return (
        "TrainTime {train_time.val:.3f} ({train_time.avg:.3f})\t"
        "EvalTime {eval_time.val:.3f} ({eval_time.avg:.3f})".format(
          train_time=self.train_time,
          eval_time=self.eval_time,
        )
      )


class PerformanceMeter(AbstractMultiMeter):
  def __init__(self) -> None:
    self.loss = AverageMeter()
    self.top1 = AverageMeter()
    self.top5 = AverageMeter()

    super().__init__()

  def reset(self) -> None:
    self.loss.reset()
    self.top1.reset()
    self.top5.reset()

  def update(
    self, output: torch.Tensor, target: torch.Tensor, loss: torch.Tensor
  ) -> None:
    acc1, acc5 = accuracy(output, target, topk=(1, 5))
    self.loss.update(loss.item(), target.size(0))
    self.top1.update(acc1[0], target.size(0))
    self.top5.update(acc5[0], target.size(0))

  def reduce_all(self, device: torch.device):
    self.loss.reduce_all(device)
    self.top1.reduce_all(device)
    self.top5.reduce_all(device)

  def to_dict(self) -> dict[str, float]:
    return {
      "loss": self.loss.avg,
      "acc1": self.top1.avg,
      "acc5": self.top5.avg,
    }

  def log_str(self) -> str:
    return (
      "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
      "Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t"
      "Acc@5 {top5.val:.3f} ({top5.avg:.3f})".format(
        loss=self.loss,
        top1=self.top1,
        top5=self.top5,
      )
    )


class GeneralMeter(AbstractMultiMeter):
  def __init__(self) -> None:
    self.timer = TimerMeter()
    self.perf = PerformanceMeter()

    super().__init__()

  def reset(self) -> None:
    self.timer.reset()
    self.perf.reset()

  def to_dict(self, no_train_time: bool = False) -> dict[str, float]:
    res = self.timer.to_dict(no_train_time)
    res.update(self.perf.to_dict())
    return res

  def reduce_all(self, device: torch.device) -> None:
    self.timer.reduce_all(device)
    self.perf.reduce_all(device)

  def log_str(self, no_train_time: bool = False) -> str:
    return self.timer.log_str(no_train_time) + "\t" + self.perf.log_str()


def combine_cascade_output(
  defer_output: torch.Tensor,
  small_output: torch.Tensor,
  large_output: torch.Tensor,
) -> torch.Tensor:
  """
  Assumes quantites are pre-computed, calculates combined output
  """
  num_classes = small_output.shape[1]
  defer_vec = defer_output.float().unsqueeze(1).expand(-1, num_classes)

  return (1 - defer_vec) * small_output + defer_vec * large_output


def _combine_cascade_output(
  defer_output: torch.Tensor,
  small_output: torch.Tensor,
  large_output: torch.Tensor,
) -> torch.Tensor:
  """
  Assumes quantites are pre-computed, calculates combined output
  """
  batch_size = small_output.shape[0]
  num_classes = small_output.shape[1]

  device = small_output.device
  output = torch.empty(batch_size, num_classes).to(device)
  output[defer_output] = large_output[defer_output]
  output[~defer_output] = small_output[~defer_output]

  return output


class DeferPerformanceMeter(AbstractMultiMeter):
  def __init__(self) -> None:
    self.defer_rate = AverageMeter()
    self.top1 = AverageMeter()
    self.top5 = AverageMeter()

    super().__init__()

  def reset(self) -> None:
    self.defer_rate.reset()
    self.top1.reset()
    self.top5.reset()

  def update(
    self,
    defer_output: torch.Tensor,
    small_output: torch.Tensor,
    large_output: torch.Tensor,
    target: torch.Tensor,
  ) -> None:
    cascade_output = combine_cascade_output(defer_output, small_output, large_output)

    batch_size = target.size(0)
    acc1, acc5 = accuracy(cascade_output, target, topk=(1, 5))
    defer_rate = 100 * defer_output.float().mean()
    self.defer_rate.update(defer_rate, batch_size)
    self.top1.update(acc1[0], batch_size)
    self.top5.update(acc5[0], batch_size)

  def reduce_all(self, device: torch.device) -> None:
    self.defer_rate.reduce_all(device)
    self.top1.reduce_all(device)
    self.top5.reduce_all(device)

  def to_dict(self) -> dict[str, float]:
    return {
      "defer_rate": self.defer_rate.avg,
      "acc1": self.top1.avg,
      "acc5": self.top5.avg,
    }


class TradeOffMeter(AbstractMeter):
  def __init__(
    self,
    n_defer_rates: int = 101,
  ) -> None:
    self.n_defer_rates = n_defer_rates
    self.defer_rates = torch.linspace(0.0, 100.0, n_defer_rates)

    self.meters: dict[float, DeferPerformanceMeter] = {}
    for defer_rate in self.defer_rates:
      self.meters[defer_rate.item()] = DeferPerformanceMeter()

    super().__init__()

  def reset(self) -> None:
    for defer_rate in self.defer_rates:
      self.meters[defer_rate.item()].reset()

  def update(
    self,
    defer_values: torch.Tensor,
    small_output: torch.Tensor,
    large_output: torch.Tensor,
    target: torch.Tensor,
  ) -> None:
    batch_size = target.size(0)
    sorted_indices = torch.argsort(defer_values, stable=True)

    for defer_rate in self.defer_rates:
      k = round(batch_size * defer_rate.item() / 100.0)
      defer_output = torch.zeros(batch_size, dtype=torch.bool, device=target.device)
      if k > 0:
        defer_output[sorted_indices[:k]] = True

      rate_meter = self.meters[defer_rate.item()]
      rate_meter.update(defer_output, small_output, large_output, target)

  def reduce_all(self, device) -> None:
    for defer_rate in self.defer_rates:
      self.meters[defer_rate.item()].reduce_all(device)

  def to_tradeoff_iterator(
    self,
    metric: str = "acc1",
  ) -> Iterator[tuple[float, float]]:
    if metric not in {"acc1", "acc5"}:
      raise ValueError(f"Unknown trade-off metric: {metric}")

    for defer_rate in self.defer_rates:
      rate_meter = self.meters[defer_rate.item()]
      rate_meter_res = rate_meter.to_dict()
      actual_defer_rate = rate_meter_res["defer_rate"]
      accuracy = rate_meter_res[metric]

      yield actual_defer_rate, accuracy


class DeferMeter(AbstractMultiMeter):
  def __init__(self):
    self.timer = TimerMeter()
    self.defer_loss = AverageMeter()
    self.trade_off = TradeOffMeter()

    super().__init__()

  def reset(self) -> None:
    self.timer.reset()
    self.defer_loss.reset()
    self.trade_off.reset()

  def to_dict(self, no_train_time: bool = False) -> dict[str, float]:
    res = self.timer.to_dict(no_train_time)
    res["defer_loss"] = self.defer_loss.avg
    return res

  def reduce_all(self, device: torch.device) -> None:
    self.timer.reduce_all(device)
    self.defer_loss.reduce_all(device)
    self.trade_off.reduce_all(device)

  def log_str(self, no_train_time: bool = False) -> str:
    s = self.timer.log_str(no_train_time)
    s += f"\tDeferLoss {self.defer_loss.val:.4f} ({self.defer_loss.avg:.4f})"
    return s
