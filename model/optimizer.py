import torch.nn as nn
import torch.optim as optim
from torchvision.models.mobilenetv2 import MobileNetV2

OptSchTuple = tuple[optim.Optimizer, optim.lr_scheduler.LRScheduler]


def optimizer_selector(model: nn.Module, dataset_str: str) -> OptSchTuple:
  """Selector for the optimizer."""
  match dataset_str:
    case "cifar100":
      opt_sch = cifar100_optimizer(model)
    case "pathmnist" | "dermamnist":
      opt_sch = medmnist_optimizer(model)
    case "imagenet2012":
      if isinstance(model, MobileNetV2):
        opt_sch = mobilenet_optimizer(model)
      else:
        opt_sch = efficientnet_optimizer(model)
    case _:
      raise ValueError(f"Unknown dataset str: {dataset_str}")

  return opt_sch


def gen_warmup_anneal(
  optimizer: optim.Optimizer, step_lr_milestones: list[int]
) -> optim.lr_scheduler.LRScheduler:
  """Generate a warmup anneal scheduler (by epoch)"""

  # Scheduler parameters (by epoch)
  warmup_steps = step_lr_milestones[0]
  warmup_start = 1.0 / warmup_steps

  # Scheduler: Warm up + Annealing
  warmup_scheduler = optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=warmup_start,
    end_factor=1.0,
    total_iters=warmup_steps,
  )
  step_scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=step_lr_milestones[1:], gamma=0.1
  )
  scheduler = optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, step_scheduler],
    milestones=[warmup_steps],
  )
  return scheduler


###############################################################################


def cifar100_optimizer(model: nn.Module) -> OptSchTuple:
  optimizer = optim.SGD(
    model.parameters(),
    lr=1.0,
    momentum=0.9,
    weight_decay=1e-4,
  )

  multi_step_lr_milestones = [15, 96, 192, 224]
  scheduler = gen_warmup_anneal(optimizer, multi_step_lr_milestones)

  return optimizer, scheduler


def medmnist_optimizer(model: nn.Module) -> OptSchTuple:
  optimizer = optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4,
  )

  multi_step_lr_milestones = [5, 60, 90, 110]
  scheduler = gen_warmup_anneal(optimizer, multi_step_lr_milestones)

  return optimizer, scheduler


def mobilenet_optimizer(model: nn.Module) -> OptSchTuple:
  optimizer = optim.SGD(
    model.parameters(),
    lr=0.05,
    momentum=0.9,
    weight_decay=1e-4,
  )

  multi_step_lr_milestones = [5, 30, 60, 80]
  scheduler = gen_warmup_anneal(optimizer, multi_step_lr_milestones)

  return optimizer, scheduler


def efficientnet_optimizer(model: nn.Module) -> OptSchTuple:
  optimizer = optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4,
  )

  # Max is set here, we are always training to 90 for imagenet
  scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=128)

  return optimizer, scheduler


###############################################################################


def cifar100_optimizer_alt(model: nn.Module) -> OptSchTuple:
  """Weird that this doesn't actually work well for higher corruptions."""
  optimizer = optim.SGD(
    model.parameters(),
    lr=1.0,
    momentum=0.9,
    weight_decay=1e-4,
  )

  multi_step_lr_milestones = [15, 96, 192, 224]
  scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=multi_step_lr_milestones, gamma=0.1
  )

  return optimizer, scheduler


def dermamnist_optimizer(model: nn.Module) -> OptSchTuple:
  """Works better, but not in a meaningful way."""
  optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-2,
  )

  multi_step_lr_milestones = [5, 60, 90, 110]
  scheduler = gen_warmup_anneal(optimizer, multi_step_lr_milestones)

  return optimizer, scheduler
