import random
from itertools import filterfalse
from typing import Any, Iterator, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel as parallel


def get_device(multi_gpu: bool = False) -> torch.device:
  if multi_gpu:
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
  else:
    if torch.cuda.is_available():
      device = torch.device("cuda")
    elif torch.mps.is_available():
      device = torch.device("mps")
    else:
      device = torch.device("cpu")

  return device


def multi_gpu_setup(use_multi_gpu: bool = False) -> tuple[int, int]:
  if use_multi_gpu:
    backend = "nccl"
    dist.init_process_group(backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
  else:
    rank = 0
    world_size = 1

  return rank, world_size


def set_random_seed(seed: int) -> None:
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(seed)
  random.seed(seed)


def model_to_ddp(model: nn.Module, rank: int) -> nn.Module:
  model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
  model = parallel.DistributedDataParallel(model, device_ids=[rank])
  return model


class PretrainWrapper(nn.Module):
  def __init__(self, model: nn.Module) -> None:
    super().__init__()
    self.model_to_wrap = model

  def forward(self, *args: Any, **kwds: Any) -> Any:
    self.model_to_wrap.eval()

    with torch.no_grad():
      return self.model_to_wrap.forward(*args, **kwds)

  def named_parameters(
    self,
    prefix: str = "",
    recurse: bool = True,
    remove_duplicate: bool = True,
  ) -> Iterator[Tuple[str, nn.Parameter]]:
    """Remove small model parameters from parameter list."""
    named_params = super().named_parameters(prefix, recurse, remove_duplicate)
    return filterfalse(lambda k_v: k_v[0].startswith("model_to_wrap."), named_params)


def to_probs(logits: torch.Tensor, temp: float, dim: int = 1) -> torch.Tensor:
  return F.softmax(logits / temp, dim=dim)


def to_log_probs(logits: torch.Tensor, temp: float, dim: int = 1) -> torch.Tensor:
  return F.log_softmax(logits / temp, dim=dim)
