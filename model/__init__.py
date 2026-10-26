import functools
from typing import Any

from model.imagenet import my_efficient_net_b0, my_mobilenet_v2
from model.optimizer import optimizer_selector

from .resnet import (
  resnet8,
  resnet8x4,
  resnet14,
  resnet20,
  resnet32,
  resnet32x4,
  resnet44,
  resnet56,
  resnet110,
)

# Dict selector for train scripts
model_dict = {
  "resnet8": resnet8,
  "resnet14": resnet14,
  "resnet20": resnet20,
  "resnet32": resnet32,
  "resnet44": resnet44,
  "resnet56": resnet56,
  "resnet110": resnet110,
  "resnet8x4": resnet8x4,
  "resnet32x4": resnet32x4,
  "mobilenet": my_mobilenet_v2,
  "efficientnet": my_efficient_net_b0,
  "pretrained-mobilenet": functools.partial(my_mobilenet_v2, weights="DEFAULT"),
  "pretrained-efficientnet": functools.partial(my_efficient_net_b0, weights="DEFAULT"),
}

def model_hparams_selector(model_str: str, dataset_str: str) -> dict[str, Any]:
  base_dataset_str, *_ = dataset_str.split("_")

  if model_str.startswith("resnet") and base_dataset_str in {"pathmnist", "dermamnist"}:
    return {"avg_pool_size": 7}

  return {}

__all__ = ["model_dict", "model_hparams_selector", "optimizer_selector"]
