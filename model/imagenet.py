"""
Helper functions to initialize imagenet models
"""

import typing

import torch.nn as nn
from torchvision.models import efficientnet_b0, mobilenet_v2


def my_mobilenet_v2(
  num_classes: int = 1000, weights: typing.Any | None = None
) -> nn.Module:
  return mobilenet_v2(num_classes=num_classes, weights=weights)


def my_efficient_net_b0(
  num_classes: int = 1000, weights: typing.Any | None = None
) -> nn.Module:
  return efficientnet_b0(num_classes=num_classes, weights=weights)
