""" """

import torch
import torchvision.transforms.v2 as transforms


def create_dataset_transform(builder_str: str, is_train: bool = True):
  """Selects the transform"""
  base_dataset_str, *_ = builder_str.split("_")
  match base_dataset_str:
    case "cifar100":
      transform = cifar100_transform(is_train)
    case "imagenet2012":
      transform = imagenet2012_transform(is_train)
    case "pathmnist":
      transform = pathmnist_transform(is_train)
    case "dermamnist":
      transform = dermamnist_transform(is_train)
    case _:
      raise ValueError(f"Unknown dataset str: {base_dataset_str}")

  return transform


###############################################################################


def cifar100_transform(is_train: bool) -> transforms.Transform:
  cifar100_mean = [0.50707483, 0.48654887, 0.44091749]
  cifar100_std = [0.26733416, 0.25643855, 0.27615049]

  normalize_transform = transforms.Normalize(cifar100_mean, cifar100_std)
  if is_train:
    transform = transforms.Compose(
      [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        normalize_transform,
      ]
    )
  else:
    transform = transforms.Compose(
      [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        normalize_transform,
      ]
    )

  return transform


def imagenet2012_transform(is_train: bool) -> transforms.Transform:
  imagenet2012_mean = [0.485, 0.456, 0.406]
  imagenet2012_std = [0.229, 0.224, 0.225]

  normalize_transform = transforms.Normalize(imagenet2012_mean, imagenet2012_std)
  if is_train:
    transform = transforms.Compose(
      [
        transforms.ToImage(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToDtype(torch.float32, scale=True),
        normalize_transform,
      ]
    )
  else:
    transform = transforms.Compose(
      [
        transforms.ToImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToDtype(torch.float32, scale=True),
        normalize_transform,
      ]
    )

  return transform


def pathmnist_transform(is_train: bool) -> transforms.Transform:
  pathmnist_mean = [0.740545, 0.53298219, 0.70582885]
  pathmnist_std = [0.12368222, 0.17676253, 0.12443067]

  normalize_transform = transforms.Normalize(pathmnist_mean, pathmnist_std)
  if is_train:
    transform = transforms.Compose(
      [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        normalize_transform,
      ]
    )
  else:
    transform = transforms.Compose(
      [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        normalize_transform,
      ]
    )

  return transform


def dermamnist_transform(is_train: bool) -> transforms.Transform:
  dermamnist_mean = [0.7631127, 0.53808818, 0.56138814]
  dermamnist_std = [0.13660513, 0.15425383, 0.16921601]

  normalize_transform = transforms.Normalize(dermamnist_mean, dermamnist_std)
  if is_train:
    transform = transforms.Compose(
      [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        normalize_transform,
      ]
    )
  else:
    transform = transforms.Compose(
      [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        normalize_transform,
      ]
    )

  return transform
