import typing

import numpy as np
import numpy.typing as npt
import tensorflow_datasets as tfds
import torch.utils.data as data
import torchvision.transforms.v2 as transforms
from PIL import Image
from tensorflow_datasets.core import FileFormat

from train.transform import create_dataset_transform


class DataSourceWrapper(data.Dataset):
  def __init__(
    self,
    data_source: typing.Any,
    transform: transforms.Transform | None = None,
    pil_preprocess: bool = True,
    extra_feature: str | None = None,
  ):
    self.data_source = data_source
    self.transform = transform

    self.pil_preprocess = pil_preprocess
    self.extra_feature = extra_feature

  def __len__(self):
    return len(self.data_source)

  def __getitem__(self, idx: int) -> typing.Any:
    example = self.data_source[idx]

    image = example["image"]
    label = example["label"]

    # Convert image to a PIL image for PyTorch transforms
    if self.pil_preprocess:
      image = Image.fromarray(image)
      image = image.convert("RGB")

    if self.transform:
      image = self.transform(image)

    if self.extra_feature is None:
      return image, label
    return image, label, example[self.extra_feature]


class NumpyDatasetWrapper(data.Dataset):
  def __init__(
    self,
    images: npt.NDArray[np.float32],
    labels: npt.NDArray[np.float32],
    transform: transforms.Transform | None = None,
    pil_preprocess: bool = True,
    extra_values: npt.NDArray[typing.Any] | None = None,
  ):
    self.images = images
    self.labels = labels
    self.transform = transform

    self.pil_preprocess = pil_preprocess
    self.extra_values = extra_values

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx: int) -> typing.Any:
    image = self.images[idx]
    label = self.labels[idx]

    # Convert image to a PIL image for PyTorch transforms
    if self.pil_preprocess:
      image = Image.fromarray(image)

    if self.transform:
      image = self.transform(image)

    if self.extra_values is None:
      return image, label
    return image, label, self.extra_values[idx]


def create_dataloader(
  builder_str: str,
  builder_part: str,
  is_train: bool,
  batch_size: int,
  num_workers: int,
  persistent_workers: bool,
  drop_last: bool,
  use_multi_gpu: bool = False,
  extra_feature: str | None = None,
) -> tuple[
  data.Sampler[typing.Any], data.DataLoader[typing.Any], tfds.core.DatasetBuilder
]:
  """Unified method for create dataloaders."""
  builder = tfds.builder(builder_str)
  transform = create_dataset_transform(builder_str, is_train=is_train)

  # Two ways of doing this depending on file format
  if builder.info.file_format == FileFormat.ARRAY_RECORD:
    data_source = builder.as_data_source(builder_part)
    torch_dataset = DataSourceWrapper(
      data_source,
      transform,
      extra_feature=extra_feature,
    )
  else:
    dataset = builder.as_dataset(
      builder_part,
      as_supervised=extra_feature is None,
    )

    # Just doing something crude to switch to torch dataloaders...
    # Should be a better way, but could not get it to work (shards)
    images = []
    labels = []
    extra_values = []
    for example in dataset.as_numpy_iterator():  # pyright: ignore
      if extra_feature is None:
        image, label = example
      else:
        image = example["image"]
        label = example["label"]
        extra_values.append(example[extra_feature])
      if isinstance(image, np.ndarray):
        images.append(image)
        labels.append(label)
      else:
        images.append(image.numpy())
        labels.append(label.numpy())

    images = np.array(images)
    labels = np.array(labels)

    torch_dataset = NumpyDatasetWrapper(
      images,
      labels,
      transform,
      extra_values=(
        None if extra_feature is None else np.array(extra_values)
      ),
    )

  if use_multi_gpu:
    sampler = data.DistributedSampler(torch_dataset, shuffle=is_train)
  elif is_train:
    sampler = data.RandomSampler(torch_dataset)
  else:
    sampler = data.SequentialSampler(torch_dataset)

  dataloader = data.DataLoader(
    torch_dataset,
    sampler=sampler,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=drop_last,
    pin_memory=True,
    persistent_workers=persistent_workers and num_workers > 0,
    prefetch_factor=4 if num_workers > 0 else None,
  )

  return sampler, dataloader, builder
