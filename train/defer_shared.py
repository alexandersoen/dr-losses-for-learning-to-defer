from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.data as utils_data
import torch.utils.tensorboard as tensorboard
from absl import logging

from configs.base import DeferTrainConfig
from defer.base import DeferModel, DeferTrainer
from model import model_hparams_selector
from train import utils
from train.checkpointing import (
  load_cascade_element_by_config,
  load_latest_checkpoint,
  save_checkpoint,
)
from train.loaders import create_dataloader
from train.meters import DeferMeter


@dataclass(frozen=True)
class DeferSetup:
  rank: int
  world_size: int
  device: torch.device
  train_path: Path


@dataclass(frozen=True)
class DeferData:
  train_sampler: utils_data.Sampler[Any]
  train_dataloader: utils_data.DataLoader[Any]
  test_sampler: utils_data.Sampler[Any]
  test_dataloader: utils_data.DataLoader[Any]
  num_classes: int


@dataclass(frozen=True)
class CascadeModels:
  small_model: nn.Module
  large_model: nn.Module


def setup_defer_training(
  workdir: Path,
  savedir_postfix: str,
  dataset_postfix: str,
  model_postfix: str,
  config: DeferTrainConfig,
  use_multi_gpu: bool,
) -> DeferSetup:
  ###########################################################################

  rank, world_size = utils.multi_gpu_setup(use_multi_gpu)

  device = utils.get_device(use_multi_gpu)
  logging.info(f"Using device {device}")

  utils.set_random_seed(config.seed)
  logging.info(f"Using random seed {config.seed}")

  train_path = Path(
    workdir,
    (f"defer_models-{savedir_postfix}" if len(savedir_postfix) > 0 else "defer_models"),
    (
      f"{config.base_dataset_config.builder_str}-{dataset_postfix}"
      if len(dataset_postfix) > 0
      else config.base_dataset_config.builder_str
    ),
    (
      f"{config.defer_algo_str}-{model_postfix}"
      if len(model_postfix) > 0
      else config.defer_algo_str
    ),
  )

  return DeferSetup(
    rank=rank,
    world_size=world_size,
    device=device,
    train_path=train_path,
  )


def create_defer_data(
  config: DeferTrainConfig,
  world_size: int,
  use_multi_gpu: bool,
) -> DeferData:
  ###########################################################################

  batch_size = config.batch_size // world_size

  train_sampler, train_dataloader, train_builder = create_dataloader(
    config.train_dataset_config.builder_str,
    config.train_dataset_config.builder_part,
    is_train=True,
    num_workers=config.num_workers,
    persistent_workers=config.persistent_workers,
    batch_size=batch_size,
    drop_last=True,
    use_multi_gpu=use_multi_gpu,
  )
  test_sampler, test_dataloader, _ = create_dataloader(
    config.test_dataset_config.builder_str,
    config.test_dataset_config.builder_part,
    is_train=False,
    num_workers=config.num_workers,
    persistent_workers=config.persistent_workers,
    batch_size=batch_size,
    drop_last=False,
    use_multi_gpu=use_multi_gpu,
  )
  class_label_feature = train_builder.info.features["label"]  # pyright: ignore
  num_classes = cast(int, class_label_feature.num_classes)

  return DeferData(
    train_sampler=train_sampler,
    train_dataloader=train_dataloader,
    test_sampler=test_sampler,
    test_dataloader=test_dataloader,
    num_classes=num_classes,
  )


def load_cascade_models(
  config: DeferTrainConfig,
  num_classes: int,
  device: torch.device,
) -> CascadeModels:
  ###########################################################################
  small_model_hparams = model_hparams_selector(
    config.small_model_config.model_str, config.base_dataset_config.builder_str
  )
  large_model_hparams = model_hparams_selector(
    config.large_model_config.model_str, config.base_dataset_config.builder_str
  )

  logging.info("Loading small model ...")
  small_model = load_cascade_element_by_config(
    config.small_model_config,
    num_classes=num_classes,
    model_hparams=small_model_hparams,
  )

  logging.info("Loading large model ...")
  large_model = load_cascade_element_by_config(
    config.large_model_config,
    num_classes=num_classes,
    model_hparams=large_model_hparams,
  )

  return CascadeModels(
    small_model=small_model.to(device),
    large_model=large_model.to(device),
  )


def run_defer_training_loop(
  setup: DeferSetup,
  data: DeferData,
  config: DeferTrainConfig,
  defer_model: DeferModel,
  defer_trainer: DeferTrainer[Any],
  use_multi_gpu: bool,
) -> None:
  ###########################################################################

  start_epoch = 1

  if config.checkpoint_resume:
    checkpoint = load_latest_checkpoint(
      setup.train_path,
      defer_model,
      defer_trainer.optimizer,
      defer_trainer.scheduler,
      setup.device,
    )
    if checkpoint is not None:
      cp_epoch, *_ = checkpoint
      start_epoch = 1 + cp_epoch
      defer_trainer.defer_model = defer_model

  if use_multi_gpu:
    defer_model.model_to_ddp(setup.rank)

  ###########################################################################

  metrics = DeferMeter()

  ###########################################################################

  if setup.rank == 0:
    writer = tensorboard.writer.SummaryWriter(setup.train_path)
  else:
    writer = None

  ###########################################################################

  for epoch in range(start_epoch, config.num_epochs + 1):
    is_last = epoch == config.num_epochs

    defer_trainer.train_epoch_step(
      config=config,
      metrics=metrics,
      device=setup.device,
      train_sampler=data.train_sampler,
      train_loader=data.train_dataloader,
      epoch=epoch,
      rank=setup.rank,
      use_multi_gpu=use_multi_gpu,
    )

    if use_multi_gpu:
      metrics.reduce_all(setup.device)

    if writer:
      for k, v in metrics.to_dict().items():
        writer.add_scalar(f"train/{k}", v, epoch)
      for dr, acc in metrics.trade_off.to_tradeoff_iterator(metric="acc1"):
        writer.add_scalar(f"train_trade_off_top1/epoch_{epoch}", acc, dr)
        writer.add_scalar(f"train_trade_off/epoch_{epoch}", acc, dr)
      for dr, acc in metrics.trade_off.to_tradeoff_iterator(metric="acc5"):
        writer.add_scalar(f"train_trade_off_top5/epoch_{epoch}", acc, dr)
      writer.flush()

    defer_trainer.test_epoch_step(
      config=config,
      metrics=metrics,
      device=setup.device,
      test_sampler=data.test_sampler,
      test_loader=data.test_dataloader,
      epoch=epoch,
      rank=setup.rank,
      use_multi_gpu=use_multi_gpu,
    )

    if use_multi_gpu:
      metrics.reduce_all(setup.device)

    if writer:
      for k, v in metrics.to_dict(no_train_time=True).items():
        writer.add_scalar(f"test/{k}", v, epoch)
      for dr, acc in metrics.trade_off.to_tradeoff_iterator(metric="acc1"):
        writer.add_scalar(f"test_trade_off_top1/epoch_{epoch}", acc, dr)
        writer.add_scalar(f"test_trade_off/epoch_{epoch}", acc, dr)
      for dr, acc in metrics.trade_off.to_tradeoff_iterator(metric="acc5"):
        writer.add_scalar(f"test_trade_off_top5/epoch_{epoch}", acc, dr)
      writer.flush()

    if setup.rank == 0 and (not epoch % config.checkpoint_every_epochs or is_last):
      save_checkpoint(
        setup.train_path,
        f"epoch_{epoch}",
        epoch,
        defer_model,
        defer_trainer.optimizer,
        defer_trainer.scheduler,
        use_multi_gpu=use_multi_gpu,
      )

  if writer:
    writer.close()

  if use_multi_gpu:
    dist.destroy_process_group()
