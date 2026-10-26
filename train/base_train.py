from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard as tensorboard
from absl import logging

from configs.base import CascadeElementTrainConfig
from model import model_dict, model_hparams_selector, optimizer_selector
from train import utils
from train.checkpointing import load_latest_checkpoint, save_checkpoint
from train.loaders import create_dataloader
from train.meters import GeneralMeter

Example = Any


def get_optimizer(
  builder_str: str, model: nn.Module
) -> tuple[optim.Optimizer, optim.lr_scheduler.LRScheduler]:
  base_dataset_str, *_ = builder_str.split("_")
  sch_opt_pair = optimizer_selector(model, base_dataset_str)

  return sch_opt_pair


def train_epoch_step(
  config: CascadeElementTrainConfig,
  metrics: GeneralMeter,
  model: nn.Module,
  criterion: nn.Module,
  device: torch.device,
  train_sampler: torch.utils.data.Sampler[Example],
  train_loader: torch.utils.data.DataLoader[Example],
  optimizer: optim.Optimizer,
  scheduler: optim.lr_scheduler.LRScheduler,
  epoch: int,
  rank: int,
  use_multi_gpu: bool,
) -> None:
  # Set up sampler for DDP
  if use_multi_gpu and isinstance(train_sampler, torch.utils.data.DistributedSampler):
    train_sampler.set_epoch(epoch)

  # Set train
  model.train()

  # Reset metrics
  metrics.reset()

  for batch_idx, (data, target) in enumerate(train_loader):  # noqa: F402
    metrics.timer.mark_batch_start()

    data = data.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    output = model(data)
    loss = criterion(output, target)

    metrics.perf.update(output, target, loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Time batch
    metrics.timer.update_eval_time()

    # print info
    if rank == 0 and batch_idx % config.log_every_steps == 0:
      train_log = f"Epoch [{epoch}][{batch_idx}/{len(train_loader)}]:\t"
      metric_log = metrics.log_str(no_train_time=True)
      logging.info(train_log + metric_log)

  scheduler.step()


def eval_epoch_step(
  config: CascadeElementTrainConfig,
  metrics: GeneralMeter,
  model: nn.Module,
  criterion: nn.Module,
  device: torch.device,
  test_sampler: torch.utils.data.Sampler[Example],
  test_loader: torch.utils.data.DataLoader[Example],
  epoch: int,
  rank: int,
  use_multi_gpu: bool,
) -> None:
  # Set up sampler for DDP
  if use_multi_gpu and isinstance(test_sampler, torch.utils.data.DistributedSampler):
    test_sampler.set_epoch(epoch)

  model.eval()

  # Reset metrics
  metrics.reset()
  with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):  # noqa: F402
      metrics.timer.mark_batch_start()

      data = data.to(device, non_blocking=True)
      target = target.to(device, non_blocking=True)

      output = model(data)
      loss = criterion(output, target)

      # Time batch & reset
      metrics.timer.update_eval_time()

      # Compute metrics
      metrics.perf.update(output, target, loss)

      # print info
      if rank == 0 and batch_idx % config.log_every_steps == 0:
        test_log = f"Test [{batch_idx}/{len(test_loader)}]:\t"
        metric_log = metrics.log_str(no_train_time=True)
        logging.info(test_log + metric_log)


def train(
  workdir: Path,
  savedir_postfix: str,
  dataset_postfix: str,
  model_postfix: str,
  config: CascadeElementTrainConfig,
  use_multi_gpu: bool = False,
) -> None:
  logging.info("base model training initiated")

  ###########################################################################

  # Set up multi gpu setup
  rank, world_size = utils.multi_gpu_setup(use_multi_gpu)

  # Device
  device = utils.get_device(use_multi_gpu)
  logging.info(f"Using device {device}")

  # Set random seed
  utils.set_random_seed(config.seed)
  logging.info(f"Using random seed {config.seed}")

  # Training setting vars
  train_path = Path(
    workdir,
    (f"base_models-{savedir_postfix}" if len(savedir_postfix) > 0 else "base_models"),
    (
      f"{config.train_dataset_config.builder_str}-{dataset_postfix}"
      if len(dataset_postfix) > 0
      else config.train_dataset_config.builder_str
    ),
    (
      f"{config.model_str}-{model_postfix}"
      if len(model_postfix) > 0
      else config.model_str
    ),
  )

  ###########################################################################

  # Dataloaders
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

  ###########################################################################

  # Setup training objects
  model_hparams = model_hparams_selector(
    config.model_str, config.train_dataset_config.builder_str
  )
  model = model_dict[config.model_str](
    num_classes=num_classes, **model_hparams
  ).to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer, scheduler = get_optimizer(config.train_dataset_config.builder_str, model)

  start_epoch = 1

  ###########################################################################

  # Resume if checkpoint exists
  if config.checkpoint_resume:
    checkpoint = load_latest_checkpoint(train_path, model, optimizer, scheduler, device)
    if checkpoint is not None:
      cp_epoch, *_ = checkpoint  # other return are inplace updated
      start_epoch = 1 + cp_epoch

  if use_multi_gpu:
    model = utils.model_to_ddp(model, rank)

  ###########################################################################

  # Set up metric
  metrics = GeneralMeter()

  ###########################################################################

  # Set up logger
  if rank == 0:
    writer = tensorboard.writer.SummaryWriter(train_path)
  else:
    writer = None

  ###########################################################################

  # Train loop
  for epoch in range(start_epoch, config.num_epochs + 1):
    is_last = epoch == config.num_epochs

    # Train step
    train_epoch_step(
      config=config,
      metrics=metrics,
      model=model,
      criterion=criterion,
      device=device,
      train_sampler=train_sampler,
      train_loader=train_dataloader,
      optimizer=optimizer,
      scheduler=scheduler,
      epoch=epoch,
      rank=rank,
      use_multi_gpu=use_multi_gpu,
    )

    # Sync the metrics
    if use_multi_gpu:
      metrics.reduce_all(device)

    # Train metrics to writer
    if writer:
      for k, v in metrics.to_dict().items():
        writer.add_scalar(f"train/{k}", v, epoch)

      writer.flush()

    # Eval test
    eval_epoch_step(
      config=config,
      metrics=metrics,
      model=model,
      criterion=criterion,
      device=device,
      test_sampler=test_sampler,
      test_loader=test_dataloader,
      epoch=epoch,
      rank=rank,
      use_multi_gpu=use_multi_gpu,
    )

    # Sync the metrics
    if use_multi_gpu:
      metrics.reduce_all(device)

    # Test metrics to writer
    if writer:
      for k, v in metrics.to_dict(no_train_time=True).items():
        writer.add_scalar(f"test/{k}", v, epoch)

      writer.flush()

    # Checkpointing
    if rank == 0 and (not epoch % config.checkpoint_every_epochs or is_last):
      save_checkpoint(
        train_path,
        f"epoch_{epoch}",
        epoch,
        model,
        optimizer,
        scheduler,
        use_multi_gpu=use_multi_gpu,
      )

  if writer:
    writer.close()

  if use_multi_gpu:
    dist.destroy_process_group()
