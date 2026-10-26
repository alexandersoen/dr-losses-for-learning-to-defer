import json
import pickle
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from absl import logging

from configs.base import CascadeElementConfig
from model import model_dict
from train.utils import PretrainWrapper

# Checkpoint return type, first element is the epoch
CheckpointReturn = tuple[
  int,
  nn.Module,
  optim.Optimizer | None,
  optim.lr_scheduler.LRScheduler | None,
]


def load_trusted_checkpoint(
  checkpoint_path: Path,
  map_location: torch.device,
) -> dict[str, Any]:
  """Load a checkpoint saved by this project across torch weights_only defaults.

  Newer PyTorch versions default to a stricter weights-only unpickler, but our
  optimizer state may contain trusted Python objects such as collections.Counter.
  """
  try:
    return torch.load(checkpoint_path, weights_only=True, map_location=map_location)
  except pickle.UnpicklingError:
    logging.warning(
      "weights_only checkpoint load failed for %s; retrying with full unpickling",
      checkpoint_path,
    )
    return torch.load(checkpoint_path, weights_only=False, map_location=map_location)


def set_latest_json(base_dir: Path, latest_subpath_str: str) -> None:
  """Save the `latest` field as a subdirectory."""
  latest_json_file = Path(base_dir, "latest.json")
  latest_dict = {"latest": latest_subpath_str}

  logging.info(f"update latest model folder to: {latest_subpath_str}")
  with latest_json_file.open("w") as f:
    json.dump(latest_dict, f, indent=4, sort_keys=True)


def get_latest_json(base_dir: Path) -> str | None:
  """Save the `latest` field as a subdirectory."""
  latest_json_file = Path(base_dir, "latest.json")

  if not latest_json_file.exists():
    return

  logging.info(f"get latest model folder from: {latest_json_file}")
  with latest_json_file.open() as f:
    latest_dict = json.load(f)

  return latest_dict["latest"]


def ddp_key_replace(s: str) -> str:
  return s.replace("module.", "")


def from_distributed_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
  return {ddp_key_replace(k): v for k, v in state_dict.items()}


def save_checkpoint(
  base_dir: Path,
  save_subdir_str: str,
  epoch: int,
  model: nn.Module,
  optimizer: optim.Optimizer | None,
  scheduler: optim.lr_scheduler.LRScheduler | None,
  other_state: dict[str, str] | None = None,
  use_multi_gpu: bool = False,
) -> None:
  """Save all the things."""

  # Paths
  base_checkpoint_path = Path(base_dir, "checkpoints")
  checkpoint_path = Path(base_checkpoint_path, f"checkpoint_{save_subdir_str}.pt")

  # Save all the things
  logging.info(f"save model, optimizer, and scheduler to {checkpoint_path}")
  base_checkpoint_path.mkdir(parents=True, exist_ok=True)

  # Generate and save combined states
  state: dict[str, Any] = {} if other_state is None else other_state
  state["epoch"] = epoch

  # State dict, but gotta handle ddp case
  state["model_state"] = model.state_dict()
  if use_multi_gpu:
    state["model_state"] = from_distributed_state_dict(state["model_state"])

  # Other state dicts
  if optimizer is not None:
    state["optimizer_state"] = optimizer.state_dict()
  if scheduler is not None:
    state["scheduler_state"] = scheduler.state_dict()
  torch.save(state, checkpoint_path)

  # Generate latest checkpoint path
  set_latest_json(base_checkpoint_path, f"checkpoint_{save_subdir_str}.pt")


def load_latest_checkpoint(
  base_dir: Path,
  model: nn.Module,
  optimizer: optim.Optimizer | None,
  scheduler: optim.lr_scheduler.LRScheduler | None,
  device: torch.device,
) -> CheckpointReturn | None:
  # Get paths
  base_checkpoint_path = Path(base_dir, "checkpoints")
  logging.info(f"looking for checkpoint at {base_checkpoint_path}")

  latest_subdir_str = get_latest_json(base_checkpoint_path)
  if latest_subdir_str is None:
    # Early return if no checkpoints specified
    return
  checkpoint_path = Path(base_checkpoint_path, latest_subdir_str)

  # Load from latest checkpoint
  state = load_trusted_checkpoint(checkpoint_path, device)
  model.load_state_dict(state["model_state"])
  if optimizer is not None:
    optimizer.load_state_dict(state["optimizer_state"])
  if scheduler is not None:
    scheduler.load_state_dict(state["scheduler_state"])
  epoch = state["epoch"]

  logging.info(f"load model, optimizer, and scheduler from epoch {epoch}")

  return epoch, model, optimizer, scheduler


def load_latest_model(base_dir: Path, model: nn.Module) -> None:
  # Get paths
  base_checkpoint_path = Path(base_dir, "checkpoints")
  logging.info(f"looking for checkpoint at {base_checkpoint_path}")

  latest_subdir_str = get_latest_json(base_checkpoint_path)
  if latest_subdir_str is None:
    # Early return if no checkpoints specified
    logging.info("no checkpoint found")
    return

  checkpoint_path = Path(base_checkpoint_path, latest_subdir_str)

  # Load from latest checkpoint
  state = load_trusted_checkpoint(checkpoint_path, torch.device("cpu"))
  model.load_state_dict(state["model_state"])

  logging.info(f"checkpoint found, load model from {checkpoint_path}")


def load_cascade_element_by_config(
  cascade_config: CascadeElementConfig,
  num_classes: int,
  model_hparams: dict[str, Any] | None = None,
) -> nn.Module:
  if model_hparams is None:
    model_hparams = {}

  _model = model_dict[cascade_config.model_str](
    num_classes=num_classes, **model_hparams
  )
  checkpoint_path = Path(cascade_config.model_checkpoint_path)
  load_latest_model(checkpoint_path, _model)

  # Stop any potential grads from getting pushed through
  model = PretrainWrapper(_model)

  return model
