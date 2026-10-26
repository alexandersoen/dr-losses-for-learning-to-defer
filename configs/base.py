import dataclasses
from abc import ABC, abstractmethod
from typing import Any


@dataclasses.dataclass(unsafe_hash=True)
class BaseConfig(ABC):
  def to_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)


@dataclasses.dataclass(unsafe_hash=True)
class DatasetConfig(BaseConfig):
  builder_str: str
  builder_part: str


@dataclasses.dataclass(unsafe_hash=True)
class ObjectConfig(BaseConfig):
  object_class: str = dataclasses.field(init=False)

  def __post_init__(self):
    self.object_class = self.object_class_val

  @property
  @abstractmethod
  def object_class_val(self) -> str:
    pass

  def to_params(self) -> dict[str, Any]:
    param_dict = self.to_dict()
    del param_dict["object_class"]
    return param_dict

  def to_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)


@dataclasses.dataclass(unsafe_hash=True)
class OptimizerConfig(ObjectConfig):
  pass


@dataclasses.dataclass(unsafe_hash=True)
class SchedulerConfig(ObjectConfig):
  pass


@dataclasses.dataclass(unsafe_hash=True)
class CascadeElementConfig(BaseConfig):
  model_str: str
  model_checkpoint_path: str


@dataclasses.dataclass(unsafe_hash=True)
class CascadeElementTrainConfig(BaseConfig):
  """Cascade element model training config."""

  # Dataset related stuff
  train_dataset_config: DatasetConfig
  test_dataset_config: DatasetConfig

  # Model related stuff
  model_str: str

  # Training loop related stuff
  num_epochs: int
  batch_size: int
  num_workers: int

  # Randomization
  seed: int

  # Logging / checkpoint related stuff
  log_every_steps: int
  checkpoint_every_epochs: int
  checkpoint_resume: bool

  # Save postfix
  save_dataset_postfix: str = ""
  save_model_postfix: str = ""
  persistent_workers: bool = True


@dataclasses.dataclass(unsafe_hash=True)
class DeferTrainConfig(BaseConfig):
  """Cascade element model training config."""

  # Dataset related stuff
  base_dataset_config: DatasetConfig  # Not used in this training
  train_dataset_config: DatasetConfig
  test_dataset_config: DatasetConfig

  # Cascade model related stuff
  small_model_config: CascadeElementConfig
  large_model_config: CascadeElementConfig

  # Defer algorithm
  defer_algo_str: str
  defer_algo_hparam_str: str

  # Training loop related stuff
  num_epochs: int
  batch_size: int
  num_workers: int

  # Randomization
  seed: int

  # Logging / checkpoint related stuff
  log_every_steps: int
  checkpoint_every_epochs: int
  checkpoint_resume: bool

  # Eval
  eval_tradeoff: bool

  # Save prefix
  save_dataset_postfix: str = ""
  save_model_postfix: str = ""
  persistent_workers: bool = True


@dataclasses.dataclass(unsafe_hash=True)
class DRCPETrainConfig(DeferTrainConfig):
  criterion: str = "cross-entropy"
  partial_loss: str = "log"
  temp: float = 1.0
  small_gamma: float = 1.0
  large_gamma: float = 1.0
  large_posthoc: bool = False
  threshold_target: str = "scorer"
