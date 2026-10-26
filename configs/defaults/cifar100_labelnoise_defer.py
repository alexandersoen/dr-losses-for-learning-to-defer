from configs.base import (
  CascadeElementConfig,
  DRCPETrainConfig,
  DatasetConfig,
  DeferTrainConfig,
)


def get_config(defer_noise_str: str = "conf_0_8_14") -> DeferTrainConfig:
  defer_algo_str, label_noise, small, large = defer_noise_str.split("_")

  # Used for bookkeeping only; keep the shifted dataset identity in result paths.
  base_dataset_config = DatasetConfig(
    f"cifar100_labelnoise/first_{label_noise}", "train"
  )
  train_dataset_config = DatasetConfig("cifar100_labelnoise/first_0", "validation")
  test_dataset_config = DatasetConfig("cifar100_labelnoise/first_0", "test")

  small_model_config = CascadeElementConfig(
    f"resnet{small}",
    f"res/base_models/cifar100_labelnoise/first_{label_noise}/resnet{small}",
  )
  large_model_config = CascadeElementConfig(
    f"resnet{large}",
    f"res/base_models/cifar100_labelnoise/first_{label_noise}/resnet{large}",
  )

  if defer_algo_str == "dr-cpe":
    config_cls = DRCPETrainConfig
  else:
    config_cls = DeferTrainConfig

  return config_cls(
    base_dataset_config=base_dataset_config,
    train_dataset_config=train_dataset_config,
    test_dataset_config=test_dataset_config,
    small_model_config=small_model_config,
    large_model_config=large_model_config,
    defer_algo_str=defer_algo_str,
    defer_algo_hparam_str="{}",
    save_dataset_postfix=f"{small}_{large}",
    num_epochs=32,
    batch_size=128,
    num_workers=8,
    seed=42,
    log_every_steps=1,
    checkpoint_every_epochs=1,
    checkpoint_resume=True,  # Overrides results!
    eval_tradeoff=False,
  )
