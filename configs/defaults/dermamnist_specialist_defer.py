from configs.base import (
  CascadeElementConfig,
  DatasetConfig,
  DeferTrainConfig,
  DRCPETrainConfig,
)


def get_config(defer_noise_str: str = "conf_100_8_32") -> DeferTrainConfig:
  defer_algo_str, nonspecialist_perc, small, large = defer_noise_str.split("_")

  base_dataset_config = DatasetConfig(
    f"dermamnist_specialist/nonspecialist_{nonspecialist_perc}", "train"
  )
  train_dataset_config = DatasetConfig(
    "dermamnist_specialist/nonspecialist_100", "validation"
  )
  test_dataset_config = DatasetConfig(
    "dermamnist_specialist/nonspecialist_100", "test"
  )

  small_model_config = CascadeElementConfig(
    f"resnet{small}",
    f"res/base_models/dermamnist_specialist/nonspecialist_100/resnet{small}",
  )
  large_model_config = CascadeElementConfig(
    f"resnet{large}",
    "res/base_models/dermamnist_specialist/"
    f"nonspecialist_{nonspecialist_perc}/resnet{large}",
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
    checkpoint_resume=True,
    eval_tradeoff=False,
  )
