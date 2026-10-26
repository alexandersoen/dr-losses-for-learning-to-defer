from configs.base import (
  CascadeElementConfig,
  DRCPETrainConfig,
  DatasetConfig,
  DeferTrainConfig,
)


def get_config(defer_noise_str: str = "conf_100") -> DeferTrainConfig:
  defer_algo_str, non_dog_perc = defer_noise_str.split("_")

  base_dataset_config = DatasetConfig(
    f"imagenet2012_dogs/nondog_{non_dog_perc}", "train"
  )
  train_dataset_config = DatasetConfig(
    "imagenet2012_dogs/nondog_100", "validation[:10000]"
  )
  test_dataset_config = DatasetConfig(
    "imagenet2012_dogs/nondog_100", "validation[10000:]"
  )

  small_model_config = CascadeElementConfig(
    "mobilenet",
    "res/base_models/imagenet2012_dogs/nondog_100/mobilenet",
  )
  large_model_config = CascadeElementConfig(
    "efficientnet",
    f"res/base_models/imagenet2012_dogs/nondog_{non_dog_perc}/efficientnet",
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
    num_epochs=32,
    batch_size=128,
    num_workers=8,
    seed=42,
    log_every_steps=1,
    checkpoint_every_epochs=1,
    checkpoint_resume=True,  # Overrides results!
    eval_tradeoff=False,
  )
