from configs.base import CascadeElementTrainConfig, DatasetConfig


def get_config(
  model_nondog_str: str = "mobilenet_100",
) -> CascadeElementTrainConfig:
  model_str, non_dog_perc = model_nondog_str.split("_")

  train_dataset_config = DatasetConfig(
    f"imagenet2012_dogs/nondog_{non_dog_perc}", "train"
  )
  test_dataset_config = DatasetConfig("imagenet2012_dogs/nondog_100", "validation")

  return CascadeElementTrainConfig(
    train_dataset_config=train_dataset_config,
    test_dataset_config=test_dataset_config,
    model_str=model_str,
    num_epochs=128,
    batch_size=64,
    num_workers=16,
    seed=42,
    log_every_steps=64,
    checkpoint_every_epochs=32,
    checkpoint_resume=True,
  )
