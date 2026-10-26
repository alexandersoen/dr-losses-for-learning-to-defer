from configs.base import CascadeElementTrainConfig, DatasetConfig


def get_config(
  model_noise_str: str = "resnet8_100",
) -> CascadeElementTrainConfig:
  model_str, n_classes = model_noise_str.split("_")

  train_dataset_config = DatasetConfig(f"cifar100_longtail/head_{n_classes}", "train")
  test_dataset_config = DatasetConfig("cifar100_longtail/head_100", "test")

  return CascadeElementTrainConfig(
    train_dataset_config=train_dataset_config,
    test_dataset_config=test_dataset_config,
    model_str=model_str,
    num_epochs=256,
    batch_size=1024,
    num_workers=8,
    seed=42,
    log_every_steps=1,
    checkpoint_every_epochs=32,
    checkpoint_resume=True,
  )
