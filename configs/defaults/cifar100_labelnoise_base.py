from configs.base import CascadeElementTrainConfig, DatasetConfig


def get_config(
  model_noise_str: str = "resnet8_0",
) -> CascadeElementTrainConfig:
  model_str, label_noise = model_noise_str.split("_")

  train_dataset_config = DatasetConfig(
    f"cifar100_labelnoise/first_{label_noise}", "train"
  )
  test_dataset_config = DatasetConfig("cifar100_labelnoise/first_0", "test")

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
