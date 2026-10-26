from configs.base import CascadeElementTrainConfig, DatasetConfig


def get_config(
  model_specialist_str: str = "resnet8_100",
) -> CascadeElementTrainConfig:
  model_str, nonspecialist_perc = model_specialist_str.split("_")

  train_dataset_config = DatasetConfig(
    f"pathmnist_specialist/nonspecialist_{nonspecialist_perc}", "train"
  )
  test_dataset_config = DatasetConfig("pathmnist_specialist/nonspecialist_100", "test")

  return CascadeElementTrainConfig(
    train_dataset_config=train_dataset_config,
    test_dataset_config=test_dataset_config,
    model_str=model_str,
    num_epochs=128,
    batch_size=512,
    num_workers=8,
    seed=42,
    log_every_steps=1,
    checkpoint_every_epochs=16,
    checkpoint_resume=True,
  )
