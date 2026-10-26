import ast
from pathlib import Path

from absl import logging

from configs.base import DeferTrainConfig
from defer import defer_model_train_class_dict
from train.defer_shared import (
  create_defer_data,
  load_cascade_models,
  run_defer_training_loop,
  setup_defer_training,
)


def train(
  workdir: Path,
  savedir_postfix: str,
  dataset_postfix: str,
  model_postfix: str,
  config: DeferTrainConfig,
  use_multi_gpu: bool = False,
):
  logging.info("defer model training initiated")

  setup = setup_defer_training(
    workdir=workdir,
    savedir_postfix=savedir_postfix,
    dataset_postfix=dataset_postfix,
    model_postfix=model_postfix,
    config=config,
    use_multi_gpu=use_multi_gpu,
  )

  ###########################################################################

  data = create_defer_data(
    config=config,
    world_size=setup.world_size,
    use_multi_gpu=use_multi_gpu,
  )

  ###########################################################################

  models = load_cascade_models(
    config=config,
    num_classes=data.num_classes,
    device=setup.device,
  )

  ###########################################################################

  # Define the defer algo + its trainer
  MyDeferModel, MyDeferTrainer, hparams = defer_model_train_class_dict[
    config.defer_algo_str
  ]

  hparams = dict(hparams)
  hparam_override = ast.literal_eval(config.defer_algo_hparam_str)
  hparams.update(hparam_override)

  defer_model = MyDeferModel(num_classes=data.num_classes, **hparams).to(setup.device)
  defer_trainer = MyDeferTrainer(
    models.small_model,
    models.large_model,
    defer_model,
    eval_tradeoff=config.eval_tradeoff,
  )

  ###########################################################################

  logging.info(f"Training defer model: {defer_model}")

  run_defer_training_loop(
    setup=setup,
    data=data,
    config=config,
    defer_model=defer_model,
    defer_trainer=defer_trainer,
    use_multi_gpu=use_multi_gpu,
  )
