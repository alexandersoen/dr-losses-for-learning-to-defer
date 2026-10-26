from pathlib import Path

from absl import logging

from configs.base import DRCPETrainConfig
from defer.dr_cpe import (
  DRCPEConfig,
  DRCPEDefer,
  DRCPEDeferTrainer,
)
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
  config: DRCPETrainConfig,
  use_multi_gpu: bool = False,
):
  logging.info("dr_cpe defer model training initiated")

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

  dr_cpe_config = DRCPEConfig.from_strings(
    criterion=config.criterion,
    partial_loss=config.partial_loss,
    num_classes=data.num_classes,
    temp=config.temp,
    small_gamma=config.small_gamma,
    large_gamma=config.large_gamma,
    large_posthoc=config.large_posthoc,
    threshold_target=config.threshold_target,
  )
  defer_model = DRCPEDefer(dr_cpe_config).to(setup.device)
  defer_trainer = DRCPEDeferTrainer(
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
