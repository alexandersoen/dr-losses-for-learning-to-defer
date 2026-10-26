import logging as py_logging
import pathlib
import sys
import typing

import triton  # Putting first so things don't explode
from absl import app, flags, logging
from clu import platform
from ml_collections import config_flags

import train.base_train as base_train
import train.defer_train as defer_train
import train.dr_cpe_train as dr_cpe_train
from configs.base import (
  CascadeElementTrainConfig,
  DeferTrainConfig,
  DRCPETrainConfig,
)

FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", None, "Directory to store model data.")
flags.DEFINE_bool("use_multi_gpu", False, "Flag to utilize multi-gpu.")
config_flags.DEFINE_config_file(
  "config",
  None,
  "File path to the training hyperparameter configuration.",
  lock_config=True,
)
flags.DEFINE_string("spostfix", "", "Prefix string on savedir string name (on save).")
flags.DEFINE_string("dpostfix", None, "Prefix string on dataset string name (on save).")
flags.DEFINE_string("mpostfix", None, "Prefix string on model string name (on save).")


def _resolve_model_postfix(config: typing.Any) -> str:
  if FLAGS.mpostfix:
    return FLAGS.mpostfix
  return config.save_model_postfix


def main(argv: typing.Any) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # Create work dir
  workdir_path = pathlib.Path(FLAGS.workdir)
  workdir_path.mkdir(parents=True, exist_ok=True)

  # Log to file
  _handler = logging.get_absl_handler()
  _handler = typing.cast(logging.ABSLHandler, _handler)
  _handler.use_absl_log_file(log_dir=str(workdir_path))

  logging.set_stderrthreshold(logging.INFO)

  print("Logging to: %s" % logging.get_log_file_name())

  platform.work_unit().create_artifact(
    platform.ArtifactType.DIRECTORY, workdir_path, "workdir"
  )

  try:
    # Setup save postfix
    dataset_postfix = (
      FLAGS.dpostfix if FLAGS.dpostfix else FLAGS.config.save_dataset_postfix
    )
    model_postfix = _resolve_model_postfix(FLAGS.config)

    # Run depending on config type
    if isinstance(FLAGS.config, CascadeElementTrainConfig):
      base_train.train(
        workdir_path,
        FLAGS.spostfix,
        dataset_postfix,
        model_postfix,
        FLAGS.config,
        FLAGS.use_multi_gpu,
      )
    elif isinstance(FLAGS.config, DRCPETrainConfig):
      dr_cpe_train.train(
        workdir_path,
        FLAGS.spostfix,
        dataset_postfix,
        model_postfix,
        FLAGS.config,
        FLAGS.use_multi_gpu,
      )
    elif isinstance(FLAGS.config, DeferTrainConfig):
      defer_train.train(
        workdir_path,
        FLAGS.spostfix,
        dataset_postfix,
        model_postfix,
        FLAGS.config,
        FLAGS.use_multi_gpu,
      )
  finally:
    py_logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()


if __name__ == "__main__":
  flags.mark_flags_as_required(["config", "workdir"])
  app.run(main)
