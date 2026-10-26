# Code for "Density-Ratio Losses for Post-Hoc Learning to Defer"

This repo is the code implementation for the paper "Density-Ratio Losses for Post-Hoc Learning to Defer".
It provides the minimal code to produce the basic baseline deferral approaches and the deferral mechanisms trained via the joint DR loss objectives.

## Requirements

I would recommend to use `uv` to install requirements. The standard dependencies are all documented in `pyproject.toml`.

Only non-standard installation step is to install [alt-datasets](https://github.com/alexandersoen/alt-datasets) separately.
This will give access to the corrupted datasets (`tfds`) used in the experiments. I would clone this external repo and then run the `build_all.sh` script.
For `imagenet` datasets, you need to separately download the data as one would for the standard `tfds` imagenet dataset. Then you need to "link" the data folder and run the specific corrupted dataset build command.

## Usage

Usage comes in two stages. First, one is required to train the base and expert models.


One can achieve this via:

```bash
TODO
```

Then one can train the deferral mechanism via:

```bash
TODO
```

There are some additional example scripts in ...


## Citation

For usage of the repo, please cite the following:

```bibtex
@article{soen2026density,
  title={Density-Ratio Losses for Post-Hoc Learning to Defer},
  author={Soen, Alexander and Thobaben, Ragnar and Jald{\'e}n, Joakim and Nock, Richard},
  journal={Advances in Neural Information Processing Systems},
  volume={39},
  year={2026}
}
```
