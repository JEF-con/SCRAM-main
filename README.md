# SCRAM

## Paper

**SCRAM: A Dual-Server Aggregation for Privacy-Preserving and Robust Federated Learning Against Backdoor Attacks**

**Authors:** Kaijie Huang, Lingling Xu, Xinglin Zhang, Ying Gao, Xiaofeng Chen

## Overview

This repository contains the core code of `SCRAM`, a dual-server aggregation framework for robust and privacy-preserving federated learning under backdoor attacks.

The open-source version is organized around the following goals:

- reproduce the main SCRAM defense pipeline in federated learning
- run SCRAM-oriented training experiments under backdoor attacks
- benchmark the runtime cost of major SCRAM stages
- keep the public codebase lightweight and suitable for GitHub release

This public version focuses on the **core training and benchmark code**. Local data analysis scripts, private experimental logs, cached datasets, and local outputs are not intended to be uploaded to GitHub.

## Repository Structure

```text
SCRAM-main/
  main_fed.py               # Main federated learning entry
  models/                   # Models, local updates, attacks, evaluation
  utils/                    # SCRAM core implementation and utilities
  scripts/
    benchmark/              # SCRAM benchmark and simulation scripts
    plotting/               # Benchmark plotting scripts
  docs/                     # Auxiliary documentation
  results/
    benchmark/              # Example benchmark summaries
  archive/
    legacy/                 # Archived historical prototypes
```

## Environment

Recommended environment:

- Python `>= 3.9`
- PyTorch `>= 1.10`

Install dependencies with:

```bash
pip install -r requirements.txt
```

The original environment installation script is kept in [install_requirements.sh](./install_requirements.sh).

## Data

This repository does **not** include:

- raw training datasets
- cached dataset files
- private experiment logs
- local analysis outputs

Public datasets such as MNIST, FashionMNIST, and CIFAR-10 can be downloaded automatically through `torchvision`.

See [data/README.md](./data/README.md) for more details.

## Main Training

The main training entry is [main_fed.py](./main_fed.py).

Example:

```bash
python main_fed.py ^
  --dataset fashion_mnist ^
  --model cnn ^
  --attack dba ^
  --defence scram ^
  --epochs 200 ^
  --num_users 100 ^
  --frac 1 ^
  --malicious 0.1 ^
  --poison_frac 1.0 ^
  --local_ep 2 ^
  --local_bs 64 ^
  --attack_begin 0 ^
  --attack_label 5 ^
  --attack_goal -1 ^
  --iid 1 ^
  --gpu 0 ^
  --save save/my_experiments
```

Notes:

- `--defence scram` is the main method in this repository
- `--defence scram_without_privacy` can be used as an ablation baseline
- `cnn` is mainly used for FashionMNIST
- `VGG` and `resnet` are mainly used for CIFAR-10

## SCRAM Benchmark

Single benchmark run:

```bash
python scripts/benchmark/benchmark_scram.py --clients 50 --size 1200000 --device cpu
```

Batch benchmark:

```bash
python scripts/benchmark/run_benchmark_experiments.py --device cpu
```

Benchmark outputs are written by default to:

- `results/benchmark/`

## Benchmark Plotting

```bash
python scripts/plotting/plot_scram_clients_lines.py
python scripts/plotting/plot_param_scale_lines.py
```

Generated figures are written to:

- `results/benchmark/figures/`

## Core Files

- [main_fed.py](./main_fed.py): federated learning training entry
- [utils/scram.py](./utils/scram.py): core SCRAM implementation
- [scripts/benchmark/benchmark_scram.py](./scripts/benchmark/benchmark_scram.py): runtime benchmark for SCRAM stages
- [scripts/benchmark/run_benchmark_experiments.py](./scripts/benchmark/run_benchmark_experiments.py): batch benchmark runner
- [docs/file_manifest.md](./docs/file_manifest.md): additional file-level description

## Public Release Notes

For the GitHub release, the repository is intentionally limited to code that is directly relevant to the core method and reproducible experiments.

The following items are excluded from public upload by default:

- dataset files and cached split files
- local training outputs and checkpoints
- local logs
- analysis scripts under `scripts/analysis/`
- generated analysis figures and intermediate artifacts

See [.gitignore](./.gitignore) for the ignored paths.

## Citation

If you use this repository, please cite your SCRAM paper. You can replace the following entry with the final published version:

```text
@article{huang_scram,
  title   = {SCRAM: A Dual-Server Aggregation for Privacy-Preserving and Robust Federated Learning Against Backdoor Attacks},
  author  = {Kaijie Huang and Lingling Xu and Xinglin Zhang and Ying Gao and Xiaofeng Chen}
}
```
