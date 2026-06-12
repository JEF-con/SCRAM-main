# SCRAM

## Paper

**SCRAM: A Dual-Server Aggregation for Privacy-Preserving and Robust Federated Learning Against Backdoor Attacks**

**Authors:** Kaijie Huang, Lingling Xu, Xinglin Zhang, Ying Gao, Xiaofeng Chen

## Overview

This repository contains the public code release of `SCRAM`, a dual-server aggregation framework for privacy-preserving and robust federated learning against backdoor attacks.

The public version includes the main local training pipeline, the core SCRAM implementation, and the benchmark scripts used to evaluate the runtime cost of major SCRAM stages.

## Repository Structure

```text
SCRAM-main/
  data/                     # Data notes and optional split metadata
  docs/                     # Supplementary documentation
  models/                   # Model, local update, attack, and evaluation modules
  scripts/                  # Benchmark and plotting scripts
  utils/                    # Core SCRAM implementation and shared utilities
  .gitignore                # Ignore rules for local data/logs/outputs
  install_requirements.sh   # Original environment installation script
  LICENSE                   # License for public release
  main_fed.py               # Main federated learning training entry
  README.md                 # Repository overview
  requirements.txt          # Python dependency list
```

## Environment

Recommended environment:

- Python `>= 3.9`
- PyTorch `>= 1.10`
- NumPy
- scikit-learn
- matplotlib
- hdbscan
- phe

Install dependencies with:

```bash
pip install -r requirements.txt
```

The original environment installation script is also provided in [install_requirements.sh](./install_requirements.sh).

## Data

The `data/` folder is included in the public release for documentation and optional split metadata.

Public datasets such as MNIST, FashionMNIST, and CIFAR-10 can be downloaded automatically through `torchvision`.

Please refer to [data/README.md](./data/README.md) for details about:

- supported public datasets
- optional client split index files
- data-related files that are not included in the public release

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

Benchmark result files are generated locally when the scripts are executed.

## Benchmark Plotting

```bash
python scripts/plotting/plot_scram_clients_lines.py
python scripts/plotting/plot_param_scale_lines.py
```

Generated figures are written to local output directories created by the scripts.

## Core Files

- [main_fed.py](./main_fed.py): federated learning training entry
- [utils/scram.py](./utils/scram.py): core SCRAM implementation
- [models/](./models): model and local update related modules
- [scripts/benchmark/benchmark_scram.py](./scripts/benchmark/benchmark_scram.py): runtime benchmark for SCRAM stages
- [scripts/benchmark/run_benchmark_experiments.py](./scripts/benchmark/run_benchmark_experiments.py): batch benchmark runner
- [docs/](./docs): supplementary documentation for the public release

## Public Release Notes

This public release keeps the files that are directly relevant to the training pipeline, SCRAM method, and benchmark scripts.

The following items are not intended to be uploaded to GitHub:

- private experiment logs
- local checkpoints and outputs
- local analysis scripts
- generated figures and intermediate artifacts

See [.gitignore](./.gitignore) for the ignored paths used in the public release.

## License

This repository is released under the license specified in [LICENSE](./LICENSE).

## Citation

If you use this repository, please cite your SCRAM paper. You can replace the following entry with the final published version:

```text
@article{huang_scram,
  title   = {SCRAM: A Dual-Server Aggregation for Privacy-Preserving and Robust Federated Learning Against Backdoor Attacks},
  author  = {Kaijie Huang and Lingling Xu and Xinglin Zhang and Ying Gao and Xiaofeng Chen}
}
```
