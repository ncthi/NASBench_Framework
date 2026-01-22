# NASBench Framework

A lightweight Python framework for querying neural architecture search (NAS) benchmarks from a unified interface.

## Goal

This project aims to provide a single, consistent API to **query architectures and their benchmarked results** from three popular NAS benchmarks:

- **NAS-Bench-101**
- **NAS-Bench-201**
- **NAS-Bench-301**

The intent is to make it easy to:

- Fetch benchmarked metrics (e.g., accuracy, training time, params, FLOPs when available)
- Standardize architecture representations and conversions across benchmarks
- Build reproducible evaluation pipelines for NAS research

## Key Features (Planned)

- Unified `query()` interface across benchmarks
- Benchmark-specific adapters with a shared schema for returned results
- Pluggable architecture encoders/decoders (e.g., cell graphs, operation strings)
- Local caching and fast lookup

## Repository Layout (Suggested)

The repository is intentionally minimal at the moment. A typical layout will look like:

```
# NASBench Framework

This repository provides a simple command-line interface (CLI) to query multiple NAS benchmarks from one place:

- NAS-Bench-101
- NAS-Bench-201
- NAS-Bench-301
- HW-NAS-Bench

## Repository layout

The current project structure is:

```
.
├── environment.yml            # Conda environment definition
├── main.py                    # CLI entry point (argparse)
├── README.md
├── src/                       # CLI implementation (dispatches to per-benchmark helpers)
│   ├── __init__.py            # get_top_arch(args) router
│   ├── get_nasbench101.py
│   ├── get_nasbench201.py
│   ├── get_nasbench301.py
│   └── get_hwnasbench.py
└── nasbench/                  # Benchmark APIs + bundled assets
    ├── nasbench101/           # NAS-Bench-101 API code (+ TFRecord in this repo)
    ├── nasbench201/           # NAS-Bench-201 API code (+ .pth in this repo)
    ├── nasbench301/           # NAS-Bench-301 API code (+ surrogate model files)
    └── hw_nas_bench/          # HW-NAS-Bench API code (+ search spaces)
```

## 1) Conda environment setup

The recommended way to install dependencies is using `environment.yml`.

### Prerequisites

- Miniconda/Anaconda installed
- Linux/macOS recommended

### Create the environment

From the repository root:

```bash
conda env create -f environment.yml
conda activate nasbench-framework
```

Notes:

- The environment uses Python 3.7 and installs TensorFlow 1.15.5 (required by NAS-Bench-101 code).
- PyTorch is installed as CPU-only by default (`cpuonly`). If you want CUDA, edit `environment.yml` accordingly.

### Update the environment

```bash
conda env update -f environment.yml --prune
```

### Quick sanity check

```bash
python -c "import torch; import tensorflow as tf; print('torch', torch.__version__, 'tf', tf.__version__)"
```

## 2) Using the framework (CLI)

All functionality is exposed via `main.py` using subcommands.

### General help

```bash
python main.py -h
python main.py nasbench101 -h
python main.py nasbench201 -h
python main.py nasbench301 -h
python main.py hwnasbench -h
```

### Command overview

#### NAS-Bench-101

```bash
python main.py nasbench101 --epochs 108 --k 10
```

Options:

- `--epochs` (int, default: `108`, choices: `4, 12, 36, 108`): training budget to query.
- `--k` (int, default: `10`): print top-k architectures by validation accuracy.

#### NAS-Bench-201

```bash
python main.py nasbench201 --dataset cifar10 --hp 12 --setname x-valid --k 10
```

Options:

- `--dataset` (str, default: `cifar10`): e.g. `cifar10`, `cifar10-valid`, `cifar100`, `ImageNet16-120`.
- `--hp` (str, default: `12`, choices: `12, 200`): training regime used in the benchmark.
- `--setname` (str, default: `train`, choices: `train, x-valid, x-test, ori-test`): which split to read metrics from.
- `--is-random` (flag): query metrics for random architectures.
- `--k` (int, default: `10`): print top-k architectures by accuracy.

#### NAS-Bench-301 (surrogate ensemble)

```bash
python main.py nasbench301 --num-samples 100 --seed 0 --version 1.0 --k 10
```

Options:

- `--num-samples` (int, default: `100`): number of random samples to draw from the surrogate ensemble.
- `--seed` (int, default: `0`): RNG seed for sampling.
- `--version` (float, default: `1.0`, choices: `1.0, 2.0`): surrogate ensemble version.
- `--with_noise` (flag): toggles the surrogate noise behavior (see `-h` output / implementation).
- `--k` (int, default: `10`): print top-k architectures by predicted accuracy.

#### HW-NAS-Bench

Example (per-device ranking):

```bash
python main.py hwnasbench \
  --device raspi4 \
  --metric latency \
  --dataset cifar10 \
  --search_space nasbench201 \
  --split x-valid \
  --k 10
```

Example (aggregate across devices + JSON output):

```bash
python main.py hwnasbench --mode aggregate --agg mean --metric latency --json --k 10
```

Options:

- `--device` (str, choices: `edgegpu, edgetpu, eyeriss, fpga, pixel3, raspi4`): target device (recommended for `--mode per_device`).
- `--metric` (str, default: `latency`, choices: `latency, energy, peak_power, avg_power, inference_time`).
- `--dataset` (str, default: `cifar10`).
- `--hp` (str, default: `12`, choices: `12, 200`).
- `--split` (str, default: `x-valid`, choices: `train, x-valid, x-test, ori-test`).
- `--is-random` (flag): query random architectures.
- `--search_space` (str, default: `nasbench201`, choices: `nasbench201, fbnet`).
- `--show-accuracy` (flag): show accuracy along with hardware metrics (only for `nasbench201` search space).
- `--mode` (str, default: `per_device`, choices: `per_device, aggregate`).
- `--agg` (str, default: `mean`, choices: `mean, max, product`): only used when `--mode aggregate`.
- `--json` (flag): print JSON output.
- `--fbnet-samples` (int, default: `2000`): (FBNet only) candidate sample size for top-k.
- `--fbnet-seed` (int, default: `None`): (FBNet only) RNG seed.
- `--k` (int, default: `10`): print top-k architectures by the selected metric.

## 3) Troubleshooting

- If TensorFlow import fails: ensure you are using the conda env `nasbench-framework` and Python 3.7.
- If you see protobuf-related errors with TF1: this environment pins `protobuf=3.20.*` intentionally.

## 4) Entry point

The CLI entry point is `main.py`, which calls `src.get_top_arch(args)`.
