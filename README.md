# NASBench Framework

This repository provides a simple command-line interface (CLI) to query multiple NAS benchmarks from one place:

- NAS-Bench-101
- NAS-Bench-201
- NAS-Bench-301
- HW-NAS-Bench
- AccelNASBench
- NAS-Bench-Suite-Zero
- NAS-Bench-X11 (NB101, NB201, NB301, and NB-NLP surrogate models)

## Repository Layout (Suggested)

The current project structure is:

```text
.
├── environment.yml            # Conda environment definition
├── main.py                    # CLI entry point (argparse)
├── README.md
├── src/                       # CLI implementation (dispatches to per-benchmark helpers)
│   ├── __init__.py            # get_top_arch(args) router
│   ├── get_nasbench101.py
│   ├── get_nasbench201.py
│   ├── get_nasbench301.py
│   ├── get_hwnasbench.py
│   ├── get_accelnasbench.py
│   ├── get_nasbench_x11.py
│   ├── get_nasbench_graph.py
│   └── get_suitezero.py
└── nasbench/                  # Benchmark APIs + bundled assets
    ├── accel_nasbench/         # AccelNASBench surrogate models + search space
    ├── hw_nas_bench/           # HW-NAS-Bench API code (+ search spaces)
    ├── nasbench101/            # NAS-Bench-101 API code (+ TFRecord in this repo)
    ├── nasbench201/            # NAS-Bench-201 API code (+ .pth in this repo)
    ├── nasbench301/            # NAS-Bench-301 API code (+ surrogate model files)
    ├── nas_bench_x11/          # NAS-Bench-X11 surrogate models for NB101/201/301/NLP
    ├── nas_bench_graph/        # NAS-Bench-Graph benchmarks
    └── nas_bench_suite_zero/   # NAS-Bench-Suite-Zero (naslib integration)
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

### Help

```bash
python main.py -h
python main.py nasbench101 -h
python main.py nasbench201 -h
python main.py nasbench301 -h
python main.py hwnasbench -h
python main.py accelnasbench -h
python main.py suitezero -h
python main.py nasbench_x11 -h
```

### NAS-Bench-101

**Usage**

```bash
python main.py nasbench101 [--epochs {4,12,36,108}] [--k K]
```

**Options**

- `--epochs` (int, default: `108`, choices: `4, 12, 36, 108`): training budget to query.
- `--k` (int, default: `10`): print top-k architectures by validation accuracy.

**Examples**

```bash
python main.py nasbench101 --epochs 108 --k 10
python main.py nasbench101 --epochs 36 --k 20
```

### NAS-Bench-201

**Usage**

```bash
python main.py nasbench201 [--dataset DATASET] [--hp {12,200}] [--setname {train,x-valid,x-test,ori-test}] [--is-random] [--k K]
```

**Options**

- `--dataset` (str, default: `cifar10`): e.g. `cifar10`, `cifar10-valid`, `cifar100`, `ImageNet16-120`.
- `--hp` (str, default: `12`, choices: `12, 200`): training regime used in the benchmark.
- `--setname` (str, default: `train`, choices: `train, x-valid, x-test, ori-test`): which split to read metrics from.
- `--is-random` (flag): if set, query metrics for random architectures.
- `--k` (int, default: `10`): print top-k architectures by accuracy.

**Examples**

```bash
python main.py nasbench201 --dataset cifar10 --hp 12 --setname x-valid --k 10
python main.py nasbench201 --dataset cifar100 --hp 200 --setname x-test --is-random --k 10
```

### NAS-Bench-301 (surrogate ensemble)

**Usage**

```bash
python main.py nasbench301 [--with_noise] [--num-samples N] [--seed SEED] [--version {1.0,2.0}] [--k K]
```

**Options**

- `--num-samples` (int, default: `100`): number of random samples to draw from the surrogate ensemble.
- `--seed` (int, default: `0`): RNG seed for sampling.
- `--version` (float, default: `1.0`, choices: `1.0, 2.0`): surrogate ensemble version.
- `--with_noise` (flag): toggles the surrogate noise behavior (see `-h` output / implementation).
- `--k` (int, default: `10`): print top-k architectures by predicted accuracy.

**Examples**

```bash
python main.py nasbench301 --num-samples 100 --seed 0 --version 1.0 --k 10
python main.py nasbench301 --num-samples 500 --seed 1 --version 2.0 --k 50
```

### HW-NAS-Bench

**Usage (per-device)**

```bash
python main.py hwnasbench --mode per_device --device DEVICE [--metric METRIC] [--dataset DATASET] [--search_space {nasbench201,fbnet}] [--hp {12,200}] [--split {train,x-valid,x-test,ori-test}] [--is-random] [--show-accuracy] [--json] [--k K]
```

**Usage (aggregate)**

```bash
python main.py hwnasbench --mode aggregate --agg {mean,max,product} [--metric METRIC] [--dataset DATASET] [--search_space {nasbench201,fbnet}] [--hp {12,200}] [--split {train,x-valid,x-test,ori-test}] [--is-random] [--show-accuracy] [--json] [--k K]
```

**Options**

- `--mode` (str, default: `per_device`, choices: `per_device, aggregate`): choose per-device query or aggregate scoring.
- `--device` (str, choices: `edgegpu, edgetpu, eyeriss, fpga, pixel3, raspi4`): target device (used in `per_device` mode).
- `--agg` (str, default: `mean`, choices: `mean, max, product`): aggregation rule (only for `aggregate` mode).
- `--metric` (str, default: `latency`, choices: `latency, energy, peak_power, avg_power, inference_time`): hardware metric.
- `--dataset` (str, default: `cifar10`).
- `--search_space` (str, default: `nasbench201`, choices: `nasbench201, fbnet`).
- `--hp` (str, default: `12`, choices: `12, 200`).
- `--split` (str, default: `x-valid`, choices: `train, x-valid, x-test, ori-test`).
- `--is-random` (flag): query random architectures.
- `--show-accuracy` (flag): show accuracy along with hardware metrics (only for `nasbench201` search space).
- `--json` (flag): print JSON output.
- `--k` (int, default: `10`): print top-k architectures.
- `--fbnet-samples` (int, default: `2000`): (FBNet only) candidate sample size for top-k.
- `--fbnet-seed` (int, default: `None`): (FBNet only) RNG seed.

**Examples**

```bash
python main.py hwnasbench --mode per_device --device raspi4 --metric latency --dataset cifar10 --search_space nasbench201 --split x-valid --k 10
python main.py hwnasbench --mode aggregate --agg mean --metric latency --json --k 10
```

### AccelNASBench (surrogate ensemble)

AccelNASBench queries a surrogate ensemble over an EfficientNet-like search space and can rank candidates by:

- `accuracy` (higher is better)
- `throughput` (higher is better)
- `latency` (lower is better)

**Usage**

```bash
python main.py accelnasbench [--seed SEED] [--num-candidates N] [--top-k K] [--sort-by {accuracy,throughput,latency}] [--throughput-device DEV] [--latency-device DEV] [--model MODEL]
```

**Options**

- `--seed` (int, default: `3`): RNG seed used when sampling candidates and loading the ensemble.
- `--num-candidates` (int, default: `200`): number of random architectures to sample before ranking.
- `--top-k` (int, default: `10`): number of best architectures to print.
- `--sort-by` (str, default: `accuracy`, choices: `accuracy, throughput, latency`): objective used for ranking.
- `--throughput-device` (str, default: `tpuv2`): device name used for throughput surrogate.
- `--latency-device` (str, default: `zcu102`): device name used for latency surrogate.
- `--model` (str, default: `xgb`): surrogate model family.

**Examples**

```bash
python main.py accelnasbench --num-candidates 200 --top-k 10 --sort-by accuracy --seed 3
python main.py accelnasbench --sort-by throughput --throughput-device tpuv2 --model xgb
python main.py accelnasbench --sort-by latency --latency-device zcu102 --model xgb
```

### NAS-Bench-Suite-Zero (SuiteZero)

This subcommand exposes several NASLib search spaces packaged in NAS-Bench-Suite-Zero and returns the top-$k$ architectures by a chosen metric.

**Usage**

```bash
python main.py [--seed SEED] suitezero \
  --search_space {nasbench101,nasbench201,nasbench301,transbench101_macro,transbench101_micro} \
  [--dataset DATASET] [--metric {val_acc,test_acc,train_time,latency,parameters,flops}] [--k K] [--num_samples N] [--jsonl]
```

Notes:

- `--seed` is a global option for `main.py` and should be placed before the subcommand.
- `--num_samples` is used for random-sampling based spaces (e.g. `nasbench301`, `transbench101_*`).
- Metric support depends on the selected search space:
  - `nasbench101`: `val_acc`, `test_acc`, `train_time`, `parameters`
  - `nasbench201`: `val_acc`, `test_acc`, `train_time`, `parameters`
  - `nasbench301`: `val_acc`, `train_time`
  - `transbench101_micro` / `transbench101_macro`: `val_acc`, `test_acc`, `train_time`

**Datasets / tasks**

- `nasbench101`: `cifar10`
- `nasbench201`: `cifar10`, `cifar100`, `ImageNet16-120`
- `nasbench301`: `cifar10`
- `transbench101_micro` / `transbench101_macro`:
  `class_scene`, `class_object`, `jigsaw`, `room_layout`, `segmentsemantic`, `normal`, `autoencoder`

**Examples**

TransBench101 Macro (Taskonomy), sample 100 candidates and take top-5:

```bash
python main.py suitezero --search_space transbench101_macro --dataset class_scene --k 5 --num_samples 100
```

NAS-Bench-201 on CIFAR-100, rank by validation accuracy:

```bash
python main.py suitezero --search_space nasbench201 --dataset cifar100 --metric val_acc --k 10
```

Machine-readable output (JSON Lines):

```bash
python main.py suitezero --search_space transbench101_micro --dataset class_scene --k 5 --num_samples 100 --jsonl
```

### NAS-Bench-X11 (surrogate models)

NAS-Bench-X11 provides unified surrogate models for predicting architecture performance across multiple search spaces:

- **nb101**: NAS-Bench-101 surrogate (predicts learning curves from architecture encodings)
- **nb201**: NAS-Bench-201 surrogate (predicts learning curves from architecture strings)
- **nb301**: NAS-Bench-301 surrogate (predicts learning curves from DARTS genotypes)
- **nbnlp**: NAS-Bench-NLP surrogate (predicts learning curves for NLP architectures)

**Usage**

```bash
python main.py nasbench_x11 --search_space {nb101,nb201,nb301,nbnlp} [--num_samples N] [--epoch EPOCH] [--k K]
```

**Options**

- `--search_space` (str, default: `nbnlp`, choices: `nb101, nb201, nb301, nbnlp`): which search space and surrogate model to use.
- `--num_samples` (int, default: `10000`): number of random architectures to sample and evaluate with the surrogate model.
  - For `nb101`: samples from all architectures in NAS-Bench-101 (up to 423,624 total).
  - For `nb201`: samples from all architectures in NAS-Bench-201 (up to 15,625 total).
  - For `nb301` and `nbnlp`: generates random architectures from the respective search spaces.
- `--epoch` (int, default: `0`): epoch index for extracting accuracy from the predicted learning curve (0-indexed).
  - For `nb201`: typically use epochs 0-199 (hp=200) or 0-11 (hp=12).
  - For `nb301` and `nbnlp`: typically use epochs 0-97 (98 epochs total).
- `--k` (int, default: `10`): print top-k architectures by predicted accuracy.

**Examples**

Query NAS-Bench-101 surrogate model (evaluates all ~423k architectures, returns actual test accuracy):

```bash
python main.py nasbench_x11 --search_space nb101 --k 10
```

Query NAS-Bench-201 surrogate model (epoch 199):

```bash
python main.py nasbench_x11 --search_space nb201 --epoch 199 --k 20
```

Query NAS-Bench-301 surrogate model (sample 10000 random DARTS genotypes):

```bash
python main.py nasbench_x11 --search_space nb301 --num_samples 10000 --epoch 97 --k 10
```

Query NAS-Bench-NLP surrogate model (sample 5000 random NLP architectures):

```bash
python main.py nasbench_x11 --search_space nbnlp --num_samples 5000 --epoch 50 --k 15
```

**Notes**

- The `nb101` search space queries all architectures from the original NAS-Bench-101 dataset and uses the surrogate to predict learning curves.
- The `nb201` search space can query up to all 15,625 architectures or sample a subset.
- The `nb301` and `nbnlp` search spaces generate random valid architectures from their respective search spaces.
- All surrogate models predict learning curves with noise enabled for more realistic predictions.
- The models are loaded from `nasbench/nas_bench_x11/models/` directory.

## 3) Troubleshooting

- If TensorFlow import fails: ensure you are using the conda env `nasbench-framework` and Python 3.7.
- If you see protobuf-related errors with TF1: this environment pins `protobuf=3.20.*` intentionally.

## 4) Entry point

The CLI entry point is `main.py`, which calls `src.get_top_arch(args)`.
