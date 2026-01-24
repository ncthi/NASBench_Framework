import argparse
from src import get_top_arch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NAS Benchmark Framework")

    subparsers = parser.add_subparsers(dest="nasbench", required=True)

    # NAS-Bench-101
    p101 = subparsers.add_parser("nasbench101", help="Query NAS-Bench-101")
    p101.add_argument(
        "--epochs",
        type=int,
        default=108,
        choices=[4, 12, 36, 108],
        help="Training epochs budget to query",
    )
    p101.add_argument(
        "--k",
        type=int,
        default=10,
        help="If set, prints top-k architectures by validation accuracy",
    )

    # NAS-Bench-201
    p201 = subparsers.add_parser("nasbench201", help="Query NAS-Bench-201")
    p201.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="Dataset name (e.g., cifar10, cifar10-valid, cifar100, ImageNet16-120)",
    )
    p201.add_argument(
        "--hp",
        type=str,
        default="12",
        choices=["12", "200"],
        help="Training hyper-parameter regime used in the benchmark",
    )
    p201.add_argument(
        "--setname",
        type=str,
        default="train",
        choices=["train", "x-valid", "x-test","ori-test"],
        help="Dataset split to query the metrics from",
    )
    p201.add_argument(
        "--is-random",
        action="store_true",
        help="If set, query metrics for random architectures",
    )
    p201.add_argument(
        "--k",
        type=int,
        default=10,
        help="If set, prints top-k architectures by accuracy",
    )
    # NAS-Bench-301
    p301 = subparsers.add_parser("nasbench301", help="Query NAS-Bench-301 surrogate ensemble")
    p301.add_argument(
        "--with_noise",
        action="store_true",
        help="If set, query ensemble mean (no noise)",
    )
    p301.add_argument(
        "--k",
        type=int,
        default=10,
        help="If set, prints top-k architectures by predicted accuracy",
    )
    p301.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of random samples to draw from the surrogate ensemble",
    )
    p301.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling from the surrogate ensemble",
    )
    p301.add_argument(
        "--version",
        type=float,
        default=1.0,
        choices=[1.0, 2.0],
        help="Version of the NAS-Bench-301 surrogate ensemble to use",
    )

    phw = subparsers.add_parser("hwnasbench", help="Query HW-NAS-Bench")
    phw.add_argument(
        "--device",
        type=str,
        default=None,
        choices=['edgegpu', 'edgetpu', 'eyeriss', 'fpga', 'pixel3', 'raspi4'],
        help="Target device to query",
    )
    phw.add_argument(
        "--metric",
        type=str,
        default="latency",
        choices=["latency", "energy", "peak_power", "avg_power", "inference_time"],
        help="Hardware metric to query",
    )
    phw.add_argument(
        "--k",
        type=int,
        default=10,
        help="If set, prints top-k architectures by the specified hardware metric",
    )
    phw.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="Dataset name (e.g., cifar10, cifar10-valid, cifar100, ImageNet16-120)",
    )
    phw.add_argument("--agg", default="mean", choices=["mean", "max", "product"], help="Only for --mode aggregate")
    phw.add_argument(
        "--hp",
        type=str,
        default="12",
        choices=["12", "200"],
        help="Training hyper-parameter regime used in the benchmark",
    )
    phw.add_argument(
        "--split",
        type=str,
        default="x-valid",
        choices=["train", "x-valid", "x-test","ori-test"],
        help="Dataset split to query the metrics from",
    )
    phw.add_argument(
        "--is-random",
        action="store_true",
        help="If set, query metrics for random architectures",
    )
    phw.add_argument(
        "--search_space",
        type=str,
        default="nasbench201",
        choices=["nasbench201", "fbnet"],
        help="Search space of the architectures",
    )
    phw.add_argument(
        "--show-accuracy",
        action="store_true",
        help="If set, shows accuracy along with hardware metrics (only for nasbench201 search space)",
    )
    phw.add_argument("--mode", default="per_device", choices=["per_device", "aggregate"])
    phw.add_argument("--json", action="store_true", help="Print JSON output")
    phw.add_argument(
        "--fbnet-samples",
        type=int,
        default=2000,
        help="(FBNet only) Number of random architectures to sample for top-k (default: 2000)",
    )
    phw.add_argument(
        "--fbnet-seed",
        type=int,
        default=None,
        help="(FBNet only) RNG seed for sampling candidates",
    )
    paccel = subparsers.add_parser("accelnasbench", help="Query AccelNASBench surrogate ensemble")
    
    paccel.add_argument("--seed", type=int, default=3)
    paccel.add_argument("--num-candidates", type=int, default=200)
    paccel.add_argument("--top-k", type=int, default=10)
    paccel.add_argument(
        "--sort-by",
        type=str,
        default="accuracy",
        choices=["accuracy", "throughput", "latency"],
        help="Which metric to optimize (accuracy/throughput: max, latency: min).",
    )
    paccel.add_argument("--throughput-device", type=str, default="tpuv2")
    paccel.add_argument("--latency-device", type=str, default="zcu102")
    paccel.add_argument("--model", type=str, default="xgb")

    pzero= subparsers.add_parser("suitezero", help="Query SuiteZero benchmark")
    pzero.add_argument(
        "--search_space",
        required=True,
        choices="nasbench101,nasbench201,nasbench301,transbench101_macro,transbench101_micro".split(","),
        help="Benchmark/search space to query.",
    )
    pzero.add_argument(
        "--task",
        required=False,
        type=str,
        help="Task/dataset name (defaults to the first supported for the selected search space).",
    )
    pzero.add_argument(
        "--metric",
        required=False,
        default="VAL_ACCURACY",
        type=str,
        choices=["VAL_ACCURACY", "TEST_ACCURACY", "TRAIN_ACCURACY", "LATENCY", "PARAMETERS", "FLOPS"],
        help="Metric to use for ranking architectures.",
    )
    pzero.add_argument(
        "--top_k",
        required=False,
        default=10,
        type=int,
        help="Number of top architectures to return.",
    )
    pzero.add_argument(
        "--max_archs",
        required=False,
        default=None,
        type=int,
        help="Maximum number of architectures to evaluate (None = all).",
    )
    pzero.add_argument(
        "--epoch",
        required=False,
        default=-1,
        type=int,
        help="Epoch index passed to graph.query (many benchmarks use -1 for last epoch).",
    )
    pzero.add_argument(
        "--jsonl",
        required=False,
        action="store_true",
        help="Print results as JSON Lines.",
    )
    return parser


def main(args: argparse.Namespace) -> int:
    get_top_arch(args)
    return 0



if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(main(args))
