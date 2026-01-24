#!/usr/bin/env python3

import argparse
import json
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

from nasbench.nas_bench_suite_zero.naslib.search_spaces import (
    NasBench101SearchSpace,
    NasBench201SearchSpace,
    NasBench301SearchSpace,
    TransBench101SearchSpaceMacro,
    TransBench101SearchSpaceMicro,
)
from nasbench.nas_bench_suite_zero.naslib.search_spaces.core.query_metrics import Metric
from nasbench.nas_bench_suite_zero.naslib.utils import get_dataset_api


SEARCH_SPACES = {
    "nasbench101": NasBench101SearchSpace,
    "nasbench201": NasBench201SearchSpace,
    "nasbench301": NasBench301SearchSpace,
    "transbench101_micro": TransBench101SearchSpaceMicro,
    "transbench101_macro": TransBench101SearchSpaceMacro,
}

TASKS = {
    "nasbench101": ["cifar10"],
    "nasbench201": ["cifar10", "cifar100", "ImageNet16-120"],
    "nasbench301": ["cifar10"],
    "transbench101_micro": [
        "class_scene",
        "class_object",
        "jigsaw",
        "room_layout",
        "segmentsemantic",
        "normal",
        "autoencoder",
    ],
    "transbench101_macro": [
        "class_scene",
        "class_object",
        "jigsaw",
        "room_layout",
        "segmentsemantic",
        "normal",
        "autoencoder",
    ],
}

SORTABLE_METRICS = {
    "VAL_ACCURACY": (Metric.VAL_ACCURACY, True),  # (metric, higher_is_better)
    "TEST_ACCURACY": (Metric.TEST_ACCURACY, True),
    "TRAIN_ACCURACY": (Metric.TRAIN_ACCURACY, True),
    "TRAIN_TIME": (Metric.TRAIN_TIME, False),
    "VAL_LOSS": (Metric.VAL_LOSS, False),
    "TEST_LOSS": (Metric.TEST_LOSS, False),
    "TRAIN_LOSS": (Metric.TRAIN_LOSS, False),
    "LATENCY": (Metric.LATENCY, False),
    "PARAMETERS": (Metric.PARAMETERS, False),
    "FLOPS": (Metric.FLOPS, False),
}


def _get_arch_genotype(graph: Any, search_space: str) -> Optional[Any]:
    """Get architecture genotype representation."""
    try:
        if search_space == "nasbench101":
            spec = graph.get_spec()
            if spec and isinstance(spec, dict):
                matrix = spec.get("matrix")
                ops = spec.get("ops")
                if matrix is not None and ops is not None:
                    try:
                        import numpy as np
                        if isinstance(matrix, np.ndarray):
                            matrix = matrix.tolist()
                    except Exception:
                        pass
                    return {"matrix": matrix, "ops": ops}
            return graph.get_hash()

        if search_space == "nasbench301":
            from naslib.search_spaces.nasbench301.conversions import (
                convert_naslib_to_genotype,
            )
            genotype = convert_naslib_to_genotype(graph)
            if hasattr(genotype, "_asdict"):
                return genotype._asdict()
            return genotype

        if search_space == "nasbench201":
            from naslib.search_spaces.nasbench201.conversions import (
                convert_naslib_to_str,
            )
            return convert_naslib_to_str(graph)

        if search_space.startswith("transbench101"):
            op_indices = graph.get_op_indices()
            if search_space == "transbench101_macro":
                from naslib.search_spaces.transbench101.conversions import (
                    convert_op_indices_macro_to_str,
                )
                return convert_op_indices_macro_to_str(op_indices)
            else:
                from naslib.search_spaces.transbench101.conversions import (
                    convert_op_indices_micro_to_str,
                )
                return convert_op_indices_micro_to_str(op_indices)
    except Exception:
        pass

    try:
        return graph.get_hash()
    except Exception:
        return None


def _maybe_query_hp_metric(
    *,
    graph: Any,
    task: str,
    epoch: int,
    dataset_api,
    metric: Metric,
) -> Optional[Any]:
    """Try to query metric from HP dict."""
    if metric not in {Metric.FLOPS, Metric.PARAMETERS, Metric.LATENCY, Metric.TRAIN_TIME}:
        return None

    try:
        hp = graph.query(Metric.HP, dataset=task, epoch=epoch, dataset_api=dataset_api)
    except Exception:
        return None

    if not isinstance(hp, dict):
        return None

    if metric == Metric.FLOPS:
        return hp.get("flops", hp.get("flop"))
    if metric == Metric.PARAMETERS:
        return hp.get("params", hp.get("parameters"))
    if metric == Metric.LATENCY:
        return hp.get("latency")
    if metric == Metric.TRAIN_TIME:
        return hp.get("train_time")

    return None


def _query_architecture(
    *,
    search_space: str,
    task: str,
    metric: Metric,
    arch_spec: Any,
    epoch: int,
    dataset_api,
) -> Optional[Dict]:
    """Query a single architecture and return its metrics."""
    try:
        graph = SEARCH_SPACES[search_space]()
        graph.set_spec(arch_spec, dataset_api=dataset_api)

        genotype = _get_arch_genotype(graph, search_space)

        # Query the metric
        try:
            value = graph.query(metric, dataset=task, epoch=epoch, dataset_api=dataset_api)
            if value == -1:
                value = _maybe_query_hp_metric(
                    graph=graph, task=task, epoch=epoch, dataset_api=dataset_api, metric=metric
                )
                if value is None:
                    return None
        except Exception:
            value = _maybe_query_hp_metric(
                graph=graph, task=task, epoch=epoch, dataset_api=dataset_api, metric=metric
            )
            if value is None:
                return None

        return {
            "genotype": genotype,
            "arch_spec": arch_spec,
            "metric_value": value,
        }
    except Exception as e:
        return None


def get_top_architectures(
    *,
    search_space: str,
    task: str,
    metric_name: str,
    top_k: int,
    epoch: int,
    max_archs: Optional[int],
    dataset_api,
) -> List[Dict]:
    """Get top-k architectures by the specified metric."""
    
    if metric_name not in SORTABLE_METRICS:
        raise ValueError(f"Metric {metric_name} not supported. Choose from: {list(SORTABLE_METRICS.keys())}")

    metric, higher_is_better = SORTABLE_METRICS[metric_name]

    print(f"Querying {search_space} on {task} for metric {metric_name}...")
    print(f"{'Maximizing' if higher_is_better else 'Minimizing'} {metric_name}")

    graph = SEARCH_SPACES[search_space]()
    
    # Check if search space has arch_iterator or needs random sampling
    has_iterator = hasattr(graph, 'get_arch_iterator') and callable(getattr(graph, 'get_arch_iterator'))
    
    results = []
    count = 0
    errors = 0
    seen_genotypes = set()  # Track unique architectures

    if has_iterator:
        # Use iterator for exhaustive search
        try:
            arch_iterator = graph.get_arch_iterator(dataset_api=dataset_api)
            # Convert to list to avoid segfault with dictionary keys iterator
            if hasattr(arch_iterator, 'keys') or not isinstance(arch_iterator, list):
                arch_iterator = list(arch_iterator)
        except Exception as e:
            raise SystemExit(f"Failed to get architecture iterator: {e}")

        for arch_spec in arch_iterator:
            if max_archs is not None and count >= max_archs:
                break

            result = _query_architecture(
                search_space=search_space,
                task=task,
                metric=metric,
                arch_spec=arch_spec,
                epoch=epoch,
                dataset_api=dataset_api,
            )

            if result is not None:
                results.append(result)
                count += 1
                if count % 100 == 0:
                    print(f"Processed {count} architectures, found {len(results)} valid, {errors} errors")
            else:
                errors += 1
    else:
        # Use random sampling for search spaces without iterator
        print(f"Using random sampling (no exhaustive iterator available)")
        
        max_samples = max_archs if max_archs is not None else 1000  # Default to 1000 samples
        max_retries = max_samples * 10  # Allow retries for duplicates
        
        attempts = 0
        while count < max_samples and attempts < max_retries:
            attempts += 1
            
            # Sample random architecture
            try:
                graph_sample = SEARCH_SPACES[search_space]()
                graph_sample.sample_random_architecture(dataset_api=dataset_api)
                arch_spec = graph_sample.get_op_indices()
                
                # Convert to tuple for hashing
                arch_key = tuple(arch_spec) if hasattr(arch_spec, '__iter__') else arch_spec
                
                # Skip duplicates
                if arch_key in seen_genotypes:
                    continue
                    
                seen_genotypes.add(arch_key)
                
            except Exception as e:
                errors += 1
                continue

            result = _query_architecture(
                search_space=search_space,
                task=task,
                metric=metric,
                arch_spec=arch_spec,
                epoch=epoch,
                dataset_api=dataset_api,
            )

            if result is not None:
                results.append(result)
                count += 1
                if count % 50 == 0:
                    print(f"Sampled {count} unique architectures, found {len(results)} valid, {errors} errors")
            else:
                errors += 1

    print(f"Total processed: {count} architectures, {len(results)} valid, {errors} errors")

    # Sort results
    results.sort(key=lambda x: x["metric_value"], reverse=higher_is_better)

    return results[:top_k]


def get_top_arch(args) -> int:

    task = args.task or TASKS[args.search_space][0]
    if args.search_space in TASKS and task not in TASKS[args.search_space]:
        valid = ", ".join(TASKS[args.search_space])
        raise SystemExit(f"Unknown task '{task}' for {args.search_space}. Valid: {valid}")

    if args.top_k <= 0:
        raise SystemExit("--top_k must be >= 1")

    dataset_api = get_dataset_api(search_space=args.search_space, dataset=task)

    top_archs = get_top_architectures(
        search_space=args.search_space,
        task=task,
        metric_name=args.metric.upper(),
        top_k=args.top_k,
        epoch=args.epoch,
        max_archs=args.max_archs,
        dataset_api=dataset_api,
    )

    if args.jsonl:
        for i, arch in enumerate(top_archs, 1):
            output = {
                "rank": i,
                "genotype": arch["genotype"],
                "metric": args.metric.upper(),
                "value": arch["metric_value"],
            }
            print(json.dumps(output, default=str))
    else:
        print(f"\n{'='*80}")
        print(f"Top {len(top_archs)} architectures by {args.metric.upper()} on {args.search_space}/{task}")
        print(f"{'='*80}\n")

        for i, arch in enumerate(top_archs, 1):
            print(f"Rank {i}: {args.metric.upper()} = {arch['metric_value']}")
            print(f"  Genotype: {arch['genotype']}")
            print()

    return 0

