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
from tqdm import tqdm
import random


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


def get_on_nb101(args):
    """Get top architectures from NASBench101"""
    print(f'\n{"="*80}')
    print(f'NASBench101 - Top {args.k} Architectures (Dataset: {args.dataset})')
    print(f'{"="*80}\n')
    
    # Initialize search space and dataset API
    search_space = NasBench101SearchSpace()
    dataset_api = get_dataset_api('nasbench101', args.dataset)
    
    print(f'Loading architectures from NASBench101...')
    
    # Collect all architectures
    results = []
    arch_iterator = search_space.get_arch_iterator(dataset_api)
    
    for arch_hash in tqdm(arch_iterator, desc="Querying architectures"):
        search_space.set_spec(arch_hash, dataset_api=dataset_api)
        
        # Query test accuracy
        test_acc = search_space.query(
            metric=Metric.TEST_ACCURACY,
            dataset=args.dataset,
            dataset_api=dataset_api,
            epoch=108
        )
        
        val_acc = search_space.query(
            metric=Metric.VAL_ACCURACY,
            dataset=args.dataset,
            dataset_api=dataset_api,
            epoch=108
        )
        
        train_time = search_space.query(
            metric=Metric.TRAIN_TIME,
            dataset=args.dataset,
            dataset_api=dataset_api,
            epoch=108
        )
        
        params = search_space.query(
            metric=Metric.PARAMETERS,
            dataset=args.dataset,
            dataset_api=dataset_api,
            epoch=108
        )
        
        results.append({
            'hash': arch_hash,
            'test_acc': test_acc,
            'val_acc': val_acc,
            'train_time': train_time,
            'params': params
        })
    
    # Sort by test accuracy
    results.sort(key=lambda x: x['test_acc'], reverse=True)
    
    # Display top K
    print(f'\n🏆 TOP {args.k} ARCHITECTURES BY TEST ACCURACY:\n')
    for i, result in enumerate(results[:args.k], 1):
        print(f'{i:2d}. Hash: {result["hash"]}')
        print(f'    📊 Test Accuracy:  {result["test_acc"]:.4f}%')
        print(f'    📈 Val Accuracy:   {result["val_acc"]:.4f}%')
        print(f'    ⏱️  Train Time:     {result["train_time"]:.2f}s')
        print(f'    🔢 Parameters:     {result["params"]:.2f}M')
        print()
    
    print_statistics(results)
    return results[:args.k]


def get_on_nb201(args):
    """Get top architectures from NASBench201"""
    print(f'\n{"="*80}')
    print(f'NASBench201 - Top {args.k} Architectures (Dataset: {args.dataset})')
    print(f'{"="*80}\n')
    
    # Initialize search space and dataset API
    search_space = NasBench201SearchSpace()
    dataset_api = get_dataset_api('nasbench201', args.dataset)
    
    print(f'Collecting all architectures from NASBench201...')
    
    # Collect all architectures
    results = []
    arch_iterator = search_space.get_arch_iterator(dataset_api)
    
    for op_indices in tqdm(list(arch_iterator), desc="Querying architectures"):
        search_space.set_spec(op_indices, dataset_api=dataset_api)
        
        # Query test accuracy
        test_acc = search_space.query(
            metric=Metric.TEST_ACCURACY,
            dataset=args.dataset,
            dataset_api=dataset_api
        )
        
        val_acc = search_space.query(
            metric=Metric.VAL_ACCURACY,
            dataset=args.dataset,
            dataset_api=dataset_api
        )
        
        train_time = search_space.query(
            metric=Metric.TRAIN_TIME,
            dataset=args.dataset,
            dataset_api=dataset_api
        )
        
        params = search_space.query(
            metric=Metric.PARAMETERS,
            dataset=args.dataset,
            dataset_api=dataset_api
        )
        
        results.append({
            'op_indices': op_indices,
            'test_acc': test_acc,
            'val_acc': val_acc,
            'train_time': train_time,
            'params': params
        })
    
    # Sort by test accuracy
    results.sort(key=lambda x: x['test_acc'], reverse=True)
    
    # Display top K
    print(f'\n🏆 TOP {args.k} ARCHITECTURES BY TEST ACCURACY:\n')
    for i, result in enumerate(results[:args.k], 1):
        print(f'{i:2d}. Op Indices: {result["op_indices"]}')
        print(f'    📊 Test Accuracy:  {result["test_acc"]:.4f}%')
        print(f'    📈 Val Accuracy:   {result["val_acc"]:.4f}%')
        print(f'    ⏱️  Train Time:     {result["train_time"]:.2f}s')
        print(f'    🔢 Parameters:     {result["params"]:.2f}M')
        print()
    
    print_statistics(results)
    return results[:args.k]


def get_on_nb301(args):
    """Get top architectures from NASBench301 via random sampling"""
    print(f'\n{"="*80}')
    print(f'NASBench301 - Top {args.k} from {args.num_samples} Random Samples (Dataset: {args.dataset})')
    print(f'{"="*80}\n')
    
    # Initialize search space and dataset API
    search_space = NasBench301SearchSpace()
    dataset_api = get_dataset_api('nasbench301', args.dataset)
    
    print(f'Sampling {args.num_samples} unique random architectures...')
    
    # Collect random architectures
    results = []
    sampled_hashes = set()
    
    pbar = tqdm(total=args.num_samples, desc="Sampling architectures", unit="arch")
    
    i = 0
    attempts = 0
    max_attempts = args.num_samples * 100
    
    while i < args.num_samples and attempts < max_attempts:
        attempts += 1
        
        # Sample random architecture
        search_space.sample_random_architecture(dataset_api=dataset_api)
        arch_hash = search_space.get_hash()
        
        # Skip if already sampled
        if arch_hash in sampled_hashes:
            continue
        
        sampled_hashes.add(arch_hash)
        
        # Query metrics
        test_acc = search_space.query(
            metric=Metric.TEST_ACCURACY,
            dataset=args.dataset,
            dataset_api=dataset_api
        )
        
        val_acc = search_space.query(
            metric=Metric.VAL_ACCURACY,
            dataset=args.dataset,
            dataset_api=dataset_api
        )
        
        train_time = search_space.query(
            metric=Metric.TRAIN_TIME,
            dataset=args.dataset,
            dataset_api=dataset_api
        )
        
        params = search_space.query(
            metric=Metric.PARAMETERS,
            dataset=args.dataset,
            dataset_api=dataset_api
        )
        
        results.append({
            'hash': arch_hash,
            'test_acc': test_acc,
            'val_acc': val_acc,
            'train_time': train_time,
            'params': params
        })
        
        i += 1
        pbar.update(1)
    
    pbar.close()
    
    # Sort by test accuracy
    results.sort(key=lambda x: x['test_acc'], reverse=True)
    
    # Display top K
    print(f'\n🏆 TOP {args.k} ARCHITECTURES BY TEST ACCURACY:\n')
    for i, result in enumerate(results[:args.k], 1):
        print(f'{i:2d}. Architecture Hash: {result["hash"]}')
        print(f'    📊 Test Accuracy:  {result["test_acc"]:.4f}%')
        print(f'    📈 Val Accuracy:   {result["val_acc"]:.4f}%')
        print(f'    ⏱️  Train Time:     {result["train_time"]:.2f}s')
        print(f'    🔢 Parameters:     {result["params"]:.2f}M')
        print()
    
    print_statistics(results)
    return results[:args.k]


def get_on_transbench101_micro(args):
    """Get top architectures from TransBench101 Micro via random sampling"""
    print(f'\n{"="*80}')
    print(f'TransBench101-Micro - Top {args.k} from {args.num_samples} Random Samples (Task: {args.dataset})')
    print(f'{"="*80}\n')
    
    # Initialize search space and dataset API
    search_space = TransBench101SearchSpaceMicro()
    dataset_api = get_dataset_api('transbench101_micro', args.dataset)
    
    print(f'Sampling {args.num_samples} unique random architectures...')
    
    # Collect random architectures
    results = []
    sampled_hashes = set()
    
    pbar = tqdm(total=args.num_samples, desc="Sampling architectures", unit="arch")
    
    i = 0
    attempts = 0
    max_attempts = args.num_samples * 100
    
    while i < args.num_samples and attempts < max_attempts:
        attempts += 1
        
        # Sample random architecture
        search_space.sample_random_architecture(dataset_api=dataset_api)
        arch_hash = search_space.get_hash()
        
        # Skip if already sampled
        if arch_hash in sampled_hashes:
            continue
        
        sampled_hashes.add(arch_hash)
        
        # Query metrics
        test_acc = search_space.query(
            metric=Metric.TEST_ACCURACY,
            dataset=args.dataset,
            dataset_api=dataset_api
        )
        
        val_acc = search_space.query(
            metric=Metric.VAL_ACCURACY,
            dataset=args.dataset,
            dataset_api=dataset_api
        )
        
        train_time = search_space.query(
            metric=Metric.TRAIN_TIME,
            dataset=args.dataset,
            dataset_api=dataset_api
        )
        
        params = search_space.query(
            metric=Metric.PARAMETERS,
            dataset=args.dataset,
            dataset_api=dataset_api
        )
        
        results.append({
            'hash': arch_hash,
            'test_acc': test_acc,
            'val_acc': val_acc,
            'train_time': train_time,
            'params': params
        })
        
        i += 1
        pbar.update(1)
    
    pbar.close()
    
    # Sort by test accuracy
    results.sort(key=lambda x: x['test_acc'], reverse=True)
    
    # Display top K
    print(f'\n🏆 TOP {args.k} ARCHITECTURES BY TEST ACCURACY:\n')
    for i, result in enumerate(results[:args.k], 1):
        print(f'{i:2d}. Architecture Hash: {result["hash"]}')
        print(f'    📊 Test Accuracy:  {result["test_acc"]:.4f}%')
        print(f'    📈 Val Accuracy:   {result["val_acc"]:.4f}%')
        print(f'    ⏱️  Train Time:     {result["train_time"]:.2f}s')
        print(f'    🔢 Parameters:     {result["params"]:.2f}M')
        print()
    
    print_statistics(results)
    return results[:args.k]


def get_on_transbench101_macro(args):
    """Get top architectures from TransBench101 Macro via random sampling"""
    print(f'\n{"="*80}')
    print(f'TransBench101-Macro - Top {args.k} from {args.num_samples} Random Samples (Task: {args.dataset})')
    print(f'{"="*80}\n')
    
    # Initialize search space and dataset API
    search_space = TransBench101SearchSpaceMacro()
    dataset_api = get_dataset_api('transbench101_macro', args.dataset)
    
    print(f'Sampling {args.num_samples} unique random architectures...')
    
    # Collect random architectures
    results = []
    sampled_hashes = set()
    
    pbar = tqdm(total=args.num_samples, desc="Sampling architectures", unit="arch")
    
    i = 0
    attempts = 0
    max_attempts = args.num_samples * 100
    
    while i < args.num_samples and attempts < max_attempts:
        attempts += 1
        
        # Sample random architecture
        search_space.sample_random_architecture(dataset_api=dataset_api)
        arch_hash = search_space.get_hash()
        
        # Skip if already sampled
        if arch_hash in sampled_hashes:
            continue
        
        sampled_hashes.add(arch_hash)
        
        # Query metrics
        test_acc = search_space.query(
            metric=Metric.TEST_ACCURACY,
            dataset=args.dataset,
            dataset_api=dataset_api
        )
        
        val_acc = search_space.query(
            metric=Metric.VAL_ACCURACY,
            dataset=args.dataset,
            dataset_api=dataset_api
        )
        
        train_time = search_space.query(
            metric=Metric.TRAIN_TIME,
            dataset=args.dataset,
            dataset_api=dataset_api
        )
        
        params = search_space.query(
            metric=Metric.PARAMETERS,
            dataset=args.dataset,
            dataset_api=dataset_api
        )
        
        results.append({
            'hash': arch_hash,
            'test_acc': test_acc,
            'val_acc': val_acc,
            'train_time': train_time,
            'params': params
        })
        
        i += 1
        pbar.update(1)
    
    pbar.close()
    
    # Sort by test accuracy
    results.sort(key=lambda x: x['test_acc'], reverse=True)
    
    # Display top K
    print(f'\n🏆 TOP {args.k} ARCHITECTURES BY TEST ACCURACY:\n')
    for i, result in enumerate(results[:args.k], 1):
        print(f'{i:2d}. Architecture Hash: {result["hash"]}')
        print(f'    📊 Test Accuracy:  {result["test_acc"]:.4f}%')
        print(f'    📈 Val Accuracy:   {result["val_acc"]:.4f}%')
        print(f'    ⏱️  Train Time:     {result["train_time"]:.2f}s')
        print(f'    🔢 Parameters:     {result["params"]:.2f}M')
        print()
    
    print_statistics(results)
    return results[:args.k]


def print_statistics(results):
    """Print statistics about the results"""
    if not results:
        return
    
    best = results[0]
    worst = results[-1]
    avg_acc = sum(r['test_acc'] for r in results) / len(results)
    
    print(f'{"="*80}')
    print(f'📈 STATISTICS')
    print(f'{"="*80}')
    print(f'Best Test Accuracy:    {best["test_acc"]:.4f}%')
    print(f'Worst Test Accuracy:   {worst["test_acc"]:.4f}%')
    print(f'Average Test Accuracy: {avg_acc:.4f}%')
    print(f'Total Architectures:   {len(results)}')
    print()



def get_top_arch(args):
    if args.search_space == "nasbench101":
        get_on_nb101(args)
    elif args.search_space == "nasbench201":
        get_on_nb201(args)
    elif args.search_space == "nasbench301":
        get_on_nb301(args)
    elif args.search_space == "transbench101_micro":
        get_on_transbench101_micro(args)
    elif args.search_space == "transbench101_macro":
        get_on_transbench101_macro(args)
    else:
        raise ValueError(f"Unsupported search space: {args.search_space}")

    