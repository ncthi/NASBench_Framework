import argparse
import os
from pathlib import Path
import heapq

from nasbench.nasbench201 import NASBench201API as API

api_file="nasbench/nasbench201/NAS-Bench-201-v1_1-096897.pth"


def get_top_arch(args):  
    api = API(api_file, verbose=False)

    heap: list[tuple[float, int, str]] = []
    missing_dataset = 0
    missing_metric = 0

    for arch_index in api.evaluated_indexes:
        try:
            info = api.query_meta_info_by_index(arch_index, hp=args.hp)
        except Exception:
            continue

        if args.dataset not in info.get_dataset_names():
            missing_dataset += 1
            continue

        try:
            metrics = info.get_metrics(args.dataset, args.setname, iepoch=None, is_random=args.is_random)
        except Exception:
            missing_metric += 1
            continue

        acc = metrics.get('accuracy', None)
        if acc is None:
            missing_metric += 1
            continue

        acc_f = float(acc)
        arch_str = api[arch_index]

        if len(heap) < args.k:
            heapq.heappush(heap, (acc_f, arch_index, arch_str))
        else:
            if acc_f > heap[0][0]:
                heapq.heapreplace(heap, (acc_f, arch_index, arch_str))

    topk = sorted(heap, key=lambda x: x[0], reverse=True)
    if missing_dataset > 0:
        print(f'Note: skipped {missing_dataset} architectures missing dataset={args.dataset!r} for hp={args.hp!r}.')
    if missing_metric > 0:
        print(f'Note: skipped {missing_metric} architectures missing metric set={args.setname!r}.')
    print(f'Dataset: {args.dataset} | set: {args.setname} | hp: {args.hp} | top-k: {args.k}')
    for rank, (acc, arch_index, arch_str) in enumerate(topk, start=1):
        print(f'{rank:2d}. acc={acc:.4f}% | index={arch_index:5d} | arch={arch_str}')

