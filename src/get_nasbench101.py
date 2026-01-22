import numpy as np
from nasbench.nasbench101 import api 




def get_top_arch(args):
    nasbench = api.NASBench('nasbench/nasbench101/nasbench_full.tfrecord')
    records = []  # save (accuracy, arch_hash)

    # Loop through the entire dataset
    for i, unique_hash in enumerate(nasbench.hash_iterator()):
        # Get metrics with specified epochs
        fixed_stats, computed_stats = nasbench.get_metrics_from_hash(unique_hash)
        
        # Get average accuracy from all runs

        for run_stats in computed_stats[108]:
            acc = run_stats['final_validation_accuracy']
            records.append((acc, unique_hash))

    print(f"Collected {len(records)} results from {i + 1} architectures")

    # Sort by accuracy in descending order
    records.sort(key=lambda x: x[0], reverse=True)

    # Get top-k
    topk = records[:args.k]

    print("\n" + "=" * 80)
    print(f"TOP {args.k} ARCHITECTURES WITH HIGHEST ACCURACY")
    print("=" * 80)

    for idx, (acc, arch_hash) in enumerate(topk):
        print(f"\nRank {idx+1}: Validation Accuracy = {acc:.6f}")
        print(f"Hash: {arch_hash}")
        
        # Get detailed information of the architecture
        fixed_stats, computed_stats = nasbench.get_metrics_from_hash(arch_hash)
        model_spec = fixed_stats['module_adjacency']
        operations = fixed_stats['module_operations']
        
        print(f"Operations: {operations}")
        print(f"Adjacency Matrix:\n{model_spec}")
        print("-" * 80)

