#!/usr/bin/env python3
"""Prebuild RECAP's ANN candidate cache for one supported large dataset."""
import argparse
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import EgoCluster
from utils import Dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=("tsocial", "dgraphfin"))
    parser.add_argument("--dims", type=int, default=32)
    parser.add_argument("--num-hops", type=int, default=4)
    parser.add_argument("--knn-k", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-dir", default="./knn_cache")
    parser.add_argument("--nlist", type=int, default=4096)
    parser.add_argument("--nprobe", type=int, default=16)
    parser.add_argument("--pq-m", type=int, default=16)
    parser.add_argument("--train-size", type=int, default=262144)
    parser.add_argument("--query-batch-size", type=int, default=4096)
    parser.add_argument("--add-batch-size", type=int, default=262144)
    parser.add_argument("--rerank-factor", type=int, default=32)
    parser.add_argument("--max-rerank-candidates", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    started = time.time()
    dataset = Dataset(args.dims, args.dataset)
    dataset.propagated(args.num_hops, args.device)
    hops = dataset.graph.x_list
    base = hops[0]
    initial_residual = torch.hstack(
        [hop - base for hop in hops[1:]]
    )
    cache_key = (
        "dataset",
        args.dataset,
        str(dataset.graph.feature_alignment_version),
        str(dataset.graph.adjacency_version),
        args.dims,
        args.num_hops,
    )
    selector = EgoCluster(
        initial_residual.shape[1],
        knn_k=args.knn_k,
        knn_cache_enabled=True,
        knn_cache_dir=args.cache_dir,
        ann_nlist=args.nlist,
        ann_nprobe=args.nprobe,
        ann_pq_m=args.pq_m,
        ann_train_size=args.train_size,
        ann_query_batch_size=args.query_batch_size,
        ann_add_batch_size=args.add_batch_size,
        ann_rerank_factor=args.rerank_factor,
        ann_max_rerank_candidates=args.max_rerank_candidates,
        ann_seed=args.seed,
    ).to(args.device)
    candidates = selector._get_knn_candidates(
        initial_residual, cache_key=cache_key
    )
    row_ids = torch.arange(
        candidates.shape[0], device=candidates.device
    ).unsqueeze(1)
    if torch.any(candidates == row_ids):
        raise RuntimeError("ANN cache contains self-neighbors")
    print(
        f"PASS dataset={args.dataset} candidates={tuple(candidates.shape)} "
        f"elapsed={time.time() - started:.2f}s cache_dir={args.cache_dir}"
    )


if __name__ == "__main__":
    main()
