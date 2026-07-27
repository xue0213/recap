"""Three-seed T-Finance -> T-Social shared-schema RECAP transfer."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import recap  # noqa: E402
from rebuttal.large_target_inference.ann import (  # noqa: E402
    build_faiss_candidates,
    validate_candidates,
)
from rebuttal.large_target_inference.protocol import ANN_CONFIG  # noqa: E402
from rebuttal.large_target_inference.common import (  # noqa: E402
    atomic_json,
    atomic_npy,
    sha256_file,
    utc_now,
)
from rebuttal.large_target_inference.data import (  # noqa: E402
    initial_residual,
    load_and_propagate,
)
from rebuttal.large_target_optimization.train import (  # noqa: E402
    SEEDS,
    _ann_kwargs,
    _sample_graph,
    _score_and_freeze,
    locked_model_config,
    prepare_training_candidates,
    train_seed,
)


def run(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset_root).resolve()
    config = locked_model_config(output_root / "knn_cache")

    # Source training is label-free and uses the accepted T-Finance feature
    # alignment, which the shared-schema preparation reproduces.
    source = load_and_propagate(
        dataset_root=dataset_root,
        name="tfinance",
        dims=config.dims,
        num_hops=config.num_hops,
        device=args.device,
    )
    source_indices = np.arange(source.node_count, dtype=np.int64)
    source_indices_path = output_root / "source_training_indices.npy"
    atomic_npy(source_indices_path, source_indices)
    source_sample_sha256 = sha256_file(source_indices_path)
    source_graph = _sample_graph(source, source_indices, args.device)
    training_candidates_path = output_root / "source_training_candidates.npy"
    training_metadata = prepare_training_candidates(
        source_graph,
        training_candidates_path,
        args.device,
        Path(args.tfinance_exact_candidates).resolve(),
    )
    training_candidates = np.load(training_candidates_path, mmap_mode="r")
    training_candidate_sha256 = sha256_file(training_candidates_path)
    training_records = []
    for seed in SEEDS:
        model, record = train_seed(
            seed=seed,
            graph=source_graph,
            candidates=training_candidates,
            config=config,
            output_root=output_root / "source_training",
            epochs=args.epochs,
            batch_size=args.train_batch_size,
            device=args.device,
            sample_sha256=source_sample_sha256,
            candidate_sha256=training_candidate_sha256,
        )
        training_records.append(record)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    del source, source_graph, training_candidates
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # The target uses T-Finance PCA axes but target-only post-zscore statistics.
    target = load_and_propagate(
        dataset_root=dataset_root,
        name="tsocial",
        dims=config.dims,
        num_hops=config.num_hops,
        device=args.device,
        aligned_features_path=Path(args.tsocial_shared_features).resolve(),
    )
    inference_candidates_path = output_root / "tsocial_shared_candidates.npy"
    if inference_candidates_path.exists():
        inference_metadata = validate_candidates(
            inference_candidates_path, target.node_count, config.knn_k
        )
        inference_metadata["state"] = "reused"
    else:
        residual_initial = initial_residual(target.graph.x_list)
        inference_metadata = build_faiss_candidates(
            residual_initial,
            k=config.knn_k,
            output_path=inference_candidates_path,
            nlist=ANN_CONFIG["nlist"],
            nprobe=64,
            pq_m=ANN_CONFIG["pq_m"],
            train_size=ANN_CONFIG["train_size"],
            query_batch_size=ANN_CONFIG["query_batch_size"],
            add_batch_size=ANN_CONFIG["add_batch_size"],
            rerank_factor=64,
            max_rerank_candidates=512,
            seed=ANN_CONFIG["seed"],
        )
        inference_metadata["state"] = "new_label_blind_shared_schema_ann"
        del residual_initial
    inference_candidates = np.load(inference_candidates_path, mmap_mode="r")
    atomic_json(
        output_root / "manifest.json",
        {
            "format": "recap_tfinance_to_tsocial_shared_schema_manifest_v1",
            "source": "tfinance",
            "target": "tsocial",
            "source_training_nodes": len(source_indices),
            "source_training_metadata": training_metadata,
            "source_training_candidate_sha256": training_candidate_sha256,
            "target_feature_path": str(
                Path(args.tsocial_shared_features).resolve()
            ),
            "target_feature_sha256": sha256_file(
                Path(args.tsocial_shared_features).resolve()
            ),
            "target_candidate_metadata": inference_metadata,
            "target_candidate_sha256": sha256_file(inference_candidates_path),
            "seeds": list(SEEDS),
            "labels_accessed": False,
            "created_at": utc_now(),
        },
    )

    frozen = []
    for seed, training in zip(SEEDS, training_records, strict=True):
        payload = torch.load(
            training["checkpoint_path"], map_location="cpu", weights_only=False
        )
        model = recap(**config.to_dict())
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model = model.to(args.device).eval()
        record = _score_and_freeze(
            model=model,
            context=target,
            candidates=inference_candidates,
            run_dir=output_root / "target_runs" / f"seed{seed}",
            checkpoint_record=training,
            score_batch_size=args.score_batch_size,
        )
        frozen.append(record)
        del model, payload
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    atomic_json(
        output_root / "all_scores_frozen.json",
        {
            "format": "recap_shared_schema_all_scores_frozen_v1",
            "source": "tfinance",
            "target": "tsocial",
            "records": frozen,
            "labels_accessed_before_global_freeze": False,
            "frozen_at": utc_now(),
        },
    )

    # First target-label access.
    labels = np.asarray(
        np.load(dataset_root / "tsocial" / "labels.npy"), dtype=np.int64
    )
    mask = np.ones(target.node_count, dtype=np.bool_)
    rows = []
    for seed, record in zip(SEEDS, frozen, strict=True):
        for route, path in record["score_paths"].items():
            score = np.load(path, mmap_mode="r")
            rows.append(
                {
                    "source": "tfinance",
                    "target": "tsocial",
                    "seed": seed,
                    "route": route,
                    "auroc": float(roc_auc_score(labels[mask], score[mask])),
                    "auprc": float(
                        average_precision_score(labels[mask], score[mask])
                    ),
                    "score_sha256": record["score_sha256"][route],
                    "checkpoint_sha256": training_records[seed][
                        "checkpoint_sha256"
                    ],
                }
            )
    csv_path = output_root / "results.csv"
    fields = sorted({key for row in rows for key in row})
    temporary = csv_path.with_name(f".{csv_path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, csv_path)
    summary = []
    for route in ("full", "adhesion_only", "context_only"):
        current = [row for row in rows if row["route"] == route]
        auroc = np.asarray([row["auroc"] for row in current])
        auprc = np.asarray([row["auprc"] for row in current])
        summary.append(
            {
                "route": route,
                "auroc_mean": float(auroc.mean()),
                "auroc_std": float(auroc.std(ddof=0)),
                "auprc_mean": float(auprc.mean()),
                "auprc_std": float(auprc.std(ddof=0)),
            }
        )
    atomic_json(
        output_root / "summary.json",
        {
            "format": "recap_shared_schema_summary_v1",
            "summary": summary,
            "labels_loaded_after_global_freeze": True,
            "results_sha256": sha256_file(csv_path),
            "created_at": utc_now(),
        },
    )
    print(json.dumps(summary, indent=2))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--dataset-root", required=True)
    value.add_argument("--tfinance-exact-candidates", required=True)
    value.add_argument("--tsocial-shared-features", required=True)
    value.add_argument("--output-root", required=True)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--epochs", type=int, default=100)
    value.add_argument("--train-batch-size", type=int, default=256)
    value.add_argument("--score-batch-size", type=int, default=1024)
    return value


if __name__ == "__main__":
    run(parser().parse_args())
