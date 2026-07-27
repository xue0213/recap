"""Independent hash and metric audit for T-Finance source transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from rebuttal.large_target_inference.common import atomic_json, sha256_file, utc_now
from rebuttal.large_target_inference.data import canonical_paths
from rebuttal.large_target_inference.protocol import MODEL_LOCK
from rebuttal.large_target_optimization.tfinance_transfer import (
    PROTOCOL_PATH,
    ROUTES,
    SEEDS,
    TARGETS,
    score_paths,
)


def run(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    freeze_path = output_root / "global_freeze.json"
    results_path = output_root / "results.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if (
        freeze.get("protocol_sha256") != sha256_file(PROTOCOL_PATH)
        or freeze.get("labels_accessed") is not False
        or int(freeze.get("score_vectors", -1)) != 27
        or results.get("global_freeze_sha256") != sha256_file(freeze_path)
    ):
        raise ValueError("global freeze boundary is invalid")

    frozen_hashes = {
        (item["target"], int(item["seed"]), item["route"]): item["sha256"]
        for item in freeze["scores"]
    }
    if len(frozen_hashes) != 27:
        raise ValueError("global freeze contains duplicate/missing scores")
    stored = {
        (row["target"], int(row["seed"]), row["route"]): row
        for row in results["metric_rows"]
    }
    if len(stored) != 27:
        raise ValueError("results contain duplicate/missing metric rows")

    maximum_difference = 0.0
    recomputed = []
    score_hashes = 0
    for target in TARGETS:
        paths = canonical_paths(dataset_root, target, MODEL_LOCK["dims"])
        labels = np.asarray(
            np.load(paths["labels"], mmap_mode="r"), dtype=np.int64
        ).reshape(-1)
        mask = (
            np.asarray(
                np.load(paths["evaluation_mask"], mmap_mode="r"),
                dtype=np.bool_,
            ).reshape(-1)
            if paths["evaluation_mask"].exists()
            else np.ones(len(labels), dtype=np.bool_)
        )
        for seed in SEEDS:
            for route, path in score_paths(output_root, target, seed).items():
                key = (target, seed, route)
                observed_hash = sha256_file(path)
                if observed_hash != frozen_hashes[key]:
                    raise ValueError(f"score hash mismatch: {key}")
                scores = np.load(path, mmap_mode="r")
                if scores.shape != labels.shape or not np.all(np.isfinite(scores)):
                    raise ValueError(f"invalid score vector: {key}")
                auroc = float(roc_auc_score(labels[mask], scores[mask]))
                auprc = float(
                    average_precision_score(labels[mask], scores[mask])
                )
                row = stored[key]
                difference = max(
                    abs(auroc - float(row["auroc"])),
                    abs(auprc - float(row["auprc"])),
                )
                maximum_difference = max(maximum_difference, difference)
                recomputed.append(
                    {
                        "target": target,
                        "seed": seed,
                        "route": route,
                        "auroc": auroc,
                        "auprc": auprc,
                        "maximum_difference": difference,
                    }
                )
                score_hashes += 1
    audit = {
        "format": "recap_tfinance_source_transfer_audit_v1",
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "global_freeze_sha256": sha256_file(freeze_path),
        "results_sha256": sha256_file(results_path),
        "score_hashes": score_hashes,
        "metric_rows": len(recomputed),
        "maximum_metric_difference": maximum_difference,
        "passed": score_hashes == 27 and maximum_difference == 0.0,
        "recomputed": recomputed,
        "created_at": utc_now(),
    }
    atomic_json(output_root / "independent_audit.json", audit)
    print(json.dumps(audit, indent=2))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    run(parser.parse_args())
