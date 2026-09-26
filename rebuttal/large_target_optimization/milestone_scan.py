"""Label-late oracle diagnostic over saved target-adaptation milestones."""

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
from rebuttal.large_target_inference.common import (  # noqa: E402
    atomic_json,
    sha256_file,
    utc_now,
)
from rebuttal.large_target_inference.data import load_and_propagate  # noqa: E402
from rebuttal.large_target_inference.protocol import DATASETS  # noqa: E402
from rebuttal.large_target_optimization.train import (  # noqa: E402
    _score_and_freeze,
)


PROTOCOL_PATH = (
    PROJECT_ROOT
    / "rebuttal"
    / "LARGE_TARGET_MILESTONE_DIAGNOSTIC_PROTOCOL.md"
)
EPOCHS = (25, 50, 75, 100)
SEEDS = (0, 1, 2)
ROUTES = ("full", "adhesion_only", "context_only")


def _checkpoint_path(root: Path, seed: int, epoch: int) -> Path:
    base = root / "runs" / f"seed{seed}" / "checkpoints"
    return base / ("final.pt" if epoch == 100 else f"epoch_{epoch}.pt")


def _atomic_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> None:
    target = args.target
    if target not in DATASETS:
        raise KeyError(target)
    source_root = Path(args.checkpoint_root).resolve() / target
    output_root = Path(args.output_root).resolve() / target
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = json.load(open(source_root / "training_manifest.json"))
    if manifest.get("labels_accessed") is not False:
        raise AssertionError("Training manifest is not label-isolated")
    context = load_and_propagate(
        dataset_root=Path(args.dataset_root).resolve(),
        name=target,
        dims=32,
        num_hops=4,
        device=args.device,
    )
    candidates = np.load(Path(args.candidates).resolve(), mmap_mode="r")
    if candidates.shape != (context.node_count, 64):
        raise ValueError(f"Candidate shape mismatch: {candidates.shape}")

    records = []
    for epoch in EPOCHS:
        for seed in SEEDS:
            checkpoint = _checkpoint_path(source_root, seed, epoch)
            payload = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )
            if (
                payload.get("format")
                != "recap_large_target_adapt_checkpoint_v1"
                or int(payload.get("seed", -1)) != seed
                or int(payload.get("epoch", -1)) != epoch
                or payload.get("labels_accessed") is not False
                or payload.get("sample_sha256") != manifest["sample_sha256"]
                or payload.get("candidate_sha256")
                != manifest["candidate_sha256"]
            ):
                raise ValueError(f"Incompatible checkpoint: {checkpoint}")
            checkpoint_sha256 = sha256_file(checkpoint)
            model = recap(**payload["model_config"])
            model.load_state_dict(payload["model_state_dict"], strict=True)
            model = model.to(args.device).eval()
            frozen = _score_and_freeze(
                model=model,
                context=context,
                candidates=candidates,
                run_dir=(
                    output_root / f"epoch_{epoch}" / f"seed{seed}"
                ),
                checkpoint_record={
                    "checkpoint_sha256": checkpoint_sha256,
                },
                score_batch_size=args.score_batch_size,
            )
            if frozen["checkpoint_sha256"] != checkpoint_sha256:
                raise AssertionError("Resumed score/checkpoint mismatch")
            records.append(
                {
                    "epoch": epoch,
                    "seed": seed,
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": checkpoint_sha256,
                    "score_paths": frozen["score_paths"],
                    "score_sha256": frozen["score_sha256"],
                    "labels_accessed": False,
                }
            )
            del payload, model, frozen
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"FROZEN epoch={epoch} seed={seed}", flush=True)

    atomic_json(
        output_root / "all_scores_frozen.json",
        {
            "format": "recap_large_milestone_scores_frozen_v1",
            "target": target,
            "epochs": list(EPOCHS),
            "seeds": list(SEEDS),
            "records": records,
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "labels_accessed_before_global_freeze": False,
            "frozen_at": utc_now(),
        },
    )

    dataset_dir = Path(args.dataset_root).resolve() / target
    labels = np.asarray(np.load(dataset_dir / "labels.npy"), dtype=np.int64)
    mask_path = dataset_dir / "evaluation_mask.npy"
    mask = (
        np.asarray(np.load(mask_path), dtype=np.bool_)
        if mask_path.exists()
        else np.ones(context.node_count, dtype=np.bool_)
    )
    rows = []
    for record in records:
        for route in ROUTES:
            scores = np.load(record["score_paths"][route], mmap_mode="r")
            rows.append(
                {
                    "target": target,
                    "epoch": record["epoch"],
                    "seed": record["seed"],
                    "route": route,
                    "auroc": float(roc_auc_score(labels[mask], scores[mask])),
                    "auprc": float(
                        average_precision_score(labels[mask], scores[mask])
                    ),
                    "score_sha256": record["score_sha256"][route],
                    "checkpoint_sha256": record["checkpoint_sha256"],
                    "selection_status": "exploratory_oracle",
                }
            )
    _atomic_csv(output_root / "results.csv", rows)
    summary = []
    for epoch in EPOCHS:
        for route in ROUTES:
            current = [
                row
                for row in rows
                if row["epoch"] == epoch and row["route"] == route
            ]
            auroc = np.asarray([row["auroc"] for row in current])
            auprc = np.asarray([row["auprc"] for row in current])
            summary.append(
                {
                    "target": target,
                    "epoch": epoch,
                    "route": route,
                    "auroc_mean": float(auroc.mean()),
                    "auroc_std": float(auroc.std(ddof=0)),
                    "auprc_mean": float(auprc.mean()),
                    "auprc_std": float(auprc.std(ddof=0)),
                    "selection_status": "exploratory_oracle",
                }
            )
    _atomic_csv(output_root / "summary.csv", summary)
    print(json.dumps(summary, indent=2))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--target", required=True, choices=tuple(DATASETS))
    value.add_argument("--dataset-root", required=True)
    value.add_argument("--checkpoint-root", required=True)
    value.add_argument("--candidates", required=True)
    value.add_argument("--output-root", required=True)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--score-batch-size", type=int, default=1024)
    return value


if __name__ == "__main__":
    run(parser().parse_args())
