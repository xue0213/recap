"""Label-late T-Finance single-source inference on three large targets."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ModelConfig  # noqa: E402
from model import recap  # noqa: E402
from rebuttal.large_target_inference.ann import validate_candidates  # noqa: E402
from rebuttal.large_target_inference.common import (  # noqa: E402
    atomic_json,
    atomic_npy,
    sha256_file,
    utc_now,
)
from rebuttal.large_target_inference.data import (  # noqa: E402
    canonical_paths,
    load_and_propagate,
)
from rebuttal.large_target_inference.protocol import (  # noqa: E402
    DATASETS,
    MODEL_LOCK,
)
from rebuttal.large_target_inference.scoring import (  # noqa: E402
    compute_score_components_chunked,
)


PROTOCOL_PATH = PROJECT_ROOT / "rebuttal" / "TFINANCE_SOURCE_TRANSFER_PROTOCOL.md"
TARGETS = ("tfinance", "dgraphfin", "tsocial")
SEEDS = (0, 1, 2)
ROUTES = ("full", "adhesion_only", "context_only")
CHECKPOINT_HASHES = {
    0: "9ab4717011ece6ef31934a3d54dd03511e79bd86d87e63c6c38e23658b0be9b0",
    1: "e71e9fde5cd11bc36e15376c1de3cb65bcbaef8feb55bb94d3e0307d2c96d1fd",
    2: "1cd24e8fde981fe90ece805e801f363902652f9299d4837fdb506a421c87ae33",
}
CANDIDATE_HASHES = {
    "tfinance": "f16f69f6dafc2d6c75d919d61916320798d320baef880884e45a4f7baf0400b8",
    "dgraphfin": "925d5c65249a3c279ce39b6920a0f0ef88d6f57d0ac3e6e5d2e6da60eda414d6",
    "tsocial": "4430c335dad8899435599c567897d65786a9eec73c0d805a46fe7db970e2f1c0",
}


def checkpoint_path(root: Path, seed: int) -> Path:
    return root / "runs" / f"seed{seed}" / "checkpoints" / "final.pt"


def score_paths(output_root: Path, target: str, seed: int) -> dict[str, Path]:
    base = output_root / "runs" / target / f"seed{seed}"
    return {route: base / f"scores_{route}.npy" for route in ROUTES}


def validate_checkpoint(root: Path, seed: int) -> tuple[dict, ModelConfig, Path]:
    path = checkpoint_path(root, seed)
    observed = sha256_file(path)
    if observed != CHECKPOINT_HASHES[seed]:
        raise ValueError(f"seed {seed} checkpoint hash mismatch: {observed}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("format") != "recap_large_target_adapt_checkpoint_v1"
        or int(payload.get("seed", -1)) != seed
        or int(payload.get("epoch", -1)) != 100
        or payload.get("labels_accessed") is not False
    ):
        raise ValueError(f"seed {seed} is not the locked label-free checkpoint")
    config = ModelConfig.from_dict(payload["model_config"])
    mismatch = {
        key: (expected, getattr(config, key))
        for key, expected in MODEL_LOCK.items()
        if getattr(config, key) != expected
    }
    if mismatch:
        raise ValueError(f"seed {seed} model lock mismatch: {mismatch}")
    return payload, config, path


def validate_candidate(path: Path, target: str) -> dict:
    if sha256_file(path) != CANDIDATE_HASHES[target]:
        raise ValueError(f"{target} candidate hash mismatch")
    result = validate_candidates(
        path, DATASETS[target]["nodes"], MODEL_LOCK["knn_k"]
    )
    result["sha256"] = CANDIDATE_HASHES[target]
    return result


def _finite(values: np.ndarray) -> bool:
    return values.ndim == 1 and bool(np.all(np.isfinite(values)))


def score_target(args: argparse.Namespace) -> None:
    target = args.target
    if target not in TARGETS:
        raise KeyError(target)
    output_root = Path(args.output_root).resolve()
    checkpoint_root = Path(args.checkpoint_root).resolve()
    candidate_path = Path(args.candidate_path).resolve()
    protocol_sha = sha256_file(PROTOCOL_PATH)
    candidate_meta = validate_candidate(candidate_path, target)
    candidates = np.load(candidate_path, mmap_mode="r")

    payload0, config, _ = validate_checkpoint(checkpoint_root, 0)
    del payload0
    context = load_and_propagate(
        dataset_root=Path(args.dataset_root).resolve(),
        name=target,
        dims=config.dims,
        num_hops=config.num_hops,
        device=args.device,
    )
    records = []
    for seed in SEEDS:
        paths = score_paths(output_root, target, seed)
        frozen_path = paths["full"].parent / "frozen.json"
        if frozen_path.exists() and all(path.exists() for path in paths.values()):
            frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
            if (
                frozen.get("protocol_sha256") == protocol_sha
                and frozen.get("candidate_sha256") == CANDIDATE_HASHES[target]
                and frozen.get("checkpoint_sha256") == CHECKPOINT_HASHES[seed]
                and all(
                    frozen["score_sha256"][route] == sha256_file(paths[route])
                    for route in ROUTES
                )
            ):
                records.append(frozen)
                print(f"REUSE target={target} seed={seed}", flush=True)
                continue
        payload, seed_config, path = validate_checkpoint(checkpoint_root, seed)
        if seed_config.to_dict() != config.to_dict():
            raise ValueError("T-Finance checkpoint model configurations differ")
        started = time.perf_counter()
        model = recap(**seed_config.to_dict()).to(args.device).eval()
        model.load_state_dict(payload["model_state_dict"], strict=True)
        with torch.inference_mode():
            model(context.graph)
            components = compute_score_components_chunked(
                residual=model._view_embeds[0],
                cluster=model.ego_clusters[0],
                candidates=candidates,
                score_batch_size=args.score_batch_size,
            )
        values = {
            "full": np.asarray(components["total"], dtype=np.float32),
            "adhesion_only": np.asarray(
                components["adhesion"], dtype=np.float32
            ),
            "context_only": np.asarray(
                components["context"], dtype=np.float32
            ),
        }
        if any(
            not _finite(values[route])
            or values[route].shape != (context.node_count,)
            for route in ROUTES
        ):
            raise FloatingPointError(f"invalid {target} seed {seed} score")
        for route in ROUTES:
            atomic_npy(paths[route], values[route])
        frozen = {
            "format": "recap_tfinance_source_transfer_frozen_v1",
            "source": "tfinance",
            "target": target,
            "seed": seed,
            "nodes": context.node_count,
            "checkpoint_path": str(path.resolve()),
            "checkpoint_sha256": CHECKPOINT_HASHES[seed],
            "candidate_path": str(candidate_path),
            "candidate_sha256": CANDIDATE_HASHES[target],
            "protocol_sha256": protocol_sha,
            "score_paths": {
                route: str(paths[route].resolve()) for route in ROUTES
            },
            "score_sha256": {
                route: sha256_file(paths[route]) for route in ROUTES
            },
            "score_seconds": time.perf_counter() - started,
            "labels_accessed": False,
            "frozen_at": utc_now(),
        }
        atomic_json(frozen_path, frozen)
        records.append(frozen)
        print(
            f"FROZEN target={target} seed={seed} "
            f"seconds={frozen['score_seconds']:.3f}",
            flush=True,
        )
        del payload, model, components, values
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    complete = {
        "format": "recap_tfinance_source_transfer_target_complete_v1",
        "source": "tfinance",
        "target": target,
        "protocol_sha256": protocol_sha,
        "candidate": candidate_meta,
        "records": records,
        "score_vectors": len(records) * len(ROUTES),
        "labels_accessed": False,
        "completed_at": utc_now(),
    }
    atomic_json(output_root / "targets" / target / "complete.json", complete)
    print(f"TARGET_COMPLETE target={target} scores={len(records)*3}", flush=True)


def _verify_global_freeze(output_root: Path) -> list[dict]:
    protocol_sha = sha256_file(PROTOCOL_PATH)
    records = []
    for target in TARGETS:
        complete_path = output_root / "targets" / target / "complete.json"
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if (
            complete.get("protocol_sha256") != protocol_sha
            or complete.get("labels_accessed") is not False
            or int(complete.get("score_vectors", -1)) != 9
        ):
            raise ValueError(f"{target} is not globally frozen")
        for seed in SEEDS:
            paths = score_paths(output_root, target, seed)
            frozen = json.loads(
                (paths["full"].parent / "frozen.json").read_text(
                    encoding="utf-8"
                )
            )
            if frozen.get("labels_accessed") is not False:
                raise AssertionError("labels accessed before global freeze")
            for route in ROUTES:
                if frozen["score_sha256"][route] != sha256_file(paths[route]):
                    raise ValueError(f"{target}/{seed}/{route} hash mismatch")
                records.append(
                    {
                        "target": target,
                        "seed": seed,
                        "route": route,
                        "path": str(paths[route].resolve()),
                        "sha256": frozen["score_sha256"][route],
                    }
                )
    if len(records) != 27:
        raise AssertionError(f"expected 27 scores, observed {len(records)}")
    return records


def evaluate(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    records = _verify_global_freeze(output_root)
    global_freeze = {
        "format": "recap_tfinance_source_transfer_global_freeze_v1",
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "scores": records,
        "score_vectors": len(records),
        "labels_accessed": False,
        "frozen_at": utc_now(),
    }
    atomic_json(output_root / "global_freeze.json", global_freeze)

    rows = []
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
                scores = np.load(path, mmap_mode="r")
                rows.append(
                    {
                        "target": target,
                        "seed": seed,
                        "route": route,
                        "evaluation_nodes": int(mask.sum()),
                        "anomalies": int((labels[mask] == 1).sum()),
                        "auroc": float(
                            roc_auc_score(labels[mask], scores[mask])
                        ),
                        "auprc": float(
                            average_precision_score(labels[mask], scores[mask])
                        ),
                        "score_sha256": sha256_file(path),
                    }
                )
    summary = []
    for target in TARGETS:
        for route in ROUTES:
            selected = [
                row
                for row in rows
                if row["target"] == target and row["route"] == route
            ]
            auroc = np.asarray([row["auroc"] for row in selected])
            auprc = np.asarray([row["auprc"] for row in selected])
            summary.append(
                {
                    "target": target,
                    "route": route,
                    "auroc_mean": float(auroc.mean()),
                    "auroc_std": float(auroc.std(ddof=0)),
                    "auprc_mean": float(auprc.mean()),
                    "auprc_std": float(auprc.std(ddof=0)),
                }
            )
    result = {
        "format": "recap_tfinance_source_transfer_results_v1",
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "global_freeze_sha256": sha256_file(output_root / "global_freeze.json"),
        "metric_rows": rows,
        "summary": summary,
        "labels_accessed_after_global_freeze": True,
        "evaluated_at": utc_now(),
    }
    atomic_json(output_root / "results.json", result)
    print(json.dumps(summary, indent=2), flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    score = sub.add_parser("score")
    score.add_argument("--target", choices=TARGETS, required=True)
    score.add_argument("--dataset-root", required=True)
    score.add_argument("--checkpoint-root", required=True)
    score.add_argument("--candidate-path", required=True)
    score.add_argument("--output-root", required=True)
    score.add_argument("--device", default="cuda:0")
    score.add_argument("--score-batch-size", type=int, default=1024)
    score.set_defaults(function=score_target)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--dataset-root", required=True)
    evaluation.add_argument("--output-root", required=True)
    evaluation.set_defaults(function=evaluate)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.function(arguments)
