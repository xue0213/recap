"""Resume-safe, label-late scan of all accepted RECAP source checkpoints."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ModelConfig  # noqa: E402
from model import recap  # noqa: E402
from rebuttal.large_target_inference.common import (  # noqa: E402
    atomic_json,
    atomic_npy,
    sha256_file,
    utc_now,
)
from rebuttal.large_target_inference.data import (  # noqa: E402
    load_and_propagate,
    load_and_propagate_cpu_csr,
)
from rebuttal.large_target_inference.protocol import (  # noqa: E402
    DATASETS,
    MODEL_LOCK,
)
from rebuttal.large_target_inference.scoring import (  # noqa: E402
    compute_score_components_chunked,
)


PROTOCOL_PATH = (
    PROJECT_ROOT / "rebuttal" / "LARGE_TARGET_OPTIMIZATION_PROTOCOL.md"
)
ROUTES = ("full", "adhesion_only", "context_only")


def _atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def discover_checkpoints(checkpoint_root: Path) -> list[dict]:
    patterns = (
        "rebuttal/artifacts/phase1/runs/*/checkpoints/final.pt",
        (
            "rebuttal/artifacts/questions_ofo_addendum/"
            "runs/*/checkpoints/final.pt"
        ),
    )
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(checkpoint_root.glob(pattern))
    records: list[dict] = []
    for path in sorted(set(paths)):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        run_spec = dict(payload.get("run_spec", {}))
        if payload.get("format") != "recap_phase1_checkpoint_v1":
            raise ValueError(f"Unexpected checkpoint format: {path}")
        if int(payload.get("epoch", -1)) != 100:
            raise ValueError(f"Checkpoint is not epoch 100: {path}")
        sources = tuple(run_spec.get("source_graphs", ()))
        seed = int(run_spec.get("seed", -1))
        setting = str(run_spec.get("setting"))
        if seed not in (0, 1, 2) or not sources:
            raise ValueError(f"Invalid run specification: {path}")
        model_config = ModelConfig.from_dict(payload["model_config"])
        mismatches = {
            key: (expected, getattr(model_config, key))
            for key, expected in MODEL_LOCK.items()
            if getattr(model_config, key) != expected
        }
        if mismatches:
            raise ValueError(f"Model lock mismatch in {path}: {mismatches}")
        records.append(
            {
                "checkpoint_id": path.parent.parent.name,
                "path": path,
                "sha256": sha256_file(path),
                "setting": setting,
                "seed": seed,
                "sources": sources,
                "source_family": (
                    f"OFA-{setting}"
                    if setting in {"A", "B", "C"}
                    else f"OFO-{sources[0]}"
                ),
                "model_config": model_config,
            }
        )
        del payload
    expected = {
        (family, seed)
        for family in (
            "OFA-A",
            "OFA-B",
            "OFA-C",
            "OFO-ACM",
            "OFO-Amazon",
            "OFO-BlogCatalog",
            "OFO-citeseer",
            "OFO-cora",
            "OFO-Facebook",
            "OFO-Flickr",
            "OFO-pubmed",
            "OFO-Reddit",
            "OFO-weibo",
            "OFO-YelpChi",
            "OFO-questions",
        )
        for seed in (0, 1, 2)
    }
    observed = {(item["source_family"], item["seed"]) for item in records}
    if observed != expected:
        raise ValueError(
            f"Expected 45 accepted checkpoints; missing={expected-observed}, "
            f"extra={observed-expected}"
        )
    return records


def _score_paths(run_dir: Path) -> dict[str, Path]:
    return {route: run_dir / f"scores_{route}.npy" for route in ROUTES}


def _score_one(
    checkpoint: dict,
    context,
    candidates: np.ndarray,
    run_dir: Path,
    device: str,
    score_batch_size: int,
) -> dict:
    metadata_path = run_dir / "frozen.json"
    paths = _score_paths(run_dir)
    if metadata_path.exists() and all(path.exists() for path in paths.values()):
        with metadata_path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        if (
            record["checkpoint_sha256"] == checkpoint["sha256"]
            and record["protocol_sha256"] == sha256_file(PROTOCOL_PATH)
            and all(
                record["score_sha256"][route] == sha256_file(path)
                for route, path in paths.items()
            )
        ):
            record["resume_state"] = "reused"
            return record

    started = time.perf_counter()
    payload = torch.load(
        checkpoint["path"], map_location="cpu", weights_only=False
    )
    model = recap(**checkpoint["model_config"].to_dict())
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model = model.to(device).eval()
    with torch.inference_mode():
        model(context.graph)
        residual = model._view_embeds[0]
        components = compute_score_components_chunked(
            residual=residual,
            cluster=model.ego_clusters[0],
            candidates=candidates,
            score_batch_size=score_batch_size,
        )
    route_scores = {
        "full": np.asarray(components["total"], dtype=np.float32),
        "adhesion_only": np.asarray(
            components["adhesion"], dtype=np.float32
        ),
        "context_only": np.asarray(components["context"], dtype=np.float32),
    }
    if any(
        values.shape != (context.node_count,)
        or not np.all(np.isfinite(values))
        for values in route_scores.values()
    ):
        raise FloatingPointError(
            f"Invalid score route for {checkpoint['checkpoint_id']}"
        )
    for route, values in route_scores.items():
        atomic_npy(paths[route], values)
    record = {
        "format": "recap_large_source_scan_frozen_v1",
        "checkpoint_id": checkpoint["checkpoint_id"],
        "checkpoint_sha256": checkpoint["sha256"],
        "source_family": checkpoint["source_family"],
        "sources": list(checkpoint["sources"]),
        "setting": checkpoint["setting"],
        "seed": checkpoint["seed"],
        "target": context.name,
        "nodes": context.node_count,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "score_paths": {key: str(value.resolve()) for key, value in paths.items()},
        "score_sha256": {
            key: sha256_file(value) for key, value in paths.items()
        },
        "score_seconds": time.perf_counter() - started,
        "frozen_at": utc_now(),
        "labels_accessed": False,
        "resume_state": "new",
    }
    atomic_json(metadata_path, record)
    del payload, model, residual, components, route_scores
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


def _ensemble_specs(target: str) -> dict[str, set[str]]:
    domain = (
        {"OFO-Amazon", "OFO-YelpChi", "OFA-A", "OFA-B"}
        if target in {"tfinance", "dgraphfin"}
        else {
            "OFO-Facebook",
            "OFO-Flickr",
            "OFO-Reddit",
            "OFO-weibo",
            "OFO-BlogCatalog",
            "OFA-A",
        }
    )
    return {
        "ENS-AllSources": {
            "OFA-A",
            "OFA-B",
            "OFA-C",
            "OFO-ACM",
            "OFO-Amazon",
            "OFO-BlogCatalog",
            "OFO-citeseer",
            "OFO-cora",
            "OFO-Facebook",
            "OFO-Flickr",
            "OFO-pubmed",
            "OFO-Reddit",
            "OFO-weibo",
            "OFO-YelpChi",
            "OFO-questions",
        },
        "ENS-OFA-ABC": {"OFA-A", "OFA-B", "OFA-C"},
        "ENS-Domain": domain,
    }


def _atomic_mean_npy(
    output_path: Path, input_paths: list[Path], length: int
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.name}.tmp.{os.getpid()}.npy"
    )
    output = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(length,)
    )
    arrays = [np.load(path, mmap_mode="r") for path in input_paths]
    for start in range(0, length, 500_000):
        stop = min(start + 500_000, length)
        total = np.zeros(stop - start, dtype=np.float64)
        for array in arrays:
            total += array[start:stop]
        output[start:stop] = (total / len(arrays)).astype(np.float32)
    output.flush()
    del output, arrays
    os.replace(temporary, output_path)


def _freeze_ensembles(
    *, target: str, records: list[dict], target_root: Path, node_count: int
) -> list[dict]:
    by_seed_family = {
        (int(item["seed"]), str(item["source_family"])): item
        for item in records
    }
    output: list[dict] = []
    for ensemble_name, families in _ensemble_specs(target).items():
        for seed in (0, 1, 2):
            missing = [
                family
                for family in sorted(families)
                if (seed, family) not in by_seed_family
            ]
            if missing:
                raise ValueError(
                    f"Missing ensemble members {ensemble_name}/seed{seed}: "
                    f"{missing}"
                )
            members = [by_seed_family[(seed, name)] for name in sorted(families)]
            run_dir = (
                target_root
                / "ensembles"
                / ensemble_name.replace("ENS-", "").lower()
                / f"seed{seed}"
            )
            paths = _score_paths(run_dir)
            member_hashes = {
                route: [item["score_sha256"][route] for item in members]
                for route in ROUTES
            }
            constituent_sha256 = hashlib.sha256(
                json.dumps(
                    member_hashes, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()
            frozen_path = run_dir / "frozen.json"
            reusable = False
            if frozen_path.exists() and all(path.exists() for path in paths.values()):
                with frozen_path.open("r", encoding="utf-8") as handle:
                    existing = json.load(handle)
                reusable = (
                    existing.get("protocol_sha256") == sha256_file(PROTOCOL_PATH)
                    and existing.get("constituent_sha256") == constituent_sha256
                    and all(
                        existing["score_sha256"][route] == sha256_file(path)
                        for route, path in paths.items()
                    )
                )
            if reusable:
                record = existing
                record["resume_state"] = "reused"
            else:
                for route in ROUTES:
                    _atomic_mean_npy(
                        paths[route],
                        [Path(item["score_paths"][route]) for item in members],
                        node_count,
                    )
                record = {
                    "format": "recap_large_source_scan_ensemble_frozen_v1",
                    "checkpoint_id": f"{ensemble_name.lower()}__seed{seed}",
                    "checkpoint_sha256": constituent_sha256,
                    "source_family": ensemble_name,
                    "sources": sorted(families),
                    "setting": "predeclared_seed_aligned_ensemble",
                    "seed": seed,
                    "target": target,
                    "nodes": node_count,
                    "protocol_sha256": sha256_file(PROTOCOL_PATH),
                    "constituent_sha256": constituent_sha256,
                    "score_paths": {
                        key: str(value.resolve()) for key, value in paths.items()
                    },
                    "score_sha256": {
                        key: sha256_file(value) for key, value in paths.items()
                    },
                    "frozen_at": utc_now(),
                    "labels_accessed": False,
                    "resume_state": "new",
                }
                atomic_json(frozen_path, record)
            output.append(record)
    return output


def _metric(labels: np.ndarray, mask: np.ndarray, scores: np.ndarray) -> tuple:
    return (
        float(roc_auc_score(labels[mask], scores[mask])),
        float(average_precision_score(labels[mask], scores[mask])),
    )


def _aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["target"], row["source_family"], row["route"])].append(
            row
        )
    output = []
    for (target, family, route), values in sorted(groups.items()):
        if len(values) != 3 or {int(x["seed"]) for x in values} != {0, 1, 2}:
            raise ValueError(f"Incomplete seed group: {target}/{family}/{route}")
        auroc = np.asarray([x["auroc"] for x in values], dtype=np.float64)
        auprc = np.asarray([x["auprc"] for x in values], dtype=np.float64)
        output.append(
            {
                "target": target,
                "source_family": family,
                "route": route,
                "auroc_mean": float(auroc.mean()),
                "auroc_std": float(auroc.std(ddof=0)),
                "auprc_mean": float(auprc.mean()),
                "auprc_std": float(auprc.std(ddof=0)),
                "seeds": 3,
                "selection_status": "exploratory_oracle_if_ranked_by_target_metric",
            }
        )
    return output


def run_scan(args: argparse.Namespace) -> None:
    if args.target not in DATASETS:
        raise KeyError(args.target)
    output_root = Path(args.output_root).resolve()
    target_root = output_root / args.target
    target_root.mkdir(parents=True, exist_ok=True)
    checkpoints = discover_checkpoints(Path(args.checkpoint_root).resolve())
    atomic_json(
        target_root / "checkpoint_manifest.json",
        {
            "format": "recap_large_source_scan_manifest_v1",
            "target": args.target,
            "protocol_path": str(PROTOCOL_PATH.resolve()),
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "checkpoint_count": len(checkpoints),
            "checkpoints": [
                {
                    key: (
                        str(value.resolve())
                        if isinstance(value, Path)
                        else value.to_dict()
                        if isinstance(value, ModelConfig)
                        else list(value)
                        if isinstance(value, tuple)
                        else value
                    )
                    for key, value in item.items()
                }
                for item in checkpoints
            ],
            "labels_accessed": False,
            "created_at": utc_now(),
        },
    )

    loader_kwargs = {
        "dataset_root": Path(args.dataset_root).resolve(),
        "name": args.target,
        "dims": MODEL_LOCK["dims"],
        "num_hops": MODEL_LOCK["num_hops"],
    }
    context = (
        load_and_propagate_cpu_csr(**loader_kwargs)
        if args.device == "cpu"
        else load_and_propagate(**loader_kwargs, device=args.device)
    )
    candidates = np.load(Path(args.candidates).resolve(), mmap_mode="r")
    if candidates.shape != (context.node_count, MODEL_LOCK["knn_k"]):
        raise ValueError(f"Candidate shape mismatch: {candidates.shape}")

    frozen = []
    for index, checkpoint in enumerate(checkpoints, start=1):
        record = _score_one(
            checkpoint,
            context,
            candidates,
            target_root / "runs" / checkpoint["checkpoint_id"],
            args.device,
            args.score_batch_size,
        )
        frozen.append(record)
        print(
            f"FROZEN {index:02d}/{len(checkpoints)} "
            f"{checkpoint['checkpoint_id']} "
            f"({record['resume_state']}, {record['score_seconds']:.2f}s)",
            flush=True,
        )
    ensembles = _freeze_ensembles(
        target=args.target,
        records=frozen,
        target_root=target_root,
        node_count=context.node_count,
    )

    freeze_manifest = {
        "format": "recap_large_source_scan_all_scores_frozen_v1",
        "target": args.target,
        "checkpoint_count": len(frozen),
        "ensemble_count": len(ensembles),
        "score_route_count": (len(frozen) + len(ensembles)) * len(ROUTES),
        "all_labels_accessed_before_freeze": False,
        "records": frozen,
        "ensembles": ensembles,
        "frozen_at": utc_now(),
    }
    atomic_json(target_root / "all_scores_frozen.json", freeze_manifest)

    # This is the first point in this process at which labels or masks load.
    dataset_dir = Path(args.dataset_root).resolve() / args.target
    labels = np.asarray(
        np.load(dataset_dir / "labels.npy", mmap_mode="r"),
        dtype=np.int64,
    ).reshape(-1)
    mask_path = dataset_dir / "evaluation_mask.npy"
    mask = (
        np.asarray(np.load(mask_path, mmap_mode="r"), dtype=np.bool_).reshape(-1)
        if mask_path.exists()
        else np.ones(context.node_count, dtype=np.bool_)
    )
    rows = []
    metric_records = [
        (checkpoint, record)
        for checkpoint, record in zip(checkpoints, frozen, strict=True)
    ] + [
        (
            {
                "checkpoint_id": record["checkpoint_id"],
                "sha256": record["checkpoint_sha256"],
                "source_family": record["source_family"],
                "sources": tuple(record["sources"]),
                "setting": record["setting"],
                "seed": record["seed"],
            },
            record,
        )
        for record in ensembles
    ]
    for checkpoint, record in metric_records:
        for route in ROUTES:
            scores = np.load(record["score_paths"][route], mmap_mode="r")
            auroc, auprc = _metric(labels, mask, scores)
            rows.append(
                {
                    "target": args.target,
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "checkpoint_sha256": checkpoint["sha256"],
                    "source_family": checkpoint["source_family"],
                    "sources": "|".join(checkpoint["sources"]),
                    "setting": checkpoint["setting"],
                    "seed": checkpoint["seed"],
                    "route": route,
                    "auroc": auroc,
                    "auprc": auprc,
                    "evaluation_nodes": int(mask.sum()),
                    "anomalies": int(labels[mask].sum()),
                    "score_sha256": record["score_sha256"][route],
                    "metric_status": (
                        "predeclared_label_blind_ensemble"
                        if checkpoint["source_family"].startswith("ENS-")
                        else "exploratory_source_sensitivity"
                    ),
                }
            )
    aggregate = _aggregate(rows)
    _atomic_csv(target_root / "per_checkpoint_results.csv", rows)
    _atomic_csv(target_root / "aggregate_results.csv", aggregate)
    atomic_json(
        target_root / "label_unlock.json",
        {
            "format": "recap_large_source_scan_label_unlock_v1",
            "target": args.target,
            "all_scores_frozen_sha256": sha256_file(
                target_root / "all_scores_frozen.json"
            ),
            "score_routes_frozen": (
                len(frozen) + len(ensembles)
            ) * len(ROUTES),
            "labels_loaded_after_global_freeze": True,
            "unlocked_at": utc_now(),
        },
    )
    top = sorted(
        (x for x in aggregate if x["route"] == "full"),
        key=lambda x: x["auroc_mean"],
        reverse=True,
    )[:5]
    print(json.dumps({"target": args.target, "top_full": top}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=tuple(DATASETS))
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-batch-size", type=int, default=1024)
    return parser


if __name__ == "__main__":
    run_scan(build_parser().parse_args())
