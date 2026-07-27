"""Shared-schema feature alignment for T-Finance -> T-Social transfer."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rebuttal.large_target_inference.common import (  # noqa: E402
    atomic_json,
    sha256_file,
    utc_now,
)
from rebuttal.large_target_inference.data import (  # noqa: E402
    FEATURE_ALIGNMENT_VERSION,
)


VERSION = "tfinance_pca_axes_target_zscore_v1"
EPS = 1e-8


def fit_source_axes(source_features: np.ndarray) -> dict:
    node_count = len(source_features)
    sample_count = min(node_count, 200_000)
    indices = np.linspace(
        0, node_count - 1, num=sample_count, dtype=np.int64
    )
    sample = np.asarray(source_features[indices], dtype=np.float32)
    median = np.median(sample, axis=0, keepdims=True)
    q75 = np.percentile(sample, 75, axis=0, keepdims=True)
    q25 = np.percentile(sample, 25, axis=0, keepdims=True)
    iqr = q75 - q25
    normalized = (sample - median) / (iqr + EPS)
    pca = PCA(n_components=normalized.shape[1], random_state=0)
    pca.fit(normalized)
    return {
        "median": median.astype(np.float32),
        "iqr": iqr.astype(np.float32),
        "components": pca.components_.astype(np.float32),
        "pca_mean": pca.mean_.astype(np.float32),
        "explained_variance": pca.explained_variance_.astype(np.float32),
        "sample_count": sample_count,
    }


def apply_axes_target_zscore(
    *,
    features: np.ndarray,
    params: dict,
    output_path: Path,
    dims: int = 32,
    chunk_size: int = 100_000,
) -> dict:
    node_count, raw_dims = features.shape
    if raw_dims != params["components"].shape[1]:
        raise ValueError(
            f"Shared raw schema mismatch: {raw_dims} versus "
            f"{params['components'].shape[1]}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.name}.tmp.{os.getpid()}.npy"
    )
    aligned = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(node_count, dims)
    )
    aligned[:] = 0
    coordinate_count = params["components"].shape[0]
    total = np.zeros(coordinate_count, dtype=np.float64)
    total_square = np.zeros(coordinate_count, dtype=np.float64)
    for start in range(0, node_count, chunk_size):
        stop = min(start + chunk_size, node_count)
        values = np.asarray(features[start:stop], dtype=np.float32)
        normalized = (values - params["median"]) / (params["iqr"] + EPS)
        projected = (
            (normalized - params["pca_mean"]) @ params["components"].T
        ).astype(np.float32, copy=False)
        aligned[start:stop, :coordinate_count] = projected
        total += projected.sum(axis=0, dtype=np.float64)
        total_square += np.square(projected, dtype=np.float64).sum(
            axis=0, dtype=np.float64
        )
    mean = total / node_count
    variance = np.maximum(total_square / node_count - mean * mean, 0)
    std = np.sqrt(variance)
    for start in range(0, node_count, chunk_size):
        stop = min(start + chunk_size, node_count)
        aligned[start:stop, :coordinate_count] = (
            aligned[start:stop, :coordinate_count] - mean
        ) / (std + EPS)
    aligned.flush()
    del aligned
    os.replace(temporary, output_path)
    return {
        "nodes": node_count,
        "raw_dims": raw_dims,
        "aligned_dims": dims,
        "coordinate_count": coordinate_count,
        "target_post_zscore_mean": mean.tolist(),
        "target_post_zscore_std": std.tolist(),
        "output_path": str(output_path.resolve()),
        "output_sha256": sha256_file(output_path),
    }


def _save_params(path: Path, params: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.npz")
    np.savez(
        temporary,
        median=params["median"],
        iqr=params["iqr"],
        components=params["components"],
        pca_mean=params["pca_mean"],
        explained_variance=params["explained_variance"],
        sample_count=np.asarray(params["sample_count"], dtype=np.int64),
        version=np.asarray(VERSION),
    )
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve()
    source_dir = dataset_root / "tfinance"
    target_dir = dataset_root / "tsocial"
    source_features_path = source_dir / "features.npy"
    target_features_path = target_dir / "features.npy"
    current_source_aligned_path = (
        source_dir
        / f"features_aligned_32_{FEATURE_ALIGNMENT_VERSION}.npy"
    )
    output_path = output_root / f"tsocial_aligned_32_{VERSION}.npy"
    params_path = output_root / f"{VERSION}.npz"
    report_path = output_root / "shared_schema_preparation.json"
    if report_path.exists() and output_path.exists() and params_path.exists():
        with report_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        if (
            report["target"]["output_sha256"] == sha256_file(output_path)
            and report["params_sha256"] == sha256_file(params_path)
        ):
            print(json.dumps(report, indent=2))
            return

    source_features = np.load(source_features_path, mmap_mode="r")
    target_features = np.load(target_features_path, mmap_mode="r")
    if source_features.shape[1] != 10 or target_features.shape[1] != 10:
        raise ValueError(
            f"T-Finance/T-Social raw schemas differ: "
            f"{source_features.shape}, {target_features.shape}"
        )
    params = fit_source_axes(source_features)
    _save_params(params_path, params)
    source_check_path = output_root / f"tfinance_check_32_{VERSION}.npy"
    source_record = apply_axes_target_zscore(
        features=source_features,
        params=params,
        output_path=source_check_path,
    )
    current = np.load(current_source_aligned_path, mmap_mode="r")
    check = np.load(source_check_path, mmap_mode="r")
    difference = np.abs(
        np.asarray(current[:, :10], dtype=np.float32)
        - np.asarray(check[:, :10], dtype=np.float32)
    )
    source_record["existing_alignment_max_abs_difference"] = float(
        difference.max()
    )
    source_record["existing_alignment_mean_abs_difference"] = float(
        difference.mean()
    )
    if source_record["existing_alignment_max_abs_difference"] > 1e-4:
        raise AssertionError(
            "Re-fitted T-Finance axes do not reproduce the accepted source "
            f"alignment: {source_record}"
        )
    target_record = apply_axes_target_zscore(
        features=target_features,
        params=params,
        output_path=output_path,
    )
    report = {
        "format": "recap_shared_schema_alignment_v1",
        "version": VERSION,
        "source": source_record,
        "target": target_record,
        "params_path": str(params_path.resolve()),
        "params_sha256": sha256_file(params_path),
        "source_features_sha256": sha256_file(source_features_path),
        "target_features_sha256": sha256_file(target_features_path),
        "labels_accessed": False,
        "created_at": utc_now(),
    }
    atomic_json(report_path, report)
    print(json.dumps(report, indent=2))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--dataset-root", required=True)
    value.add_argument("--output-root", required=True)
    return value


if __name__ == "__main__":
    run(parser().parse_args())

