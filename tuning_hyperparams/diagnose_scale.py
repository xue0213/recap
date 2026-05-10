from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST_DATASETS = [
    "Facebook",
    "cora",
    "citeseer",
    "ACM",
    "BlogCatalog",
    "weibo",
    "Reddit",
    "Amazon",
]


def add_project_root_to_path() -> None:
    import sys

    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def std(values: list[float]) -> float:
    if not values:
        return float("nan")
    mu = mean(values)
    return (sum((value - mu) ** 2 for value in values) / len(values)) ** 0.5


def safe_std(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.std(values))


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    return safe_corr(rankdata(x), rankdata(y))


def tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def metric_eval(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    try:
        return (
            float(roc_auc_score(labels, scores)),
            float(average_precision_score(labels, scores)),
        )
    except ValueError:
        return float("nan"), float("nan")


def top_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    if k <= 0 or a.size == 0 or b.size == 0:
        return float("nan")
    k = min(k, a.size, b.size)
    top_a = set(np.argpartition(-a, k - 1)[:k])
    top_b = set(np.argpartition(-b, k - 1)[:k])
    return len(top_a & top_b) / k


def weighted_between_var(values: np.ndarray, groups: np.ndarray, num_groups: int) -> float:
    if values.size == 0 or np.std(values) < 1e-12:
        return 0.0
    global_mean = float(np.mean(values))
    total = 0.0
    n = values.size
    for group_id in range(num_groups):
        mask = groups == group_id
        if not np.any(mask):
            continue
        weight = float(np.sum(mask)) / n
        group_mean = float(np.mean(values[mask]))
        total += weight * (group_mean - global_mean) ** 2
    return total


def discover_checkpoints(args) -> list[Path]:
    checkpoints: list[Path] = []
    if args.checkpoint:
        checkpoints.extend(Path(path).resolve() for path in args.checkpoint)

    if args.sensitivity_dir:
        root = args.sensitivity_dir.resolve()
        checkpoints.extend(root.glob("checkpoints/*/*/trial_*/model.pt"))

    if args.checkpoint_glob:
        for pattern in args.checkpoint_glob:
            checkpoints.extend(Path(path).resolve() for path in sorted(PROJECT_ROOT.glob(pattern)))

    checkpoints = sorted({path.resolve() for path in checkpoints if path.exists()})
    if not checkpoints:
        raise FileNotFoundError("No checkpoint files were found.")
    if args.max_checkpoints is not None:
        checkpoints = checkpoints[: args.max_checkpoints]
    return checkpoints


def checkpoint_metadata(path: Path) -> dict:
    parts = path.resolve().parts
    if "checkpoints" in parts:
        idx = parts.index("checkpoints")
        remaining = parts[idx + 1 :]
        if len(remaining) >= 4 and remaining[2].startswith("trial_"):
            return {
                "run_id": remaining[0],
                "model_name": remaining[1],
                "trial": remaining[2].replace("trial_", ""),
                "checkpoint": str(path),
            }
        if len(remaining) >= 3 and remaining[1].startswith("trial_"):
            return {
                "run_id": remaining[0],
                "model_name": remaining[0],
                "trial": remaining[1].replace("trial_", ""),
                "checkpoint": str(path),
            }

    trial = path.parent.name.replace("trial_", "") if path.parent.name.startswith("trial_") else ""
    run_id = path.parents[2].name if len(path.parents) >= 3 else path.stem
    return {
        "run_id": run_id,
        "model_name": path.parents[1].name if len(path.parents) >= 2 else "",
        "trial": trial,
        "checkpoint": str(path),
    }


def prepare_datasets(dims: int, num_hops: int, dataset_names: list[str], device: str):
    add_project_root_to_path()
    from utils import Dataset

    datasets = []
    for name in dataset_names:
        dataset = Dataset(dims, name)
        dataset.propagated(num_hops, device=device)
        datasets.append(dataset)
    return datasets


@torch.no_grad()
def diagnose_dataset(model, model_config, dataset, dataset_name: str, device: str) -> dict:
    graph = dataset.graph.to(device)
    labels = tensor_to_numpy(graph.ano_labels).astype(int)
    anomaly_count = int(labels.sum())
    anomaly_rate = float(labels.mean()) if labels.size else float("nan")

    model.eval()
    model(graph)
    E = model._view_embeds[0]
    cluster = model.ego_clusters[0]
    components = cluster.compute_score_components(
        E,
        E_init=model._view_embeds_init[0],
        cache_key=model._view_cache_keys[0],
    )

    H = cluster.cluster(E)
    hard_assign = tensor_to_numpy(H.argmax(dim=1)).astype(int)
    num_clusters = int(H.shape[1])
    hard_counts = np.bincount(hard_assign, minlength=num_clusters).astype(np.float64)
    hard_usage = hard_counts / max(float(hard_counts.sum()), 1.0)
    assigned_hard_usage = hard_usage[hard_assign]
    neglog_assigned_hard_usage = -np.log(assigned_hard_usage + float(cluster.eps))

    total = tensor_to_numpy(components["total"])
    adhesion = tensor_to_numpy(components["adhesion"])
    scale = tensor_to_numpy(components["scale"])
    beta = float(getattr(model_config, "beta", getattr(cluster, "beta", 1.0)))
    beta_scale = beta * scale

    total_auroc, total_auprc = metric_eval(labels, total)
    adhesion_auroc, adhesion_auprc = metric_eval(labels, adhesion)
    scale_auroc, scale_auprc = metric_eval(labels, scale)
    neg_scale_auroc, neg_scale_auprc = metric_eval(labels, -scale)

    adhesion_std = safe_std(adhesion)
    scale_std = safe_std(scale)
    total_std = safe_std(total)
    beta_scale_std = safe_std(beta_scale)

    scale_total_var = float(np.var(scale))
    scale_between_var = weighted_between_var(scale, hard_assign, num_clusters)
    scale_between_ratio = (
        scale_between_var / scale_total_var if scale_total_var > 1e-12 else float("nan")
    )

    k_at_anomaly_count = max(1, anomaly_count)
    k_at_5pct = max(1, int(math.ceil(0.05 * labels.size)))

    return {
        "dataset": dataset_name,
        "num_nodes": int(labels.size),
        "anomaly_count": anomaly_count,
        "anomaly_rate": anomaly_rate,
        "num_clusters": num_clusters,
        "knn_k": getattr(model_config, "knn_k", ""),
        "num_hops": getattr(model_config, "num_hops", ""),
        "tau_s": getattr(model_config, "tau_s", ""),
        "beta": beta,
        "lambda_H": getattr(model_config, "lambda_H", ""),
        "lambda_bal": getattr(model_config, "lambda_bal", ""),
        "lambda_E": getattr(model_config, "lambda_E", ""),
        "total_AUROC": total_auroc,
        "total_AUPRC": total_auprc,
        "adhesion_AUROC": adhesion_auroc,
        "adhesion_AUPRC": adhesion_auprc,
        "scale_AUROC": scale_auroc,
        "scale_AUPRC": scale_auprc,
        "neg_scale_AUROC": neg_scale_auroc,
        "neg_scale_AUPRC": neg_scale_auprc,
        "total_minus_adhesion_AUROC": total_auroc - adhesion_auroc,
        "total_minus_adhesion_AUPRC": total_auprc - adhesion_auprc,
        "scale_minus_adhesion_AUROC": scale_auroc - adhesion_auroc,
        "scale_minus_adhesion_AUPRC": scale_auprc - adhesion_auprc,
        "adhesion_mean": float(np.mean(adhesion)),
        "adhesion_std": adhesion_std,
        "adhesion_min": float(np.min(adhesion)),
        "adhesion_max": float(np.max(adhesion)),
        "scale_mean": float(np.mean(scale)),
        "scale_std": scale_std,
        "scale_min": float(np.min(scale)),
        "scale_max": float(np.max(scale)),
        "beta_scale_std": beta_scale_std,
        "total_mean": float(np.mean(total)),
        "total_std": total_std,
        "scale_std_over_adhesion_std": scale_std / adhesion_std if adhesion_std > 1e-12 else float("nan"),
        "beta_scale_std_over_adhesion_std": (
            beta_scale_std / adhesion_std if adhesion_std > 1e-12 else float("nan")
        ),
        "total_std_over_adhesion_std": total_std / adhesion_std if adhesion_std > 1e-12 else float("nan"),
        "pearson_total_adhesion": safe_corr(total, adhesion),
        "pearson_total_scale": safe_corr(total, scale),
        "pearson_adhesion_scale": safe_corr(adhesion, scale),
        "spearman_total_adhesion": spearman_corr(total, adhesion),
        "spearman_total_scale": spearman_corr(total, scale),
        "spearman_adhesion_scale": spearman_corr(adhesion, scale),
        "spearman_scale_neglog_hard_usage": spearman_corr(scale, neglog_assigned_hard_usage),
        "scale_between_hard_community_var_ratio": scale_between_ratio,
        "hard_usage_min": float(np.min(hard_usage)),
        "hard_usage_max": float(np.max(hard_usage)),
        "top_overlap_total_adhesion_at_anomaly_count": top_overlap(
            total, adhesion, k_at_anomaly_count
        ),
        "top_overlap_total_scale_at_anomaly_count": top_overlap(
            total, scale, k_at_anomaly_count
        ),
        "top_overlap_adhesion_scale_at_anomaly_count": top_overlap(
            adhesion, scale, k_at_anomaly_count
        ),
        "top_overlap_total_adhesion_at_5pct": top_overlap(total, adhesion, k_at_5pct),
        "top_overlap_total_scale_at_5pct": top_overlap(total, scale, k_at_5pct),
        "scale_metric_weak_flag": (
            max(scale_auroc, neg_scale_auroc) < 0.55
            if scale_auroc == scale_auroc and neg_scale_auroc == neg_scale_auroc
            else False
        ),
        "scale_opposite_direction_flag": (
            neg_scale_auroc > scale_auroc + 0.02
            if scale_auroc == scale_auroc and neg_scale_auroc == neg_scale_auroc
            else False
        ),
        "scale_small_variation_flag": (
            beta_scale_std / adhesion_std < 0.01 if adhesion_std > 1e-12 else False
        ),
        "scale_does_not_change_total_flag": spearman_corr(total, adhesion) > 0.999,
        "scale_between_community_dominated_flag": (
            scale_between_ratio > 0.8 if scale_between_ratio == scale_between_ratio else False
        ),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(rows: list[dict], group_keys: list[str]) -> list[dict]:
    numeric_keys = [
        key
        for key, value in rows[0].items()
        if key not in group_keys and isinstance(value, (int, float, bool, np.bool_))
    ]
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row[group_key] for group_key in group_keys)
        groups.setdefault(key, []).append(row)

    out = []
    for key, group_rows in sorted(groups.items(), key=lambda item: item[0]):
        summary = {group_key: key[idx] for idx, group_key in enumerate(group_keys)}
        summary["num_rows"] = len(group_rows)
        for numeric_key in numeric_keys:
            vals = [float(row[numeric_key]) for row in group_rows]
            vals = [value for value in vals if value == value]
            summary[f"{numeric_key}_mean"] = mean(vals)
            summary[f"{numeric_key}_std"] = std(vals)
        out.append(summary)
    return out


def print_brief_summary(rows: list[dict]) -> None:
    summary = summarize(rows, ["run_id"])
    print("\nScale diagnostic brief:")
    for row in summary:
        print(
            f"- {row['run_id']}: "
            f"total_auc={row.get('total_AUROC_mean', float('nan')):.4f}, "
            f"adh_auc={row.get('adhesion_AUROC_mean', float('nan')):.4f}, "
            f"scale_auc={row.get('scale_AUROC_mean', float('nan')):.4f}, "
            f"d_auc={row.get('total_minus_adhesion_AUROC_mean', float('nan')):+.5f}, "
            f"scale/adh_std={row.get('scale_std_over_adhesion_std_mean', float('nan')):.6f}, "
            f"rho(total,adh)={row.get('spearman_total_adhesion_mean', float('nan')):.6f}, "
            f"betweenC={row.get('scale_between_hard_community_var_ratio_mean', float('nan')):.4f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose RECAP context score contribution.")
    parser.add_argument("--checkpoint", nargs="*", default=None, help="One or more checkpoint paths.")
    parser.add_argument("--sensitivity-dir", type=Path, default=None, help="A sensitivity_results/<RUN_NAME> directory.")
    parser.add_argument("--checkpoint-glob", nargs="*", default=None, help="Glob patterns relative to project root.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_TEST_DATASETS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "tuning_hyperparams" / "diagnostics")
    parser.add_argument("--max-checkpoints", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    add_project_root_to_path()
    from model_checkpoint import load_model

    checkpoints = discover_checkpoints(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_cache = {}
    rows = []

    for ckpt_idx, checkpoint_path in enumerate(checkpoints, start=1):
        meta = checkpoint_metadata(checkpoint_path)
        print(f"\n[{ckpt_idx}/{len(checkpoints)}] Loading {checkpoint_path}")
        model, _, model_config, _ = load_model(str(checkpoint_path), device=args.device)

        dataset_key = (model_config.dims, model_config.num_hops, args.device, tuple(args.datasets))
        if dataset_key not in dataset_cache:
            print(
                f"Preparing datasets for dims={model_config.dims}, "
                f"num_hops={model_config.num_hops}: {args.datasets}"
            )
            dataset_cache[dataset_key] = prepare_datasets(
                dims=model_config.dims,
                num_hops=model_config.num_hops,
                dataset_names=args.datasets,
                device=args.device,
            )

        for dataset, dataset_name in zip(dataset_cache[dataset_key], args.datasets):
            diag = diagnose_dataset(model, model_config, dataset, dataset_name, args.device)
            rows.append({**meta, **diag})
            print(
                f"  {dataset_name}: "
                f"total_auc={diag['total_AUROC']:.4f}, "
                f"adh_auc={diag['adhesion_AUROC']:.4f}, "
                f"scale_auc={diag['scale_AUROC']:.4f}, "
                f"d_auc={diag['total_minus_adhesion_AUROC']:+.5f}, "
                f"scale/adh_std={diag['scale_std_over_adhesion_std']:.6f}, "
                f"rho(total,adh)={diag['spearman_total_adhesion']:.6f}, "
                f"betweenC={diag['scale_between_hard_community_var_ratio']:.4f}"
            )

    raw_path = args.output_dir / "scale_diagnostics_raw.csv"
    by_run_path = args.output_dir / "scale_diagnostics_by_run.csv"
    by_run_dataset_path = args.output_dir / "scale_diagnostics_by_run_dataset.csv"

    write_csv(raw_path, rows)
    write_csv(by_run_path, summarize(rows, ["run_id"]))
    write_csv(by_run_dataset_path, summarize(rows, ["run_id", "dataset"]))
    print_brief_summary(rows)
    print(f"\nWrote scale diagnostics to: {args.output_dir}")


if __name__ == "__main__":
    main()
