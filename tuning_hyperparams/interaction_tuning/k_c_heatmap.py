from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_CONFIG = PROJECT_ROOT / "params" / "recap.json"
DEFAULT_TRAIN_DATASETS = ["pubmed", "Flickr", "questions", "YelpChi"]
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

DEFAULT_K_VALUES = [16, 24, 30, 36, 48, 64, 80, 96]
DEFAULT_C_VALUES = [16, 20, 24, 28, 32, 36, 40, 48]

ALIAS_FIELDS = {
    "lambda_H": "lambda_ortho",
    "lambda_bal": "lambda_min_usage",
    "lambda_E": "lambda_diversity",
    "gamma": "min_cluster_ratio",
}


def value_token(value) -> str:
    text = str(value).replace("-", "m").replace(".", "p")
    return text.replace("/", "_").replace(" ", "")


def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def sync_aliases(config: dict) -> dict:
    for main_key, alias_key in ALIAS_FIELDS.items():
        if main_key in config:
            config[alias_key] = config[main_key]
    config["num_views"] = 1
    return config


def make_config(base_config: dict, overrides: dict) -> dict:
    config = deepcopy(base_config)
    config.update(overrides)
    return sync_aliases(config)


def build_jobs(base_config: dict, k_values: list[int], c_values: list[int]) -> list[dict]:
    jobs = []
    for knn_k in k_values:
        for num_clusters in c_values:
            overrides = {
                "knn_k": int(knn_k),
                "num_clusters": int(num_clusters),
            }
            config = make_config(base_config, overrides)
            jobs.append(
                {
                    "run_id": f"k{int(knn_k)}__C{int(num_clusters)}",
                    "knn_k": int(knn_k),
                    "num_clusters": int(num_clusters),
                    "overrides": overrides,
                    "config": config,
                }
            )
    return jobs


def torch_load(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def collect_checkpoint_rows(
    job: dict,
    checkpoint_dir: Path,
    trials: int,
    model_name: str,
) -> list[dict]:
    rows = []
    for trial in range(trials):
        ckpt_path = checkpoint_dir / model_name / f"trial_{trial}" / "model.pt"
        if not ckpt_path.exists():
            return []
        checkpoint = torch_load(ckpt_path)
        metrics = checkpoint.get("metrics", {})
        if not metrics:
            return []
        for dataset, scores in metrics.items():
            rows.append(
                {
                    "run_id": job["run_id"],
                    "knn_k": job["knn_k"],
                    "num_clusters": job["num_clusters"],
                    "trial": trial,
                    "dataset": dataset,
                    "AUROC": float(scores["AUROC"]),
                    "AUPRC": float(scores["AUPRC"]),
                    "checkpoint": str(ckpt_path),
                }
            )
    return rows


def has_complete_checkpoints(
    job: dict,
    checkpoint_dir: Path,
    trials: int,
    model_name: str,
) -> bool:
    return all(
        (checkpoint_dir / model_name / f"trial_{trial}" / "model.pt").exists()
        for trial in range(trials)
    )


def collect_reuse_rows(job: dict, args, model_name: str) -> list[dict]:
    for result_dir in args.reuse_results:
        checkpoint_dir = result_dir / "checkpoints" / job["run_id"]
        rows = collect_checkpoint_rows(job, checkpoint_dir, args.trials, model_name)
        if rows:
            print(f"[reuse] {job['run_id']} from {result_dir}")
            return rows
    return []


def run_subprocess(cmd: list[str], log_path: Path, cwd: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(cmd)}")


def run_job(job: dict, args, output_dir: Path, base_config: dict) -> list[dict]:
    model_name = base_config.get("model", "recap")
    checkpoint_dir = output_dir / "checkpoints" / job["run_id"]
    config_dir = output_dir / "configs" / job["run_id"]
    log_path = output_dir / "logs" / f"{job['run_id']}.log"

    if args.force and checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)

    existing_rows = collect_checkpoint_rows(job, checkpoint_dir, args.trials, model_name)
    if existing_rows:
        print(f"[skip] {job['run_id']} already has complete checkpoint metrics.")
        return existing_rows

    if not args.force:
        reused_rows = collect_reuse_rows(job, args, model_name)
        if reused_rows:
            return reused_rows

    config = deepcopy(job["config"])
    config["knn_cache_enabled"] = args.knn_cache_enabled
    config["knn_cache_dir"] = str(args.knn_cache_dir)
    config["knn_search_dtype"] = args.knn_search_dtype
    config = sync_aliases(config)
    write_json(config_dir / f"{model_name}.json", config)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "train.py"),
        "--trials",
        str(args.trials),
        "--epochs",
        str(args.epochs),
        "--device",
        args.device,
        "--output-dir",
        str(checkpoint_dir),
        "--json-dir",
        str(config_dir),
        "--dims",
        str(config["dims"]),
        "--train-datasets",
        *args.train_datasets,
        "--test-datasets",
        *args.test_datasets,
    ]

    print(f"\n{'=' * 80}")
    print(f"Running {job['run_id']}: k={job['knn_k']}, C={job['num_clusters']}")
    print(f"{'=' * 80}")
    run_subprocess(cmd, log_path, PROJECT_ROOT)

    rows = collect_checkpoint_rows(job, checkpoint_dir, args.trials, model_name)
    if not rows:
        raise RuntimeError(f"No metrics found after training job {job['run_id']}")
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_csv(path: Path) -> list[dict]:
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def std(values: list[float]) -> float:
    if not values:
        return float("nan")
    mu = mean(values)
    return (sum((x - mu) ** 2 for x in values) / len(values)) ** 0.5


def summarize_runs(raw_rows: list[dict]) -> list[dict]:
    by_run_trial = {}
    metadata = {}
    for row in raw_rows:
        run_id = row["run_id"]
        metadata[run_id] = {
            "run_id": run_id,
            "knn_k": int(row["knn_k"]),
            "num_clusters": int(row["num_clusters"]),
        }
        key = (run_id, int(row["trial"]))
        by_run_trial.setdefault(key, {"AUROC": [], "AUPRC": []})
        by_run_trial[key]["AUROC"].append(float(row["AUROC"]))
        by_run_trial[key]["AUPRC"].append(float(row["AUPRC"]))

    by_run = {}
    for (run_id, trial), vals in by_run_trial.items():
        by_run.setdefault(run_id, {"AUROC": [], "AUPRC": []})
        by_run[run_id]["AUROC"].append(mean(vals["AUROC"]))
        by_run[run_id]["AUPRC"].append(mean(vals["AUPRC"]))

    summary = []
    for run_id in sorted(by_run):
        row = dict(metadata[run_id])
        row.update(
            {
                "selection_score": mean(by_run[run_id]["AUROC"]) + mean(by_run[run_id]["AUPRC"]),
                "AUROC_mean": mean(by_run[run_id]["AUROC"]),
                "AUROC_std": std(by_run[run_id]["AUROC"]),
                "AUPRC_mean": mean(by_run[run_id]["AUPRC"]),
                "AUPRC_std": std(by_run[run_id]["AUPRC"]),
                "num_trials": len(by_run[run_id]["AUROC"]),
            }
        )
        summary.append(row)

    summary.sort(key=lambda row: (int(row["knn_k"]), int(row["num_clusters"])))
    return summary


def metric_matrix(
    summary: list[dict],
    k_values: list[int],
    c_values: list[int],
    metric: str,
) -> list[list[float]]:
    lookup = {
        (int(row["knn_k"]), int(row["num_clusters"])): float(row[f"{metric}_mean"])
        for row in summary
    }
    return [
        [lookup.get((int(k), int(c)), float("nan")) for c in c_values]
        for k in k_values
    ]


def plot_heatmap(
    summary: list[dict],
    k_values: list[int],
    c_values: list[int],
    output_dir: Path,
    metric: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        print(f"matplotlib/numpy unavailable; skip heatmap ({exc})")
        return

    matrix = np.array(metric_matrix(summary, k_values, c_values, metric), dtype=float)
    if matrix.size == 0:
        return

    color_map = "YlOrBr" if metric == "AUROC" else "Blues"
    fig, ax = plt.subplots(figsize=(7.6, 5.8), constrained_layout=True)
    image = ax.imshow(matrix * 100.0, cmap=color_map, aspect="auto")

    ax.set_xticks(range(len(c_values)))
    ax.set_xticklabels([str(v) for v in c_values])
    ax.set_yticks(range(len(k_values)))
    ax.set_yticklabels([str(v) for v in k_values])
    ax.set_xlabel("Number of communities C")
    ax.set_ylabel("KNN neighbors k")
    ax.set_title(f"k x C Joint Tuning ({metric})", fontsize=14, fontweight="semibold")

    finite = matrix[np.isfinite(matrix)] * 100.0
    threshold = (float(finite.min()) + float(finite.max())) / 2.0 if finite.size else 0.0
    for y, row in enumerate(matrix * 100.0):
        for x, value in enumerate(row):
            if np.isfinite(value):
                color = "white" if value > threshold else "#1F2933"
                ax.text(x, y, f"{value:.1f}", ha="center", va="center", color=color, fontsize=8)

    for spine in ax.spines.values():
        spine.set_color("#A9AFB8")
    ax.tick_params(colors="#4D4D4D")
    ax.xaxis.label.set_color("#4D4D4D")
    ax.yaxis.label.set_color("#4D4D4D")

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(f"{metric} (%)")
    colorbar.ax.tick_params(colors="#4D4D4D")

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"k_c_heatmap_{metric.lower()}.png"
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_paper_heatmaps(
    summary: list[dict],
    k_values: list[int],
    c_values: list[int],
    output_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        print(f"matplotlib/numpy unavailable; skip paper heatmap ({exc})")
        return

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), constrained_layout=True)
    metric_specs = [
        ("AUROC", "YlOrBr", "(a) AUROC"),
        ("AUPRC", "Blues", "(b) AUPRC"),
    ]
    for ax, (metric, color_map, title) in zip(axes, metric_specs):
        matrix = np.array(metric_matrix(summary, k_values, c_values, metric), dtype=float) * 100.0
        if matrix.size == 0:
            ax.axis("off")
            continue

        image = ax.imshow(matrix, cmap=color_map, aspect="auto")
        ax.set_xticks(range(len(c_values)))
        ax.set_xticklabels([str(v) for v in c_values])
        ax.set_yticks(range(len(k_values)))
        ax.set_yticklabels([str(v) for v in k_values])
        ax.set_xlabel("Number of communities C")
        ax.set_title(title, color="#4D4D4D")
        if ax is axes[0]:
            ax.set_ylabel("KNN neighbors k")

        finite = matrix[np.isfinite(matrix)]
        if finite.size:
            threshold = (float(finite.min()) + float(finite.max())) / 2.0
        else:
            threshold = 0.0

        for y, row in enumerate(matrix):
            for x, value in enumerate(row):
                if np.isfinite(value):
                    color = "white" if value > threshold else "#1F2933"
                    ax.text(x, y, f"{value:.1f}", ha="center", va="center", color=color, fontsize=7.2)

        for spine in ax.spines.values():
            spine.set_color("#A9AFB8")
        ax.tick_params(colors="#4D4D4D")
        ax.xaxis.label.set_color("#4D4D4D")
        ax.yaxis.label.set_color("#4D4D4D")

        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        colorbar.set_label(f"{metric} (%)", color="#4D4D4D")
        colorbar.ax.tick_params(colors="#4D4D4D")

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "k_c_heatmap_main.png"
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Run k x C joint tuning and heatmap generation.")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--train-datasets", nargs="+", default=DEFAULT_TRAIN_DATASETS)
    parser.add_argument("--test-datasets", nargs="+", default=DEFAULT_TEST_DATASETS)
    parser.add_argument("--knn-cache-dir", type=Path, default=PROJECT_ROOT / "knn_cache")
    parser.add_argument("--knn-search-dtype", type=str, default="auto")
    parser.add_argument("--disable-knn-cache", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--k-values", type=parse_int_list, default=DEFAULT_K_VALUES)
    parser.add_argument("--c-values", type=parse_int_list, default=DEFAULT_C_VALUES)
    parser.add_argument(
        "--reuse-results",
        nargs="*",
        type=Path,
        default=[],
        help="Existing k x C result directories whose checkpoints can be reused.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = (
            PROJECT_ROOT
            / "tuning_hyperparams"
            / "interaction_tuning"
            / "k_c_heatmap_results"
            / timestamp
        )
    args.output_dir = args.output_dir.resolve()
    args.knn_cache_dir = args.knn_cache_dir.resolve()
    args.reuse_results = [path.resolve() for path in args.reuse_results]
    args.knn_cache_enabled = not args.disable_knn_cache
    k_values = [int(v) for v in args.k_values]
    c_values = [int(v) for v in args.c_values]

    base_config = sync_aliases(load_json(args.base_config))
    jobs = build_jobs(base_config, k_values, c_values)
    manifest = {
        "base_config": str(args.base_config),
        "output_dir": str(args.output_dir),
        "device": args.device,
        "epochs": args.epochs,
        "trials": args.trials,
        "train_datasets": args.train_datasets,
        "test_datasets": args.test_datasets,
        "knn_cache_dir": str(args.knn_cache_dir),
        "knn_search_dtype": args.knn_search_dtype,
        "k_values": k_values,
        "c_values": c_values,
        "reuse_results": [str(path) for path in args.reuse_results],
        "num_jobs": len(jobs),
    }
    write_json(args.output_dir / "manifest.json", manifest)
    write_csv(
        args.output_dir / "job_plan.csv",
        [
            {
                "run_id": job["run_id"],
                "knn_k": job["knn_k"],
                "num_clusters": job["num_clusters"],
            }
            for job in jobs
        ],
        ["run_id", "knn_k", "num_clusters"],
    )

    if args.dry_run:
        model_name = base_config.get("model", "recap")
        reusable = 0
        for job in jobs:
            for result_dir in args.reuse_results:
                checkpoint_dir = result_dir / "checkpoints" / job["run_id"]
                if has_complete_checkpoints(job, checkpoint_dir, args.trials, model_name):
                    reusable += 1
                    break
        print(
            f"Dry run complete: wrote {len(jobs)} jobs to {args.output_dir} "
            f"({reusable} reusable, {len(jobs) - reusable} to train)"
        )
        return

    if args.plot_only:
        raw_path = args.output_dir / "raw_results.csv"
        if not raw_path.exists():
            raise FileNotFoundError(f"Cannot plot-only without {raw_path}")
        raw_rows = read_csv(raw_path)
    else:
        raw_rows = []
        for job in jobs:
            rows = run_job(job, args, args.output_dir, base_config)
            raw_rows.extend(rows)
            write_csv(
                args.output_dir / "raw_results.csv",
                raw_rows,
                [
                    "run_id",
                    "knn_k",
                    "num_clusters",
                    "trial",
                    "dataset",
                    "AUROC",
                    "AUPRC",
                    "checkpoint",
                ],
            )

    summary = summarize_runs(raw_rows)
    write_csv(
        args.output_dir / "summary_by_run.csv",
        summary,
        [
            "run_id",
            "knn_k",
            "num_clusters",
            "selection_score",
            "AUROC_mean",
            "AUROC_std",
            "AUPRC_mean",
            "AUPRC_std",
            "num_trials",
        ],
    )
    plot_heatmap(summary, k_values, c_values, args.output_dir / "figures", "AUROC")
    plot_heatmap(summary, k_values, c_values, args.output_dir / "figures", "AUPRC")
    plot_paper_heatmaps(summary, k_values, c_values, args.output_dir / "figures")

    print(f"\nk x C heatmap tuning complete: {args.output_dir}")


if __name__ == "__main__":
    main()
