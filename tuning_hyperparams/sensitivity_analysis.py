from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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

PARAM_SWEEPS = {
    "beta": [0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.08, 0.12],
    "lambda_H": [0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8],
    "lambda_usage_entropy": [0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8],
    "cluster_init_gain": [0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5],
    "tau_c": [0.22, 0.25, 0.28, 0.3, 0.32, 0.35, 0.38, 0.42, 0.5, 0.65],
    "tau_s": [0.03, 0.05, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3],
    "num_hops": [1, 2, 3, 4, 5, 6, 8, 10, 12, 14],
    "knn_k": [8, 12, 16, 18, 20, 22, 24, 28, 32, 40],
    "num_clusters": [12, 16, 20, 24, 28, 32, 36, 40, 48, 56],
    # Dormant compatibility knob: keep the field loadable/plot-safe, but do
    # not generate sweeps that reactivate the residual variance loss.
    "lambda_E": [0],
}

DEFAULT_SWEEP_PARAMS = [
    "beta",
    "lambda_H",
    "cluster_init_gain",
    "tau_c",
    "tau_s",
    "num_hops",
]
APPENDIX_SWEEP_PARAMS = ["knn_k", "num_clusters", "lambda_E"]

MAIN_FIGURE_PARAMS = ["beta", "lambda_H", "cluster_init_gain", "tau_c", "tau_s", "num_hops"]
APPENDIX_FIGURE_PARAMS = ["knn_k", "num_clusters", "lambda_E", "lambda_usage_entropy"]

HEATMAP_KNN_K = [16, 20, 24, 32]
HEATMAP_NUM_CLUSTERS = [20, 28, 36, 48]

PARAM_LABELS = {
    "knn_k": r"$k$",
    "num_clusters": r"$C$",
    "num_hops": r"$L$",
    "tau_s": r"$\tau_s$",
    "tau_c": r"$\tau_c$",
    "cluster_init_gain": r"$g_{init}$",
    "beta": r"$\beta$",
    "lambda_H": r"$\lambda_H$",
    "lambda_usage_entropy": r"$\lambda_{usage}$",
    "lambda_E": r"$\lambda_E$",
}

METRIC_STYLES = {
    "AUROC": {"color": "#E6862D", "marker": "s"},
    "AUPRC": {"color": "#2F6FB3", "marker": "v"},
}

PLOT_THEME = {
    "axis_gray": "#4D4D4D",
    "grid_gray": "#D6D9DE",
    "spine_gray": "#A9AFB8",
    "baseline_gray": "#7E8694",
}

ALIAS_FIELDS = {
    "lambda_H": "lambda_ortho",
    "lambda_bal": "lambda_min_usage",
    "lambda_E": "lambda_diversity",
    "gamma": "min_cluster_ratio",
}


def value_equal(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-12
    except (TypeError, ValueError):
        return str(a) == str(b)


def value_token(value) -> str:
    text = str(value).replace("-", "m").replace(".", "p")
    return text.replace("/", "_").replace(" ", "")


def value_label(value) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def ordered_values(values, baseline):
    out = list(values)
    if not any(value_equal(v, baseline) for v in out):
        out.append(baseline)
        try:
            out = sorted(out, key=float)
        except (TypeError, ValueError):
            pass
    return out


def resolve_sweep_params(params: list[str]) -> list[str]:
    selected: list[str] = []
    aliases = {
        "main": DEFAULT_SWEEP_PARAMS,
        "appendix": APPENDIX_SWEEP_PARAMS,
        "all": list(PARAM_SWEEPS.keys()),
    }
    for param in params:
        expanded = aliases.get(param, [param])
        for item in expanded:
            if item not in PARAM_SWEEPS:
                raise ValueError(
                    f"Unknown sensitivity parameter: {item}. "
                    f"Available: {', '.join(PARAM_SWEEPS.keys())}"
                )
            if item not in selected:
                selected.append(item)
    return selected


def sync_aliases(config: dict) -> dict:
    for main_key, alias_key in ALIAS_FIELDS.items():
        if main_key in config:
            config[alias_key] = config[main_key]
    config["num_views"] = 1
    return config


def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def build_sensitivity_jobs(base_config: dict, sweep_params: list[str]) -> list[dict]:
    baseline_config = sync_aliases(deepcopy(base_config))
    jobs = [
        {
            "run_id": "baseline",
            "kind": "baseline",
            "param": "baseline",
            "value": "baseline",
            "config": baseline_config,
        }
    ]

    for param in sweep_params:
        values = PARAM_SWEEPS[param]
        baseline_value = base_config.get(param)
        if baseline_value is None:
            continue
        for value in ordered_values(values, baseline_value):
            if value_equal(value, baseline_value):
                continue
            config = sync_aliases(deepcopy(base_config))
            config[param] = value
            config = sync_aliases(config)
            jobs.append(
                {
                    "run_id": f"{param}__{value_token(value)}",
                    "kind": "sensitivity",
                    "param": param,
                    "value": value,
                    "config": config,
                }
            )
    return jobs


def build_heatmap_jobs(base_config: dict) -> list[dict]:
    jobs = []
    for knn_k in HEATMAP_KNN_K:
        for num_clusters in HEATMAP_NUM_CLUSTERS:
            config = sync_aliases(deepcopy(base_config))
            config["knn_k"] = knn_k
            config["num_clusters"] = num_clusters
            jobs.append(
                {
                    "run_id": f"heatmap__knn_{knn_k}__clusters_{num_clusters}",
                    "kind": "heatmap",
                    "param": "knn_k,num_clusters",
                    "value": f"{knn_k},{num_clusters}",
                    "knn_k": knn_k,
                    "num_clusters": num_clusters,
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
                    "kind": job["kind"],
                    "param": job["param"],
                    "value": job["value"],
                    "trial": trial,
                    "dataset": dataset,
                    "AUROC": float(scores["AUROC"]),
                    "AUPRC": float(scores["AUPRC"]),
                    "checkpoint": str(ckpt_path),
                }
            )
    return rows


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
    print(f"Running {job['run_id']} ({job['kind']}): {job['param']}={job['value']}")
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


def load_sensitivity_summary_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "param": row["param"],
                    "value": float(row["value"]),
                    "is_baseline": str(row["is_baseline"]).strip().lower() == "true",
                    "AUROC_mean": float(row["AUROC_mean"]),
                    "AUROC_std": float(row["AUROC_std"]),
                    "AUPRC_mean": float(row["AUPRC_mean"]),
                    "AUPRC_std": float(row["AUPRC_std"]),
                }
            )
    return rows


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
            "kind": row["kind"],
            "param": row["param"],
            "value": row["value"],
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
                "AUROC_mean": mean(by_run[run_id]["AUROC"]),
                "AUROC_std": std(by_run[run_id]["AUROC"]),
                "AUPRC_mean": mean(by_run[run_id]["AUPRC"]),
                "AUPRC_std": std(by_run[run_id]["AUPRC"]),
                "num_trials": len(by_run[run_id]["AUROC"]),
            }
        )
        summary.append(row)
    return summary


def build_sensitivity_summary(
    run_summary: list[dict],
    base_config: dict,
    sweep_params: list[str],
) -> list[dict]:
    by_run = {row["run_id"]: row for row in run_summary}
    baseline = by_run.get("baseline")
    rows = []
    if baseline is None:
        return rows

    for param in sweep_params:
        values = PARAM_SWEEPS[param]
        baseline_value = base_config.get(param)
        if baseline_value is None:
            continue
        for value in ordered_values(values, baseline_value):
            if value_equal(value, baseline_value):
                source = baseline
            else:
                source = by_run.get(f"{param}__{value_token(value)}")
            if source is None:
                continue
            rows.append(
                {
                    "param": param,
                    "value": value,
                    "is_baseline": value_equal(value, baseline_value),
                    "AUROC_mean": source["AUROC_mean"],
                    "AUROC_std": source["AUROC_std"],
                    "AUPRC_mean": source["AUPRC_mean"],
                    "AUPRC_std": source["AUPRC_std"],
                }
            )
    return rows


def metric_ylim(values: list[float], errors: list[float]) -> tuple[float, float]:
    lowers = [100.0 * (value - err) for value, err in zip(values, errors)]
    uppers = [100.0 * (value + err) for value, err in zip(values, errors)]
    low = max(0.0, min(lowers))
    high = min(100.0, max(uppers))
    span = max(high - low, 0.2)
    pad = max(0.15, span * 0.25)
    low = max(0.0, low - pad)
    high = min(100.0, high + pad)
    if high - low < 0.5:
        center = (high + low) / 2.0
        low = max(0.0, center - 0.25)
        high = min(100.0, center + 0.25)
    return low, high


def collect_param_series(rows: list[dict], param: str, base_config: dict) -> dict:
    baseline_value = base_config.get(param)
    param_rows = [row for row in rows if row["param"] == param]

    series = {
        "xs": [],
        "labels": [],
        "AUROC_mean": [],
        "AUROC_std": [],
        "AUPRC_mean": [],
        "AUPRC_std": [],
        "baseline_idx": None,
    }
    for idx, row in enumerate(param_rows):
        value = row["value"]
        series["xs"].append(idx)
        series["labels"].append(value_label(value))
        series["AUROC_mean"].append(float(row["AUROC_mean"]))
        series["AUROC_std"].append(float(row["AUROC_std"]))
        series["AUPRC_mean"].append(float(row["AUPRC_mean"]))
        series["AUPRC_std"].append(float(row["AUPRC_std"]))
        row_is_baseline = bool(row.get("is_baseline", False))
        if row_is_baseline or value_equal(value, baseline_value):
            series["baseline_idx"] = idx
    return series


def axis_layout(num_params: int) -> tuple[int, int, tuple[float, float]]:
    if num_params <= 3:
        return 1, num_params, (4.2 * num_params, 3.2)
    if num_params <= 5:
        return 1, num_params, (3.95 * num_params, 3.55)
    return 2, 3, (12.6, 6.4)


def plot_sensitivity_dual_axis(
    rows: list[dict],
    output_path: Path,
    base_config: dict,
    params: list[str],
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:
        print(f"matplotlib is unavailable; skip sensitivity plot ({exc})")
        return

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    plot_params = [
        param for param in params
        if param in PARAM_SWEEPS and base_config.get(param) is not None
    ]
    if not plot_params:
        return

    nrows, ncols, figsize = axis_layout(len(plot_params))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        constrained_layout=True,
    )
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for plot_idx, (ax, param) in enumerate(zip(axes, plot_params)):
        col_idx = plot_idx % ncols
        series = collect_param_series(rows, param, base_config)
        xs = series["xs"]
        if not xs:
            ax.axis("off")
            continue

        auc_style = METRIC_STYLES["AUROC"]
        pr_style = METRIC_STYLES["AUPRC"]
        ax_pr = ax.twinx()
        ax.set_facecolor("white")

        auc_pct = [100.0 * value for value in series["AUROC_mean"]]
        auc_lower = [
            100.0 * max(0.0, value - err)
            for value, err in zip(series["AUROC_mean"], series["AUROC_std"])
        ]
        auc_upper = [
            100.0 * min(1.0, value + err)
            for value, err in zip(series["AUROC_mean"], series["AUROC_std"])
        ]
        pr_pct = [100.0 * value for value in series["AUPRC_mean"]]
        pr_lower = [
            100.0 * max(0.0, value - err)
            for value, err in zip(series["AUPRC_mean"], series["AUPRC_std"])
        ]
        pr_upper = [
            100.0 * min(1.0, value + err)
            for value, err in zip(series["AUPRC_mean"], series["AUPRC_std"])
        ]

        ax.fill_between(
            xs,
            auc_lower,
            auc_upper,
            color=auc_style["color"],
            alpha=0.22,
            linewidth=0,
            zorder=1,
        )
        ax_pr.fill_between(
            xs,
            pr_lower,
            pr_upper,
            color=pr_style["color"],
            alpha=0.20,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            xs,
            auc_pct,
            color=auc_style["color"],
            marker=auc_style["marker"],
            markersize=4.2,
            linewidth=2.0,
            zorder=3,
        )
        ax_pr.plot(
            xs,
            pr_pct,
            color=pr_style["color"],
            marker=pr_style["marker"],
            markersize=4.2,
            linewidth=2.0,
            zorder=3,
        )
        ax.set_title(PARAM_LABELS.get(param, param), fontsize=12)
        ax.set_xticks(xs)
        ax.set_xticklabels(series["labels"], rotation=30, ha="right")
        ax.set_ylabel("AUROC (%)" if col_idx == 0 else "")
        ax_pr.set_ylabel("AUPRC (%)" if col_idx == ncols - 1 else "")
        ax.tick_params(axis="both", colors=PLOT_THEME["axis_gray"])
        ax_pr.tick_params(axis="y", colors=PLOT_THEME["axis_gray"])
        ax.xaxis.label.set_color(PLOT_THEME["axis_gray"])
        ax.yaxis.label.set_color(PLOT_THEME["axis_gray"])
        ax_pr.yaxis.label.set_color(PLOT_THEME["axis_gray"])
        ax.title.set_color(PLOT_THEME["axis_gray"])
        ax.set_ylim(*metric_ylim(series["AUROC_mean"], series["AUROC_std"]))
        ax_pr.set_ylim(*metric_ylim(series["AUPRC_mean"], series["AUPRC_std"]))
        ax.grid(True, axis="y", color=PLOT_THEME["grid_gray"], alpha=0.9, linewidth=0.7)
        ax.grid(True, axis="x", color=PLOT_THEME["grid_gray"], alpha=0.5, linewidth=0.5)
        ax.spines["left"].set_color(PLOT_THEME["spine_gray"])
        ax.spines["bottom"].set_color(PLOT_THEME["spine_gray"])
        ax_pr.spines["right"].set_color(PLOT_THEME["spine_gray"])
        for spine in ("top",):
            ax.spines[spine].set_visible(False)
            ax_pr.spines[spine].set_visible(False)

    for ax in axes[len(plot_params):]:
        ax.axis("off")

    legend_handles = [
        Line2D([0], [0], color=METRIC_STYLES["AUROC"]["color"], marker=METRIC_STYLES["AUROC"]["marker"], linewidth=1.8, label="AUROC"),
        Line2D([0], [0], color=METRIC_STYLES["AUPRC"]["color"], marker=METRIC_STYLES["AUPRC"]["marker"], linewidth=1.8, label="AUPRC"),
    ]
    axes[0].legend(
        handles=legend_handles,
        loc="upper left",
        frameon=True,
        fancybox=False,
        framealpha=0.96,
        edgecolor=PLOT_THEME["spine_gray"],
    )
    # fig.suptitle(title, fontsize=16, fontweight="semibold", color=PLOT_THEME["axis_gray"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_sensitivity(rows: list[dict], output_dir: Path, base_config: dict) -> None:
    plot_sensitivity_dual_axis(
        rows,
        output_dir / "sensitivity_main.png",
        base_config,
        MAIN_FIGURE_PARAMS,
        "RECAP Hyperparameter Sensitivity (Main)",
    )


def plot_heatmap(run_summary: list[dict], output_dir: Path, metric: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"matplotlib is unavailable; skip heatmap ({exc})")
        return

    by_run = {row["run_id"]: row for row in run_summary}
    matrix = []
    for knn_k in HEATMAP_KNN_K:
        row_values = []
        for num_clusters in HEATMAP_NUM_CLUSTERS:
            run_id = f"heatmap__knn_{knn_k}__clusters_{num_clusters}"
            summary = by_run.get(run_id)
            row_values.append(float("nan") if summary is None else float(summary[f"{metric}_mean"]))
        matrix.append(row_values)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(HEATMAP_NUM_CLUSTERS)))
    ax.set_xticklabels([str(v) for v in HEATMAP_NUM_CLUSTERS])
    ax.set_yticks(range(len(HEATMAP_KNN_K)))
    ax.set_yticklabels([str(v) for v in HEATMAP_KNN_K])
    ax.set_xlabel("num_clusters")
    ax.set_ylabel("knn_k")
    # ax.set_title(f"knn_k x num_clusters Sensitivity ({metric})")
    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            if value == value:
                ax.text(x, y, f"{value:.3f}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax)
    path = output_dir / f"knn_clusters_heatmap_{metric.lower()}.png"
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Run RECAP hyperparameter sensitivity experiments.")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument(
        "--params",
        nargs="+",
        default=DEFAULT_SWEEP_PARAMS,
        help=(
            "Parameters to sweep. Use explicit names, or aliases: main, appendix, all. "
            "Default runs the most important community/performance parameters."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--train-datasets", nargs="+", default=DEFAULT_TRAIN_DATASETS)
    parser.add_argument("--test-datasets", nargs="+", default=DEFAULT_TEST_DATASETS)
    parser.add_argument("--knn-cache-dir", type=Path, default=PROJECT_ROOT / "knn_cache")
    parser.add_argument("--knn-search-dtype", type=str, default="auto")
    parser.add_argument("--disable-knn-cache", action="store_true")
    parser.add_argument("--include-heatmap", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = PROJECT_ROOT / "tuning_hyperparams" / "sensitivity_results" / timestamp
    args.output_dir = args.output_dir.resolve()
    args.knn_cache_dir = args.knn_cache_dir.resolve()
    args.knn_cache_enabled = not args.disable_knn_cache

    base_config = sync_aliases(load_json(args.base_config))
    sweep_params = resolve_sweep_params(args.params)
    jobs = build_sensitivity_jobs(base_config, sweep_params)
    if args.include_heatmap:
        jobs.extend(build_heatmap_jobs(base_config))

    manifest = {
        "base_config": str(args.base_config),
        "output_dir": str(args.output_dir),
        "device": args.device,
        "epochs": args.epochs,
        "trials": args.trials,
        "params": sweep_params,
        "param_sweeps": {param: PARAM_SWEEPS[param] for param in sweep_params},
        "train_datasets": args.train_datasets,
        "test_datasets": args.test_datasets,
        "knn_cache_dir": str(args.knn_cache_dir),
        "knn_search_dtype": args.knn_search_dtype,
        "include_heatmap": args.include_heatmap,
        "jobs": [
            {key: value for key, value in job.items() if key != "config"}
            for job in jobs
        ],
    }
    write_json(args.output_dir / "manifest.json", manifest)
    write_csv(
        args.output_dir / "job_plan.csv",
        [
            {
                "run_id": job["run_id"],
                "kind": job["kind"],
                "param": job["param"],
                "value": job["value"],
            }
            for job in jobs
        ],
        ["run_id", "kind", "param", "value"],
    )

    if args.dry_run:
        print(f"Dry run complete: wrote {len(jobs)} jobs to {args.output_dir}")
        return

    raw_rows = []
    if args.plot_only:
        raw_path = args.output_dir / "raw_results.csv"
        if not raw_path.exists():
            raise FileNotFoundError(f"Cannot plot-only without {raw_path}")
        with open(raw_path, "r") as f:
            raw_rows = list(csv.DictReader(f))
    else:
        for job in jobs:
            rows = run_job(job, args, args.output_dir, base_config)
            raw_rows.extend(rows)
            write_csv(
                args.output_dir / "raw_results.csv",
                raw_rows,
                ["run_id", "kind", "param", "value", "trial", "dataset", "AUROC", "AUPRC", "checkpoint"],
            )

    run_summary = summarize_runs(raw_rows)
    sensitivity_summary_from_csv = load_sensitivity_summary_csv(
        args.output_dir / "sensitivity_summary.csv"
    ) if args.plot_only else []
    sensitivity_summary = (
        [
            row for row in sensitivity_summary_from_csv
            if row["param"] in sweep_params
        ]
        if sensitivity_summary_from_csv
        else build_sensitivity_summary(run_summary, base_config, sweep_params)
    )

    write_csv(
        args.output_dir / "summary_by_run.csv",
        run_summary,
        ["run_id", "kind", "param", "value", "AUROC_mean", "AUROC_std", "AUPRC_mean", "AUPRC_std", "num_trials"],
    )
    write_csv(
        args.output_dir / "sensitivity_summary.csv",
        sensitivity_summary,
        ["param", "value", "is_baseline", "AUROC_mean", "AUROC_std", "AUPRC_mean", "AUPRC_std"],
    )
    plot_sensitivity(sensitivity_summary, args.output_dir / "figures", base_config)
    if args.include_heatmap:
        plot_heatmap(run_summary, args.output_dir / "figures", "AUROC")
        plot_heatmap(run_summary, args.output_dir / "figures", "AUPRC")

    print(f"\nSensitivity experiment complete: {args.output_dir}")


if __name__ == "__main__":
    main()
