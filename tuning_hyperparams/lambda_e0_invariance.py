from __future__ import annotations

import argparse
import csv
import json
import math
import time
from copy import deepcopy
from pathlib import Path

from sensitivity_analysis import (
    DEFAULT_TEST_DATASETS,
    DEFAULT_TRAIN_DATASETS,
    PROJECT_ROOT,
    collect_checkpoint_rows,
    mean,
    run_job,
    std,
    summarize_runs,
    sync_aliases,
    value_equal,
    value_token,
    write_csv,
    write_json,
)


DEFAULT_REFERENCE_DIR = (
    PROJECT_ROOT / "tuning_hyperparams" / "sensitivity_results" / "sensitivity_default_v2"
)
DEFAULT_REFERENCE_CONFIG = DEFAULT_REFERENCE_DIR / "configs" / "baseline" / "recap.json"
DEFAULT_REFERENCE_SUMMARY = DEFAULT_REFERENCE_DIR / "sensitivity_summary.csv"

DEFAULT_PARAMS = [
    "beta",
    "lambda_H",
    "cluster_init_gain",
    "tau_c",
    "tau_s",
    "num_hops",
]

DEFAULT_PROBE_VALUES = {
    "beta": [0.0, 0.02, 0.05],
    "lambda_H": [0.0, 0.1, 0.8],
    "cluster_init_gain": [0.8, 1.5, 3.5],
    "tau_c": [0.25, 0.3, 0.32, 0.65],
    "tau_s": [0.03, 0.08, 0.25],
    "num_hops": [2, 4, 6],
}


def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def load_summary_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
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


def dedup_values(values: list) -> list:
    out = []
    for value in values:
        if not any(value_equal(value, existing) for existing in out):
            out.append(value)
    return out


def load_probe_values(path: Path | None) -> dict[str, list]:
    if path is None:
        return deepcopy(DEFAULT_PROBE_VALUES)
    payload = load_json(path)
    return {str(param): list(values) for param, values in payload.items()}


def values_for_param(base_config: dict, probe_values: dict[str, list], param: str) -> list:
    values = list(probe_values.get(param, []))
    baseline_value = base_config.get(param)
    if baseline_value is not None:
        values.append(baseline_value)
    return dedup_values(values)


def build_probe_jobs(
    base_config: dict,
    params: list[str],
    probe_values: dict[str, list],
) -> list[dict]:
    jobs = [
        {
            "run_id": "baseline",
            "kind": "lambda_E0_baseline",
            "param": "baseline",
            "value": "baseline",
            "config": sync_aliases(deepcopy(base_config)),
        }
    ]
    for param in params:
        baseline_value = base_config.get(param)
        if baseline_value is None:
            continue
        for value in values_for_param(base_config, probe_values, param):
            if value_equal(value, baseline_value):
                continue
            config = sync_aliases(deepcopy(base_config))
            config[param] = value
            config = sync_aliases(config)
            jobs.append(
                {
                    "run_id": f"{param}__{value_token(value)}",
                    "kind": "lambda_E0_probe",
                    "param": param,
                    "value": value,
                    "config": config,
                }
            )
    return jobs


def build_probe_summary(
    run_summary: list[dict],
    base_config: dict,
    params: list[str],
    probe_values: dict[str, list],
) -> list[dict]:
    by_run = {row["run_id"]: row for row in run_summary}
    baseline = by_run.get("baseline")
    if baseline is None:
        return []

    rows = []
    for param in params:
        baseline_value = base_config.get(param)
        if baseline_value is None:
            continue
        for value in values_for_param(base_config, probe_values, param):
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


def find_ref_row(reference_rows: list[dict], param: str, value) -> dict | None:
    for row in reference_rows:
        if row["param"] == param and value_equal(row["value"], value):
            return row
    return None


def pooled_std(ref_std: float, probe_std: float) -> float:
    if math.isnan(ref_std):
        ref_std = 0.0
    if math.isnan(probe_std):
        probe_std = 0.0
    return (ref_std**2 + probe_std**2) ** 0.5


def metric_pass(
    delta: float,
    pooled: float,
    sigma_multiplier: float,
    min_abs_tol: float,
) -> bool:
    return abs(delta) <= max(min_abs_tol, sigma_multiplier * pooled)


def compare_summaries(
    reference_rows: list[dict],
    probe_rows: list[dict],
    sigma_multiplier: float,
    min_abs_tol: float,
) -> list[dict]:
    rows = []
    for probe in probe_rows:
        ref = find_ref_row(reference_rows, probe["param"], probe["value"])
        if ref is None:
            continue
        out = {
            "param": probe["param"],
            "value": probe["value"],
            "is_baseline": probe["is_baseline"],
        }
        for metric in ("AUROC", "AUPRC"):
            ref_mean = float(ref[f"{metric}_mean"])
            probe_mean = float(probe[f"{metric}_mean"])
            ref_std = float(ref[f"{metric}_std"])
            probe_std = float(probe[f"{metric}_std"])
            delta = probe_mean - ref_mean
            sigma = pooled_std(ref_std, probe_std)
            passed = metric_pass(delta, sigma, sigma_multiplier, min_abs_tol)
            out.update(
                {
                    f"ref_{metric}_mean": ref_mean,
                    f"lambda_E0_{metric}_mean": probe_mean,
                    f"delta_{metric}": delta,
                    f"abs_delta_{metric}": abs(delta),
                    f"ref_{metric}_std": ref_std,
                    f"lambda_E0_{metric}_std": probe_std,
                    f"pooled_{metric}_std": sigma,
                    f"{metric}_within_tolerance": passed,
                }
            )
        rows.append(out)
    return rows


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return float("nan")
    x_mean = mean(xs)
    y_mean = mean(ys)
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    denom = (x_var * y_var) ** 0.5
    if denom == 0:
        return float("nan")
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom


def summarize_comparison(comparison_rows: list[dict]) -> list[dict]:
    by_param: dict[str, list[dict]] = {}
    for row in comparison_rows:
        by_param.setdefault(row["param"], []).append(row)

    rows = []
    for param, param_rows in by_param.items():
        row = {"param": param, "n": len(param_rows)}
        for metric in ("AUROC", "AUPRC"):
            deltas = [float(item[f"abs_delta_{metric}"]) for item in param_rows]
            pass_values = [bool(item[f"{metric}_within_tolerance"]) for item in param_rows]
            ref_values = [float(item[f"ref_{metric}_mean"]) for item in param_rows]
            probe_values = [float(item[f"lambda_E0_{metric}_mean"]) for item in param_rows]
            row.update(
                {
                    f"max_abs_delta_{metric}": max(deltas) if deltas else float("nan"),
                    f"mean_abs_delta_{metric}": mean(deltas),
                    f"pass_rate_{metric}": mean([1.0 if value else 0.0 for value in pass_values]),
                    f"pearson_{metric}": pearson(ref_values, probe_values),
                }
            )
        rows.append(row)
    return rows


def write_markdown_report(
    path: Path,
    args,
    base_config: dict,
    comparison_rows: list[dict],
    comparison_summary: list[dict],
) -> None:
    total_checks = len(comparison_rows) * 2
    passed_checks = sum(
        int(row["AUROC_within_tolerance"]) + int(row["AUPRC_within_tolerance"])
        for row in comparison_rows
    )
    verdict = "PASS" if total_checks and passed_checks == total_checks else "REVIEW"

    worst_rows = sorted(
        comparison_rows,
        key=lambda row: max(float(row["abs_delta_AUROC"]), float(row["abs_delta_AUPRC"])),
        reverse=True,
    )[:8]

    lines = [
        "# lambda_E=0 Invariance Check",
        "",
        f"Verdict: **{verdict}** ({passed_checks}/{total_checks} metric checks within tolerance).",
        "",
        "## Setup",
        "",
        f"- Reference summary: `{args.reference_summary}`",
        f"- Reference lambda_E: `{args.reference_lambda_e}`",
        f"- Probe lambda_E: `{base_config['lambda_E']}`",
        f"- Epochs: `{args.epochs}`",
        f"- Trials: `{args.trials}`",
        f"- Train datasets: `{', '.join(args.train_datasets)}`",
        f"- Test datasets: `{', '.join(args.test_datasets)}`",
        f"- Tolerance: `max({args.min_abs_tol}, {args.sigma_multiplier} * pooled_std)`",
        "",
        "## Per-Parameter Summary",
        "",
        "| Param | n | max ΔAUROC | max ΔAUPRC | AUROC pass | AUPRC pass | r(AUROC) | r(AUPRC) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison_summary:
        lines.append(
            "| {param} | {n} | {max_auc:.6f} | {max_pr:.6f} | {auc_pass:.2f} | "
            "{pr_pass:.2f} | {auc_r:.3f} | {pr_r:.3f} |".format(
                param=row["param"],
                n=row["n"],
                max_auc=float(row["max_abs_delta_AUROC"]),
                max_pr=float(row["max_abs_delta_AUPRC"]),
                auc_pass=float(row["pass_rate_AUROC"]),
                pr_pass=float(row["pass_rate_AUPRC"]),
                auc_r=float(row["pearson_AUROC"]),
                pr_r=float(row["pearson_AUPRC"]),
            )
        )

    lines.extend(
        [
            "",
            "## Largest Pointwise Deltas",
            "",
            "| Param | Value | ΔAUROC | ΔAUPRC | AUROC ok | AUPRC ok |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in worst_rows:
        lines.append(
            "| {param} | {value} | {d_auc:.6f} | {d_pr:.6f} | {auc_ok} | {pr_ok} |".format(
                param=row["param"],
                value=row["value"],
                d_auc=float(row["delta_AUROC"]),
                d_pr=float(row["delta_AUPRC"]),
                auc_ok=row["AUROC_within_tolerance"],
                pr_ok=row["AUPRC_within_tolerance"],
            )
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a compact paired check that changing default lambda_E from 0.1 "
            "to 0.0 does not materially alter existing sensitivity conclusions."
        )
    )
    parser.add_argument("--base-config", type=Path, default=DEFAULT_REFERENCE_CONFIG)
    parser.add_argument("--reference-summary", type=Path, default=DEFAULT_REFERENCE_SUMMARY)
    parser.add_argument("--reference-lambda-e", type=float, default=0.1)
    parser.add_argument("--probe-lambda-e", type=float, default=0.0)
    parser.add_argument("--params", nargs="+", default=DEFAULT_PARAMS)
    parser.add_argument("--probe-values-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--train-datasets", nargs="+", default=DEFAULT_TRAIN_DATASETS)
    parser.add_argument("--test-datasets", nargs="+", default=DEFAULT_TEST_DATASETS)
    parser.add_argument("--knn-cache-dir", type=Path, default=PROJECT_ROOT / "knn_cache")
    parser.add_argument("--knn-search-dtype", type=str, default="auto")
    parser.add_argument("--disable-knn-cache", action="store_true")
    parser.add_argument("--sigma-multiplier", type=float, default=2.0)
    parser.add_argument("--min-abs-tol", type=float, default=0.005)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = (
            PROJECT_ROOT
            / "tuning_hyperparams"
            / "lambda_e0_invariance_results"
            / f"lambda_e0_{timestamp}"
        )
    args.output_dir = args.output_dir.resolve()
    args.knn_cache_dir = args.knn_cache_dir.resolve()
    args.knn_cache_enabled = not args.disable_knn_cache

    base_config = sync_aliases(load_json(args.base_config))
    base_config["lambda_E"] = args.probe_lambda_e
    base_config = sync_aliases(base_config)
    probe_values = load_probe_values(args.probe_values_json)
    jobs = build_probe_jobs(base_config, args.params, probe_values)

    manifest = {
        "base_config": str(args.base_config),
        "reference_summary": str(args.reference_summary),
        "output_dir": str(args.output_dir),
        "reference_lambda_E": args.reference_lambda_e,
        "probe_lambda_E": args.probe_lambda_e,
        "device": args.device,
        "epochs": args.epochs,
        "trials": args.trials,
        "params": args.params,
        "probe_values": {param: values_for_param(base_config, probe_values, param) for param in args.params},
        "train_datasets": args.train_datasets,
        "test_datasets": args.test_datasets,
        "knn_cache_dir": str(args.knn_cache_dir),
        "knn_search_dtype": args.knn_search_dtype,
        "jobs": [{key: value for key, value in job.items() if key != "config"} for job in jobs],
    }
    write_json(args.output_dir / "manifest.json", manifest)
    write_json(args.output_dir / "base_config_lambda_E0.json", base_config)
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

    model_name = base_config.get("model", "recap")
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
            if not rows:
                checkpoint_dir = args.output_dir / "checkpoints" / job["run_id"]
                rows = collect_checkpoint_rows(job, checkpoint_dir, args.trials, model_name)
            raw_rows.extend(rows)
            write_csv(
                args.output_dir / "raw_results.csv",
                raw_rows,
                [
                    "run_id",
                    "kind",
                    "param",
                    "value",
                    "trial",
                    "dataset",
                    "AUROC",
                    "AUPRC",
                    "checkpoint",
                ],
            )

    run_summary = summarize_runs(raw_rows)
    probe_summary = build_probe_summary(run_summary, base_config, args.params, probe_values)
    reference_summary = load_summary_csv(args.reference_summary)
    comparison_rows = compare_summaries(
        reference_summary,
        probe_summary,
        sigma_multiplier=args.sigma_multiplier,
        min_abs_tol=args.min_abs_tol,
    )
    comparison_summary = summarize_comparison(comparison_rows)

    write_csv(
        args.output_dir / "summary_by_run.csv",
        run_summary,
        [
            "run_id",
            "kind",
            "param",
            "value",
            "AUROC_mean",
            "AUROC_std",
            "AUPRC_mean",
            "AUPRC_std",
            "num_trials",
        ],
    )
    write_csv(
        args.output_dir / "lambda_E0_probe_summary.csv",
        probe_summary,
        ["param", "value", "is_baseline", "AUROC_mean", "AUROC_std", "AUPRC_mean", "AUPRC_std"],
    )
    write_csv(
        args.output_dir / "lambda_E0_vs_reference.csv",
        comparison_rows,
        [
            "param",
            "value",
            "is_baseline",
            "ref_AUROC_mean",
            "lambda_E0_AUROC_mean",
            "delta_AUROC",
            "abs_delta_AUROC",
            "ref_AUROC_std",
            "lambda_E0_AUROC_std",
            "pooled_AUROC_std",
            "AUROC_within_tolerance",
            "ref_AUPRC_mean",
            "lambda_E0_AUPRC_mean",
            "delta_AUPRC",
            "abs_delta_AUPRC",
            "ref_AUPRC_std",
            "lambda_E0_AUPRC_std",
            "pooled_AUPRC_std",
            "AUPRC_within_tolerance",
        ],
    )
    write_csv(
        args.output_dir / "lambda_E0_invariance_summary.csv",
        comparison_summary,
        [
            "param",
            "n",
            "max_abs_delta_AUROC",
            "mean_abs_delta_AUROC",
            "pass_rate_AUROC",
            "pearson_AUROC",
            "max_abs_delta_AUPRC",
            "mean_abs_delta_AUPRC",
            "pass_rate_AUPRC",
            "pearson_AUPRC",
        ],
    )
    write_markdown_report(
        args.output_dir / "lambda_E0_invariance_summary.md",
        args,
        base_config,
        comparison_rows,
        comparison_summary,
    )
    print(f"\nlambda_E=0 invariance check complete: {args.output_dir}")


if __name__ == "__main__":
    main()
