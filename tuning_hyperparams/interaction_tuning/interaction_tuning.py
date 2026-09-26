from __future__ import annotations

import argparse
import csv
import json
import re
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

TUNED_KEYS = [
    "knn_k",
    "num_clusters",
    "num_hops",
    "tau_s",
    "tau_c",
    "cluster_init_gain",
    "beta",
    "lambda_H",
    "lambda_bal",
    "lambda_E",
    "lambda_usage_entropy",
    "assignment_entropy_lower",
    "assignment_entropy_upper",
    "usage_entropy_lower",
    "usage_entropy_upper",
]
JOB_TUNED_KEYS = ["epochs"]

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


def tuned_subset(config: dict) -> dict:
    return {key: config.get(key) for key in TUNED_KEYS if key in config}


def make_config(base_config: dict, overrides: dict) -> dict:
    config = deepcopy(base_config)
    config.update(overrides)
    return sync_aliases(config)


def add_job(
    jobs: list[dict],
    seen: set[tuple],
    base_config: dict,
    stage: str,
    run_id: str,
    overrides: dict,
    train_overrides: dict | None = None,
) -> None:
    train_overrides = train_overrides or {}
    config = make_config(base_config, overrides)
    key = tuple((name, config.get(name)) for name in TUNED_KEYS)
    key += tuple((f"train_{name}", train_overrides.get(name)) for name in JOB_TUNED_KEYS)
    if key in seen:
        return
    seen.add(key)
    model_overrides = {key: config.get(key) for key in TUNED_KEYS}
    jobs.append(
        {
            "run_id": run_id,
            "stage": stage,
            "overrides": {**model_overrides, **train_overrides},
            "model_overrides": model_overrides,
            "train_overrides": train_overrides,
            "config": config,
        }
    )


def build_entropy_jobs(base_config: dict) -> list[dict]:
    """Build a compact interaction search for the entropy-band community objective.

    The current community objective has two known failure modes:
    - no symmetry break: assignments remain nearly uniform;
    - over-break: assignments collapse to a few communities.

    The search is therefore intentionally centered on coupled community profiles
    instead of a wide Cartesian sweep. It keeps graph anchors small, then studies
    the interaction between initialization strength, community regularization,
    usage-entropy guardrails, and the downstream context-score weight.
    """
    jobs: list[dict] = []
    seen: set[tuple] = set()

    add_job(jobs, seen, base_config, "baseline", "baseline", tuned_subset(base_config))

    common = {
        "knn_k": 20,
        "num_clusters": 28,
        "num_hops": 4,
        "tau_s": 0.05,
        "tau_c": 0.3,
        "cluster_init_gain": 2.0,
        "beta": 0.02,
        "lambda_H": 0.35,
        "lambda_bal": 0.0,
        "lambda_E": 0.0,
        "lambda_usage_entropy": 0.5,
        "assignment_entropy_lower": 0.45,
        "assignment_entropy_upper": 0.85,
        "usage_entropy_lower": 0.65,
        "usage_entropy_upper": 0.9,
    }

    # Stage 1: graph anchors. Previous searches liked k=20,C=28,L=4 and
    # k=24,C=36,L=3; add two nearby controls without exploding the grid.
    for knn_k, num_clusters, num_hops in [
        (20, 28, 4),
        (24, 36, 3),
        (16, 28, 4),
        (20, 36, 3),
    ]:
        overrides = {
            **common,
            "knn_k": knn_k,
            "num_clusters": num_clusters,
            "num_hops": num_hops,
        }
        run_id = f"graph__k{knn_k}__C{num_clusters}__L{num_hops}"
        add_job(jobs, seen, base_config, "graph_anchor", run_id, overrides)

    anchor = dict(common)

    # Stage 2: coupled community profiles. These are curated profiles, not a
    # Cartesian product: the goal is to find a stable middle between uniform and
    # few-community collapse.
    community_profiles = [
        {
            "name": "soft_break",
            "cluster_init_gain": 1.5,
            "lambda_H": 0.35,
            "lambda_usage_entropy": 0.35,
        },
        {
            "name": "gain2_lH025",
            "cluster_init_gain": 2.0,
            "lambda_H": 0.25,
            "lambda_usage_entropy": 0.5,
        },
        {
            "name": "gain2_lH035",
            "cluster_init_gain": 2.0,
            "lambda_H": 0.35,
            "lambda_usage_entropy": 0.5,
        },
        {
            "name": "gain2_lH05",
            "cluster_init_gain": 2.0,
            "lambda_H": 0.5,
            "lambda_usage_entropy": 0.5,
        },
        {
            "name": "gain25_lH035",
            "cluster_init_gain": 2.5,
            "lambda_H": 0.35,
            "lambda_usage_entropy": 0.5,
        },
        {
            "name": "gain2_loose_usage",
            "cluster_init_gain": 2.0,
            "lambda_H": 0.35,
            "lambda_usage_entropy": 0.25,
            "usage_entropy_lower": 0.6,
            "usage_entropy_upper": 0.92,
        },
        {
            "name": "gain2_strict_usage",
            "cluster_init_gain": 2.0,
            "lambda_H": 0.35,
            "lambda_usage_entropy": 0.75,
            "usage_entropy_lower": 0.72,
            "usage_entropy_upper": 0.92,
        },
        {
            "name": "gain2_soft_assign",
            "cluster_init_gain": 2.0,
            "lambda_H": 0.35,
            "lambda_usage_entropy": 0.5,
            "assignment_entropy_lower": 0.55,
            "assignment_entropy_upper": 0.9,
        },
        {
            "name": "gain2_hard_assign",
            "cluster_init_gain": 2.0,
            "lambda_H": 0.35,
            "lambda_usage_entropy": 0.5,
            "assignment_entropy_lower": 0.35,
            "assignment_entropy_upper": 0.8,
        },
    ]
    for profile in community_profiles:
        name = profile["name"]
        overrides = {**anchor, **{k: v for k, v in profile.items() if k != "name"}}
        run_id = f"comm__{name}"
        add_job(jobs, seen, base_config, "community_profile", run_id, overrides)

    # Stage 3: re-calibrate beta after community assignments stop being uniform.
    beta_anchor = {
        **anchor,
        "cluster_init_gain": 2.0,
        "lambda_H": 0.35,
        "lambda_usage_entropy": 0.5,
    }
    for beta in [0, 0.005, 0.01, 0.02, 0.05]:
        overrides = {**beta_anchor, "beta": beta}
        run_id = f"beta_recal__b{value_token(beta)}"
        add_job(jobs, seen, base_config, "beta_recalibration", run_id, overrides)

    # Stage 4: light controls around the selected profile. lambda_E is kept
    # dormant globally, so these controls focus on active graph/community knobs.
    for tau_s in [0.03, 0.05, 0.08]:
        overrides = {**beta_anchor, "tau_s": tau_s}
        run_id = f"ctrl__tau{value_token(tau_s)}"
        add_job(jobs, seen, base_config, "controls", run_id, overrides)

    return jobs


def build_performance_v2_jobs(base_config: dict) -> list[dict]:
    """Build a performance-recovery search with non-trivial beta.

    This plan assumes the entropy-band objective is structurally useful but too
    strong when trained for too long. It therefore studies training length,
    lighter community regularization, and beta values that are still large
    enough to demonstrate the local community-difference score.
    """
    jobs: list[dict] = []
    seen: set[tuple] = set()

    common = {
        "knn_k": 20,
        "num_clusters": 28,
        "num_hops": 4,
        "tau_s": 0.08,
        "tau_c": 0.3,
        "cluster_init_gain": 2.0,
        "beta": 0.02,
        "lambda_H": 0.2,
        "lambda_bal": 0.0,
        "lambda_E": 0.0,
        "lambda_usage_entropy": 0.25,
        "assignment_entropy_lower": 0.45,
        "assignment_entropy_upper": 0.85,
        "usage_entropy_lower": 0.65,
        "usage_entropy_upper": 0.9,
    }

    add_job(
        jobs,
        seen,
        base_config,
        "baseline",
        "baseline_perf_v2",
        common,
        {"epochs": 100},
    )

    # Stage 1: training length. Prior runs suggested 200 epochs hurts ranking
    # after community structure has already stabilized.
    for epochs in [80, 100, 120, 150]:
        run_id = f"epoch__e{epochs}"
        add_job(
            jobs,
            seen,
            base_config,
            "epoch_budget",
            run_id,
            common,
            {"epochs": epochs},
        )

    # Stage 2: lighter community regularization profiles. Keep beta non-trivial.
    profiles = [
        {
            "name": "light_lH01_lU01",
            "lambda_H": 0.1,
            "lambda_usage_entropy": 0.1,
            "cluster_init_gain": 1.5,
        },
        {
            "name": "light_lH02_lU01",
            "lambda_H": 0.2,
            "lambda_usage_entropy": 0.1,
            "cluster_init_gain": 2.0,
        },
        {
            "name": "mid_lH02_lU025",
            "lambda_H": 0.2,
            "lambda_usage_entropy": 0.25,
            "cluster_init_gain": 2.0,
        },
        {
            "name": "mid_lH035_lU025",
            "lambda_H": 0.35,
            "lambda_usage_entropy": 0.25,
            "cluster_init_gain": 2.0,
        },
        {
            "name": "gain25_lH02_lU025",
            "lambda_H": 0.2,
            "lambda_usage_entropy": 0.25,
            "cluster_init_gain": 2.5,
        },
        {
            "name": "soft_assign_lH02",
            "lambda_H": 0.2,
            "lambda_usage_entropy": 0.25,
            "assignment_entropy_lower": 0.55,
            "assignment_entropy_upper": 0.9,
        },
        {
            "name": "hard_assign_lH02",
            "lambda_H": 0.2,
            "lambda_usage_entropy": 0.25,
            "assignment_entropy_lower": 0.35,
            "assignment_entropy_upper": 0.8,
        },
        {
            "name": "loose_usage_lH02",
            "lambda_H": 0.2,
            "lambda_usage_entropy": 0.1,
            "usage_entropy_lower": 0.6,
            "usage_entropy_upper": 0.92,
        },
    ]
    for profile in profiles:
        name = profile["name"]
        overrides = {**common, **{k: v for k, v in profile.items() if k != "name"}}
        run_id = f"perf__{name}"
        add_job(
            jobs,
            seen,
            base_config,
            "regularization_profile",
            run_id,
            overrides,
            {"epochs": 100},
        )

    # Stage 3: beta recalibration, excluding beta=0 and very tiny beta. The
    # point is to retain a visible local community-difference contribution.
    beta_anchor = {
        **common,
        "lambda_H": 0.2,
        "lambda_usage_entropy": 0.25,
        "cluster_init_gain": 2.0,
    }
    for beta in [0.02, 0.03, 0.05, 0.08]:
        run_id = f"beta_nontrivial__b{value_token(beta)}"
        add_job(
            jobs,
            seen,
            base_config,
            "beta_nontrivial",
            run_id,
            {**beta_anchor, "beta": beta},
            {"epochs": 100},
        )

    graph_anchor = {**beta_anchor, "beta": 0.02}

    # Stage 4: community softmax temperature. Keep this narrow because tau_c
    # has been the most sensitive parameter around the uniform/collapse edge.
    for tau_c in [0.28, 0.3, 0.32, 0.35]:
        run_id = f"tauc_perf__tc{value_token(tau_c)}"
        add_job(
            jobs,
            seen,
            base_config,
            "tau_c_perf",
            run_id,
            {**graph_anchor, "tau_c": tau_c},
            {"epochs": 100},
        )

    # Stage 5: small graph and tau_s controls around the likely performance
    # profile. Keep this narrow so the round remains affordable.
    for knn_k, num_clusters, num_hops in [
        (16, 28, 4),
        (20, 28, 4),
        (24, 28, 4),
        (20, 36, 3),
    ]:
        run_id = f"graph_perf__k{knn_k}__C{num_clusters}__L{num_hops}"
        add_job(
            jobs,
            seen,
            base_config,
            "graph_perf",
            run_id,
            {
                **graph_anchor,
                "knn_k": knn_k,
                "num_clusters": num_clusters,
                "num_hops": num_hops,
            },
            {"epochs": 100},
        )

    for tau_s in [0.05, 0.08, 0.1, 0.12]:
        run_id = f"tau_perf__tau{value_token(tau_s)}"
        add_job(
            jobs,
            seen,
            base_config,
            "tau_perf",
            run_id,
            {**graph_anchor, "tau_s": tau_s},
            {"epochs": 100},
        )

    return jobs


def build_jobs(base_config: dict, plan: str) -> list[dict]:
    if plan == "entropy":
        return build_entropy_jobs(base_config)
    if plan == "performance_v2":
        return build_performance_v2_jobs(base_config)
    raise ValueError(f"Unknown interaction tuning plan: {plan}")


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
                    "stage": job["stage"],
                    "trial": trial,
                    "dataset": dataset,
                    "AUROC": float(scores["AUROC"]),
                    "AUPRC": float(scores["AUPRC"]),
                    "overrides": json.dumps(job["overrides"], sort_keys=True),
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
    job_epochs = int(job.get("train_overrides", {}).get("epochs", args.epochs))

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
        str(job_epochs),
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
    print(f"Running {job['run_id']} ({job['stage']})")
    print(json.dumps(job["overrides"], sort_keys=True))
    print(f"epochs={job_epochs}")
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


DIAGNOSTIC_FIELDS = [
    "H_ent_norm",
    "H_maxp",
    "pi_ent_norm",
    "pi_std",
    "effC",
    "Lvar",
    "Lvar_active",
    "scale_adh_std",
]


def parse_diagnostic_line(line: str) -> dict[str, float]:
    payload = line.split("Diagnostics:", 1)[1]
    payload = payload.replace("scale/adh_std", "scale_adh_std")
    values = {}
    for key, value in re.findall(r"([A-Za-z0-9_]+)=([-+0-9.eE]+)", payload):
        values[key] = float(value)
    return values


def parse_final_diagnostics(log_path: Path) -> list[dict[str, float]]:
    """Return the last diagnostics line for each trial in a training log."""
    if not log_path.exists():
        return []

    trial_re = re.compile(r"Trial\s+(\d+)\s+开始训练")
    current_trial: int | None = None
    by_trial: dict[int, dict[str, float]] = {}

    with open(log_path, "r") as f:
        for line in f:
            trial_match = trial_re.search(line)
            if trial_match:
                current_trial = int(trial_match.group(1))
                continue
            if "Diagnostics:" not in line or current_trial is None:
                continue
            by_trial[current_trial] = parse_diagnostic_line(line)

    return [by_trial[idx] for idx in sorted(by_trial)]


def summarize_diagnostics(jobs: list[dict], output_dir: Path) -> list[dict]:
    rows = []
    for job in jobs:
        diagnostics = parse_final_diagnostics(output_dir / "logs" / f"{job['run_id']}.log")
        row = {
            "run_id": job["run_id"],
            "stage": job["stage"],
            "num_diag_trials": len(diagnostics),
            "overrides": json.dumps(job["overrides"], sort_keys=True),
        }
        for field in DIAGNOSTIC_FIELDS:
            vals = [diag[field] for diag in diagnostics if field in diag]
            row[f"{field}_mean"] = mean(vals)
            row[f"{field}_std"] = std(vals)
        rows.append(row)
    return rows


def summarize_runs(raw_rows: list[dict], auroc_weight: float, auprc_weight: float) -> list[dict]:
    by_run_trial = {}
    metadata = {}
    for row in raw_rows:
        run_id = row["run_id"]
        metadata[run_id] = {
            "run_id": run_id,
            "stage": row["stage"],
            "overrides": row["overrides"],
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
        auroc_mean = mean(by_run[run_id]["AUROC"])
        auprc_mean = mean(by_run[run_id]["AUPRC"])
        row = dict(metadata[run_id])
        row.update(
            {
                "selection_score": auroc_weight * auroc_mean + auprc_weight * auprc_mean,
                "AUROC_mean": auroc_mean,
                "AUROC_std": std(by_run[run_id]["AUROC"]),
                "AUPRC_mean": auprc_mean,
                "AUPRC_std": std(by_run[run_id]["AUPRC"]),
                "num_trials": len(by_run[run_id]["AUROC"]),
            }
        )
        summary.append(row)

    summary.sort(key=lambda row: float(row["selection_score"]), reverse=True)
    for rank, row in enumerate(summary, start=1):
        row["rank"] = rank
    return summary


def write_best_config(output_dir: Path, summary: list[dict], base_config: dict) -> None:
    if not summary:
        return
    best = summary[0]
    overrides = json.loads(best["overrides"])
    model_overrides = {key: value for key, value in overrides.items() if key in TUNED_KEYS}
    train_overrides = {key: value for key, value in overrides.items() if key in JOB_TUNED_KEYS}
    best_config = make_config(base_config, model_overrides)
    write_json(output_dir / "best_config.json", best_config)
    write_json(
        output_dir / "best_result.json",
        {
            "run_id": best["run_id"],
            "stage": best["stage"],
            "rank": best["rank"],
            "selection_score": best["selection_score"],
            "AUROC_mean": best["AUROC_mean"],
            "AUROC_std": best["AUROC_std"],
            "AUPRC_mean": best["AUPRC_mean"],
            "AUPRC_std": best["AUPRC_std"],
            "overrides": model_overrides,
            "train_overrides": train_overrides,
        },
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run RECAP interaction hyperparameter tuning.")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument(
        "--plan",
        choices=["entropy", "performance_v2"],
        default="performance_v2",
        help="Tuning plan to run. performance_v2 keeps beta non-trivial and targets AUROC+AUPRC.",
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
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--auroc-weight", type=float, default=1.0)
    parser.add_argument("--auprc-weight", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = (
            PROJECT_ROOT
            / "tuning_hyperparams"
            / "interaction_tuning"
            / "results"
            / timestamp
        )
    args.output_dir = args.output_dir.resolve()
    args.knn_cache_dir = args.knn_cache_dir.resolve()
    args.knn_cache_enabled = not args.disable_knn_cache

    base_config = sync_aliases(load_json(args.base_config))
    jobs = build_jobs(base_config, args.plan)

    manifest = {
        "base_config": str(args.base_config),
        "plan": args.plan,
        "output_dir": str(args.output_dir),
        "device": args.device,
        "epochs": args.epochs,
        "trials": args.trials,
        "train_datasets": args.train_datasets,
        "test_datasets": args.test_datasets,
        "knn_cache_dir": str(args.knn_cache_dir),
        "knn_search_dtype": args.knn_search_dtype,
        "auroc_weight": args.auroc_weight,
        "auprc_weight": args.auprc_weight,
        "tuned_keys": TUNED_KEYS,
        "job_tuned_keys": JOB_TUNED_KEYS,
        "num_jobs": len(jobs),
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
                "stage": job["stage"],
                **job["overrides"],
                "overrides": json.dumps(job["overrides"], sort_keys=True),
            }
            for job in jobs
        ],
        ["run_id", "stage", *TUNED_KEYS, *JOB_TUNED_KEYS, "overrides"],
    )

    if args.dry_run:
        print(f"Dry run complete: wrote {len(jobs)} jobs to {args.output_dir}")
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
                    "stage",
                    "trial",
                    "dataset",
                    "AUROC",
                    "AUPRC",
                    "overrides",
                    "checkpoint",
                ],
            )

    summary = summarize_runs(raw_rows, args.auroc_weight, args.auprc_weight)
    diagnostics_summary = summarize_diagnostics(jobs, args.output_dir)
    diagnostics_by_run = {row["run_id"]: row for row in diagnostics_summary}
    diagnostic_summary_fields = [
        "num_diag_trials",
        *[
            item
            for field in DIAGNOSTIC_FIELDS
            for item in (f"{field}_mean", f"{field}_std")
        ],
    ]
    for row in summary:
        diag = diagnostics_by_run.get(row["run_id"], {})
        for field in diagnostic_summary_fields:
            row[field] = diag.get(field, "")

    write_csv(
        args.output_dir / "diagnostics_by_run.csv",
        diagnostics_summary,
        [
            "run_id",
            "stage",
            "num_diag_trials",
            *[
                item
                for field in DIAGNOSTIC_FIELDS
                for item in (f"{field}_mean", f"{field}_std")
            ],
            "overrides",
        ],
    )

    summary_fields = [
        "rank",
        "run_id",
        "stage",
        "selection_score",
        "AUROC_mean",
        "AUROC_std",
        "AUPRC_mean",
        "AUPRC_std",
        "num_trials",
        *diagnostic_summary_fields,
        "overrides",
    ]
    write_csv(
        args.output_dir / "summary_by_run.csv",
        summary,
        summary_fields,
    )
    write_csv(
        args.output_dir / "best_configs_top20.csv",
        summary[:20],
        summary_fields,
    )
    write_best_config(args.output_dir, summary, base_config)

    print(f"\nInteraction tuning complete: {args.output_dir}")
    if summary:
        print("Best run:")
        print(json.dumps(summary[0], indent=2))


if __name__ == "__main__":
    main()
