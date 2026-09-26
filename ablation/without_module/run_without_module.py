#!/usr/bin/env python3
"""Run without-module ablations for RECAP.

The script reuses the production detector/model and injects narrowly scoped
runtime variants. This keeps the ablation faithful to the current code path
while avoiding permanent changes to the main implementation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MethodType

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import create_default_configs  # noqa: E402
from detector import recapDetector  # noqa: E402
from utils import prepare_datasets, set_seed  # noqa: E402


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

METHODS = (
    "RECAP",
    "RECAP w/o Residual",
    "RECAP w/o L_con",
    "RECAP w/o L_H",
    "RECAP w/o Adhesion Score",
    "RECAP w/o Context Score",
    "RECAP C=1",
)


@dataclass
class MethodResult:
    method: str
    trial: int
    dataset: str
    AUROC: float
    AUPRC: float
    train_seconds: float
    eval_seconds: float
    status: str = "ok"
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RECAP without-module ablation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", default="cpu", help="Training/evaluation device")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--dims", type=int, default=32)
    parser.add_argument("--model", default="recap_auprc_best")
    parser.add_argument("--json-dir", default="params")
    parser.add_argument("--output-dir", default="ablation/without_module/results")
    parser.add_argument("--train-datasets", nargs="+", default=None)
    parser.add_argument("--test-datasets", nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--num-clusters", type=int, default=None)
    parser.add_argument("--knn-k", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--cluster-lr-multiplier", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--no-diagnostics", action="store_true")
    parser.add_argument("--diagnostics-interval", type=int, default=10)
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke run: one source graph, two target graphs, one seed, <=2 epochs",
    )
    return parser.parse_args()


def apply_quick_mode(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.trials = 1
    args.epochs = min(args.epochs, 2)
    if args.train_datasets is None:
        args.train_datasets = ["pubmed"]
    if args.test_datasets is None:
        args.test_datasets = ["Facebook", "cora"]


def ensure_device_available(device: str) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device={device}, but CUDA is unavailable. Use --device cpu here."
        )


def apply_config_overrides(model_config, args: argparse.Namespace) -> None:
    if args.num_clusters is not None:
        model_config.num_clusters = args.num_clusters
    if args.knn_k is not None:
        model_config.knn_k = args.knn_k
    if args.lr is not None:
        model_config.lr = args.lr
    if args.cluster_lr_multiplier is not None:
        model_config.cluster_lr_multiplier = args.cluster_lr_multiplier
    if args.gamma is not None:
        model_config.gamma = args.gamma
        model_config.min_cluster_ratio = args.gamma


def non_residual_embed(self, hop_list):
    if len(hop_list) - 1 != self.num_hops:
        raise ValueError(
            f"hop_list length should be num_hops+1={self.num_hops + 1}, got {len(hop_list)}"
        )

    processed = list(hop_list)
    for i, layer in enumerate(self.layers):
        if i != 0:
            processed = [self.dropout(x) for x in processed]
        processed = [layer(x) for x in processed]
        if i != len(self.layers) - 1:
            processed = [self.act(x) for x in processed]
    return torch.hstack(processed[1:])


def non_residual_initial_embed(self, hop_list):
    if len(hop_list) - 1 != self.num_hops:
        raise ValueError(
            f"hop_list length should be num_hops+1={self.num_hops + 1}, got {len(hop_list)}"
        )
    return torch.hstack(hop_list[1:])


def compute_losses_without_l_con(self, E, E_init=None, cache_key=None):
    del E_init, cache_key
    H = self.cluster(E)
    l_H, _, _ = self._compute_H_loss(H)
    l_var = self._compute_var_loss(E) if self.lambda_E != 0 else E.new_tensor(0.0)
    return self.lambda_H * l_H + self.lambda_E * l_var


def compute_losses_without_l_H(self, E, E_init=None, cache_key=None):
    edge_index, edge_weight = self.build_ego_graph(E, E_init=E_init, cache_key=cache_key)
    H = self.cluster(E)
    l_con = self._compute_con_loss(H, edge_index, edge_weight)
    l_var = self._compute_var_loss(E) if self.lambda_E != 0 else E.new_tensor(0.0)
    return l_con + self.lambda_E * l_var


def compute_scores_without_adhesion(self, E, E_init=None, cache_key=None):
    return self.compute_score_components(E, E_init=E_init, cache_key=cache_key)["scale"]


def compute_scores_without_context(self, E, E_init=None, cache_key=None):
    return self.compute_score_components(E, E_init=E_init, cache_key=cache_key)["adhesion"]


def inject_ablation(detector: recapDetector, method: str) -> None:
    model = detector.model

    if method == "RECAP":
        return

    if method == "RECAP C=1":
        return

    if method == "RECAP w/o Residual":
        model._compute_residual_embed = MethodType(non_residual_embed, model)
        model._compute_initial_residual_embed = MethodType(non_residual_initial_embed, model)
        return

    if method == "RECAP w/o Adhesion Score":
        for cluster in model.ego_clusters:
            cluster.compute_scores = MethodType(compute_scores_without_adhesion, cluster)
        return

    if method == "RECAP w/o Context Score":
        model.beta = 0.0
        for cluster in model.ego_clusters:
            cluster.beta = 0.0
            cluster.compute_scores = MethodType(compute_scores_without_context, cluster)
        return

    if method == "RECAP w/o L_con":
        replacement = compute_losses_without_l_con
    elif method == "RECAP w/o L_H":
        replacement = compute_losses_without_l_H
    else:
        raise ValueError(f"Unknown ablation method: {method}")

    for cluster in model.ego_clusters:
        cluster.compute_losses = MethodType(replacement, cluster)


def configure_method_model(base_model_config, method: str):
    model_config = copy.deepcopy(base_model_config)
    if method == "RECAP w/o Context Score":
        model_config.beta = 0.0
    elif method == "RECAP C=1":
        model_config.num_clusters = 1
        # With a single community, entropy range penalties become constants and
        # do not describe a meaningful module contribution.
        model_config.lambda_H = 0.0
        model_config.lambda_ortho = 0.0
        model_config.lambda_usage_entropy = 0.0
        model_config.assignment_entropy_lower = None
        model_config.assignment_entropy_upper = None
        model_config.usage_entropy_lower = None
        model_config.usage_entropy_upper = None
    return model_config


def run_one_method(
    method: str,
    trial: int,
    train_config,
    model_config,
    data_train: list,
    data_test: list,
    dataset_names: list[str],
    args: argparse.Namespace,
) -> list[MethodResult]:
    print(f"\n--- {method} | trial {trial} ---")
    trial_train_config = copy.deepcopy(train_config)
    trial_train_config.device = args.device
    trial_train_config.epochs = args.epochs
    trial_train_config.trials = args.trials
    trial_train_config.save_checkpoint = False
    trial_train_config.output_dir = args.output_dir
    trial_train_config.early_stop = args.early_stop
    trial_train_config.patience = args.patience
    trial_train_config.log_diagnostics = not args.no_diagnostics
    trial_train_config.diagnostics_interval = args.diagnostics_interval

    data = {"train": data_train, "test": data_test}
    detector = recapDetector(trial_train_config, model_config, data)
    inject_ablation(detector, method)

    train_start = time.perf_counter()
    detector.train(verbose=True)
    train_seconds = time.perf_counter() - train_start

    eval_start = time.perf_counter()
    scores = detector.evaluate(data_list=data_test, dataset_names=dataset_names, verbose=True)
    eval_seconds = time.perf_counter() - eval_start
    per_dataset_eval = eval_seconds / max(1, len(dataset_names))

    rows = []
    for dataset_name in dataset_names:
        metric = scores[dataset_name]
        rows.append(
            MethodResult(
                method=method,
                trial=trial,
                dataset=dataset_name,
                AUROC=float(metric["AUROC"]),
                AUPRC=float(metric["AUPRC"]),
                train_seconds=train_seconds,
                eval_seconds=per_dataset_eval,
            )
        )
    print(f"  {method} train_seconds={train_seconds:.1f} eval_seconds={eval_seconds:.1f}")
    return rows


def run_experiment(args: argparse.Namespace, train_config, base_model_config, data_train, data_test, dataset_names):
    records: list[MethodResult] = []
    for trial in range(args.trials):
        set_seed(trial)
        print("\n" + "=" * 70)
        print(f"Trial {trial}")
        print("=" * 70)
        for method in args.methods:
            set_seed(trial)
            model_config = configure_method_model(base_model_config, method)
            try:
                records.extend(
                    run_one_method(
                        method=method,
                        trial=trial,
                        train_config=train_config,
                        model_config=model_config,
                        data_train=data_train,
                        data_test=data_test,
                        dataset_names=dataset_names,
                        args=args,
                    )
                )
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                print(f"  {method} FAILED: {exc}")
                for dataset_name in dataset_names:
                    records.append(
                        MethodResult(
                            method=method,
                            trial=trial,
                            dataset=dataset_name,
                            AUROC=math.nan,
                            AUPRC=math.nan,
                            train_seconds=math.nan,
                            eval_seconds=math.nan,
                            status="failed",
                            error=str(exc),
                        )
                    )
    return records


def summarize(records: list[MethodResult]) -> list[dict]:
    grouped: dict[tuple[str, str], list[MethodResult]] = defaultdict(list)
    for record in records:
        grouped[(record.method, record.dataset)].append(record)

    rows = []
    for (method, dataset), items in sorted(grouped.items()):
        aurocs = np.asarray([r.AUROC for r in items], dtype=np.float64)
        auprcs = np.asarray([r.AUPRC for r in items], dtype=np.float64)
        train_seconds = np.asarray([r.train_seconds for r in items], dtype=np.float64)
        eval_seconds = np.asarray([r.eval_seconds for r in items], dtype=np.float64)
        rows.append(
            {
                "method": method,
                "dataset": dataset,
                "n": int(np.sum(~np.isnan(aurocs))),
                "AUROC_mean": float(np.nanmean(aurocs)),
                "AUROC_std": float(np.nanstd(aurocs)),
                "AUPRC_mean": float(np.nanmean(auprcs)),
                "AUPRC_std": float(np.nanstd(auprcs)),
                "train_seconds_mean": float(np.nanmean(train_seconds)),
                "eval_seconds_mean": float(np.nanmean(eval_seconds)),
            }
        )
    return rows


def summarize_average(records: list[MethodResult]) -> list[dict]:
    per_trial: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(
        lambda: {"AUROC": [], "AUPRC": []}
    )
    for record in records:
        if not math.isnan(record.AUROC):
            per_trial[(record.method, record.trial)]["AUROC"].append(record.AUROC)
        if not math.isnan(record.AUPRC):
            per_trial[(record.method, record.trial)]["AUPRC"].append(record.AUPRC)

    by_method: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"AUROC": [], "AUPRC": []})
    for (method, _trial), values in per_trial.items():
        if values["AUROC"]:
            by_method[method]["AUROC"].append(float(np.mean(values["AUROC"])))
        if values["AUPRC"]:
            by_method[method]["AUPRC"].append(float(np.mean(values["AUPRC"])))

    rows = []
    for method, values in sorted(by_method.items()):
        aurocs = np.asarray(values["AUROC"], dtype=np.float64)
        auprcs = np.asarray(values["AUPRC"], dtype=np.float64)
        rows.append(
            {
                "method": method,
                "dataset": "Average",
                "n": int(len(aurocs)),
                "AUROC_mean": float(np.nanmean(aurocs)),
                "AUROC_std": float(np.nanstd(aurocs)),
                "AUPRC_mean": float(np.nanmean(auprcs)),
                "AUPRC_std": float(np.nanstd(auprcs)),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def metric_cell(row: dict, metric: str) -> str:
    return f"{row[f'{metric}_mean']:.4f} +/- {row[f'{metric}_std']:.4f}"


def write_markdown(path: Path, average_rows: list[dict], summary_rows: list[dict]) -> None:
    lines = [
        "# Without-Module Ablation Results",
        "",
        "## Average Across Target Graphs",
        "",
        "| Method | AUROC | AUPRC | Seeds |",
        "|---|---:|---:|---:|",
    ]
    for row in average_rows:
        lines.append(
            f"| {row['method']} | {metric_cell(row, 'AUROC')} | "
            f"{metric_cell(row, 'AUPRC')} | {row['n']} |"
        )

    lines.extend(
        [
            "",
            "## Per Dataset",
            "",
            "| Method | Dataset | AUROC | AUPRC | Seeds |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {row['dataset']} | {metric_cell(row, 'AUROC')} | "
            f"{metric_cell(row, 'AUPRC')} | {row['n']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def write_outputs(output_dir: Path, args: argparse.Namespace, model_config, records: list[MethodResult]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = summarize(records)
    average_rows = summarize_average(records)

    payload = {
        "config": {
            **vars(args),
            "project_root": str(PROJECT_ROOT),
            "model_config": model_config.to_dict(),
        },
        "records": [asdict(record) for record in records],
        "summary": summary_rows,
        "average": average_rows,
    }
    raw_path = output_dir / "without_module_raw.json"
    summary_path = output_dir / "without_module_summary.csv"
    average_path = output_dir / "without_module_average.csv"
    markdown_path = output_dir / "without_module_table.md"

    raw_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    write_csv(summary_path, summary_rows)
    write_csv(average_path, average_rows)
    write_markdown(markdown_path, average_rows, summary_rows)

    print("\n" + "=" * 70)
    print("Saved outputs")
    print("=" * 70)
    print(f"  Raw JSON : {raw_path}")
    print(f"  Summary  : {summary_path}")
    print(f"  Average  : {average_path}")
    print(f"  Markdown : {markdown_path}")


def main() -> None:
    args = parse_args()
    apply_quick_mode(args)
    ensure_device_available(args.device)
    os.chdir(PROJECT_ROOT)

    train_datasets = args.train_datasets or DEFAULT_TRAIN_DATASETS
    test_datasets = args.test_datasets or DEFAULT_TEST_DATASETS

    train_config, model_config = create_default_configs(
        model_name=args.model,
        json_dir=args.json_dir,
        dims=args.dims,
    )
    apply_config_overrides(model_config, args)

    print("\n" + "=" * 70)
    print("Without-Module Ablation")
    print("=" * 70)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Methods      : {args.methods}")
    print(f"Train graphs : {train_datasets}")
    print(f"Test graphs  : {test_datasets}")
    print(f"Trials       : {args.trials}")
    print(f"Epochs       : {args.epochs}")
    print(f"Device       : {args.device}")

    data_train, data_test = prepare_datasets(
        dims=args.dims,
        train_datasets=train_datasets,
        test_datasets=test_datasets,
        num_hops=model_config.num_hops,
    )

    records = run_experiment(
        args=args,
        train_config=train_config,
        base_model_config=model_config,
        data_train=data_train,
        data_test=data_test,
        dataset_names=test_datasets,
    )
    write_outputs(Path(args.output_dir), args, model_config, records)


if __name__ == "__main__":
    main()
