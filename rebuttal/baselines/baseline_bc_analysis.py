"""Audit and report the user-requested B/C baseline supplement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .baseline_analysis import (
    _calibrations,
    _csv,
    _lookup,
    _paired,
    _phase1_recap_macros,
    _timings,
    aggregate,
    audit_runs,
)
from .baseline_bc_protocol import (
    build_supplement_manifest,
    validate_supplement_manifest,
)
from .baseline_bc_runner import (
    DEFAULT_OUTPUT_ROOT,
    SUPPLEMENT_PROTOCOL_PATH,
)
from .baseline_common import atomic_json
from .baseline_protocol import DATASETS, SETTINGS
from .baseline_runner import PROJECT_ROOT


PRIMARY_ROOT = (
    PROJECT_ROOT / "rebuttal" / "artifacts" / "phase2_baselines"
)
PHASE1_MACRO_PATH = (
    PROJECT_ROOT
    / "rebuttal"
    / "artifacts"
    / "phase1"
    / "analysis"
    / "metric_macros.csv"
)
REPORT_PATH = (
    PROJECT_ROOT
    / "rebuttal"
    / "reports"
    / "PHASE2_BC_BASELINE_SUPPLEMENT_REPORT.md"
)
METHOD_ORDER = ("AnomalyGFM-ZS", "IA-GGAD", "ARC", "UNPrompt")


def _load_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{path}: expected a JSON list")
    return value


def render_report(
    supplement_dataset_rows: list[dict[str, Any]],
    supplement_macro_rows: list[dict[str, Any]],
    audit: dict[str, Any],
    primary_root: Path,
    output_root: Path,
) -> str:
    primary_dataset_rows = _load_json(
        primary_root / "analysis" / "dataset_summary.json"
    )
    primary_macro_rows = _load_json(
        primary_root / "analysis" / "macro_summary.json"
    )
    combined_dataset_rows = [
        row
        for row in primary_dataset_rows
        if row["setting"] in {"B", "C"}
        and row["method"] in {"ARC", "IA-GGAD"}
    ] + supplement_dataset_rows
    combined_macro_rows = [
        row
        for row in primary_macro_rows
        if row["setting"] in {"B", "C"}
        and row["method"] in {"ARC", "IA-GGAD"}
    ] + supplement_macro_rows
    recap_macros = _phase1_recap_macros(PHASE1_MACRO_PATH)
    calibrations = _calibrations(output_root)
    timings = _timings(
        _load_json(output_root / "analysis" / "raw_records.json")
    )

    lines = [
        "# Phase 2 B/C Baseline Completion Supplement",
        "",
        "Date: 2026-07-26",
        "",
        "Status: **PASS**" if audit["passed"] else "Status: **FAIL**",
        "",
        "This user-revised confirmatory supplement adds UNPrompt and",
        "AnomalyGFM-ZS to Settings B and C. Original ARC/IA-GGAD and RECAP",
        "artifacts are reused only for reporting; no original run was modified.",
        "All values are mean±population standard deviation over seeds 0/1/2,",
        "in percent.",
    ]

    for setting, title, macro_label in (
        ("B", "Setting B — Leave-Social-Domain-Out", "Social Macro"),
        ("C", "Setting C — Citation-only Source Transfer", "Dataset-Macro"),
    ):
        targets = SETTINGS[setting]["targets"]
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Method | "
                + " | ".join(DATASETS[name]["display"] for name in targets)
                + f" | {macro_label}"
                + (" | Domain-Macro |" if setting == "C" else " |"),
                "|---|" + "---:|" * (len(targets) + 1 + (setting == "C")),
            ]
        )
        for method in METHOD_ORDER:
            cells = [
                _paired(
                    _lookup(
                        combined_dataset_rows,
                        setting=setting,
                        method=method,
                        target_graph=target,
                    )
                )
                for target in targets
            ]
            cells.append(
                _paired(
                    _lookup(
                        combined_macro_rows,
                        setting=setting,
                        method=method,
                        aggregation="dataset_macro",
                    )
                )
            )
            if setting == "C":
                cells.append(
                    _paired(
                        _lookup(
                            combined_macro_rows,
                            setting=setting,
                            method=method,
                            aggregation="domain_macro",
                        )
                    )
                )
            lines.append(f"| {method} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Dataset-macro comparison with RECAP",
            "",
            "| Setting | RECAP | ARC | IA-GGAD | UNPrompt | AnomalyGFM-ZS |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for setting in ("B", "C"):
        cells = [
            _paired(
                _lookup(
                    recap_macros,
                    setting=setting,
                    aggregation="dataset_macro",
                )
            )
        ]
        for method in ("ARC", "IA-GGAD", "UNPrompt", "AnomalyGFM-ZS"):
            cells.append(
                _paired(
                    _lookup(
                        combined_macro_rows,
                        setting=setting,
                        method=method,
                        aggregation="dataset_macro",
                    )
                )
            )
        lines.append(f"| {setting} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Supplement artifact audit",
            "",
            f"- Training runs: {audit['training_runs_found']}/{audit['training_runs_expected']}",
            f"- Recomputed evaluations: {audit['evaluations_verified']}/{audit['evaluations_expected']}",
            f"- Frozen score files: {audit['target_score_files_verified']}",
            f"- Label-audit events: {audit['label_audit_events']}",
            f"- Maximum checkpoint reload difference: {audit['checkpoint_reload_max_abs_diff']:.8g}",
            f"- Audit problems: {len(audit['problems'])}",
            "",
            "## AnomalyGFM source-only calibration locks",
            "",
            "| Lock | Selected weight | Seed | Grid size |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in calibrations:
        lines.append(
            f"| {row['name']} | {row['selected_weight']:.6g} | "
            f"{row['calibration_seed']} | {row['grid_size']} |"
        )

    lines.extend(
        [
            "",
            "## Supplement resource totals",
            "",
            "| Method | Runs | Preparation (s) | Training (s) | Evaluation (s) | Peak GPU GiB |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in timings:
        lines.append(
            f"| {row['method']} | {row['runs']} | "
            f"{row['preparation_seconds']:.2f} | {row['training_seconds']:.2f} | "
            f"{row['evaluation_seconds']:.2f} | {row['peak_gpu_gib']:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def analyze(
    output_root: Path,
    dataset_dir: Path,
    primary_root: Path,
) -> dict[str, Any]:
    specs = build_supplement_manifest()
    validate_supplement_manifest(specs)
    records, audit = audit_runs(
        output_root,
        dataset_dir,
        specs,
        protocol_path=SUPPLEMENT_PROTOCOL_PATH,
    )
    analysis_root = output_root / "analysis"
    atomic_json(analysis_root / "artifact_audit.json", audit)
    if not audit["passed"]:
        raise RuntimeError(
            f"Supplement audit failed with {len(audit['problems'])} problems"
        )
    dataset_rows, macro_rows = aggregate(records)
    if len(dataset_rows) != 20 or len(macro_rows) != 6:
        raise ValueError(
            f"Expected 20 dataset rows and 6 macro rows, got "
            f"{len(dataset_rows)} and {len(macro_rows)}"
        )
    atomic_json(analysis_root / "raw_records.json", records)
    _csv(analysis_root / "raw_records.csv", records)
    atomic_json(analysis_root / "dataset_summary.json", dataset_rows)
    _csv(analysis_root / "dataset_summary.csv", dataset_rows)
    atomic_json(analysis_root / "macro_summary.json", macro_rows)
    _csv(
        analysis_root / "macro_summary.csv",
        [{k: v for k, v in row.items() if k != "seed_values"} for row in macro_rows],
    )
    report = render_report(
        dataset_rows,
        macro_rows,
        audit,
        primary_root,
        output_root,
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    payload = {
        "audit": audit,
        "dataset_rows": len(dataset_rows),
        "macro_rows": len(macro_rows),
        "report": str(REPORT_PATH),
    }
    atomic_json(analysis_root / "analysis_complete.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--dataset-dir", default="/root/autodl-tmp/recap/dataset"
    )
    parser.add_argument("--primary-root", default=str(PRIMARY_ROOT))
    args = parser.parse_args()
    result = analyze(
        Path(args.output_root),
        Path(args.dataset_dir),
        Path(args.primary_root),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
