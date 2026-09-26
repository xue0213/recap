from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_DIR = (
    PROJECT_ROOT / "tuning_hyperparams" / "sensitivity_results" / "sensitivity_default_v2"
)

DIAG_RE = re.compile(
    r"Diagnostics: .*?Lvar=(?P<lvar>[-+0-9.eE]+), "
    r"Lvar_active=(?P<active>[-+0-9.eE]+)"
)


def read_job_plan(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path, "r") as f:
        return {row["run_id"] for row in csv.DictReader(f)}


def parse_log(path: Path) -> dict:
    lvars: list[float] = []
    active_ratios: list[float] = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            match = DIAG_RE.search(line)
            if match:
                lvars.append(float(match.group("lvar")))
                active_ratios.append(float(match.group("active")))

    return {
        "run_id": path.stem,
        "diagnostic_points": len(lvars),
        "max_lvar": max(lvars) if lvars else float("nan"),
        "mean_lvar": sum(lvars) / len(lvars) if lvars else float("nan"),
        "max_lvar_active": max(active_ratios) if active_ratios else float("nan"),
        "mean_lvar_active": (
            sum(active_ratios) / len(active_ratios) if active_ratios else float("nan")
        ),
        "log_path": str(path),
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_report(path: Path, rows: list[dict], args) -> None:
    valid_rows = [row for row in rows if row["diagnostic_points"] > 0]
    max_lvar = max((row["max_lvar"] for row in valid_rows), default=float("nan"))
    max_active = max((row["max_lvar_active"] for row in valid_rows), default=float("nan"))
    pass_lvar = max_lvar <= args.max_lvar_tol
    pass_active = max_active <= args.max_active_tol
    verdict = "PASS" if pass_lvar and pass_active else "REVIEW"
    worst = sorted(
        valid_rows,
        key=lambda row: (row["max_lvar"], row["max_lvar_active"]),
        reverse=True,
    )[:10]

    lines = [
        "# lambda_E Activity Check",
        "",
        f"Verdict: **{verdict}**",
        "",
        "This check scans the original sensitivity training diagnostics and measures whether",
        "`L_var` was active. If `Lvar` and `Lvar_active` stay near zero, the",
        "`lambda_E * L_var` term was effectively dormant in the logged optimization states.",
        "",
        "## Summary",
        "",
        f"- Result dir: `{args.result_dir}`",
        f"- Runs scanned: `{len(valid_rows)}`",
        f"- Diagnostic points: `{sum(row['diagnostic_points'] for row in valid_rows)}`",
        f"- Max Lvar: `{max_lvar:.8g}`",
        f"- Max Lvar active ratio: `{max_active:.8g}`",
        f"- Tolerances: `Lvar <= {args.max_lvar_tol}`, `active <= {args.max_active_tol}`",
        "",
        "## Worst Runs",
        "",
        "| run_id | diagnostic points | max Lvar | max active | mean Lvar | mean active |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in worst:
        lines.append(
            "| {run_id} | {n} | {max_lvar:.8g} | {max_active:.8g} | "
            "{mean_lvar:.8g} | {mean_active:.8g} |".format(
                run_id=row["run_id"],
                n=row["diagnostic_points"],
                max_lvar=row["max_lvar"],
                max_active=row["max_lvar_active"],
                mean_lvar=row["mean_lvar"],
                mean_active=row["mean_lvar_active"],
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan sensitivity logs to verify whether lambda_E * L_var was active."
    )
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--include-extra-logs", action="store_true")
    parser.add_argument("--max-lvar-tol", type=float, default=1e-5)
    parser.add_argument("--max-active-tol", type=float, default=1e-3)
    return parser.parse_args()


def main():
    args = parse_args()
    args.result_dir = args.result_dir.resolve()
    if args.output_dir is None:
        args.output_dir = (
            PROJECT_ROOT
            / "tuning_hyperparams"
            / "lambda_e0_invariance_results"
            / f"{args.result_dir.name}_activity_check"
        )
    args.output_dir = args.output_dir.resolve()

    log_dir = args.result_dir / "logs"
    expected_run_ids = read_job_plan(args.result_dir / "job_plan.csv")
    rows = []
    for log_path in sorted(log_dir.glob("*.log")):
        if expected_run_ids and not args.include_extra_logs and log_path.stem not in expected_run_ids:
            continue
        rows.append(parse_log(log_path))

    fieldnames = [
        "run_id",
        "diagnostic_points",
        "max_lvar",
        "mean_lvar",
        "max_lvar_active",
        "mean_lvar_active",
        "log_path",
    ]
    write_csv(args.output_dir / "lambda_E_activity_by_run.csv", rows, fieldnames)
    write_report(args.output_dir / "lambda_E_activity_summary.md", rows, args)
    print(f"lambda_E activity check complete: {args.output_dir}")


if __name__ == "__main__":
    main()
