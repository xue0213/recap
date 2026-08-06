"""Resumable runner for the user-requested B/C baseline supplement."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch

from .baseline_bc_protocol import (
    build_supplement_manifest,
    expected_supplement_evaluations,
    validate_supplement_manifest,
)
from .baseline_common import atomic_json, sha256_file, utc_now
from .baseline_protocol import BaselineRunSpec
from .baseline_runner import (
    DEFAULT_DATASET_DIR,
    DEFAULT_VENDOR_ROOT,
    PROJECT_ROOT,
    UPSTREAM_MANIFEST_PATH,
    execute_spec,
    run_dir,
    select_spec,
)


DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "rebuttal" / "artifacts" / "phase2_bc_supplement"
)
SUPPLEMENT_PROTOCOL_PATH = (
    PROJECT_ROOT / "rebuttal" / "BASELINE_BC_SUPPLEMENT_PROTOCOL.md"
)
PARENT_PROTOCOL_PATH = (
    PROJECT_ROOT / "rebuttal" / "BASELINE_OFA_REPROTOCOL.md"
)


def save_manifest(output_root: Path) -> Path:
    specs = build_supplement_manifest()
    validate_supplement_manifest(specs)
    path = output_root / "manifest.json"
    atomic_json(
        path,
        {
            "format": "recap_phase2_bc_supplement_manifest_v1",
            "created_at": utc_now(),
            "classification": "confirmatory_user_revised_scope",
            "protocol_path": str(SUPPLEMENT_PROTOCOL_PATH),
            "protocol_sha256": sha256_file(SUPPLEMENT_PROTOCOL_PATH),
            "parent_protocol_sha256": sha256_file(PARENT_PROTOCOL_PATH),
            "upstream_manifest_sha256": sha256_file(
                UPSTREAM_MANIFEST_PATH
            ),
            "original_phase2_artifacts_immutable": True,
            "training_runs": len(specs),
            "final_evaluations": expected_supplement_evaluations(),
            "runs": [spec.to_dict() for spec in specs],
        },
    )
    return path


def load_manifest(output_root: Path) -> list[BaselineRunSpec]:
    path = output_root / "manifest.json"
    if not path.exists():
        save_manifest(output_root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "recap_phase2_bc_supplement_manifest_v1":
        raise ValueError("Supplement manifest format mismatch")
    if payload.get("protocol_sha256") != sha256_file(
        SUPPLEMENT_PROTOCOL_PATH
    ):
        raise ValueError("Supplement protocol changed after manifest creation")
    if payload.get("parent_protocol_sha256") != sha256_file(
        PARENT_PROTOCOL_PATH
    ):
        raise ValueError("Parent protocol hash mismatch")
    if payload.get("upstream_manifest_sha256") != sha256_file(
        UPSTREAM_MANIFEST_PATH
    ):
        raise ValueError("Upstream manifest hash mismatch")
    specs = [
        BaselineRunSpec.from_dict(value) for value in payload.get("runs", [])
    ]
    validate_supplement_manifest(specs)
    if (
        int(payload.get("training_runs", -1)) != len(specs)
        or int(payload.get("final_evaluations", -1))
        != expected_supplement_evaluations()
    ):
        raise ValueError("Supplement manifest count mismatch")
    return specs


def status(output_root: Path) -> dict[str, Any]:
    specs = load_manifest(output_root)
    complete: list[str] = []
    partial: list[str] = []
    pending: list[str] = []
    evaluations = 0
    for spec in specs:
        directory = run_dir(output_root, spec)
        complete_path = directory / "complete.json"
        if complete_path.exists():
            complete.append(spec.run_id)
            payload = json.loads(complete_path.read_text(encoding="utf-8"))
            evaluations += len(payload.get("target_results", {}))
        elif directory.exists():
            partial.append(spec.run_id)
        else:
            pending.append(spec.run_id)
    return {
        "training_complete": len(complete),
        "training_expected": len(specs),
        "evaluations_complete": evaluations,
        "evaluations_expected": expected_supplement_evaluations(),
        "complete": complete,
        "partial": partial,
        "pending": pending,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR
    )
    parser.add_argument(
        "--vendor-root", type=Path, default=DEFAULT_VENDOR_ROOT
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )
    parser.add_argument("--device", default="cuda:0")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("manifest")
    subparsers.add_parser("status")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--run-id", required=True)
    run_parser.add_argument("--smoke-epochs", type=int)
    pending_parser = subparsers.add_parser("run-pending")
    pending_parser.add_argument("--method")
    pending_parser.add_argument("--setting")
    pending_parser.add_argument("--max-runs", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.command == "manifest":
        print(save_manifest(output_root))
        return
    if args.command == "status":
        print(json.dumps(status(output_root), indent=2))
        return

    specs = load_manifest(output_root)
    device = torch.device(args.device)
    if args.command == "run":
        spec = select_spec(specs, args.run_id)
        result = execute_spec(
            spec,
            dataset_dir=args.dataset_dir.resolve(),
            vendor_root=args.vendor_root.resolve(),
            output_root=output_root,
            device=device,
            smoke_epochs=args.smoke_epochs,
            protocol_path=SUPPLEMENT_PROTOCOL_PATH,
        )
        print(json.dumps({"run_id": spec.run_id, "status": result["status"]}))
        return

    selected = [
        spec
        for spec in specs
        if (args.method is None or spec.method == args.method)
        and (args.setting is None or spec.setting == args.setting)
        and not (run_dir(output_root, spec) / "complete.json").exists()
    ]
    if args.max_runs is not None:
        selected = selected[: args.max_runs]
    for index, spec in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] {spec.run_id}", flush=True)
        execute_spec(
            spec,
            dataset_dir=args.dataset_dir.resolve(),
            vendor_root=args.vendor_root.resolve(),
            output_root=output_root,
            device=device,
            protocol_path=SUPPLEMENT_PROTOCOL_PATH,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(json.dumps(status(output_root), indent=2))


if __name__ == "__main__":
    main()
