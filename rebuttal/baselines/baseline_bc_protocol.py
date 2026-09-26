"""Declarative user-requested B/C completion manifest."""

from __future__ import annotations

from .baseline_protocol import BaselineRunSpec, SEEDS, SETTINGS, method_slug


METHODS = ("UNPrompt", "AnomalyGFM-ZS")


def build_supplement_manifest() -> list[BaselineRunSpec]:
    specs: list[BaselineRunSpec] = []
    for setting in ("B", "C"):
        definition = SETTINGS[setting]
        for method in METHODS:
            for seed in SEEDS:
                specs.append(
                    BaselineRunSpec(
                        run_id=(
                            f"ofa_{setting.lower()}_supplement__"
                            f"{method_slug(method)}__seed{seed}"
                        ),
                        method=method,
                        setting=setting,
                        seed=seed,
                        source_graphs=definition["sources"],
                        target_graphs=definition["targets"],
                    )
                )
    return specs


def expected_supplement_evaluations() -> int:
    return sum(
        len(spec.target_graphs) for spec in build_supplement_manifest()
    )


def validate_supplement_manifest(specs: list[BaselineRunSpec]) -> None:
    if len(specs) != 12:
        raise ValueError(f"Expected 12 supplementary runs, found {len(specs)}")
    if len({spec.run_id for spec in specs}) != len(specs):
        raise ValueError("Duplicate supplementary run IDs")
    if sum(len(spec.target_graphs) for spec in specs) != 60:
        raise ValueError("Expected 60 supplementary evaluations")
    for spec in specs:
        definition = SETTINGS[spec.setting]
        if spec.setting not in {"B", "C"} or spec.method not in METHODS:
            raise ValueError(f"{spec.run_id}: outside supplementary scope")
        if spec.source_graphs != definition["sources"]:
            raise ValueError(f"{spec.run_id}: source split drift")
        if spec.target_graphs != definition["targets"]:
            raise ValueError(f"{spec.run_id}: target split drift")
        if spec.seed not in SEEDS:
            raise ValueError(f"{spec.run_id}: unexpected seed")
