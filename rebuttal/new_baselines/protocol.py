"""Declarative manifest for the three-baseline extension."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from rebuttal.baselines.baseline_protocol import SETTINGS
from rebuttal.ofo_baselines.protocol import DATASETS


SEEDS = (0, 1, 2)
OFO_METHODS = ("DiffGAD", "GUIDE")
OFA_METHOD = "OWLEYE"


@dataclass(frozen=True)
class ExtensionRunSpec:
    run_id: str
    method: str
    seed: int
    dataset: str | None = None
    setting: str | None = None
    source_graphs: tuple[str, ...] = ()
    target_graphs: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "ExtensionRunSpec":
        payload = dict(value)
        payload["source_graphs"] = tuple(payload.get("source_graphs", ()))
        payload["target_graphs"] = tuple(payload.get("target_graphs", ()))
        return cls(**payload)


def method_slug(method: str) -> str:
    return method.lower().replace("-", "_")


def build_manifest() -> list[ExtensionRunSpec]:
    specs: list[ExtensionRunSpec] = []
    for method in OFO_METHODS:
        for dataset in DATASETS:
            for seed in SEEDS:
                specs.append(
                    ExtensionRunSpec(
                        run_id=(
                            f"ofo__{method_slug(method)}__{dataset}__seed{seed}"
                        ),
                        method=method,
                        dataset=dataset,
                        seed=seed,
                    )
                )
    for setting in ("A", "B", "C"):
        definition = SETTINGS[setting]
        for seed in SEEDS:
            specs.append(
                ExtensionRunSpec(
                    run_id=f"ofa_{setting.lower()}__owleye__seed{seed}",
                    method=OFA_METHOD,
                    setting=setting,
                    seed=seed,
                    source_graphs=definition["sources"],
                    target_graphs=definition["targets"],
                )
            )
    validate_manifest(specs)
    return specs


def expected_evaluations() -> int:
    return 72 + sum(
        len(SETTINGS[setting]["targets"]) * len(SEEDS)
        for setting in ("A", "B", "C")
    )


def validate_manifest(specs: list[ExtensionRunSpec]) -> None:
    if len(specs) != 81:
        raise ValueError(f"Expected 81 training runs, found {len(specs)}")
    if len({spec.run_id for spec in specs}) != len(specs):
        raise ValueError("Duplicate extension run IDs")
    ofo = [spec for spec in specs if spec.method in OFO_METHODS]
    ofa = [spec for spec in specs if spec.method == OFA_METHOD]
    if len(ofo) != 72 or len(ofa) != 9:
        raise ValueError("Expected 72 OFO and 9 OFA training runs")
    for spec in ofo:
        if spec.dataset not in DATASETS or spec.setting is not None:
            raise ValueError(f"{spec.run_id}: invalid OFO scope")
        if spec.source_graphs or spec.target_graphs:
            raise ValueError(f"{spec.run_id}: OFO graph lists must be empty")
    for spec in ofa:
        if spec.setting not in SETTINGS or spec.dataset is not None:
            raise ValueError(f"{spec.run_id}: invalid OFA scope")
        definition = SETTINGS[spec.setting]
        if spec.source_graphs != definition["sources"]:
            raise ValueError(f"{spec.run_id}: source split drift")
        if spec.target_graphs != definition["targets"]:
            raise ValueError(f"{spec.run_id}: target split drift")
    if any(spec.seed not in SEEDS for spec in specs):
        raise ValueError("Unexpected seed")
    if expected_evaluations() != 126:
        raise ValueError("Expected 126 final evaluations")


if __name__ == "__main__":
    manifest = build_manifest()
    print(f"training_runs={len(manifest)} evaluations={expected_evaluations()}")
