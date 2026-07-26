"""Declarative manifest for the locked 12-dataset OFO baseline study."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from rebuttal.baselines.baseline_protocol import DATASETS


METHODS = (
    "GCN",
    "GAT",
    "BWGNN",
    "XGBGraph",
    "DOMINANT",
    "AnomalyDAE",
    "CoLA",
    "ADA-GAD",
)
DATASET_ORDER = tuple(DATASETS)
SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class OFOBaselineRun:
    run_id: str
    method: str
    dataset: str
    seed: int
    supervised: bool

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "OFOBaselineRun":
        return cls(**value)


def method_slug(method: str) -> str:
    return method.lower().replace("-", "_")


def build_manifest() -> list[OFOBaselineRun]:
    specs: list[OFOBaselineRun] = []
    for method in METHODS:
        for seed in SEEDS:
            for dataset in DATASET_ORDER:
                specs.append(
                    OFOBaselineRun(
                        run_id=(
                            f"ofo__{method_slug(method)}__"
                            f"{dataset}__seed{seed}"
                        ),
                        method=method,
                        dataset=dataset,
                        seed=seed,
                        supervised=method
                        in {"GCN", "GAT", "BWGNN", "XGBGraph"},
                    )
                )
    return specs


def validate_manifest(specs: list[OFOBaselineRun]) -> None:
    if len(specs) != 288:
        raise ValueError(f"Expected 288 runs, found {len(specs)}")
    if len({spec.run_id for spec in specs}) != 288:
        raise ValueError("Duplicate OFO baseline run ID")
    expected = {
        (method, dataset, seed)
        for method in METHODS
        for dataset in DATASET_ORDER
        for seed in SEEDS
    }
    actual = {(spec.method, spec.dataset, spec.seed) for spec in specs}
    if actual != expected:
        raise ValueError("OFO baseline manifest Cartesian product drift")
    for spec in specs:
        expected_supervision = spec.method in {
            "GCN",
            "GAT",
            "BWGNN",
            "XGBGraph",
        }
        if spec.supervised != expected_supervision:
            raise ValueError(f"{spec.run_id}: supervision flag drift")


if __name__ == "__main__":
    manifest = build_manifest()
    validate_manifest(manifest)
    print(f"training_runs={len(manifest)} evaluations={len(manifest)}")
