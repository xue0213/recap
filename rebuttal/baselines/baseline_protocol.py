"""Declarative Phase 2 OFA baseline manifest.

The scope is locked in ``rebuttal/BASELINE_OFA_REPROTOCOL.md``. This module
contains no model or result-dependent logic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


SEEDS = (0, 1, 2)

DATASETS = {
    "pubmed": {"display": "PubMed", "domain": "Citation", "file": "pubmed.mat"},
    "cora": {"display": "Cora", "domain": "Citation", "file": "cora.mat"},
    "citeseer": {
        "display": "CiteSeer",
        "domain": "Citation",
        "file": "citeseer.mat",
    },
    "ACM": {"display": "ACM", "domain": "Citation", "file": "ACM.mat"},
    "Flickr": {"display": "Flickr", "domain": "Social", "file": "Flickr.mat"},
    "BlogCatalog": {
        "display": "BlogCatalog",
        "domain": "Social",
        "file": "BlogCatalog.mat",
    },
    "Facebook": {
        "display": "Facebook",
        "domain": "Social",
        "file": "Facebook.mat",
    },
    "weibo": {"display": "Weibo", "domain": "Social", "file": "weibo.mat"},
    "Reddit": {"display": "Reddit", "domain": "Social", "file": "Reddit.mat"},
    "questions": {
        "display": "Questions",
        "domain": "Q&A",
        "file": "questions.mat",
    },
    "YelpChi": {
        "display": "YelpChi",
        "domain": "E-commerce",
        "file": "YelpChi.mat",
    },
    "Amazon": {
        "display": "Amazon",
        "domain": "E-commerce",
        "file": "Amazon.mat",
    },
}

SETTINGS = {
    "A": {
        "sources": ("pubmed", "Flickr", "questions", "YelpChi"),
        "targets": (
            "cora",
            "citeseer",
            "ACM",
            "BlogCatalog",
            "Facebook",
            "weibo",
            "Reddit",
            "Amazon",
        ),
        "methods": ("ARC", "UNPrompt", "AnomalyGFM-ZS", "IA-GGAD"),
    },
    "B": {
        "sources": ("pubmed", "cora", "questions", "YelpChi"),
        "targets": ("Flickr", "BlogCatalog", "Facebook", "weibo", "Reddit"),
        "methods": ("ARC", "IA-GGAD"),
    },
    "C": {
        "sources": ("pubmed", "cora", "citeseer", "ACM"),
        "targets": ("BlogCatalog", "Flickr", "Reddit", "Amazon", "questions"),
        "methods": ("ARC", "IA-GGAD"),
    },
}


@dataclass(frozen=True)
class BaselineRunSpec:
    run_id: str
    method: str
    setting: str
    seed: int
    source_graphs: tuple[str, ...]
    target_graphs: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "BaselineRunSpec":
        payload = dict(value)
        payload["source_graphs"] = tuple(payload["source_graphs"])
        payload["target_graphs"] = tuple(payload["target_graphs"])
        return cls(**payload)


def method_slug(method: str) -> str:
    return method.lower().replace("-", "_")


def build_manifest() -> list[BaselineRunSpec]:
    """Return the 24 locked primary training runs.

    Seed zero is first for each method/setting so method smoke gates and
    source-only fusion calibration can complete before the remaining seeds.
    """

    specs: list[BaselineRunSpec] = []
    for setting, definition in SETTINGS.items():
        for method in definition["methods"]:
            for seed in SEEDS:
                specs.append(
                    BaselineRunSpec(
                        run_id=(
                            f"ofa_{setting.lower()}__{method_slug(method)}__seed{seed}"
                        ),
                        method=method,
                        setting=setting,
                        seed=seed,
                        source_graphs=definition["sources"],
                        target_graphs=definition["targets"],
                    )
                )
    return specs


def expected_evaluations() -> int:
    return sum(len(spec.target_graphs) for spec in build_manifest())


def validate_manifest(specs: list[BaselineRunSpec]) -> None:
    if len(specs) != 24:
        raise ValueError(f"Expected 24 training runs, found {len(specs)}")
    if len({spec.run_id for spec in specs}) != len(specs):
        raise ValueError("Duplicate baseline run IDs")
    if sum(len(spec.target_graphs) for spec in specs) != 156:
        raise ValueError("Expected 156 method/target/seed evaluations")
    for spec in specs:
        definition = SETTINGS[spec.setting]
        if spec.method not in definition["methods"]:
            raise ValueError(f"{spec.run_id}: method outside locked setting scope")
        if spec.source_graphs != definition["sources"]:
            raise ValueError(f"{spec.run_id}: source split drift")
        if spec.target_graphs != definition["targets"]:
            raise ValueError(f"{spec.run_id}: target split drift")
        if spec.seed not in SEEDS:
            raise ValueError(f"{spec.run_id}: unexpected seed")


if __name__ == "__main__":
    manifest = build_manifest()
    validate_manifest(manifest)
    print(f"training_runs={len(manifest)} evaluations={expected_evaluations()}")

