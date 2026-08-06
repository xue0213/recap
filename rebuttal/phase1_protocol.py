"""Locked RECAP Phase 1 experiment manifest.

This module is intentionally declarative. Formal run IDs and source/target
splits are generated from constants that were locked before any result run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


SEEDS = (0, 1, 2)
DIAGNOSTIC_EPOCHS = (1, 10, 25, 50, 75, 100)

# Values are the exact case-sensitive names accepted by utils.Dataset.
DATASETS = {
    "pubmed": {
        "display": "PubMed",
        "domain": "Citation",
        "file": "pubmed.mat",
    },
    "cora": {
        "display": "Cora",
        "domain": "Citation",
        "file": "cora.mat",
    },
    "citeseer": {
        "display": "CiteSeer",
        "domain": "Citation",
        "file": "citeseer.mat",
    },
    "ACM": {
        "display": "ACM",
        "domain": "Citation",
        "file": "ACM.mat",
    },
    "Flickr": {
        "display": "Flickr",
        "domain": "Social",
        "file": "Flickr.mat",
    },
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
    "weibo": {
        "display": "Weibo",
        "domain": "Social",
        "file": "weibo.mat",
    },
    "Reddit": {
        "display": "Reddit",
        "domain": "Social",
        "file": "Reddit.mat",
    },
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

OFO_DATASETS = (
    "pubmed",
    "cora",
    "citeseer",
    "ACM",
    "Flickr",
    "BlogCatalog",
    "Facebook",
    "weibo",
    "Reddit",
    "YelpChi",
    "Amazon",
)

OFA_SETTINGS = {
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
    },
    "B": {
        "sources": ("pubmed", "cora", "questions", "YelpChi"),
        "targets": ("Flickr", "BlogCatalog", "Facebook", "weibo", "Reddit"),
    },
    "C": {
        "sources": ("pubmed", "cora", "citeseer", "ACM"),
        "targets": ("BlogCatalog", "Flickr", "Reddit", "Amazon", "questions"),
    },
}


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    method: str
    paradigm: str
    setting: str
    seed: int
    source_graphs: tuple[str, ...]
    target_graphs: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "RunSpec":
        value = dict(value)
        value["source_graphs"] = tuple(value["source_graphs"])
        value["target_graphs"] = tuple(value["target_graphs"])
        return cls(**value)


def build_manifest() -> list[RunSpec]:
    """Return the 42 locked formal training runs in gate-friendly order."""
    ofo_specs = [
        RunSpec(
            run_id=f"ofo__{dataset.lower()}__seed{seed}",
            method="RECAP-OFO",
            paradigm="one-for-one",
            setting="OFO",
            seed=seed,
            source_graphs=(dataset,),
            target_graphs=(dataset,),
        )
        for seed in SEEDS
        for dataset in OFO_DATASETS
    ]
    ofa_specs = [
        RunSpec(
            run_id=f"ofa__{setting.lower()}__seed{seed}",
            method="RECAP",
            paradigm="one-for-all",
            setting=setting,
            seed=seed,
            source_graphs=tuple(split["sources"]),
            target_graphs=tuple(split["targets"]),
        )
        for seed in SEEDS
        for setting, split in OFA_SETTINGS.items()
    ]

    # First two jobs are the formal seed-0 correctness gates. They are not
    # extra runs and remain part of the 42-run manifest.
    gate_ids = ("ofo__cora__seed0", "ofa__a__seed0")
    by_id = {spec.run_id: spec for spec in (*ofo_specs, *ofa_specs)}
    ordered = [by_id[run_id] for run_id in gate_ids]
    ordered.extend(
        spec
        for spec in (*ofo_specs, *ofa_specs)
        if spec.run_id not in gate_ids
    )
    validate_manifest(ordered)
    return ordered


def validate_manifest(specs: Iterable[RunSpec]) -> None:
    specs = list(specs)
    run_ids = [spec.run_id for spec in specs]
    if len(specs) != 42:
        raise ValueError(f"Expected 42 formal training runs, got {len(specs)}")
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("Duplicate run IDs in formal manifest")

    ofo = [spec for spec in specs if spec.setting == "OFO"]
    ofa = [spec for spec in specs if spec.setting != "OFO"]
    if len(ofo) != 33 or len(ofa) != 9:
        raise ValueError(f"Expected 33 OFO and 9 OFA runs, got {len(ofo)} and {len(ofa)}")
    if any("questions" in spec.source_graphs for spec in ofo):
        raise ValueError("Questions must not appear in OFO")
    if not any("questions" in (*spec.source_graphs, *spec.target_graphs) for spec in ofa):
        raise ValueError("Questions was accidentally removed from OFA")

    final_evaluations = sum(len(spec.target_graphs) for spec in specs)
    if final_evaluations != 87:
        raise ValueError(f"Expected 87 final evaluations, got {final_evaluations}")


def display_name(dataset: str) -> str:
    return str(DATASETS[dataset]["display"])


def dataset_domain(dataset: str) -> str:
    return str(DATASETS[dataset]["domain"])
