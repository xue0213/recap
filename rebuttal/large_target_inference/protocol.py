"""Declarative manifest for the locked large-target inference experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass


SEEDS = (0, 1, 2)
TARGETS = ("tfinance", "dgraphfin", "tsocial")

CHECKPOINTS = {
    0: {
        "relative_path": (
            "rebuttal/artifacts/phase1/runs/ofa__a__seed0/checkpoints/final.pt"
        ),
        "sha256": (
            "ca111ebd17e53d63c4b722c4aa47caff"
            "069f81a9525b969070f4d5bf2ee3fe44"
        ),
    },
    1: {
        "relative_path": (
            "rebuttal/artifacts/phase1/runs/ofa__a__seed1/checkpoints/final.pt"
        ),
        "sha256": (
            "702d4b65f5b7b85fea0286d5980954ea"
            "8fdb34978cf24c1ac212405da42d6c6d"
        ),
    },
    2: {
        "relative_path": (
            "rebuttal/artifacts/phase1/runs/ofa__a__seed2/checkpoints/final.pt"
        ),
        "sha256": (
            "def7d06e558a0897c105b021924b2a7f"
            "bd1e1e52b9b547a15c01b68218311cdd"
        ),
    },
}

DATASETS = {
    "tfinance": {
        "display": "T-Finance",
        "domain": "Finance",
        "nodes": 39_357,
        "adjacency_nnz": 42_445_086,
        "raw_features": 10,
        "anomalies": 1_804,
        "evaluation_nodes": 39_357,
        "primary_knn": "exact",
    },
    "dgraphfin": {
        "display": "DGraph-Fin",
        "domain": "Finance",
        "nodes": 3_700_550,
        "adjacency_nnz": 7_994_520,
        "raw_features": 17,
        "anomalies": 15_509,
        "evaluation_nodes": 1_225_601,
        "primary_knn": "faiss_ivfpq",
    },
    "tsocial": {
        "display": "T-Social",
        "domain": "Social",
        "nodes": 5_781_065,
        "adjacency_nnz": 146_211_016,
        "raw_features": 10,
        "anomalies": 174_280,
        "evaluation_nodes": 5_781_065,
        "primary_knn": "faiss_ivfpq",
    },
}

ANN_CONFIG = {
    "backend": "faiss_ivfpq",
    "nlist": 4096,
    "nprobe": 16,
    "pq_m": 16,
    "train_size": 262_144,
    "query_batch_size": 4096,
    "add_batch_size": 262_144,
    "rerank_factor": 32,
    "max_rerank_candidates": 256,
    "seed": 0,
}

MODEL_LOCK = {
    "dims": 32,
    "num_hops": 4,
    "h_feats": 256,
    "num_layers": 2,
    "num_clusters": 36,
    "knn_k": 64,
    "tau_s": 0.3,
    "tau_c": 0.3,
    "tau_e": 1.0,
    "beta": 0.02,
    "lambda_H": 0.1,
    "lambda_E": 0.0,
}


@dataclass(frozen=True)
class InferenceSpec:
    run_id: str
    seed: int
    target: str
    checkpoint_relative_path: str
    checkpoint_sha256: str
    primary_knn: str

    def to_dict(self) -> dict:
        return asdict(self)


def build_manifest() -> list[InferenceSpec]:
    manifest = [
        InferenceSpec(
            run_id=f"ofa_a__seed{seed}__{target}",
            seed=seed,
            target=target,
            checkpoint_relative_path=CHECKPOINTS[seed]["relative_path"],
            checkpoint_sha256=CHECKPOINTS[seed]["sha256"],
            primary_knn=DATASETS[target]["primary_knn"],
        )
        for target in TARGETS
        for seed in SEEDS
    ]
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: list[InferenceSpec]) -> None:
    if len(manifest) != 9:
        raise ValueError(f"Expected 9 primary evaluations, got {len(manifest)}")
    ids = [item.run_id for item in manifest]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate large-target inference run ID")
    observed = {(item.target, item.seed) for item in manifest}
    expected = {(target, seed) for target in TARGETS for seed in SEEDS}
    if observed != expected:
        raise ValueError(
            f"Large-target manifest mismatch: missing={expected-observed}, "
            f"extra={observed-expected}"
        )
