#!/usr/bin/env python3
"""Structural acceptance checks for a canonical RECAP large-dataset bundle."""
import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--expected-nodes", type=int)
    parser.add_argument("--expected-features", type=int)
    args = parser.parse_args()
    meta = json.loads((args.bundle / "metadata.json").read_text())
    x = np.load(args.bundle / "features.npy", mmap_mode="r")
    y = np.load(args.bundle / "labels.npy", mmap_mode="r")
    adj = sp.load_npz(args.bundle / "adjacency.npz")
    assert x.ndim == 2 and y.shape == (x.shape[0],)
    assert adj.shape == (x.shape[0], x.shape[0])
    assert np.isfinite(x).all()
    assert set(np.unique(y)).issubset({0.0, 1.0})
    # DGraph-Fin is explicitly symmetrized during conversion. BWGNN topology
    # is preserved exactly to avoid multiple edge-sized copies under tight RAM.
    if not meta.get("topology_preserved_from_source"):
        assert (adj != adj.T).nnz == 0
    assert meta["num_nodes"] == x.shape[0]
    mask_path = args.bundle / "evaluation_mask.npy"
    if mask_path.exists():
        mask = np.load(mask_path, mmap_mode="r")
        assert mask.dtype == np.bool_ and mask.shape == y.shape
        assert mask.any()
    if args.expected_nodes:
        assert x.shape[0] == args.expected_nodes
    if args.expected_features:
        assert x.shape[1] == args.expected_features
    print(json.dumps({"status": "PASS", **meta}, indent=2))


if __name__ == "__main__":
    main()
