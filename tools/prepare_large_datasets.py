#!/usr/bin/env python3
"""Convert BWGNN and DGraph-Fin releases into RECAP's canonical bundle.

The output layout is:
  <output>/<name>/features.npy
  <output>/<name>/labels.npy
  <output>/<name>/adjacency.npz
  <output>/<name>/metadata.json

BWGNN's T-Finance/T-Social files use DGL's graph serialization and therefore
need a conversion environment with DGL installed. DGraph-Fin is read directly
from the official ``dgraphfin.npz`` archive and has no DGL dependency.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def _binary_labels(values: np.ndarray, dataset: str) -> np.ndarray:
    labels = np.asarray(values)
    # The official T-Finance release stores a two-column one-hot label.
    if labels.ndim == 2 and labels.shape[1] > 1:
        labels = labels.argmax(axis=1)
    else:
        labels = labels.reshape(-1)
    if dataset == "dgraphfin":
        # Official classes 0/1 are normal/fraud; 2/3 are background nodes.
        labels = (labels == 1).astype(np.float32)
    else:
        labels = (labels != 0).astype(np.float32)
    return labels


def _adjacency(src: np.ndarray, dst: np.ndarray, num_nodes: int) -> sp.csr_matrix:
    values = np.ones(src.shape[0], dtype=np.float32)
    adj = sp.coo_matrix((values, (src, dst)), shape=(num_nodes, num_nodes))
    # RECAP is currently single-view and uses an undirected normalized graph.
    adj = adj.maximum(adj.T).tocsr()
    adj.sum_duplicates()
    adj.data[:] = 1.0
    return adj


def _write_bundle(
    output_root: Path,
    name: str,
    features: np.ndarray,
    labels: np.ndarray,
    adjacency: sp.csr_matrix,
    metadata: dict,
    evaluation_mask: np.ndarray | None = None,
) -> None:
    target = output_root / name
    target.mkdir(parents=True, exist_ok=True)
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32).reshape(-1)
    if adjacency.shape != (features.shape[0], features.shape[0]):
        raise ValueError("Adjacency and feature node counts differ")
    if labels.shape[0] != features.shape[0]:
        raise ValueError("Label and feature node counts differ")
    if not np.isfinite(features).all():
        raise ValueError("Features contain NaN or infinity")
    np.save(target / "features.npy", features, allow_pickle=False)
    np.save(target / "labels.npy", labels, allow_pickle=False)
    if evaluation_mask is not None:
        evaluation_mask = np.asarray(evaluation_mask, dtype=np.bool_).reshape(-1)
        if evaluation_mask.shape != labels.shape:
            raise ValueError("Evaluation mask and labels differ")
        np.save(target / "evaluation_mask.npy", evaluation_mask, allow_pickle=False)
    sp.save_npz(target / "adjacency.npz", adjacency, compressed=True)
    info = {
        "format_version": 1,
        "name": name,
        "num_nodes": int(features.shape[0]),
        "num_edges_undirected_csr": int(adjacency.nnz),
        "num_features": int(features.shape[1]),
        "num_anomalies": int(labels.sum()),
        **metadata,
    }
    (target / "metadata.json").write_text(
        json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def convert_bwgnn(source: Path, output_root: Path, name: str) -> None:
    try:
        from dgl.data.utils import load_graphs
    except ImportError as exc:
        raise SystemExit(
            "BWGNN conversion requires DGL. Install DGL in a separate conversion "
            "environment; RECAP's runtime itself does not need DGL."
        ) from exc
    print(f"[{name}] loading DGL graph", flush=True)
    graphs, _ = load_graphs(str(source))
    if len(graphs) != 1:
        raise ValueError(f"Expected one graph in {source}, found {len(graphs)}")
    graph = graphs[0]
    feature_key = next((k for k in ("feature", "feat") if k in graph.ndata), None)
    label_key = next((k for k in ("label", "labels") if k in graph.ndata), None)
    if feature_key is None or label_key is None:
        raise KeyError(f"Unexpected DGL node fields: {list(graph.ndata.keys())}")
    print(
        f"[{name}] loaded nodes={graph.num_nodes()} edges={graph.num_edges()}",
        flush=True,
    )
    features = graph.ndata[feature_key].detach().cpu().numpy()
    labels = _binary_labels(graph.ndata[label_key].detach().cpu().numpy(), name)
    # Reuse DGL's native CSR arrays. Creating COO source/destination arrays and
    # then symmetrizing them requires several edge-sized copies and exceeds
    # memory on large releases.
    graph = graph.formats(["csr"])
    indptr, indices, _ = graph.adj_tensors("csr")
    indptr_np = indptr.detach().cpu().numpy()
    indices_np = indices.detach().cpu().numpy()
    adj = sp.csr_matrix(
        (
            np.ones(indices_np.shape[0], dtype=np.float32),
            indices_np,
            indptr_np,
        ),
        shape=(graph.num_nodes(), graph.num_nodes()),
        copy=False,
    )
    print(f"[{name}] writing canonical CSR bundle", flush=True)
    _write_bundle(
        output_root, name, features, labels, adj,
        {
            "source_format": "BWGNN DGL graph",
            "source_file": source.name,
            "topology_preserved_from_source": True,
        },
    )


def convert_dgraphfin(source: Path, output_root: Path) -> None:
    archive = np.load(source, allow_pickle=False)
    required = {"x", "y", "edge_index"}
    missing = required.difference(archive.files)
    if missing:
        raise KeyError(f"DGraph-Fin archive is missing fields: {sorted(missing)}")
    edges = np.asarray(archive["edge_index"])
    if edges.shape[0] != 2 and edges.shape[1] == 2:
        edges = edges.T
    if edges.shape[0] != 2:
        raise ValueError(f"Unexpected edge_index shape: {edges.shape}")
    features = np.asarray(archive["x"], dtype=np.float32)
    raw_labels = np.asarray(archive["y"]).reshape(-1)
    labels = _binary_labels(raw_labels, "dgraphfin")
    evaluation_mask = raw_labels < 2
    adj = _adjacency(edges[0], edges[1], features.shape[0])
    _write_bundle(
        output_root, "dgraphfin", features, labels, adj,
        {
            "source_format": "official DGraph-Fin NPZ",
            "source_file": source.name,
            "background_labels_excluded_from_metrics": True,
            "available_fields": sorted(archive.files),
        },
        evaluation_mask=evaluation_mask,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=("tfinance", "tsocial", "dgraphfin"))
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-root", default="dataset_large/processed", type=Path)
    args = parser.parse_args()
    if not args.source.is_file():
        parser.error(f"source file does not exist: {args.source}")
    if args.dataset == "dgraphfin":
        convert_dgraphfin(args.source, args.output_root)
    else:
        convert_bwgnn(args.source, args.output_root, args.dataset)


if __name__ == "__main__":
    main()
