"""Shared artifact, hashing, label-vault, and resource helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(array), allow_pickle=False)
    os.replace(temporary, path)


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def synchronize(device: str | torch.device) -> None:
    resolved = torch.device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(resolved)


def current_rss_bytes() -> int:
    try:
        with Path("/proc/self/status").open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


class _RSSPoller:
    def __init__(self, interval_seconds: float = 0.05):
        self.interval_seconds = interval_seconds
        self.peak = current_rss_bytes()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.peak = max(self.peak, current_rss_bytes())

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        self._thread.join(timeout=2)
        self.peak = max(self.peak, current_rss_bytes())
        return self.peak


@contextmanager
def measured_phase(
    name: str, device: str | torch.device
) -> Iterator[dict[str, Any]]:
    """Measure synchronized wall time, process RSS, and CUDA peak memory."""
    resolved = torch.device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        synchronize(resolved)
        torch.cuda.reset_peak_memory_stats(resolved)
    record: dict[str, Any] = {
        "phase": name,
        "started_at": utc_now(),
        "rss_start_bytes": current_rss_bytes(),
    }
    poller = _RSSPoller()
    poller.start()
    started = time.perf_counter()
    try:
        yield record
    finally:
        synchronize(resolved)
        record["seconds"] = time.perf_counter() - started
        record["rss_peak_bytes"] = poller.stop()
        record["rss_end_bytes"] = current_rss_bytes()
        if resolved.type == "cuda" and torch.cuda.is_available():
            record["gpu_peak_allocated_bytes"] = int(
                torch.cuda.max_memory_allocated(resolved)
            )
            record["gpu_peak_reserved_bytes"] = int(
                torch.cuda.max_memory_reserved(resolved)
            )
        else:
            record["gpu_peak_allocated_bytes"] = 0
            record["gpu_peak_reserved_bytes"] = 0
        record["finished_at"] = utc_now()


class LabelVault:
    """Keep target labels inaccessible until every declared score is frozen."""

    def __init__(
        self,
        *,
        labels_path: Path,
        evaluation_mask_path: Path | None,
        node_count: int,
        events_path: Path,
    ):
        self.labels_path = labels_path
        self.evaluation_mask_path = evaluation_mask_path
        self.node_count = int(node_count)
        self.events_path = events_path
        self._frozen: dict[str, dict] = {}
        self._unlocked = False

    def freeze(
        self,
        *,
        route: str,
        scores_path: Path,
        scores_sha256: str,
        mask_path: Path,
        mask_sha256: str,
    ) -> None:
        if self._unlocked:
            raise AssertionError("Cannot freeze a score after labels were unlocked")
        if route in self._frozen:
            raise AssertionError(f"Duplicate score-freeze route: {route}")
        event = {
            "event": "scores_frozen",
            "route": route,
            "scores_path": str(scores_path.resolve()),
            "scores_sha256": scores_sha256,
            "mask_path": str(mask_path.resolve()),
            "mask_sha256": mask_sha256,
            "at": utc_now(),
        }
        self._frozen[route] = event
        append_jsonl(self.events_path, event)

    def unlock(
        self, required_routes: tuple[str, ...]
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._unlocked:
            raise AssertionError("Labels may be unlocked only once per run")
        missing = [route for route in required_routes if route not in self._frozen]
        if missing:
            raise AssertionError(
                f"Label access requested before all scores froze: {missing}"
            )
        append_jsonl(
            self.events_path,
            {
                "event": "labels_unlocked",
                "required_routes": list(required_routes),
                "at": utc_now(),
            },
        )
        labels = np.asarray(
            np.load(self.labels_path, mmap_mode="r"), dtype=np.int64
        ).reshape(-1)
        if labels.shape != (self.node_count,):
            raise ValueError(
                f"Label length mismatch: {labels.shape} vs {self.node_count}"
            )
        if self.evaluation_mask_path is None:
            mask = np.ones(self.node_count, dtype=np.bool_)
        else:
            mask = np.asarray(
                np.load(self.evaluation_mask_path, mmap_mode="r"),
                dtype=np.bool_,
            ).reshape(-1)
        if mask.shape != (self.node_count,):
            raise ValueError(
                f"Evaluation-mask mismatch: {mask.shape} vs {self.node_count}"
            )
        self._unlocked = True
        return labels, mask
