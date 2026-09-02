#!/usr/bin/env python3
"""Re-rank frozen centroid lags at 0/2/10 ms without rereading raw ECoG."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.group_event_analysis import lag_rank_from_centroids  # noqa: E402


RESULT = Path("results/topic5_ecog_physical_neighborhood_rnn_v0_1")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean_field(ranks: np.ndarray) -> np.ndarray:
    values = np.asarray(ranks)
    score = np.zeros(values.shape, dtype=float)
    for event, row in enumerate(values):
        keep = row >= 0
        if not np.any(keep):
            continue
        length = int(row[keep].max()) + 1
        score[event, keep] = (length - row[keep]) / max(length, 1)
    return score.mean(0)


def rerank(subject: str, tolerance_ms: float) -> tuple[dict[tuple[str, int], np.ndarray], dict[str, object]]:
    per_block = RESULT / f"cache/{subject}/per_block"
    event_map: dict[tuple[str, int], np.ndarray] = {}
    split_map: dict[tuple[str, int], int] = {}
    split_code = {"train": 0, "validation": 1, "test": 2}
    channel_names = None
    for path in sorted(per_block.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            names = [str(value) for value in data["channel_names"].tolist()]
            if channel_names is None:
                channel_names = names
            elif channel_names != names:
                raise RuntimeError(f"channel order drift: {path}")
            participation = np.asarray(data["participation"], dtype=bool)
            lag = np.asarray(data["lag_sec"], dtype=float)
            _, rank = lag_rank_from_centroids(
                lag.T, participation.T, align="first_centroid", tie_tol_ms=float(tolerance_ms)
            )
            rank = rank.T.astype(np.int16)
            stem = str(data["block_stem"].item())
            split = split_code[str(data["split"].item())]
            source = np.asarray(data["source_event_index"], dtype=int)
            for index in range(len(rank)):
                if int(rank[index].max()) < 1:
                    continue
                key = (stem, int(source[index]))
                event_map[key] = rank[index]
                split_map[key] = split
    assert channel_names is not None
    keys = sorted(event_map)
    ranks = np.stack([event_map[key] for key in keys])
    split = np.asarray([split_map[key] for key in keys], dtype=np.int8)
    output = RESULT / f"tie_tolerance_sensitivity/{tolerance_ms:g}ms/{subject}"
    output.mkdir(parents=True, exist_ok=True)
    event_path = output / "events.npz"
    np.savez_compressed(
        event_path, schema_version=np.asarray("topic5_ecog_tie_sensitivity_v0.1"),
        subject=np.asarray(subject), tolerance_ms=np.asarray(float(tolerance_ms)),
        channel_names=np.asarray(channel_names), ranks=ranks, split=split,
        block_stem=np.asarray([key[0] for key in keys]),
        source_event_index=np.asarray([key[1] for key in keys], dtype=np.int32),
    )
    payload: dict[str, object] = {
        "subject": subject, "tolerance_ms": tolerance_ms, "n_events": len(keys),
        "split_counts": {name: int(np.sum(split == code)) for name, code in split_code.items()},
        "participant_count_median": float(np.median(np.sum(ranks >= 0, axis=1))),
        "rank_set_count_median": float(np.median(np.max(ranks, axis=1) + 1)),
        "heldout_empirical_field": mean_field(ranks[split == 2]).tolist(),
        "events_sha256": sha256_file(event_path),
    }
    (output / "SUMMARY.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return event_map, payload


def main() -> None:
    summary = []
    for subject in ("958", "1084"):
        with np.load(RESULT / f"cache/{subject}/events.npz", allow_pickle=False) as base:
            base_field = mean_field(np.asarray(base["ranks"])[np.asarray(base["split"]) == 2])
        for tolerance in (0.0, 2.0, 10.0):
            _, payload = rerank(subject, tolerance)
            field = np.asarray(payload.pop("heldout_empirical_field"), dtype=float)
            payload["heldout_field_spearman_vs_5ms"] = float(spearmanr(base_field, field).statistic)
            summary.append(payload)
    output = RESULT / "summary/TIE_TOLERANCE_SENSITIVITY.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "schema": "topic5_ecog_tie_tolerance_sensitivity_v0.1",
        "complete": True,
        "scope": "data-construction sensitivity only; no model is selected or retrained from these fields",
        "results": summary,
    }, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
