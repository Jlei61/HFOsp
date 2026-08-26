#!/usr/bin/env python3
"""Build and cache the per-patient tensors every downstream job consumes.

Running this once keeps every worker's start-up cost to a file read instead of a
Plackett-Luce fit per patient.  The cache carries the input hashes so a stale
cache cannot be silently reused after the dataset changes.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.topic5_epi_prssm.contracts import (  # noqa: E402
    OUTPUT_ROOT, atomic_write_json, code_revision, package_hash, sha256_obj,
)
from src.topic5_epi_prssm.event_marks import available_subjects, load_patient  # noqa: E402
from src.topic5_epi_prssm.graph_templates import build_patient_graph  # noqa: E402
from src.topic5_epi_prssm.patient_baseline import (  # noqa: E402
    estimate_baseline, variance_decomposition,
)

CACHE = OUTPUT_ROOT / "cache/cohort_v0_1.pt"


def build(force: bool = False) -> Path:
    if CACHE.exists() and not force:
        print(f"cache exists: {CACHE}")
        return CACHE
    payload: dict[str, dict] = {}
    variance_rows = []
    started = time.time()
    for subject in available_subjects():
        events = load_patient(subject)
        graph = build_patient_graph(events, split="train")
        baseline = estimate_baseline(events, split="train")
        variance_rows.append(variance_decomposition(events, baseline, split="train"))
        payload[subject] = {
            "subject": subject, "dataset": events.dataset,
            "participation": torch.as_tensor(events.participation),
            "group_ids": torch.as_tensor(events.group_ids.astype(np.int64)),
            "n_groups": torch.as_tensor(events.group_count.astype(np.int64)),
            "marks": torch.as_tensor(events.node_marks()),
            "delta_t_raw": torch.as_tensor(events.delta_t),
            "session_opening": torch.as_tensor(events.session_opening),
            "load": torch.as_tensor(events.load),
            "split": torch.as_tensor(events.split.astype(np.int64)),
            "event_time": torch.as_tensor(events.event_time),
            "session_index": torch.as_tensor(events.sessions.session_index.astype(np.int64)),
            "adjacency": torch.as_tensor(graph.stack()),
            "node_features": torch.as_tensor(events.contact_features),
            "contact_names": list(events.contact_names),
            "baseline_order": torch.as_tensor(baseline.order_score),
            "baseline_participation": torch.as_tensor(baseline.participation_logit),
            "baseline_stop": float(baseline.stop_logit),
            "baseline_mean_load": float(baseline.mean_load),
            "n_contacts": events.n_contacts, "n_events": events.n_events,
            "geometry_available": bool(graph.geometry_available),
            "n_geometry_mapped": int(graph.n_geometry_mapped),
            "length_scale_mm": float(graph.length_scale_mm),
            "source_hashes": events.source_hashes,
            "n_sessions": int(events.sessions.n_sessions),
        }
        print(f"  {subject:22s} E={events.n_events:7d} N={events.n_contacts:3d} "
              f"geometry={graph.geometry_available}", flush=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, CACHE)
    input_hash = sha256_obj({s: d["source_hashes"] for s, d in payload.items()})
    atomic_write_json(CACHE.with_suffix(".meta.json"), {
        "contract": "topic5_epi_prssm_v0_1_cohort_cache",
        "n_subjects": len(payload), "input_hash": input_hash,
        "code_revision": code_revision(), "package_hash": package_hash(),
        "build_seconds": time.time() - started,
        "variance_decomposition": variance_rows,
    })
    print(f"wrote {CACHE} ({CACHE.stat().st_size/1e6:.0f} MB) in {time.time()-started:.0f}s")
    return CACHE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    build(parser.parse_args().force)


if __name__ == "__main__":
    main()
