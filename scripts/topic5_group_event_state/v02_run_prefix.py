#!/usr/bin/env python3
"""H2a same-prefix continuation for one or more patients (Agent A, A3).

For events that began the same way, does adding the frozen state improve the
prediction of how they continued?  Four outcomes are kept apart: whether
recruitment continued at all, which further contacts it reached, how far it got
(extent / STOP), and the later multiband expression.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from src.topic5_group_event_state.v02.contract_paths import (  # noqa: E402
    DATASET_ROOT,
)
from src.topic5_group_event_state.v02.prefix import (  # noqa: E402
    EARLY_WINDOW_SECONDS,
    OUTCOMES,
    build_prefix_data,
    fit_and_score_outcome,
)
from src.topic5_group_event_state.v02.readout import (  # noqa: E402
    ReadoutConfig,
    block_circular_shift,
)
from src.topic5_group_event_state.v02.registry import atomic_write_json  # noqa: E402
from src.topic5_group_event_state.v02.runtime import (  # noqa: E402
    already_done,
    config_fingerprint,
    save_result,
)
from src.topic5_group_event_state.v02.subject import (  # noqa: E402
    SubjectTimelineConfig,
    load_subject_timeline,
)
from src.topic5_group_event_state.contract import TIE_TOLERANCE_SECONDS  # noqa: E402

DEFAULT_OUT = Path("/data/hfosp_group_event_state_v0_2/agent_a/prefix")

# The state used as a wrong-time control is rolled within its recorded session by
# at least this much physical time -- the same idea as the future-block shift.
SHIFT_SECONDS = 1800.0


def early_waveform_summary(root: Path, raw_positions: np.ndarray, index: dict,
                           chunk: int = 2000) -> np.ndarray:
    """Mean |x| and peak-to-peak over the first 100 ms of the event core.

    ``(N, C, 2V)`` -- two numbers per reference view per contact.  This is the
    "early waveform" half of the prefix descriptor (SP A3).
    """

    fs = float(index["native_rate_hz"])
    pre = int(round(0.250 * fs))
    width = max(int(round(EARLY_WINDOW_SECONDS * fs)), 4)
    n_ctx = int(index["n_context_samples"])
    stop = min(pre + width, n_ctx)
    wave = np.load(root / "waveform.npy", mmap_mode="r")
    out = np.zeros((raw_positions.size, wave.shape[1], 2 * wave.shape[2]),
                   dtype=np.float32)
    for lo in range(0, raw_positions.size, chunk):
        idx = raw_positions[lo:lo + chunk]
        block = np.asarray(wave[idx][:, :, :, pre:stop], dtype=np.float32)
        out[lo:lo + chunk] = np.concatenate(
            [np.abs(block).mean(-1), block.max(-1) - block.min(-1)], axis=-1
        )
    return out


def _states_for(state_specs, subject: str, n_events: int) -> dict[str, np.ndarray]:
    """Per-event states, read straight from each training run directory.

    Event states are ~70 MB per (patient, producer, seed); copying them into a
    parallel directory tree would duplicate ~11 GB for no benefit, so the runner
    reads ``runs/<subject>/<producer>/seed<k>/event_state.npz`` in place.
    """

    out: dict[str, np.ndarray] = {}
    for name, path in state_specs:
        path = Path(path)
        if not path.exists():
            continue
        with np.load(path) as z:
            values = np.asarray(z["state"], dtype=np.float64)
        if values.shape[0] != n_events:
            raise ValueError(f"{path}: {values.shape[0]} states for {n_events} events")
        out[name] = values
    return out


def _event_state_specs(producer_root: Path, subject: str, producers, seeds):
    return [
        (f"{producer}_seed{seed}",
         Path(producer_root) / "runs" / subject / producer / f"seed{seed}"
         / "event_state.npz")
        for producer in producers for seed in seeds
    ]


def _fit_arm(x, data, tr, te, config) -> dict:
    arm: dict[str, dict] = {}
    arm["continues"] = fit_and_score_outcome(
        x[tr], data.continues[tr].astype(float), np.ones(tr.size, bool),
        x[te], data.continues[te].astype(float), np.ones(te.size, bool),
        kind="bernoulli", config=config,
    )
    arm["later_participation"] = fit_and_score_outcome(
        x[tr], data.later_participation[tr].astype(float), data.later_valid[tr],
        x[te], data.later_participation[te].astype(float), data.later_valid[te],
        kind="bernoulli", config=config,
    )
    arm["extent"] = fit_and_score_outcome(
        x[tr], data.extent[tr], np.ones((tr.size, data.extent.shape[1]), bool),
        x[te], data.extent[te], np.ones((te.size, data.extent.shape[1]), bool),
        kind="gaussian", config=config,
    )
    mb_tr = np.broadcast_to(data.later_multiband_valid[tr][:, None],
                            (tr.size, data.later_multiband.shape[1]))
    mb_te = np.broadcast_to(data.later_multiband_valid[te][:, None],
                            (te.size, data.later_multiband.shape[1]))
    arm["later_multiband"] = fit_and_score_outcome(
        x[tr], data.later_multiband[tr], mb_tr,
        x[te], data.later_multiband[te], mb_te,
        kind="gaussian", config=config,
    )
    return arm


def _run_one(args: tuple) -> dict:
    subject, out_root, producer_root, producers, seeds, cfg_hash, max_iter = args
    started = time.time()
    result_path = Path(out_root) / "per_subject" / f"{subject}.json"
    if already_done(result_path, cfg_hash):
        return {"subject": subject, "status": "skipped_done"}
    try:
        tl = load_subject_timeline(subject, config=SubjectTimelineConfig())
        root = DATASET_ROOT / subject
        scalars = np.load(root / "scalars.npz")
        order = np.asarray(scalars["interictal_index"], dtype=np.int64)
        raw_positions = order[tl.stream_positions]
        band_keep = np.flatnonzero(np.asarray(tl.index["band_available"], dtype=bool))

        participation = np.asarray(
            np.load(root / "participation.npy", mmap_mode="r")[raw_positions])
        relative_delay = np.asarray(
            np.load(root / "relative_delay.npy", mmap_mode="r")[raw_positions])
        band_features = np.asarray(
            np.load(root / "band_features.npy", mmap_mode="r")[raw_positions])
        early = early_waveform_summary(root, raw_positions, dict(tl.index))

        train_events = np.flatnonzero(tl.event_times < tl.split.boundary_epochs[0])
        data = build_prefix_data(
            participation, relative_delay, band_features, early,
            band_keep=band_keep, train_positions=train_events,
            tie_tolerance_seconds=float(tl.index.get("tie_tolerance_seconds",
                                                     TIE_TOLERANCE_SECONDS)),
        )
        del participation, relative_delay, band_features, early

        onehot = np.zeros((data.n_events, len(data.bucket_labels)))
        onehot[np.arange(data.n_events), data.bucket] = 1.0
        x_prefix = np.concatenate([np.ones((data.n_events, 1)), data.features, onehot], 1)

        te = np.flatnonzero(tl.event_times >= tl.split.boundary_epochs[1])
        tr = train_events
        config = ReadoutConfig(max_iter=max_iter)

        bucket_counts = {
            label: int((data.bucket[tr] == i).sum())
            for i, label in enumerate(data.bucket_labels)
        }
        result = {
            "subject": subject,
            "dataset": tl.dataset,
            "n_events": int(data.n_events),
            "n_train_events": int(tr.size),
            "n_test_events": int(te.size),
            "n_buckets": len(data.bucket_labels),
            "train_events_per_bucket": bucket_counts,
            "fraction_test_events_that_continue": float(data.continues[te].mean()),
            "median_later_scoreable_contacts": float(np.median(data.later_valid.sum(1))),
            "arms": {"prefix": _fit_arm(x_prefix, data, tr, te, config)},
            "producers_present": [],
        }

        states = _states_for(
            _event_state_specs(Path(producer_root), subject, producers, seeds),
            subject, data.n_events,
        )
        dt = np.median(np.diff(tl.event_times[tl.event_times < tl.event_times[-1]]))
        shift_events = max(int(round(SHIFT_SECONDS / max(dt, 1e-3))), 1)
        for name, values in sorted(states.items()):
            result["producers_present"].append(name)
            x_state = np.concatenate([x_prefix, values], 1)
            result["arms"][f"prefix+S({name})"] = _fit_arm(x_state, data, tr, te, config)
            shifted = block_circular_shift(
                values, tl.event_segment, tl.event_times, shift_events
            )
            result["arms"][f"prefix+shift(S({name}))"] = _fit_arm(
                np.concatenate([x_prefix, shifted], 1), data, tr, te, config
            )
            result.setdefault("shift", {})[name] = {
                "shift_events": shift_events,
                "target_seconds": SHIFT_SECONDS,
                "median_inter_event_seconds": float(dt),
            }
        result["seconds"] = round(time.time() - started, 1)
        save_result(result_path, result, cfg_hash)
        return {"subject": subject, "status": "ok", "seconds": result["seconds"]}
    except Exception as exc:
        payload = {"subject": subject, "status": "failed",
                   "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc(limit=8)}
        atomic_write_json(Path(out_root) / "failures" / f"{subject}.json", payload)
        return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--producer-root", type=Path,
                        default=Path("/data/hfosp_group_event_state_v0_2/agent_a/producers/main"))
    parser.add_argument("--producers", nargs="+", default=["P_local", "P_slow"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--tag", default="main")
    args = parser.parse_args()

    subjects = args.subjects or sorted(
        p.name for p in DATASET_ROOT.iterdir() if (p / "index.json").exists()
    )
    out_root = Path(args.out_root) / args.tag
    (out_root / "per_subject").mkdir(parents=True, exist_ok=True)
    (out_root / "failures").mkdir(parents=True, exist_ok=True)
    cfg_hash = config_fingerprint(
        SubjectTimelineConfig().as_dict(), args.max_iter, SHIFT_SECONDS,
        sorted(args.producers), sorted(args.seeds),
    )
    payload = [(s, str(out_root), str(args.producer_root), list(args.producers),
                list(args.seeds), cfg_hash, args.max_iter) for s in subjects]
    results = []
    started = time.time()
    with mp.get_context("spawn").Pool(processes=max(1, args.workers)) as pool:
        for i, res in enumerate(pool.imap_unordered(_run_one, payload), start=1):
            results.append(res)
            print(f"[{i}/{len(subjects)}] {res['subject']}: {res['status']} "
                  f"{res.get('seconds', '')}", flush=True)
    atomic_write_json(out_root / "manifest.json", {
        "tag": args.tag, "subjects": sorted(subjects), "results": results,
        "config_hash": cfg_hash, "producer_root": str(args.producer_root),
        "producers": list(args.producers), "seeds": list(args.seeds),
        "n_ok": sum(1 for r in results if r["status"] in ("ok", "skipped_done")),
        "n_failed": sum(1 for r in results if r["status"] == "failed"),
        "elapsed_seconds": round(time.time() - started, 1),
    })
    print(json.dumps({"n_ok": sum(1 for r in results if r["status"] != "failed"),
                      "n_failed": sum(1 for r in results if r["status"] == "failed")}))


if __name__ == "__main__":
    main()
