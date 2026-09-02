#!/usr/bin/env python3
"""Data-boundary audit on real timelines (v0.3.3 plan Task 3, clause B6).  Read-only.

For every patient with a v0.3.2 measurement entry it counts, from the real
window builder (``build_carry_segments`` / ``build_anchor_grid``) and the frozen
boundary contract (``v033_evaluator.boundaries``): how many events fall in a
seizure / immediate-postictal interval (never written into the state), how many
target windows are invalid and why, and on how many anchors the mainline
(session carry, autonomous flow across a seizure) differs from the
``sensitivity_hard_seizure_reset`` variant (= v0.3.2 segment carry).
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.topic5_group_event_state.v02.timeline import sessions_from_inventory  # noqa: E402
from src.topic5_group_event_state.v032_eval.contract import load_eval_config  # noqa: E402
from src.topic5_group_event_state.v032_eval.timeline import load_eval_timeline  # noqa: E402
from src.topic5_group_event_state.v033_evaluator import boundaries as B  # noqa: E402


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


def audit_subject(args: tuple[str, str]) -> dict:
    subject, config_path = args
    try:
        cfg = load_eval_config(Path(config_path))
        tl = load_eval_timeline(subject, cfg)
        root = Path(cfg["dataset_root"]) / subject
        scalars = np.load(root / "scalars.npz")
        order = np.asarray(scalars["interictal_index"], dtype=np.int64)
        t_stream = np.asarray(scalars["t_abs"], dtype=np.float64)[order]
        rows = [r for r in csv.DictReader(Path(cfg["session_inventory"]).open()) if r["subject"] == subject]
        sessions = sessions_from_inventory(rows)
        postictal = float(cfg["timeline"]["postictal_exclusion_seconds"])
        units = B.state_carry_units(sessions)
        segments = tl.segments
        ev_unit = B.anchor_carry_index(t_stream, units)
        ev_seg = B.anchor_carry_index(t_stream, segments)
        update = B.event_update_mask(t_stream, tl.seizures, postictal_seconds=postictal)
        inside_session = ev_unit >= 0
        t_anchor = tl.grid.t_anchor
        an_unit = B.anchor_carry_index(t_anchor, units)
        an_seg = B.anchor_carry_index(t_anchor, segments)
        last_main = B.carry_last_event(t_stream, ev_unit, update, t_anchor, an_unit)
        last_hard = B.carry_last_event(t_stream, ev_seg, update, t_anchor, an_seg)
        # v0.3.2 grid.last_event_pos indexes the kept stream; map to raw stream positions
        v032_last = np.where(tl.grid.last_event_pos >= 0,
                             tl.stream_positions[np.clip(tl.grid.last_event_pos, 0, None)], -1)
        per_horizon = {}
        for h_i, horizon in enumerate(tl.horizons_seconds):
            valid = B.target_window_valid(t_anchor, horizon, segments, tl.partition)
            v032 = tl.grid.eligible[:, h_i] & tl.partition.window_within_phase(t_anchor, horizon)
            seg_stop = np.asarray([segments[i].stop_epoch if i >= 0 else np.nan for i in an_seg])
            crosses_segment = (an_seg < 0) | ((t_anchor + horizon) > seg_stop)
            crosses_phase = ~tl.partition.window_within_phase(t_anchor, horizon)
            per_horizon[str(int(horizon))] = {
                "n_valid_target_window": int(valid.sum()),
                "n_invalid_segment_end_or_gap_or_seizure": int(crosses_segment.sum()),
                "n_invalid_phase_crossing": int((crosses_phase & ~crosses_segment).sum()),
                "matches_v032_eligibility": bool(np.array_equal(valid, v032)),
            }
        # target segments that follow a seizure inside the same session
        post_seizure_segments = []
        for seg in segments:
            same_session_earlier = [s for s in segments if s.session_id == seg.session_id and s.stop_epoch <= seg.start_epoch]
            if same_session_earlier:
                post_seizure_segments.append(int(seg.segment_id))
        excluded_seconds = 0.0
        for sz in tl.seizures:
            onset, stop = float(sz["onset_epoch"]), max(float(sz["offset_epoch"]), float(sz["onset_epoch"])) + postictal
            for u in units:
                excluded_seconds += max(0.0, min(stop, u.stop_epoch) - max(onset, u.start_epoch))
        differs = last_main != last_hard
        kept_positions = np.asarray(tl.stream_positions, dtype=np.int64)
        kept_update_ok = bool(update[kept_positions].all()) if kept_positions.size else True
        return {
            "subject": subject, "status": "ok",
            "n_sessions": len(units), "n_target_segments": len(segments), "n_seizures": len(tl.seizures),
            "postictal_exclusion_seconds": postictal,
            "recorded_seconds_in_sessions": float(sum(u.stop_epoch - u.start_epoch for u in units)),
            "target_segment_seconds": float(sum(s.duration_seconds for s in segments)),
            "excluded_seizure_postictal_seconds_in_sessions": excluded_seconds,
            "events": {
                "n_interictal_stream": int(t_stream.size),
                "n_inside_sessions": int(inside_session.sum()),
                "n_outside_sessions": int((~inside_session).sum()),
                "n_in_target_segments": int((ev_seg >= 0).sum()),
                "n_excluded_seizure_or_postictal_inside_sessions": int((inside_session & ~update).sum()),
                "n_kept_by_v032_timeline": int(tl.n_events),
                "kept_equals_in_target_segments": bool(int((ev_seg >= 0).sum()) == tl.n_events),
                "no_updating_event_inside_seizure_or_postictal": kept_update_ok,
            },
            "anchors": {"n_grid": int(t_anchor.size),
                        "all_inside_a_target_segment": bool((an_seg >= 0).all()),
                        "per_horizon": per_horizon},
            "carry": {
                "n_anchors_with_history_mainline": int((last_main >= 0).sum()),
                "n_anchors_with_history_hard_reset": int((last_hard >= 0).sum()),
                "n_anchors_where_variants_differ": int(differs.sum()),
                "fraction_anchors_where_variants_differ": float(differs.mean()) if t_anchor.size else None,
                "n_post_seizure_target_segments_in_same_session": len(post_seizure_segments),
                "hard_reset_variant_equals_v032_last_event_pos": bool(np.array_equal(last_hard, v032_last)),
            },
        }
    except Exception as exc:  # pragma: no cover - surfaced in the JSON, never silenced
        return {"subject": subject, "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/topic5_group_event_state_v032_eval.json")
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "results/group_event_state/v0_3_3/evaluator_assay/data_boundary_audit.json")
    parser.add_argument("--mirror", type=Path,
                        default=Path("/data/hfosp_group_event_state_v0_3_3/agent_a/data_boundary_audit.json"))
    args = parser.parse_args()
    cfg = load_eval_config(args.config)
    if args.subjects:
        subjects = list(args.subjects)
    else:
        table = Path(cfg["data_root"]) / "measurement/patient_learnability_table.csv"
        subjects = sorted(r["subject"] for r in csv.DictReader(table.open()))
    jobs = [(s, str(args.config)) for s in subjects]
    results = []
    with get_context("spawn").Pool(processes=max(1, min(args.workers, len(jobs)))) as pool:
        for res in pool.imap_unordered(audit_subject, jobs):
            results.append(res)
            print(f"{res['subject']}: {res['status']}", flush=True)
    results.sort(key=lambda r: r["subject"])
    ok = [r for r in results if r["status"] == "ok"]
    cohort = {
        "n_subjects": len(results), "n_ok": len(ok),
        "n_events_excluded_seizure_or_postictal": int(sum(r["events"]["n_excluded_seizure_or_postictal_inside_sessions"] for r in ok)),
        "n_anchors_where_variants_differ": int(sum(r["carry"]["n_anchors_where_variants_differ"] for r in ok)),
        "n_anchors_total": int(sum(r["anchors"]["n_grid"] for r in ok)),
        "all_matches_v032_eligibility": bool(all(v["matches_v032_eligibility"] for r in ok for v in r["anchors"]["per_horizon"].values())),
        "all_hard_reset_equals_v032_last_event_pos": bool(all(r["carry"]["hard_reset_variant_equals_v032_last_event_pos"] for r in ok)),
        "all_kept_equals_in_target_segments": bool(all(r["events"]["kept_equals_in_target_segments"] for r in ok)),
        "all_state_events_exclude_seizure_and_postictal": bool(
            all(r["events"]["no_updating_event_inside_seizure_or_postictal"] for r in ok)),
    }
    payload = {
        "format": "group_event_state_v0_3_3_data_boundary_audit",
        "generated": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "contract": {"default_variant": B.DEFAULT_VARIANT, "variants": B.VARIANTS,
                     "clauses": {
                         "B1": "target window [t,t+h) lies inside one target segment (session cut at seizure onset, postictal exclusion removed) and inside one partition phase",
                         "B2": "events inside [onset, max(offset,onset)+postictal) never update the state",
                         "B3": "mainline: across a seizure inside one recorded session the state keeps its autonomous decay (no reset)",
                         "B4": "a recorded gap / session edge is a hard reset (state starts from 0)",
                         "B5": "hard reset at seizure onset (v0.3.2 segment carry) is the named sensitivity variant, never the default",
                     }},
        "config_sha256": cfg["_config_sha256"],
        "cohort": cohort,
        "subjects": results,
        "evidence_label": "DIAGNOSTIC (data contract audit; no scientific result)",
        "sealed_partition_opened": False,
    }
    _atomic_json(args.out, payload)
    _atomic_json(args.mirror, payload)
    print(json.dumps(cohort, indent=2))


if __name__ == "__main__":
    main()
