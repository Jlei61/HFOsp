"""Round 1 of the rev5 objective: score every field we already have, no new runs.

Before spending a night optimising against a new objective it has to be shown
that the objective ranks the fields we already understand, and that the target
is reachable at all. Both questions are answerable from artefacts on disk.

The feature is signed monotonicity, sign(slope) * r2, one scale-free number per
event. Slope magnitude is deliberately excluded: the model recruits about seven
contacts where the patient recruits twelve, so any slope-magnitude feature would
partly be fitting that mismatch rather than the field.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from src.interictal_propagation import load_subject_propagation_events  # noqa: E402
from src.topic4_core_field_profile import (event_shape,  # noqa: E402
                                           split_by_block)
from src.topic4_core_field_runner import (_placement, atomic_write_json,  # noqa: E402
                                          provenance)

OUT = "results/topic4_sef_hfo/data_driven_core_field_stage3"
STAGE2 = "results/topic4_sef_hfo/data_driven_core_field"
RUN = "results/topic4_sef_hfo/field_swap_subject_snn"
SWEEP = f"{OUT}/cells/sigma1.2"
PATIENT = "/mnt/epilepsia_data/interilca_inter_results/all_data_lns/1146/all_recs"

# frozen before the round
EDGES = np.linspace(-1.0, 1.0, 21)
HELD_OUT_FRAC, SPLIT_SEED = 0.3, 20260808


def axial_map():
    cfg = json.load(open(f"{STAGE2}/config/stage_config.json"))
    reg = _placement(cfg)
    u, c = reg["axis_unit_vec"], reg["center"]
    return {str(n): float((p - c) @ u) for n, p in
            zip(reg["montage_sheet"].names,
                np.asarray(reg["montage_sheet"].contacts, float))}


def signed_monotonicity(ranks, axial):
    s = event_shape(ranks, axial)
    return None if s is None else float(np.sign(s["slope"]) * s["r2"])


def density(values, edges=EDGES):
    h, _ = np.histogram(np.asarray(values, float), bins=edges)
    return h / h.sum() if h.sum() else h


def distance(a, b, edges=EDGES):
    """Total variation between two signed-monotonicity densities on frozen bins."""
    return 0.5 * float(np.abs(density(a, edges) - density(b, edges)).sum())


def patient_events(axial):
    d = load_subject_propagation_events(PATIENT)
    names, R, B, blocks = (d["channel_names"], d["ranks"], d["bools"],
                           np.asarray(d["block_ids"]))
    vals, keep = [], []
    for j in range(R.shape[1]):
        v = signed_monotonicity(
            {names[i]: float(R[i, j]) for i in np.flatnonzero(B[:, j])}, axial)
        if v is not None:
            vals.append(v); keep.append(j)
    return np.asarray(vals), blocks[np.asarray(keep)]


def arm_values(paths, axial):
    out = []
    for p in paths:
        rec = json.load(open(p))
        for ev in rec.get("events", []):
            v = signed_monotonicity(ev.get("ranks"), axial)
            if v is not None:
                out.append(v)
    return np.asarray(out)


def describe(v):
    v = np.asarray(v, float)
    return dict(n=int(len(v)),
                frac_extreme=float(np.mean(np.abs(v) > 0.8)) if len(v) else None,
                frac_middle=float(np.mean(np.abs(v) < 0.3)) if len(v) else None,
                frac_positive=float(np.mean(v > 0)) if len(v) else None,
                median=float(np.median(v)) if len(v) else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    axial = axial_map()

    vals, blocks = patient_events(axial)
    tr, te = split_by_block(blocks, HELD_OUT_FRAC, SPLIT_SEED)
    p_train, p_test = vals[tr], vals[te]
    print(f"patient {len(vals)} events over {len(np.unique(blocks))} recordings "
          f"-> train {len(p_train)} / held out {len(p_test)}\n")

    arms = {}
    arms["hand_placed_two_cores_driven"] = sorted(glob.glob(
        f"{RUN}/readout_epilepsiae_1146_paired_tsrc_highn_s*_20260721.json"))
    arms["hand_placed_two_cores_spontaneous"] = [
        f"{RUN}/readout_epilepsiae_1146_gradient_shared_corefrozen_cr1p5_s5_20260722.json"]
    arms["learned_filament"] = glob.glob(
        f"{RUN}/readout_epilepsiae_1146_learned_core_field_pool_s*.json")

    sweep = json.load(open(f"{OUT}/config/sweep_config.json"))
    for i, c in enumerate(sweep["grid"]["centers"]):
        paths = glob.glob(f"{SWEEP}/c{i:03d}_s*.json")
        if paths:
            arms[f"single_blob@({c[0]:.1f},{c[1]:.1f})"] = paths

    rows = []
    for name, paths in arms.items():
        v = arm_values(paths, axial)
        if len(v) < 20:
            continue
        rows.append(dict(arm=name, **describe(v),
                         distance_train=distance(v, p_train),
                         distance_heldout=distance(v, p_test)))
    rows.sort(key=lambda r: r["distance_train"])

    print(f"{'field':>34} {'n':>5} {'|v|>0.8':>8} {'|v|<0.3':>8} "
          f"{'dist(train)':>12} {'dist(held out)':>14}")
    for r in rows[:12]:
        print(f"{r['arm']:>34} {r['n']:5d} {r['frac_extreme']:8.1%} "
              f"{r['frac_middle']:8.1%} {r['distance_train']:12.3f} "
              f"{r['distance_heldout']:14.3f}")
    print("   ...")
    for r in rows[-3:]:
        print(f"{r['arm']:>34} {r['n']:5d} {r['frac_extreme']:8.1%} "
              f"{r['frac_middle']:8.1%} {r['distance_train']:12.3f} "
              f"{r['distance_heldout']:14.3f}")

    best = rows[0]
    print(f"\nbest reachable distance among fields already on disk: "
          f"{best['distance_train']:.3f} ({best['arm']})")
    print(f"patient: {describe(p_train)}")

    atomic_write_json(dict(
        stage="stage3_rev5_round1",
        feature="signed monotonicity = sign(slope) * r2, one number per event",
        why_not_slope_magnitude=(
            "the model recruits ~7 contacts where the patient recruits ~12, so a "
            "slope-magnitude feature would partly fit that mismatch; r2 and the "
            "sign are scale free"),
        edges=EDGES.tolist(), split=dict(frac=HELD_OUT_FRAC, seed=SPLIT_SEED,
                                         unit="recording block"),
        patient=dict(train=describe(p_train), held_out=describe(p_test),
                     n_recordings=int(len(np.unique(blocks)))),
        arms=rows, provenance=provenance()),
        f"{a.out}/profile_round1.json")
    print(f"\nwrote {a.out}/profile_round1.json")


if __name__ == "__main__":
    main()
