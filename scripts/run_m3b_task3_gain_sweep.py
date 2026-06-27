"""M3B Task 3 — "same field, two gains" (SECONDARY instrument readout).

P1-1 LOCK: instrument-probe readout of the KICK-DRIVEN LIF RATE FIELD. NOT a spontaneous mechanism,
NOT a seizure-like phase transition. Per plan §2/§3: read the PROPAGATED shape (event-aligned window,
source excluded), NOT the radial seed.

GAIN-KNOB CHOICE (plan §3 "pick in the pilot, log the choice"): the kick-AMPLITUDE sweep (amp 5→16)
was ALL-OR-NONE — the event was byte-identical (axis 3.3°, n_part 12, support 3869px, placement 0.844
at every amplitude). Above the trigger threshold the propagation is set by the OPERATING POINT, not
the kick strength, so kick-amplitude is NOT a gain knob. The effective-recruitment-gain knob is
EXCITABILITY (the operating point `mean_field(ratio)`; fixed supra-threshold kick). This is an
instrument gain knob, NOT the h(W)-coupled μ mechanism (forbidden, plan §1).

Leg-readout (plan §2): as excitability rises and the event EXTENT (field support area) grows, does the
AXIS/SHAPE stay invariant (shape-stable, gain-variant)? Each gain should also land in the real
interictal cohort. If shape changes with gain -> "same scaffold at two gains" is falsified.

Run from worktree root: python scripts/run_m3b_task3_gain_sweep.py
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.getcwd())
from src import propagation_contact_plane_readout as R                            # noqa: E402
import scripts.run_sef_hfo_obs_increment3a as inc3a                              # noqa: E402
from scripts.run_contact_plane_readout import build_record_from_events           # noqa: E402
from src.sef_hfo_lif import mean_field                                           # noqa: E402

REAL_DIR = ("/home/honglab/leijiaxin/HFOsp/results/spatial_modulation/"
            "propagation_geometry/observation_readout/real_subjects")
OUT = "results/topic4_sef_hfo/m3b_bridge/task3_gain_sweep"
RATIOS = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]   # excitability gain knob (logged)
FIXED_AMP = 8.0                                        # supra-threshold trigger (kick-amp is all-or-none)
THETA = 45.0
L, N, PITCH, NCON = 24.0, 96, 4.0, 6
SHAFTS = (15.0, 75.0, 135.0)
TOTAL_PIX = R.GRID_N * R.GRID_N
X, Y = R.make_plane_grid()


def _field(rec):
    f = R.smooth_field(rec, X, Y, sigma_xy=None, scalar="rank", s_thresh=R.S_THRESH)
    return f["T"], f["S"]


def _load_reals():
    items = []
    for fn in sorted(os.listdir(REAL_DIR)):
        if not fn.endswith(".json"):
            continue
        r = json.loads(open(os.path.join(REAL_DIR, fn)).read())
        if r.get("status") in ("no_events", "descriptive_only") or not r.get("channels"):
            continue
        items.append((r["dataset"], r["subject"], _field(r)))
    return items


def _subject_first_median_corr(mf, real_items):
    rows = []
    for ds, subj, rf in real_items:
        c = R.corr_pair_mirror_invariant(mf[0], mf[1], rf[0], rf[1],
                                         s_thresh=R.S_THRESH, overlap_min=R.OVERLAP_MIN)["corr"]
        if c is not None and np.isfinite(c):
            rows.append({"dataset": ds, "subject": subj, "corr": abs(c)})
    folded = R.subject_first_fold(rows, "corr")
    return float(np.median(folded)) if folded else float("nan")


def _record_from_artifact(art):
    coords3d = np.column_stack([np.asarray(art.contact_coords, float),
                                np.zeros(len(art.contact_coords))])
    n_ch = len(art.names)
    return build_record_from_events(
        dataset="model", subject="lif_rate_45deg", template_id="t_a",
        names=list(art.names), ranks=np.asarray(art.ranks, float),
        bools=np.asarray(art.bools, bool), lag_raw=np.asarray(art.lag_raw, float),
        coords=coords3d, mapped=np.ones(n_ch, bool), soz_core=set(),
        montage="single", lag_time_unit="ms", spacing_mm=PITCH)


def run():
    os.makedirs(OUT, exist_ok=True)
    inc3a.PULSE = dict(radius=2.0, amp=FIXED_AMP, t_on=0.0, t_off=30.0)
    center = np.zeros(2)
    half = L / 2.0
    montage = inc3a._montage(center, PITCH, NCON, SHAFTS)
    kick45 = center - 0.6 * half * np.array([np.cos(np.deg2rad(THETA)), np.sin(np.deg2rad(THETA))])
    real_items = _load_reals()

    rows = []
    fields = []   # (ratio, (T,S), axis_err, support_px)
    for ratio in RATIOS:
        op = mean_field(ratio)
        r = inc3a._read(op, np.deg2rad(THETA), 2.0, kick45, montage, np.deg2rad(THETA),
                        N, L, PITCH, save_diag=True)
        if "_diag" not in r or r.get("axis_err") is None:
            rows.append({"ratio": ratio, "status": "no_event_or_no_axis", "n_part": r.get("n_part")})
            continue
        art = r["_diag"]["artifact"]
        rec = _record_from_artifact(art)
        mf = _field(rec)
        s_area = int((mf[1] >= R.S_THRESH).sum())
        runaway = s_area > 0.8 * TOTAL_PIX
        place = _subject_first_median_corr(mf, real_items)
        rows.append({"ratio": ratio, "status": ("runaway" if runaway else rec.get("status", "ok")),
                     "axis_err_deg": r["axis_err"], "n_part": int(r["n_part"]),
                     "readability": r.get("readability"), "field_support_pixels": s_area,
                     "support_frac": round(s_area / TOTAL_PIX, 3),
                     "placement_corr_to_interictal_cohort": place})
        if not runaway:
            fields.append((ratio, mf, r["axis_err"], s_area))

    consec = []
    for i in range(len(fields) - 1):
        a, fa = fields[i][0], fields[i][1]
        b, fb = fields[i + 1][0], fields[i + 1][1]
        c = R.corr_pair_mirror_invariant(fa[0], fa[1], fb[0], fb[1],
                                         s_thresh=R.S_THRESH, overlap_min=R.OVERLAP_MIN)["corr"]
        consec.append({"ratio_pair": [a, b], "field_corr": (float(c) if c is not None else None)})

    supports = [f[3] for f in fields]
    axerrs = [f[2] for f in fields]
    axis_range = (float(max(axerrs) - min(axerrs)) if axerrs else float("nan"))
    min_consec = min([c["field_corr"] for c in consec if c["field_corr"] is not None], default=float("nan"))
    support_range = [min(supports), max(supports)] if supports else None
    support_growth = (max(supports) / max(min(supports), 1) if supports else float("nan"))

    shape_stable = (np.isfinite(axis_range) and axis_range <= 15.0
                    and np.isfinite(min_consec) and min_consec >= 0.85)
    gain_variant = (support_range is not None and support_growth >= 1.3)
    if shape_stable and gain_variant:
        verdict = ("instrument-probe SAME-FIELD-TWO-GAINS supported: as excitability rises the event "
                   "extent grows (support {} px, x{:.2f}) while the axis stays ~45deg (range {:.1f}deg) "
                   "and consecutive fields stay correlated (min {:.2f}). NOT a mechanism phase transition."
                   .format(support_range, support_growth, axis_range, min_consec))
    elif not gain_variant:
        verdict = ("INCONCLUSIVE: excitability did not give a graded recruitment-extent range "
                   "(support {} px, x{:.2f}); the model's event size is ~fixed at this operating regime "
                   "(consistent with the static-μ flat-event caveat). 'Two gains' not accessible in this "
                   "instrument — NOT evidence against the scaffold, an instrument-range limit."
                   .format(support_range, support_growth))
    else:
        verdict = ("FALSIFIES same-scaffold: the field shape/axis CHANGES with gain (axis range {:.1f}deg, "
                   "min consec corr {:.2f}).".format(axis_range, min_consec))

    out = {
        "task": "M3B Task 3 — same field two gains (SECONDARY instrument readout)",
        "scope_lock": "kick-rate-field instrument probe; NOT a spontaneous mechanism / phase transition (P1-1)",
        "gain_knob": "excitability mean_field(ratio), fixed supra-threshold kick amp=8",
        "kick_amp_is_all_or_none": "confirmed: amp 5->16 gave a byte-identical event (axis 3.3, n_part 12, "
                                   "support 3869px, placement 0.844) -> kick amplitude is NOT a gain knob",
        "ratios": RATIOS,
        "per_gain": rows,
        "consecutive_field_corr": consec,
        "support_pixels_range": support_range,
        "support_growth_factor": support_growth,
        "axis_err_range_deg": axis_range,
        "min_consecutive_field_corr": min_consec,
        "shape_stable": bool(shape_stable),
        "gain_variant": bool(gain_variant),
        "verdict": verdict,
    }
    with open(os.path.join(OUT, "task3_gain_sweep.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    run()
