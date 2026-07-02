"""Candidate screen for the Topic 4 off-axis surround stimulation experiment (NO SNN simulation).

For each predeclared patient-like subject, read its committed subject-SNN artifact
(`figdata_*.npz` geometry + `readout_*.json` event counts), compute the pathological-axis frame,
planar-geometry metrics, and the off-axis / on-axis contact selections, then decide geometry
eligibility and rank the candidates. Writes `candidate_screen.json`.

This screen ONLY checks geometry + already-recorded readout counts. The two simulation-dependent
gates -- ">=2 readable events BEFORE stim_on" and "no runaway/tonic before the stim window" -- are
NOT decided here; they are checked in the pilot runner (plan Task 5). ECoG-like subjects are chosen
by planar geometry, NOT by clinical channel-type labels (plan boundary).

    python scripts/screen_topic4_offaxis_surround_candidates.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic4_offaxis_surround_stim import (axis_frame, project_contacts,  # noqa: E402
    select_offaxis_surround_contacts, select_onaxis_corridor_contacts, onaxis_effective_halfwidth)

FIGDATA_DIR = ROOT / "results" / "topic4_sef_hfo" / "field_swap_subject_snn"
OUT = ROOT / "results" / "topic4_sef_hfo" / "offaxis_surround_stim" / "candidate_screen.json"

# Predeclared candidates (plan): ECoG-like Epilepsiae by GEOMETRY first, yuquan as geometry backups.
CANDIDATES = ["epilepsiae_916", "epilepsiae_590", "epilepsiae_442", "epilepsiae_1150",
              "yuquan_zhaojinrui", "yuquan_zhaochenxi"]

# v1 defaults (plan §Geometry), locked here; corridor/offaxis scale with the inter-core distance.
N = 4
STIM_RADIUS_MM = 2.0
PCA_MIN = 0.45
OFFAXIS_SPAN_MIN_MM = 8.0
MIN_CLEAN_EVENTS = 6


def _find(prefix, subject):
    hits = sorted(glob.glob(str(FIGDATA_DIR / f"{prefix}_{subject}_cohort_*")))
    if not hits:
        hits = sorted(glob.glob(str(FIGDATA_DIR / f"{prefix}_{subject}_*")))
    return hits[0] if hits else None


def _pca_ratio(pts):
    """minor/major principal-std ratio of the 2-D contact cloud (1.0 = isotropic, ~0 = a line)."""
    ev = np.linalg.eigvalsh(np.cov(np.asarray(pts, float).T))
    ev = np.clip(ev, 0.0, None)
    return float(np.sqrt(ev.min() / ev.max())) if ev.max() > 0 else 0.0


def screen_one(subject):
    fd_path = _find("figdata", subject)
    ro_path = _find("readout", subject)
    if fd_path is None:
        return {"subject": subject, "eligible": False, "reasons": ["no figdata artifact found"]}
    fd = np.load(fd_path, allow_pickle=True)
    ro = json.loads(Path(ro_path).read_text()) if ro_path else {}
    contacts = np.asarray(fd["contacts"], float)
    valid = np.asarray(fd["valid"], bool) if "valid" in fd else np.ones(len(contacts), bool)
    foci = np.asarray(fd["foci"], float)
    core_r = float(fd["core_r"])
    frame = axis_frame(foci[0], foci[1])
    inter_core = frame["inter_core_mm"]
    corridor_hw = max(1.5, 0.15 * inter_core)
    offaxis_min = max(2.5, corridor_hw)
    # exclude source/sink core contacts AND invalid contacts from stimulation selection
    core = ((np.linalg.norm(contacts - foci[0], axis=1) <= core_r)
            | (np.linalg.norm(contacts - foci[1], axis=1) <= core_r))
    exclude = core | ~valid
    pr = project_contacts(contacts, frame)
    offaxis_span = float(pr["off"][valid].max() - pr["off"][valid].min()) if valid.any() else 0.0
    pca = _pca_ratio(contacts[valid]) if valid.sum() >= 2 else 0.0
    n_clean = int(ro.get("n_clean", 0))

    # ELIGIBILITY = planar geometry + off-axis surround (MAIN arm) selectable + enough readable events.
    # The on-axis comparator is NOT an eligibility gate (plan) -- it is reported best-effort below.
    reasons = []
    off_idx = None
    try:
        off_idx = select_offaxis_surround_contacts(contacts, frame, exclude, N, corridor_hw, offaxis_min).tolist()
    except ValueError as e:
        reasons.append(f"off-axis surround (main arm): {e}")
    if pca < PCA_MIN:
        reasons.append(f"pca_ratio {pca:.2f} < {PCA_MIN} (not planar/broad enough)")
    if offaxis_span < OFFAXIS_SPAN_MIN_MM:
        reasons.append(f"off-axis span {offaxis_span:.1f} < {OFFAXIS_SPAN_MIN_MM} mm")
    if n_clean < MIN_CLEAN_EVENTS:
        reasons.append(f"n_clean {n_clean} < {MIN_CLEAN_EVENTS}")
    eligible = not reasons

    # on-axis corridor comparator: best-effort (robust nearest-axis fallback), flag if degraded
    on_idx = None
    onaxis = {}
    try:
        on_idx = select_onaxis_corridor_contacts(contacts, frame, exclude, N, corridor_hw).tolist()
        eff = onaxis_effective_halfwidth(contacts, frame, on_idx)
        onaxis = {"effective_halfwidth_mm": round(eff, 2), "nominal_corridor_hw_mm": round(corridor_hw, 2),
                  "degraded": bool(eff > offaxis_min)}   # as off-axis as the surround -> weak comparator
    except ValueError as e:
        onaxis = {"error": str(e)}

    names = [str(x) for x in fd["names"]] if "names" in fd else [str(i) for i in range(len(contacts))]
    return {
        "subject": subject, "montage": ro.get("montage"), "figdata": Path(fd_path).name,
        "inter_core_mm": round(inter_core, 2), "pca_ratio": round(pca, 3),
        "offaxis_span_mm": round(offaxis_span, 2), "corridor_halfwidth_mm": round(corridor_hw, 2),
        "offaxis_min_mm": round(offaxis_min, 2), "core_r_mm": core_r,
        "n_valid_contacts": int(valid.sum()), "n_core_contacts": int(core.sum()),
        "n_clean": n_clean, "n_events": int(ro.get("n_events", 0)), "bidirectional": ro.get("bidirectional"),
        "offaxis_contacts": [names[i] for i in off_idx] if off_idx else None,
        "onaxis_contacts": [names[i] for i in on_idx] if on_idx else None,
        "offaxis_idx": off_idx, "onaxis_idx": on_idx, "onaxis_comparator": onaxis,
        "eligible_geometry": eligible, "reasons": reasons,
        "sim_gated_checks": ">=2 readable events before stim_on AND no runaway before stim window -> pilot only",
    }


def main():
    rows = [screen_one(s) for s in CANDIDATES]
    eligible = [r for r in rows if r.get("eligible_geometry")]
    eligible.sort(key=lambda r: (-r["n_clean"], -r["pca_ratio"]))          # most events, then most planar
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "defaults": dict(N=N, stim_radius_mm=STIM_RADIUS_MM, pca_min=PCA_MIN,
                         offaxis_span_min_mm=OFFAXIS_SPAN_MIN_MM, min_clean_events=MIN_CLEAN_EVENTS),
        "ranked_eligible": [r["subject"] for r in eligible],
        "candidates": rows}, indent=2))
    print(f"wrote {OUT}")
    print("RANKED ELIGIBLE (geometry):", [r["subject"] for r in eligible] or "NONE")
    for r in rows:
        tag = "OK " if r.get("eligible_geometry") else "REJ"
        oc = r.get("onaxis_comparator", {})
        on_flag = (f" on={r.get('onaxis_contacts')} (deg={oc.get('degraded')}, eff_hw={oc.get('effective_halfwidth_mm')})"
                   if r.get("onaxis_contacts") else f" on=FAIL({oc.get('error','')})")
        print(f"  [{tag}] {r['subject']:22s} pca={r.get('pca_ratio')} off_span={r.get('offaxis_span_mm')} "
              f"n_clean={r.get('n_clean')} off={r.get('offaxis_contacts')}{on_flag}"
              + ("" if r.get("eligible_geometry") else f" | REJ: {'; '.join(r['reasons'])}"))
    print("DONE_OFFAXIS_SCREEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
