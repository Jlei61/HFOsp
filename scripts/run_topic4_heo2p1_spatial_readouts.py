"""FCXR-HEO2.1 — recompute the three DE-CONFLATED spatial readouts on the 48 Phase-0 cells + 6 Phase-1
arms (compute-free; the per-contact lfp_trace (T,15) is stored). Splits the old single `coverage` into:
active_recruitment (how many contacts active) · broadband_coverage_1_80 (how many broadband-shaped, 1-80
Hz only) · phase_coherence (oscillation_probe cross-contact coherence = synchronous?). Also applies the
tail-segmented relabels to the arms. Augments phase0_state_map.json + phase1_verdict.json in place.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import src.topic4_mz_fcxr_heo2 as H2  # noqa: E402
from topic4_mz_fcxr_heo1 import build_baseline_reference, band_db_field, oscillation_probe  # noqa: E402

DT = 0.05
MZ = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_full_conductance_spatial_relay")
HEO1 = os.path.join(MZ, "high_energy_oscillatory_branch")
OUT = os.path.join(MZ, "broadband_diagnostic")


def _readouts(lfp, rate, ref):
    ddb = band_db_field(lfp, DT, ref)                       # (15, 6) per-contact six-band ΔdB
    probe = oscillation_probe(lfp, rate, DT)
    return dict(active_recruitment=H2.active_recruitment(ddb),
                broadband_coverage_1_80=H2.broadband_coverage_1_80(ddb),
                phase_coherence=round(float(probe["coherence_med"]), 3),
                phase_span_deg=round(float(probe["phase_span_deg"]), 1),
                frac_contacts_common=round(float(probe["frac_contacts_common"]), 3))


def main():
    d0 = np.load(os.path.join(HEO1, "baseline_lfp_seed1.npz"), allow_pickle=True)
    ref = build_baseline_reference(np.asarray(d0["lfp_trace"], float), np.asarray(d0["rate_E"], float), DT)

    # -------- Phase 0: 48 cells (full trace) --------
    cells = json.load(open(os.path.join(OUT, "phase0_state_map.json")))
    for c in cells:
        t = np.load(os.path.join(HEO1, "screen_cells", c["label"] + "_trace.npz"), allow_pickle=True)
        c.update(_readouts(np.asarray(t["lfp_trace"], float), np.asarray(t["rate_E"], float), ref))
    json.dump(cells, open(os.path.join(OUT, "phase0_state_map.json"), "w"), indent=1)

    # -------- Phase 1: 6 arms (post-enable window for dyn arms) --------
    verd = json.load(open(os.path.join(OUT, "phase1_verdict.json")))
    arm_readouts = {}
    for lab in verd:
        t = np.load(os.path.join(OUT, "arms", lab + "_trace.npz"), allow_pickle=True)
        rate = np.asarray(t["rate_E"], float); lfp = np.asarray(t["lfp_trace"], float)
        men = float(t["m_enable_ms"]); k = int(men / DT) if men > 0 else 0
        post_lfp = lfp[k:] if lfp[k:].shape[0] > int(0.5 / DT * 1000) else lfp
        post_rate = rate[k:] if rate[k:].size > int(0.5 / DT * 1000) else rate
        ro = _readouts(post_lfp, post_rate, ref)
        ro["segment_label"] = H2.segment_state_label(rate, 1000.0 / DT, m_enable_ms=(men or None), dt=DT)
        verd[lab].update(ro); arm_readouts[lab] = ro
    json.dump(verd, open(os.path.join(OUT, "phase1_verdict.json"), "w"), indent=1)

    # -------- honest decomposition summary --------
    from collections import Counter
    print("=== Phase-0 de-conflated readouts by class ===")
    for cls in ("sparse_event_train", "transitional", "tonic_16Hz_cycle"):
        sub = [c for c in cells if c["class"] == cls]
        rec = np.array([c["active_recruitment"] for c in sub]); bb = np.array([c["broadband_coverage_1_80"] for c in sub])
        coh = np.array([c["phase_coherence"] for c in sub])
        print(f"  {cls:20s} n={len(sub):2d}  recruit(med) {np.median(rec):4.0f}/15  "
              f"broadband_1-80(med) {np.median(bb):4.0f}/15  coherence(med) {np.median(coh):.2f}")
    anc = [c for c in cells if c["label"] == "gq0.999_A8_D0.15_nokick"][0]
    print(f"\n  ANCHOR gq0.999_A8_D0.15_nokick: recruit {anc['active_recruitment']}/15  "
          f"broadband_1-80 {anc['broadband_coverage_1_80']}/15  coherence {anc['phase_coherence']}  "
          f"phase_span {anc['phase_span_deg']}deg  -> widely-recruited / narrowband / synchronous")
    best = [c for c in cells if c["label"] == "gq0.999_A8_D0.13_nokick"][0]
    print(f"  CLOSEST gq0.999_A8_D0.13_nokick: recruit {best['active_recruitment']}/15  "
          f"broadband_1-80 {best['broadband_coverage_1_80']}/15  coherence {best['phase_coherence']}")

    print("\n=== Phase-1 arms: readouts + relabel ===")
    for lab in ["m_off", "dyn_tau250_frac0.05", "dyn_tau250_frac0.1", "dyn_tau750_frac0.05",
                "dyn_tau750_frac0.1", "static_K"]:
        r = arm_readouts[lab]
        print(f"  {lab:22s} old={verd[lab]['verdict']:16s} -> seg={r['segment_label']:32s} "
              f"recruit {r['active_recruitment']:2d}/15 broadband {r['broadband_coverage_1_80']:2d}/15 coh {r['phase_coherence']}")
    print("\n[heo2.1] augmented phase0_state_map.json + phase1_verdict.json")


if __name__ == "__main__":
    main()
