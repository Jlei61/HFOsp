"""Stage 3 — WHY do two parameter-equal foci (same core_mean/core_std at the two ends of
the EE axis) still produce a neg-vs-pos SOURCE ASYMMETRY in the read-out?

Cheap phase (NO re-runs): three artifact-only audits to localize the asymmetry before
spending 35-min/13GB manipulation runs (paired-swap / mirror).

  (1) IS-IT-SYSTEMATIC  — from event_types.csv, break the asymmetry down by cell & seed:
      neg/pos local source-core ignition, neg/pos local duration, readable sign-by-source.
      Consistent sign across seeds => SYSTEMATIC (chase a mechanism); sporadic => per-run luck.
  (2) STRUCTURAL audit  — from per_event/rep_*.npz (the static vth field is the run's field):
      core masks = E-neurons within patch_r of foci[0]=neg / foci[1]=pos; Vth quantiles per
      core, n_E_core, contact-to-core distances. Does the realized THRESHOLD field favor one
      core, or average out across runs? (seed+7 neg / seed+8 pos => should average out.)
  (3) READOUT / FULL-FIELD (rep event) — onset_core gives per-neuron first-spike in the rep
      event: full-field spatial extent (std of posE over fired neurons) + which core led, vs
      the 12-contact n_part/sign. Is the montage symmetric about the two cores? Does a pos-led
      event have the neg core co-active (biasing the endpoint-centroid axis)?

Outputs a JSON verdict + per-run table. Read-only on existing artifacts.
"""
import os, glob, json
import numpy as np
import pandas as pd

ROOT = "results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous"
PE = os.path.join(ROOT, "per_event")


# ---------------------------------------------------------------- (1) systematic?
def audit_systematic():
    d = pd.read_csv(os.path.join(ROOT, "event_types.csv"))
    loc = d[(d.propagation_class == "local") & (d.source.isin(["neg", "pos"]))]
    rg = d[d.propagation_class == "readable_global"]
    out = {"per_seed": [], "per_cell": []}

    def block(df, keycol):
        rows = []
        for key, g in df.groupby(keycol):
            gl = g[(g.propagation_class == "local") & (g.source.isin(["neg", "pos"]))]
            gn = gl[gl.source == "neg"]; gp = gl[gl.source == "pos"]
            grg = g[g.propagation_class == "readable_global"]
            rn = grg[grg.source == "neg"]; rp = grg[grg.source == "pos"]
            rows.append(dict(
                key=str(key),
                neg_local_ignite=round(float(gn.source_core_ignite_frac.median()), 4) if len(gn) else None,
                pos_local_ignite=round(float(gp.source_core_ignite_frac.median()), 4) if len(gp) else None,
                ignite_neg_minus_pos=(round(float(gn.source_core_ignite_frac.median()
                                       - gp.source_core_ignite_frac.median()), 4)
                                      if len(gn) and len(gp) else None),
                neg_local_dur=float(gn.duration.median()) if len(gn) else None,
                pos_local_dur=float(gp.duration.median()) if len(gp) else None,
                neg_rg_fwd=f"{int((rn.sign==1).sum())}/{len(rn)}" if len(rn) else "0/0",
                pos_rg_fwd=f"{int((rp.sign==1).sum())}/{len(rp)}" if len(rp) else "0/0",
                n_neg_local=len(gn), n_pos_local=len(gp)))
        return rows

    # seed extracted from tag suffix _sN
    d2 = d.copy()
    d2["seedkey"] = d2.tag.str.extract(r"_s(\d+)$")[0]
    out["per_seed"] = block(d2.dropna(subset=["seedkey"]), "seedkey")
    out["per_cell"] = block(d, "cell")

    # systematic test: sign of (neg_local_ignite - pos_local_ignite) across cells
    diffs = [r["ignite_neg_minus_pos"] for r in out["per_cell"] if r["ignite_neg_minus_pos"] is not None]
    pos_count = sum(1 for x in diffs if x > 0)
    out["ignite_asym_systematic"] = dict(
        n_cells_with_both=len(diffs),
        n_cells_neg_higher=pos_count,
        median_diff=round(float(np.median(diffs)), 4) if diffs else None,
        verdict=("SYSTEMATIC neg>pos" if diffs and pos_count >= 0.75 * len(diffs)
                 else "SYSTEMATIC pos>neg" if diffs and pos_count <= 0.25 * len(diffs)
                 else "MIXED/sporadic"))
    return out


# ---------------------------------------------------------------- (2) structural
def core_masks(z):
    posE = z["posE"]; isE = z["is_E"].astype(bool); foci = z["foci"]; r = float(z["patch_r"])
    neg = (np.linalg.norm(posE - foci[0], axis=1) <= r) & isE
    pos = (np.linalg.norm(posE - foci[1], axis=1) <= r) & isE
    return neg, pos


def audit_structural():
    rows = []
    for f in sorted(glob.glob(os.path.join(PE, "rep_*.npz"))):
        tag = os.path.basename(f)[4:-4]
        z = np.load(f, allow_pickle=True)
        if str(z["lesion"]) != "twoend_equal":
            continue
        vth = z["vth"]; contacts = z["contacts"]; foci = z["foci"]
        neg, pos = core_masks(z)
        d_neg = float(np.median(np.linalg.norm(contacts - foci[0], axis=1)))
        d_pos = float(np.median(np.linalg.norm(contacts - foci[1], axis=1)))
        rows.append(dict(
            tag=tag, n_E_neg=int(neg.sum()), n_E_pos=int(pos.sum()),
            vth_neg_med=round(float(np.median(vth[neg])), 4) if neg.sum() else None,
            vth_pos_med=round(float(np.median(vth[pos])), 4) if pos.sum() else None,
            vth_neg_q10=round(float(np.percentile(vth[neg], 10)), 4) if neg.sum() else None,
            vth_pos_q10=round(float(np.percentile(vth[pos], 10)), 4) if pos.sum() else None,
            vth_neg_minus_pos_med=(round(float(np.median(vth[neg]) - np.median(vth[pos])), 4)
                                   if neg.sum() and pos.sum() else None),
            contact_dist_neg=round(d_neg, 3), contact_dist_pos=round(d_pos, 3),
            contact_dist_neg_minus_pos=round(d_neg - d_pos, 3)))
    # systematic? lower Vth = more excitable
    dv = [r["vth_neg_minus_pos_med"] for r in rows if r["vth_neg_minus_pos_med"] is not None]
    dc = [r["contact_dist_neg_minus_pos"] for r in rows if r["contact_dist_neg_minus_pos"] is not None]
    verdict = dict(
        n_runs=len(rows),
        vth_neg_minus_pos_median=round(float(np.median(dv)), 4) if dv else None,
        n_runs_neg_lower_vth=sum(1 for x in dv if x < 0),
        contact_dist_neg_minus_pos_median=round(float(np.median(dc)), 4) if dc else None,
        note=("if vth_neg_minus_pos averages ~0 across runs => threshold field does NOT "
              "systematically favor a core (asymmetry is elsewhere); if consistently <0 => "
              "neg core systematically more excitable (a real structural asymmetry to explain)."))
    return rows, verdict


# ---------------------------------------------------------------- (3) readout / full-field rep
def audit_readout_fullfield():
    rows = []
    for f in sorted(glob.glob(os.path.join(PE, "rep_*.npz"))):
        tag = os.path.basename(f)[4:-4]
        z = np.load(f, allow_pickle=True)
        if str(z["lesion"]) != "twoend_equal":
            continue
        posE = z["posE"]; onset = z["onset_core"]; foci = z["foci"]
        fired = np.isfinite(onset)
        neg, pos = core_masks(z)
        # which core led: earliest finite onset neuron's nearer core
        led = None
        if fired.any():
            i0 = int(np.nanargmin(np.where(fired, onset, np.inf)))
            led = "neg" if np.linalg.norm(posE[i0] - foci[0]) < np.linalg.norm(posE[i0] - foci[1]) else "pos"
        # full-field spatial extent (std of fired E positions) vs virtual n_part
        ext = float(np.std(np.linalg.norm(posE[fired] - posE[fired].mean(0), axis=1))) if fired.sum() > 1 else 0.0
        # co-activation of the OTHER core during this event
        neg_fired = float((fired & neg).mean()) if neg.sum() else 0.0
        pos_fired = float((fired & pos).mean()) if pos.sum() else 0.0
        rows.append(dict(
            tag=tag, rep_led_core=led, sign=float(z["sign"]),
            n_fired_E=int(fired.sum()), fullfield_extent_mm=round(ext, 3),
            neg_core_fired_frac=round(neg_fired, 3), pos_core_fired_frac=round(pos_fired, 3),
            other_core_coactive=round(pos_fired if led == "neg" else neg_fired, 3) if led else None,
            dur_ms=round(float(z["event_t_off"] - z["event_t_on"]), 1)))
    return rows


def main():
    sysd = audit_systematic()
    struct_rows, struct_verdict = audit_structural()
    rf_rows = audit_readout_fullfield()
    summary = dict(systematic=sysd["ignite_asym_systematic"],
                   structural_verdict=struct_verdict,
                   structural_per_run=struct_rows,
                   readout_fullfield_per_run=rf_rows,
                   systematic_per_cell=sysd["per_cell"], systematic_per_seed=sysd["per_seed"])
    outp = os.path.join(ROOT, "stage3_source_asymmetry_audit.json")
    json.dump(summary, open(outp, "w"), indent=2, default=lambda o: None)

    print("=== (1) IS THE IGNITION ASYMMETRY SYSTEMATIC? (local events, by cell) ===")
    print(json.dumps(sysd["ignite_asym_systematic"], indent=2))
    print("\nper-seed neg vs pos local ignition (median):")
    for r in sysd["per_seed"]:
        print(f"  seed {r['key']}: neg={r['neg_local_ignite']} pos={r['pos_local_ignite']} "
              f"(Δneg-pos={r['ignite_neg_minus_pos']}) | rg fwd neg={r['neg_rg_fwd']} pos={r['pos_rg_fwd']}")

    print("\n=== (2) STRUCTURAL: realized Vth per core (twoend rep-NPZ runs) ===")
    print(json.dumps(struct_verdict, indent=2))
    for r in struct_rows:
        print(f"  {r['tag']}: n_E neg{r['n_E_neg']}/pos{r['n_E_pos']} | "
              f"vth_med neg{r['vth_neg_med']}/pos{r['vth_pos_med']} (Δ={r['vth_neg_minus_pos_med']}) | "
              f"contact_dist Δneg-pos={r['contact_dist_neg_minus_pos']}")

    print("\n=== (3) READOUT / FULL-FIELD (rep event) ===")
    for r in rf_rows:
        print(f"  {r['tag']}: led={r['rep_led_core']} sign={r['sign']} n_fired_E={r['n_fired_E']} "
              f"ext={r['fullfield_extent_mm']}mm | neg_core_fired={r['neg_core_fired_frac']} "
              f"pos_core_fired={r['pos_core_fired_frac']} other_coactive={r['other_core_coactive']}")
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
