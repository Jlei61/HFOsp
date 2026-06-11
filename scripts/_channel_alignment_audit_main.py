#!/usr/bin/env python3
"""Channel-name alignment audit across SOZ / rate / geometry (masked) sources.

Bridge logic handles per-subject rate montage variation: rate ch_name may be
bipolar ('A1-A2') or single ('A6'). Geometry & SOZ are always single contacts.
"""
import json
import os
import statistics

ROOT = "/home/honglab/leijiaxin/HFOsp"
OUT = f"{ROOT}/results/topic4_sef_hfo/soz_localization/_channel_alignment_audit_main.json"
RATE_THRESH = 30  # n_events >= 30 in BOTH exact and bridge paths


def load_json(p):
    with open(p) as f:
        return json.load(f)


def classify_montage(names):
    if not names:
        return "single"
    frac = sum("-" in n for n in names) / len(names)
    if frac >= 0.8:
        return "bipolar"
    if frac == 0:
        return "single"
    return "mixed"


def build_cohort():
    soz = {ds: load_json(f"{ROOT}/results/{ds}_soz_core_channels.json")
           for ds in ["yuquan", "epilepsiae"]}
    geom_dir = f"{ROOT}/results/interictal_propagation_masked/rank_displacement/per_subject"
    geom_files = set(os.listdir(geom_dir))
    cohort = []
    for ds in ["yuquan", "epilepsiae"]:
        rate_dir = f"{ROOT}/results/spatial_modulation/per_channel_metrics/{ds}"
        rate_subj = {f.replace("_perchannel.json", "") for f in os.listdir(rate_dir)}
        for s, soz_list in sorted(soz[ds].items()):
            if not soz_list:
                continue
            if s in rate_subj and f"{ds}_{s}.json" in geom_files:
                cohort.append((ds, s))
    return cohort, soz


def audit_subject(ds, subject, soz_core):
    rate = load_json(f"{ROOT}/results/spatial_modulation/per_channel_metrics/{ds}/{subject}_perchannel.json")
    geom = load_json(f"{ROOT}/results/interictal_propagation_masked/rank_displacement/per_subject/{ds}_{subject}.json")

    # --- rate name maps ---
    rate_names = [c["ch_name"].strip() for c in rate["channel_metrics"]]
    rate_n_events = {}
    for c in rate["channel_metrics"]:
        nm = c["ch_name"].strip()
        rate_n_events[nm] = c.get("n_events", 0)
    rate_montage = classify_montage(rate_names)

    # exact-name set (rate name as a single token), gated by n_events >= 30
    exact_ok = {nm for nm in rate_names if rate_n_events.get(nm, 0) >= RATE_THRESH}

    # endpoint -> set of qualifying bipolar rate channels it belongs to (gated >=30)
    endpoint_to_bip = {}
    for nm in rate_names:
        if "-" not in nm:
            continue
        if rate_n_events.get(nm, 0) < RATE_THRESH:
            continue  # gate bipolar path too
        a, b = nm.split("-", 1)
        a, b = a.strip(), b.strip()
        for ep in (a, b):
            endpoint_to_bip.setdefault(ep, set()).add(nm)

    # --- geom_valid contacts from pairs[0] (zip names+joint_valid) ---
    p0 = geom["pairs"][0]
    p0_names = [n.strip() for n in p0["channel_names"]]
    p0_valid = p0["joint_valid"]
    geom_valid = [nm for nm, v in zip(p0_names, p0_valid) if v]
    n_geom_valid = len(geom_valid)

    # --- U_exact: geom_valid contacts that ARE an exact rate name with n_events>=30 ---
    U_exact = {nm for nm in geom_valid if nm in exact_ok}

    # --- U_bridge: exact OR endpoint of a qualifying bipolar rate channel ---
    U_bridge = set()
    n_ambiguous_bridge = 0
    for nm in geom_valid:
        is_exact = nm in exact_ok
        bip_set = endpoint_to_bip.get(nm, set())  # already n_events>=30 gated
        if is_exact or bip_set:
            U_bridge.add(nm)
        if len(bip_set) >= 2:
            n_ambiguous_bridge += 1

    # --- SOZ coverage ---
    soz_set = set(s.strip() for s in soz_core)
    n_soz_core = len(soz_set)
    cov_exact = len(soz_set & U_exact) / n_soz_core if n_soz_core else 0.0
    cov_bridge = len(soz_set & U_bridge) / n_soz_core if n_soz_core else 0.0

    # --- note: diagnose odd things ---
    notes = []
    if rate_montage == "mixed":
        notes.append("mixed-montage rate")
    if any("'" in n for n in soz_set):
        notes.append("apostrophe contacts in SOZ")
    elif any("'" in n for n in geom_valid):
        notes.append("apostrophe contacts in geom")
    # diagnose SOZ contacts NOT covered by bridge -> classify the gap cause
    uncovered = soz_set - U_bridge
    if uncovered:
        geom_valid_set = set(geom_valid)
        not_geom = sorted(c for c in uncovered if c not in geom_valid_set)
        in_geom_no_match = []
        in_geom_lowevt = []
        for c in uncovered:
            if c in geom_valid_set:
                # matched by name but failed threshold?
                exact_lowevt = (c in rate_n_events and rate_n_events[c] < RATE_THRESH)
                bip_lowevt = any(rate_n_events.get(n, 0) < RATE_THRESH
                                 for n in rate_names if "-" in n and c in n.split("-", 1))
                if exact_lowevt or bip_lowevt:
                    in_geom_lowevt.append(c)
                else:
                    in_geom_no_match.append(c)
        if not_geom:
            notes.append(f"{len(not_geom)} SOZ not geom_valid: {not_geom}")
        if in_geom_no_match:
            notes.append(f"{len(in_geom_no_match)} SOZ geom_valid but no rate-name match: {sorted(in_geom_no_match)}")
        if in_geom_lowevt:
            notes.append(f"{len(in_geom_lowevt)} SOZ matched but n_events<30: {sorted(in_geom_lowevt)}")
    if not p0_valid or not all(p0_valid):
        notes.append(f"joint_valid not all True ({sum(p0_valid)}/{len(p0_valid)})")
    if not notes:
        notes.append("ok")

    return {
        "subject": subject,
        "dataset": ds,
        "rate_montage": rate_montage,
        "n_rate": len(rate_names),
        "n_geom_valid": n_geom_valid,
        "n_soz_core": n_soz_core,
        "u_exact": len(U_exact),
        "u_bridge": len(U_bridge),
        "soz_cov_exact": round(cov_exact, 6),
        "soz_cov_bridge": round(cov_bridge, 6),
        "n_ambiguous_bridge": n_ambiguous_bridge,
        "note": "; ".join(notes),
    }


def main():
    cohort, soz = build_cohort()
    rows = []
    for ds, subj in cohort:
        rows.append(audit_subject(ds, subj, soz[ds][subj]))

    # ---------- verification harness (from advisor) ----------
    for r in rows:
        # universal monotonicity
        assert r["u_bridge"] >= r["u_exact"], r
        assert r["soz_cov_bridge"] >= r["soz_cov_exact"] - 1e-9, r
        if r["rate_montage"] == "single":
            assert r["u_bridge"] == r["u_exact"], r
            assert abs(r["soz_cov_bridge"] - r["soz_cov_exact"]) < 1e-9, r
            assert r["n_ambiguous_bridge"] == 0, r
        if r["rate_montage"] == "bipolar":
            assert r["u_exact"] == 0, r
            assert r["soz_cov_exact"] == 0.0, r

    # ---------- summary ----------
    def med(vals):
        return round(statistics.median(vals), 6) if vals else None

    fully_aligned = [r for r in rows
                     if r["soz_cov_exact"] == 1.0 or r["u_exact"] == r["n_geom_valid"]]
    gain_bridge = [r for r in rows if r["soz_cov_bridge"] > r["soz_cov_exact"] + 1e-9]
    remaining = [r for r in rows if r["soz_cov_bridge"] < 1.0 - 1e-9]

    per_ds = {}
    for ds in ["yuquan", "epilepsiae"]:
        drows = [r for r in rows if r["dataset"] == ds]
        per_ds[ds] = {
            "n": len(drows),
            "n_fully_aligned": sum(1 for r in drows if r["soz_cov_exact"] == 1.0 or r["u_exact"] == r["n_geom_valid"]),
            "n_gain_from_bridge": sum(1 for r in drows if r["soz_cov_bridge"] > r["soz_cov_exact"] + 1e-9),
            "n_remaining_cov_lt1": sum(1 for r in drows if r["soz_cov_bridge"] < 1.0 - 1e-9),
            "montage_counts": {m: sum(1 for r in drows if r["rate_montage"] == m)
                               for m in ["bipolar", "single", "mixed"]},
            "median_soz_cov_exact": med([r["soz_cov_exact"] for r in drows]),
            "median_soz_cov_bridge": med([r["soz_cov_bridge"] for r in drows]),
        }

    summary = {
        "cohort_n": len(rows),
        "n_fully_aligned_by_exact": len(fully_aligned),
        "fully_aligned_subjects": [r["subject"] for r in fully_aligned],
        "n_gain_from_bridge": len(gain_bridge),
        "gain_from_bridge_subjects": [r["subject"] for r in gain_bridge],
        "n_remaining_soz_cov_lt1_after_bridge": len(remaining),
        "remaining_subjects": [f"{r['dataset']}:{r['subject']}({r['soz_cov_bridge']})" for r in remaining],
        "per_dataset": per_ds,
        "ambiguity_definition": "n_ambiguous_bridge counts geom contacts mapping to >=2 distinct bipolar rate channels, each already gated n_events>=30 (consistent with bridge threshold)",
        "interpretation": "single-montage subjects with soz_cov<1 are a structural gap the bridge cannot fix (SOZ contact absent from geom-valid or no rate-name match), not a montage artifact",
    }

    out = {"cohort_n": len(rows), "rows": rows, "summary": summary}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print("ALL ASSERTIONS PASSED")
    print(f"wrote {OUT}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
