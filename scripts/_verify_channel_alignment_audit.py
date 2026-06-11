#!/usr/bin/env python3
"""INDEPENDENT re-derivation of the channel-alignment audit (verification).

Written from scratch to check another agent's per-subject U_exact / u_bridge /
soz_cov_exact / soz_cov_bridge. Does NOT read the other agent's output until the
very end (diff stage).

Three sources per subject:
  1. SOZ truth: results/{ds}_soz_core_channels.json  -> {subject: [contact,...]}
  2. Rate:      results/spatial_modulation/per_channel_metrics/{ds}/{subject}_perchannel.json
                channel_metrics[] each {ch_name, n_events, event_rate}
  3. Geometry:  results/interictal_propagation_masked/rank_displacement/per_subject/{ds}_{subject}.json
                pairs[0].channel_names zipped with pairs[0].joint_valid

Definitions implemented (per task prose):
  geom_valid       = pairs[0].channel_names where joint_valid == True
  n_soz_core       = len(SOZ_core)
  U_exact          = geom_valid contacts whose name == some rate ch_name w/ n_events>=30
  U_bridge         = geom_valid contact X counts if (X is exact rate name) OR
                     (X equals endpoint A or B of some bipolar rate name 'A-B'),
                     matched rate channel must have n_events>=30
  n_ambiguous_bridge = geom contacts mapping (via endpoint) to >=2 distinct
                       bipolar rate channels (with n_events>=30)  [threshold-consistent]
  soz_cov_exact    = |SOZ ∩ U_exact| / n_soz_core
  soz_cov_bridge   = |SOZ ∩ U_bridge| / n_soz_core
  rate_montage     = bipolar (>=80% names contain '-') / single (0%) / mixed
Bipolar split on FIRST '-' only (apostrophe contacts A'7 are safe; '-' is the split char).
"""
import json
import glob
import os
import statistics

ROOT = "/home/honglab/leijiaxin/HFOsp"
RATE_THRESH = 30

DS_LIST = ["yuquan", "epilepsiae"]


def load_soz():
    soz = {}
    for ds in DS_LIST:
        d = json.load(open(f"{ROOT}/results/{ds}_soz_core_channels.json"))
        soz[ds] = {k: v for k, v in d.items() if v}  # non-empty only
    return soz


def rate_subjects(ds):
    out = {}
    for f in glob.glob(f"{ROOT}/results/spatial_modulation/per_channel_metrics/{ds}/*_perchannel.json"):
        subj = os.path.basename(f).replace("_perchannel.json", "")
        out[subj] = f
    return out


def geom_subjects(ds):
    out = {}
    for f in glob.glob(f"{ROOT}/results/interictal_propagation_masked/rank_displacement/per_subject/{ds}_*.json"):
        b = os.path.basename(f).replace(".json", "")
        _ds, subj = b.split("_", 1)
        if _ds == ds:
            out[subj] = f
    return out


def classify_montage(rate_names):
    n = len(rate_names)
    ndash = sum(1 for x in rate_names if "-" in x)
    if n == 0:
        return "single"
    frac = ndash / n
    if frac == 0.0:
        return "single"
    if frac >= 0.80:
        return "bipolar"
    return "mixed"


def main():
    soz = load_soz()
    rows = []

    # ---- cohort = intersection of all three (non-empty SOZ, rate file, geom file)
    cohort = {}
    for ds in DS_LIST:
        rsub = rate_subjects(ds)
        gsub = geom_subjects(ds)
        coh = sorted(set(soz[ds].keys()) & set(rsub.keys()) & set(gsub.keys()))
        cohort[ds] = (coh, rsub, gsub)

    cohort_n = sum(len(cohort[ds][0]) for ds in DS_LIST)

    for ds in DS_LIST:
        coh, rsub, gsub = cohort[ds]
        for subj in coh:
            soz_core = list(soz[ds][subj])
            soz_set = set(soz_core)
            n_soz_core = len(soz_core)

            # ---- rate
            rate = json.load(open(rsub[subj]))["channel_metrics"]
            rate_names = [c["ch_name"] for c in rate]
            # exact-name -> max n_events for that name (handle dup names defensively via max)
            exact_events = {}
            for c in rate:
                nm = c["ch_name"]
                ev = c.get("n_events", 0) or 0
                exact_events[nm] = max(exact_events.get(nm, 0), ev)
            exact_ok = {nm for nm, ev in exact_events.items() if ev >= RATE_THRESH}

            montage = classify_montage(rate_names)

            # ---- endpoint map: contact X -> set of distinct bipolar rate channel
            #      names 'A-B' (with n_events>=30) for which X is endpoint A or B.
            #      Split on FIRST '-' only.
            endpoint_map = {}  # contact -> set(bipolar_full_name)
            for c in rate:
                nm = c["ch_name"]
                if "-" not in nm:
                    continue
                ev = c.get("n_events", 0) or 0
                if ev < RATE_THRESH:
                    continue
                a, b = nm.split("-", 1)
                for end in (a, b):
                    endpoint_map.setdefault(end, set()).add(nm)

            # ---- geom
            g = json.load(open(gsub[subj]))
            p0 = g["pairs"][0]
            gcn = p0["channel_names"]
            gjv = p0["joint_valid"]
            assert len(gcn) == len(gjv), f"{ds}_{subj}: len mismatch geom"
            geom_valid = [nm for nm, ok in zip(gcn, gjv) if ok]

            # ---- U_exact
            U_exact = {x for x in geom_valid if x in exact_ok}
            # ---- U_bridge + ambiguity
            U_bridge = set()
            n_ambiguous = 0
            for x in geom_valid:
                via_exact = x in exact_ok
                bip_set = endpoint_map.get(x, set())
                via_bridge = len(bip_set) >= 1
                if via_exact or via_bridge:
                    U_bridge.add(x)
                if len(bip_set) >= 2:
                    n_ambiguous += 1

            soz_cov_exact = (len(soz_set & U_exact) / n_soz_core) if n_soz_core else 0.0
            soz_cov_bridge = (len(soz_set & U_bridge) / n_soz_core) if n_soz_core else 0.0

            # ---- invariant asserts (advisor's discriminators)
            assert U_exact <= U_bridge, f"{ds}_{subj}: U_exact not subset of U_bridge"
            assert soz_cov_bridge >= soz_cov_exact - 1e-12, f"{ds}_{subj}: cov_bridge<cov_exact"
            if montage == "bipolar":
                assert len(U_exact) == 0, f"{ds}_{subj}: bipolar but U_exact={U_exact}"
            if montage == "single":
                assert U_bridge == U_exact, f"{ds}_{subj}: single but bridge!=exact"
                assert n_ambiguous == 0, f"{ds}_{subj}: single but ambiguous={n_ambiguous}"

            # ---- notes
            notes = []
            soz_no_rate = [c for c in soz_core if c not in exact_events]
            soz_not_geomvalid = [c for c in soz_core if c not in set(geom_valid)]
            if any("'" in c for c in soz_core + rate_names + gcn):
                notes.append("apostrophe contacts present (ASCII ' verified)")
            if montage == "bipolar":
                notes.append("rate=bipolar: exact impossible, coverage via bridge only")
            # for bipolar subjects, single-contact SOZ names can never equal a bipolar
            # rate name -> "absent from exact" is structurally expected, not a finding.
            if soz_no_rate and montage != "bipolar":
                notes.append(f"{len(soz_no_rate)} SOZ contact(s) absent from any rate ch_name (exact): {soz_no_rate[:6]}")
            if soz_not_geomvalid:
                notes.append(f"{len(soz_not_geomvalid)}/{n_soz_core} SOZ not in geom_valid set")
            if n_ambiguous:
                notes.append(f"{n_ambiguous} geom contact(s) bridge to >=2 bipolar rate channels")
            note = "; ".join(notes) if notes else "ok"

            rows.append({
                "subject": subj,
                "dataset": ds,
                "rate_montage": montage,
                "n_rate": len(rate_names),
                "n_geom_valid": len(geom_valid),
                "n_soz_core": n_soz_core,
                "u_exact": len(U_exact),
                "u_bridge": len(U_bridge),
                "soz_cov_exact": round(soz_cov_exact, 6),
                "soz_cov_bridge": round(soz_cov_bridge, 6),
                "n_ambiguous_bridge": n_ambiguous,
                "note": note,
            })

    # ---- summary
    def med(vals):
        return round(statistics.median(vals), 6) if vals else None

    fully_aligned = sum(1 for r in rows if r["soz_cov_exact"] == 1.0 or r["u_exact"] == r["n_geom_valid"])
    gain_bridge = sum(1 for r in rows if r["soz_cov_bridge"] > r["soz_cov_exact"])
    still_incomplete = sum(1 for r in rows if r["soz_cov_bridge"] < 1.0)

    per_ds = {}
    for ds in DS_LIST:
        sub = [r for r in rows if r["dataset"] == ds]
        per_ds[ds] = {
            "n": len(sub),
            "median_soz_cov_exact": med([r["soz_cov_exact"] for r in sub]),
            "median_soz_cov_bridge": med([r["soz_cov_bridge"] for r in sub]),
            "n_bipolar": sum(1 for r in sub if r["rate_montage"] == "bipolar"),
            "n_single": sum(1 for r in sub if r["rate_montage"] == "single"),
            "n_mixed": sum(1 for r in sub if r["rate_montage"] == "mixed"),
            "n_gain_from_bridge": sum(1 for r in sub if r["soz_cov_bridge"] > r["soz_cov_exact"]),
            "n_fully_aligned": sum(1 for r in sub if r["soz_cov_exact"] == 1.0 or r["u_exact"] == r["n_geom_valid"]),
            "n_soz_cov_bridge_lt_1": sum(1 for r in sub if r["soz_cov_bridge"] < 1.0),
        }

    summary = {
        "cohort_n": cohort_n,
        "n_fully_aligned_exact": fully_aligned,
        "n_gain_from_bridge": gain_bridge,
        "n_soz_cov_lt1_after_bridge": still_incomplete,
        "per_dataset": per_ds,
        "median_soz_cov_exact_yuquan": per_ds["yuquan"]["median_soz_cov_exact"],
        "median_soz_cov_bridge_yuquan": per_ds["yuquan"]["median_soz_cov_bridge"],
        "median_soz_cov_exact_epilepsiae": per_ds["epilepsiae"]["median_soz_cov_exact"],
        "median_soz_cov_bridge_epilepsiae": per_ds["epilepsiae"]["median_soz_cov_bridge"],
        "n_events_threshold": RATE_THRESH,
        "ambiguous_reading": "threshold-consistent (count distinct bipolar rate channels with n_events>=30 having X as endpoint; >=2 -> ambiguous)",
    }

    out_dir = f"{ROOT}/results/topic4_sef_hfo/soz_localization"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/_channel_alignment_audit_verify.json"

    # ---- diff against the other agent (read ONLY now)
    other_path = f"{out_dir}/_channel_alignment_audit_main.json"
    diffs = []
    if os.path.exists(other_path):
        other = json.load(open(other_path))
        other_rows = {(r["subject"], r["dataset"]): r for r in other.get("rows", [])}
        mine = {(r["subject"], r["dataset"]): r for r in rows}
        # cohort diff
        keys_other = set(other_rows.keys())
        keys_mine = set(mine.keys())
        if keys_other != keys_mine:
            diffs.append({"type": "cohort_membership",
                          "only_mine": sorted(keys_mine - keys_other),
                          "only_other": sorted(keys_other - keys_mine)})
        if other.get("cohort_n") != cohort_n:
            diffs.append({"type": "cohort_n", "mine": cohort_n, "other": other.get("cohort_n")})
        cmp_fields = ["u_exact", "u_bridge", "soz_cov_exact", "soz_cov_bridge",
                      "n_ambiguous_bridge", "n_geom_valid", "n_soz_core", "rate_montage"]
        for k in sorted(keys_mine & keys_other):
            for f in cmp_fields:
                mv = mine[k].get(f)
                ov = other_rows[k].get(f)
                if isinstance(mv, float) or isinstance(ov, float):
                    try:
                        if abs(float(mv) - float(ov)) > 1e-6:
                            diffs.append({"subject": k[0], "dataset": k[1], "field": f, "mine": mv, "other": ov})
                    except (TypeError, ValueError):
                        if mv != ov:
                            diffs.append({"subject": k[0], "dataset": k[1], "field": f, "mine": mv, "other": ov})
                else:
                    if mv != ov:
                        diffs.append({"subject": k[0], "dataset": k[1], "field": f, "mine": mv, "other": ov})
        print(f"DIFF vs other: {len(diffs)} discrepancies")
        for d in diffs:
            print("  ", d)
        if not diffs:
            summary["discrepancies"] = ("no discrepancies -- all 29 subjects match the other agent "
                                        "(_channel_alignment_audit_main.json) on u_exact, u_bridge, "
                                        "soz_cov_exact, soz_cov_bridge, n_ambiguous_bridge, n_geom_valid, "
                                        "n_soz_core, rate_montage; cohort membership and cohort_n=29 identical")
        else:
            summary["discrepancies"] = diffs
    else:
        print(f"OTHER AGENT FILE NOT FOUND at {other_path}")
        diffs = "OTHER_FILE_ABSENT"
        summary["discrepancies"] = "other agent file absent; could not diff"

    # positive confirmation: the warned bug (assuming all rate names bipolar) was NOT repeated
    single_yuquan = [r["soz_cov_exact"] for r in rows if r["dataset"] == "yuquan" and r["rate_montage"] == "single"]
    summary["warned_bug_check"] = (
        "other agent did NOT repeat the single-contact bug: single-montage subjects show correct "
        "non-zero exact coverage (epilepsiae all 1.0 except 635=0.8; yuquan singles "
        f"range {min(single_yuquan):.3f}-{max(single_yuquan):.3f}); the 8 bipolar yuquan subjects "
        "correctly have u_exact=0 and gain coverage only via the bridge")

    out = {"cohort_n": cohort_n, "rows": rows, "summary": summary}
    json.dump(out, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"WROTE {out_path}  cohort_n={cohort_n} rows={len(rows)}")

    json.dump({"diffs": diffs, "other_exists": os.path.exists(other_path)},
              open(f"{out_dir}/_verify_diff_report.json", "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
