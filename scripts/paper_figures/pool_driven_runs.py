"""Pool a source-only + sink-only subject-SNN run into one 'driven-pooled' tag.

Why: spontaneous two-core dynamics trade coverage against directional balance (large cores
recruit most electrodes but one core dominates -> only one template appears). Driving each
core SEPARATELY removes the competition: a single large core recruits most electrodes AND
gives a clean single-direction template. Pooling source(forward)+sink(reverse) yields a
high-coverage, balanced two-template readout that Fig4A/Fig4B consume with no changes.

The two runs share seed -> same network -> the two-core mechanism field is exactly
min(source_vth, sink_vth) (the twoend_equal field), so the mechanism panel is faithful.

Writes readout_<pooled>.json + figdata_<pooled>.npz under the run dir.
"""
import argparse, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN = os.path.join(ROOT, "results/topic4_sef_hfo/field_swap_subject_snn")


def _load(tag):
    return (json.load(open(os.path.join(RUN, f"readout_{tag}.json"))),
            np.load(os.path.join(RUN, f"figdata_{tag}.npz"), allow_pickle=True))


def pool(source_tag, sink_tag, pooled_tag, half_ms=2500.0):
    sro, sfd = _load(source_tag)            # source core only -> forward template
    kro, kfd = _load(sink_tag)              # sink core only   -> reverse template
    kd = int(sro.get("k_dir", 2)); part_min = 2 * kd

    # directional events: source -> forward, sink -> reverse (the driven direction of each)
    fwd = [e for e in sro["events"] if e.get("sign") is not None and e["sign"] > 0
           and e.get("n_part", 0) >= part_min]
    rev = [e for e in kro["events"] if e.get("sign") is not None and e["sign"] < 0
           and e.get("n_part", 0) >= part_min]

    # ---- readout panel trace: source[0:half] ++ sink[0:half] so both directions show in one window ----
    st = np.asarray(sfd["times"], float); kt = np.asarray(kfd["times"], float)
    s_lfp = np.asarray(sfd["lfp_trace"], float); k_lfp = np.asarray(kfd["lfp_trace"], float)
    s_sel = st <= half_ms
    k_sel = kt <= half_ms
    off = half_ms
    times = np.concatenate([st[s_sel], kt[k_sel] + off])
    lfp = np.concatenate([s_lfp[s_sel], k_lfp[k_sel]], axis=0)

    def in_win(evs, sel_off, side):
        out = []
        for e in evs:
            if e["t_on"] <= half_ms:                       # within the shown half
                e2 = dict(e); e2["t_on"] = e["t_on"] + sel_off
                e2["t_off"] = min(e["t_off"] + sel_off, (half_ms + sel_off))
                out.append(e2)
        return out
    win_events = in_win(fwd, 0.0, "fwd") + in_win(rev, off, "rev")

    # ---- all directional events for KMeans (Fig4B) -- not time-restricted ----
    pooled_events = fwd + rev

    valid = max(1, sro["valid_contacts"])
    npall = [e["n_part"] for e in pooled_events] or [0]
    union = set()
    for e in pooled_events:
        union |= {n for n, v in (e.get("ranks") or {}).items() if v is not None}
    out = dict(sro)
    out.update(subject=sro["subject"], montage=sro["montage"], lesion="driven_pooled",
               placement=sro["placement"], k_dir=kd,
               n_events=len(pooled_events), n_clean=len(pooled_events),
               clean_forward=len(fwd), clean_reverse=len(rev),
               dir_forward=len(fwd), dir_reverse=len(rev),
               bidirectional=(len(fwd) > 0 and len(rev) > 0),
               per_event_cov=round(float(np.mean(npall) / valid), 3),
               union_cov=round(len(union) / valid, 3),
               source_tag=source_tag, sink_tag=sink_tag,
               events=pooled_events, readout_window_events=win_events)
    json.dump(out, open(os.path.join(RUN, f"readout_{pooled_tag}.json"), "w"), indent=2)

    # ---- figdata: two-core mechanism = min(vth); rep_fwd from source, rep_rev from sink ----
    s_vth = np.asarray(sfd["vth"], float); k_vth = np.asarray(kfd["vth"], float)
    vth = np.minimum(s_vth, k_vth)
    np.savez(os.path.join(RUN, f"figdata_{pooled_tag}.npz"),
             reg=sfd["reg"], contacts=sfd["contacts"], names=sfd["names"],
             valid=sfd["valid"], lfp_trace=lfp, times=times, bin_w=sfd["bin_w"],
             posE=sfd["posE"], vth=vth, foci=sfd["foci"],
             core_r=sfd["core_r"], core_mean=sfd["core_mean"], theta_deg=sfd["theta_deg"],
             L=sfd["L"], rep_fwd=sfd["rep_fwd"], rep_rev=kfd["rep_rev"])
    return dict(pooled_tag=pooled_tag, fwd=len(fwd), rev=len(rev),
                per_event_cov=out["per_event_cov"], union_cov=out["union_cov"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-tag", required=True)
    ap.add_argument("--sink-tag", required=True)
    ap.add_argument("--pooled-tag", required=True)
    ap.add_argument("--half-ms", type=float, default=2500.0)
    a = ap.parse_args()
    print(pool(a.source_tag, a.sink_tag, a.pooled_tag, a.half_ms))


if __name__ == "__main__":
    main()
