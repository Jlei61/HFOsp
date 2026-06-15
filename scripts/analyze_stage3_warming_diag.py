"""Stage 3 B-prime diagnostic (2026-06-15, review-directed). The post-gate T=30000 'no clean global'
is suspected to be a record-length-dependent detector artifact (legacy record_peak bar). Separate the
TWO questions (review: must NOT conflate):

  (1) RECOVERY (read-out question): under a length-stable bar (prefix_peak from the first cal_prefix_ms),
      do the EARLY (t<cal_prefix) events recover clean global (n_part ~8) that the legacy record_peak bar
      clipped to n_part ~2? Matched by event index (deterministic -> same events, same order).
  (2) WARMING (dynamics question): bar-INDEPENDENT rolling p50/p95 of af over fixed time bins — does the
      network's baseline activity actually rise across the long record? This supersedes the contaminated
      true_inter_event_floor (which masks the [high-bar -> narrow] windows so event tails leak in).

Usage: analyze_stage3_warming_diag.py --af af_<tag>.npz --record-peak sidecar_<rp>.json --prefix sidecar_<px>.json
"""
import json
import argparse
import numpy as np


def warming(af_npz, window_ms=1000.0):
    z = np.load(af_npz)
    af = z["af"]; bin_w = float(z["bin_w"])
    nb = max(1, int(round(window_ms / bin_w)))
    rows = []
    for i in range(len(af) // nb):
        seg = af[i * nb:(i + 1) * nb]
        rows.append(dict(t0_ms=int(i * nb * bin_w),
                         p50=round(float(np.percentile(seg, 50)), 5),
                         p95=round(float(np.percentile(seg, 95)), 5)))
    return rows, dict(bar=float(z["bar"]), peak=float(z["peak"]), bar_source=str(z["bar_source"]))


def recovery(rp_sidecar, px_sidecar, tmax):
    rp = [e for e in json.load(open(rp_sidecar))["events"] if e["t_on"] < tmax]
    px = [e for e in json.load(open(px_sidecar))["events"] if e["t_on"] < tmax]
    rows = []
    for i in range(min(len(rp), len(px))):
        rows.append(dict(idx=i, rp_ton=rp[i]["t_on"], px_ton=px[i]["t_on"],
                         rp_npart=rp[i]["n_part"], px_npart=px[i]["n_part"],
                         rp_clean=rp[i]["clean_for_timing"], px_clean=px[i]["clean_for_timing"],
                         rp_hidden=rp[i]["hidden_source_label"], px_hidden=px[i]["hidden_source_label"]))
    summ = dict(tmax=tmax, n_rp=len(rp), n_px=len(px),
                rp_clean_global=sum(1 for e in rp if e["clean_for_timing"]),
                px_clean_global=sum(1 for e in px if e["clean_for_timing"]),
                rp_npart_median=float(np.median([e["n_part"] for e in rp])) if rp else None,
                px_npart_median=float(np.median([e["n_part"] for e in px])) if px else None)
    return rows, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--af", required=True)
    ap.add_argument("--record-peak", required=True, help="legacy record_peak run sidecar")
    ap.add_argument("--prefix", required=True, help="prefix_peak run sidecar")
    ap.add_argument("--tmax", type=float, default=3000.0)
    ap.add_argument("--window-ms", type=float, default=1000.0)
    a = ap.parse_args()

    wrows, wmeta = warming(a.af, a.window_ms)
    rrows, rsumm = recovery(a.record_peak, a.prefix, a.tmax)

    print("=== (1) RECOVERY: early events (t<%.0f), record_peak bar vs prefix bar ===" % a.tmax)
    print("  ", rsumm)
    for r in rrows[:12]:
        print(f"   idx{r['idx']}: rp n_part={r['rp_npart']}/clean={r['rp_clean']}({r['rp_hidden']}) "
              f"-> px n_part={r['px_npart']}/clean={r['px_clean']}({r['px_hidden']})")
    recovered = (rsumm["px_clean_global"] > rsumm["rp_clean_global"])
    print(f"  RECOVERY VERDICT: {'YES — prefix bar restores early clean global (=> record_peak artifact)' if recovered else 'NO — early clean global not restored by prefix bar'}")

    print("\n=== (2) WARMING: bar-independent rolling af (window=%.0fms) — meta %s ===" % (a.window_ms, wmeta))
    for r in wrows:
        print(f"   t0={r['t0_ms']:>6}ms  p50={r['p50']}  p95={r['p95']}")
    if len(wrows) >= 2:
        first, last = wrows[0]["p50"], wrows[-1]["p50"]
        print(f"  WARMING VERDICT: p50 baseline {first} -> {last} "
              f"({'RISES (real non-stationarity)' if last > 2 * max(first, 1e-9) else 'stable (no clear warming)'})")


if __name__ == "__main__":
    main()
