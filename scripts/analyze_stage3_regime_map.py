"""Stage 3 local-global REGIME MAP (2026-06-15, review-corrected round 2). Aggregate each
core_mean x sep_frac cell's sidecars into mutually-exclusive regime buckets + a hard ADVANCE GATE.

Buckets (§A FATAL-1/2; mutually exclusive, priority by hidden_source_label):
  collision         hidden_source_label == "collision"                       (two-end co-ignition)
  clean_global_neg  hidden in {neg} AND clean_for_timing                      (clean readable from -end)
  clean_global_pos  hidden in {pos} AND clean_for_timing                      (clean readable from +end)
  local             else AND n_part < 7                                      (STRICT: did not spread)
  dirty_global      else (n_part >= 7 but unclean/unreadable/ambiguous source)
  -> local NEVER absorbs n_part>=7 unreadable events (FATAL-1); clean uses build_sidecar's
     clean_for_timing contract verbatim, NOT a re-invented readable test (FATAL-2; src/sef_hfo_stage3
     :212-217 already = single-source AND readable AND axis_err<25, readable implies n_part>=part_min).

ADVANCE GATE (locked §A; tested in tests/test_stage3_regime_map.py) — verdict only says "worth a
longer POST-GATE run", NOT "found bidirectional templates". THREE fail directions (FATAL-3):
  too_cold/undersampled  n < MIN_EV (not judged)
  fail:collision_dominated   collision_rate >= COLL_MAX          (HOT)
  fail:local_dominated       local_frac >= LOCAL_MAX             (COLD: long run only repeats locals)
  PASS:relay_both_ends       cg_neg>=K_END AND cg_pos>=K_END
  fail:one_end_or_no_clean_relay  else (MIDDLE: low-collision, non-local, but one-end-dominant or no
                                  clean relay) -- mid_reason in {one_end_dominant, no_clean_relay}
Seed stability (FATAL-4): per cell output n_seeds_{neg,pos}_clean + n_seeds_both so a single seed
lighting up a whole cell is visible.
"""
import os, re, glob, json
import numpy as np

ROOT = "results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/regime_map"
PART_MIN = 7
K_END, COLL_MAX, LOCAL_MAX, MIN_EV = 2, 0.30, 0.90, 10   # ADVANCE GATE thresholds (locked)


def cell_of(tag):                       # rm_m17.0_sep0.7_s3 -> (17.0, 0.7)
    m = re.search(r"_m([\d.]+)_sep([\d.]+)_s", tag)
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)


def seed_of(tag):
    m = re.search(r"_s(\d+)$", tag)
    return int(m.group(1)) if m else -1


def classify_event(ev):
    """One of 5 mutually-exclusive buckets. hidden_source_label priority; local STRICT n_part<7;
    clean reuses clean_for_timing (NOT a re-derived readable)."""
    hid = ev.get("hidden_source_label")
    if hid == "collision":
        return "collision"
    if hid in ("neg", "pos") and ev.get("clean_for_timing"):
        return f"clean_global_{hid}"
    if (ev.get("n_part") or 0) < PART_MIN:
        return "local"
    return "dirty_global"


def gate(d):
    """d keys: n, local, collision, cg_neg, cg_pos, dirty_global. Returns (verdict, mid_reason)."""
    n = d["n"]
    if n < MIN_EV:
        return "too_cold/undersampled", None
    if d["collision"] / n >= COLL_MAX:
        return "fail:collision_dominated", None
    if d["local"] / n >= LOCAL_MAX:
        return "fail:local_dominated", None
    if d["cg_neg"] >= K_END and d["cg_pos"] >= K_END:
        return "PASS:relay_both_ends", None
    mid = "one_end_dominant" if (d["cg_neg"] >= K_END or d["cg_pos"] >= K_END) else "no_clean_relay"
    return "fail:one_end_or_no_clean_relay", mid


def audit_provenance(provs):
    """provs: list of per-readout provenance dicts. A map MUST be single-provenance — mixed
    git_sha / engine_sha means a code/engine drift silently spanned the grid (it happened
    2026-06-15 when the engine was edited mid-scout). Returns single_provenance + the breakdown."""
    from collections import Counter
    gits = Counter(p.get("git_sha") for p in provs)
    engs = Counter((p.get("engine_sha") or {}).get("kick_probe.py") for p in provs)
    return dict(n_readouts=len(provs), single_provenance=(len(gits) <= 1 and len(engs) <= 1),
                distinct_git_sha={(k[:9] if k else k): v for k, v in gits.items()},
                distinct_kick_probe_engine_sha=dict(engs))


def read_provenance(root=None):
    root = root or ROOT
    out = []
    for r in sorted(glob.glob(os.path.join(root, "readout_rm_*.json"))):
        try:
            out.append(json.load(open(r)).get("provenance", {}))
        except Exception:
            pass
    return out


def cell_metrics(root=None):
    root = root or ROOT          # read module ROOT at call-time (not def-time) so overrides take
    cells = {}
    for sc in sorted(glob.glob(os.path.join(root, "sidecar_rm_*.json"))):
        tag = os.path.basename(sc)[8:-5]
        key = cell_of(tag); seed = seed_of(tag)
        try:
            ev = json.load(open(sc)).get("events", [])
        except Exception:
            continue
        d = cells.setdefault(key, dict(n=0, local=0, collision=0, cg_neg=0, cg_pos=0,
                                       dirty_global=0, seeds=set(), seeds_neg=set(), seeds_pos=set()))
        d["seeds"].add(seed)
        for e in ev:
            b = classify_event(e)
            d["n"] += 1
            if b == "clean_global_neg":
                d["cg_neg"] += 1; d["seeds_neg"].add(seed)
            elif b == "clean_global_pos":
                d["cg_pos"] += 1; d["seeds_pos"].add(seed)
            else:
                d[b] += 1
    return cells


def build_rows(cells):
    rows = []
    for (mean, sep), d in sorted(cells.items()):
        n = max(1, d["n"]); verdict, mid = gate(d)
        rows.append(dict(core_mean=mean, sep_frac=sep, n_events=d["n"], n_seeds=len(d["seeds"]),
                         local_frac=round(d["local"] / n, 3), collision_rate=round(d["collision"] / n, 3),
                         clean_global_neg=d["cg_neg"], clean_global_pos=d["cg_pos"],
                         dirty_global=d["dirty_global"],
                         n_seeds_neg_clean=len(d["seeds_neg"]), n_seeds_pos_clean=len(d["seeds_pos"]),
                         n_seeds_both=len(d["seeds_neg"] & d["seeds_pos"]),
                         verdict=verdict, mid_reason=mid))
    return rows


def _figure(rows, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    means = sorted({r["core_mean"] for r in rows}); seps = sorted({r["sep_frac"] for r in rows})

    def grid(fn):
        g = np.full((len(means), len(seps)), np.nan)
        for r in rows:
            g[means.index(r["core_mean"]), seps.index(r["sep_frac"])] = fn(r)
        return g

    panels = [(grid(lambda r: r["local_frac"]), "事件停在局部的比例", "viridis"),
              (grid(lambda r: r["collision_rate"]), "两端共点火(碰撞)比例", "magma"),
              (grid(lambda r: min(r["clean_global_neg"], r["clean_global_pos"])),
               "两端中继下限 min(neg,pos) 干净大事件", "cividis")]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for a, (g, ttl, cm) in zip(ax, panels):
        im = a.imshow(g, origin="lower", aspect="auto", cmap=cm)
        a.set_xticks(range(len(seps))); a.set_xticklabels([f"sep{s}" for s in seps])
        a.set_yticks(range(len(means))); a.set_yticklabels([f"m{m}" for m in means])
        a.set_title(ttl, fontsize=10); fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle("两端等强病灶 cm-SNN：局部 / 中继 / 共点火 三态地图（短 T scout）", fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    prov = audit_provenance(read_provenance())
    rows = build_rows(cell_metrics())
    passing = [r for r in rows if r["verdict"].startswith("PASS")]
    summary = dict(gate_thresholds=dict(K_END=K_END, COLL_MAX=COLL_MAX, LOCAL_MAX=LOCAL_MAX, MIN_EV=MIN_EV),
                   provenance_audit=prov,
                   completeness=dict(n_cells=len(rows), n_readouts=prov["n_readouts"],
                                     seeds_per_cell={f"m{r['core_mean']}/sep{r['sep_frac']}": r["n_seeds"] for r in rows}),
                   cells=rows, n_passing=len(passing),
                   passing_cells=[(r["core_mean"], r["sep_frac"]) for r in passing])
    json.dump(summary, open(os.path.join(ROOT, "regime_map_summary.json"), "w"), indent=2, default=str)
    if not prov["single_provenance"]:
        print("!!! MIXED PROVENANCE — map spans >1 engine/code version, cells NOT comparable:")
        print("   ", prov, "\n")
    else:
        print(f"[provenance OK] {prov['n_readouts']} readouts, single git+engine "
              f"({list(prov['distinct_git_sha'])[0]} / {list(prov['distinct_kick_probe_engine_sha'])[0]})")
    print(f"=== REGIME MAP ({len(rows)} cells) ===")
    for r in rows:
        print(f"  m{r['core_mean']}/sep{r['sep_frac']}: n={r['n_events']:>3} "
              f"local={r['local_frac']} coll={r['collision_rate']} dirty={r['dirty_global']} "
              f"cg neg{r['clean_global_neg']}(s{r['n_seeds_neg_clean']})/pos{r['clean_global_pos']}(s{r['n_seeds_pos_clean']}) "
              f"both_s={r['n_seeds_both']} -> {r['verdict']}" + (f" [{r['mid_reason']}]" if r['mid_reason'] else ""))
    print(f"\nPASSING (relay both ends): {summary['passing_cells'] or 'NONE'}")
    if rows:
        _figure(rows, os.path.join(ROOT, "figures", "stage3_regime_map.png"))
        print("wrote figures/stage3_regime_map.png")
    return summary


if __name__ == "__main__":
    main()
