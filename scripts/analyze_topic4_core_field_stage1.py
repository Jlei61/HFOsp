"""Score every Stage 1 run under all scoring combinations and hand the table to
the descriptive report.

Exploratory posture: this prints numbers and a recommendation. It does not gate.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.getcwd())
from src.topic4_core_field_report import (  # noqa: E402
    COMPARISONS, PRIMARY_KEY, SCORE_KEYS, SEEDS, SIM_ARMS, _arm_n_dir,
    arm_value, concordance, stage1_report, tiered_paired_stats)
from src.topic4_core_field_runner import canonical_checksum  # noqa: E402
from src.topic4_core_field_scoring import (  # noqa: E402
    adversarial_gain, assignment_invariant_S, axis_only_templates,
    balanced_pair_score, coverage_matched_axis_only, load_patient_templates,
    model_templates, sim_matrix)

OUT = "results/topic4_sef_hfo/data_driven_core_field"
RUN = "results/topic4_sef_hfo/field_swap_subject_snn"


def _axial_projection(subject):
    """Contact positions projected on the frozen shared axis (mm)."""
    fd = np.load(os.path.join(
        RUN, f"figdata_{subject}_gradient_shared_corefrozen_cr1p5_s5_20260722.npz"),
        allow_pickle=True)
    names = [str(x) for x in fd["names"]]
    coords = np.asarray(fd["contacts"], float)
    reg = fd["reg"].item()
    u = np.asarray(reg["axis_unit"], float)
    u = u / np.linalg.norm(u)
    proj = (coords - np.asarray(reg["center"], float)[None, :]) @ u
    return names, coords, reg, {n: float(p) for n, p in zip(names, proj)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    probe = os.path.join(a.out, "stage1_variance_probe")
    fig_dir = os.path.join(probe, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    cfg = json.load(open(os.path.join(a.out, "config", "stage_config.json")))
    recomputed = canonical_checksum(cfg)
    if recomputed != cfg["checksum"]:
        raise SystemExit(f"config checksum mismatch: stored={cfg['checksum'][:12]} "
                         f"recomputed={recomputed[:12]}")
    support = cfg["support"]
    targets = {s: load_patient_templates(cfg["subject"], s) for s in cfg["sources"]}
    names, coords, reg, proj = _axial_projection(cfg["subject"])
    ao_full = axis_only_templates(names, coords, np.asarray(reg["center"]),
                                  np.asarray(reg["axis_unit"]))

    runs, rows, matched = {}, [], []
    for seed in cfg["seeds"]:
        for arm in SIM_ARMS:
            rec = json.load(open(os.path.join(probe, "per_run", str(seed), f"{arm}.json")))
            if rec.get("config_checksum") != cfg["checksum"]:
                raise SystemExit(f"run {arm}/{seed} was produced under a different config")
            m = model_templates(rec["events"], support, part_min=cfg["part_min"])
            ao_m = coverage_matched_axis_only(m, proj, support=support)
            for src in cfg["sources"]:
                for rule in cfg["missing_rules"]:
                    S = assignment_invariant_S(sim_matrix(m, targets[src], support, rule))
                    P = balanced_pair_score(m, targets[src], support)
                    common = dict(n_dir=m["n_dir"],
                                  coverage_forward=m["coverage_forward"],
                                  coverage_reverse=m["coverage_reverse"])
                    runs[(arm, seed, src, rule, "spearman")] = dict(S_rank=S, **common)
                    runs[(arm, seed, src, rule, "pair")] = dict(S_rank=P, **common)
                    rows.append(dict(arm=arm, seed=seed, source=src, missing_rule=rule,
                                     n_dir=m["n_dir"], S_spearman=S, S_pair=P,
                                     coverage_forward=m["coverage_forward"],
                                     coverage_reverse=m["coverage_reverse"],
                                     coverage_union=m["coverage_union"],
                                     mean_n_part=m["mean_n_part"],
                                     adversarial_gain=adversarial_gain(
                                         m, targets[src], support, rule)["gain"],
                                     n_events=rec["n_events"], h_sum=rec["h_sum"]))
                    if ao_m is not None:
                        s_ao = assignment_invariant_S(
                            sim_matrix(ao_m, targets[src], support, rule))
                        matched.append(dict(arm=arm, seed=seed, source=src,
                                            missing_rule=rule, model_S=S,
                                            matched_axis_only_S=s_ao,
                                            delta=S - s_ao))

    with open(os.path.join(probe, "per_run.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted(rows[0])); w.writeheader(); w.writerows(rows)

    comp_rows = []
    for key in SCORE_KEYS:
        for comp in COMPARISONS:
            if comp["b"] == "axis_only":
                continue
            pairs = [(_arm_n_dir(runs, comp["a"], s, key),
                      arm_value(runs, comp["a"], s, key, "S_rank"),
                      _arm_n_dir(runs, comp["b"], s, key),
                      arm_value(runs, comp["b"], s, key, "S_rank")) for s in SEEDS]
            comp_rows.append(dict(comparison=comp["name"], group=comp["group"],
                                  purpose=comp["purpose"], source=key[0],
                                  missing_rule=key[1], score_def=key[2],
                                  **tiered_paired_stats(pairs)))
    with open(os.path.join(probe, "prespecified_comparisons.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted(comp_rows[0]))
        w.writeheader(); w.writerows(comp_rows)

    with open(os.path.join(probe, "concordance.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["source", "missing_rule", "score_def", "concordance"])
        for key in SCORE_KEYS:
            w.writerow([*key, concordance(runs, key)])

    with open(os.path.join(probe, "axis_only_comparison.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted(matched[0])); w.writeheader(); w.writerows(matched)

    # unmatched reference, kept only so the incomparability is on the record
    with open(os.path.join(probe, "axis_only_unmatched.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "missing_rule", "axis_only_S_full_coverage", "note"])
        for src in cfg["sources"]:
            for rule in cfg["missing_rules"]:
                w.writerow([src, rule,
                            assignment_invariant_S(sim_matrix(ao_full, targets[src],
                                                              support, rule)),
                            "coverage 1.0 by construction; NOT comparable with the arms"])

    report = stage1_report(runs, cfg)
    md = [r for r in matched
          if r["source"] == PRIMARY_KEY[0] and r["missing_rule"] == PRIMARY_KEY[1]]
    by_arm = {}
    for arm in SIM_ARMS:
        d = [r["delta"] for r in md if r["arm"] == arm and np.isfinite(r["delta"])]
        if d:
            by_arm[arm] = dict(n=len(d), mean_delta=float(np.mean(d)),
                               sd=float(np.std(d, ddof=1)) if len(d) > 1 else float("nan"),
                               n_above=int(sum(x > 0 for x in d)))
    report["coverage_matched_axis_only"] = by_arm
    json.dump(report, open(os.path.join(probe, "stage1_report.json"), "w"),
              indent=2, default=str)

    print(f"[stage1] integrity = {report['integrity']['status']}")
    if report["integrity"]["status"] != "ok":
        print(f"[stage1] {report['integrity'].get('reason')}")
        return
    print(f"[stage1] shape separates = {report['recommendation']['shape_separates']} "
          f"{report['recommendation']['separating_dimensions']}")
    print(f"[stage1] equivalence A={report['equivalence']['A']['equivalent']} "
          f"A2={report['equivalence']['A2']['equivalent']}")
    print(f"[stage1] low-coverage arms = {report['coverage']['low_coverage_arms']}")
    print(f"[stage1] uninformative seeds = {report['scorable']['uninformative_seeds']}")
    print("[stage1] model minus coverage-matched axis-only:")
    for arm, v in by_arm.items():
        print(f"    {arm:18s} {v['mean_delta']:+.3f} +/- {v['sd']:.3f}  "
              f"above in {v['n_above']}/{v['n']}")

    key = PRIMARY_KEY
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for i, arm in enumerate(SIM_ARMS):
        v = [runs[(arm, s) + key]["S_rank"] for s in SEEDS]
        v = [x for x in v if np.isfinite(x)]
        ax.scatter(np.full(len(v), i) + rng.uniform(-.12, .12, len(v)), v, s=22, alpha=.75)
        if v:
            ax.hlines(np.mean(v), i - .28, i + .28, lw=2.2, color="k")
    ax.set_xticks(range(len(SIM_ARMS)))
    ax.set_xticklabels(SIM_ARMS, rotation=28, ha="right", fontsize=8)
    ax.set_ylabel("assignment-invariant rank match")
    ax.set_title("Stage 1 arms, 12 paired network seeds")
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "stage1_arm_scores.pdf")); plt.close(fig)

    shape = [c for c in COMPARISONS if c["group"] == "shape"]
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for i, comp in enumerate(shape):
        d = [arm_value(runs, comp["a"], s, key, "S_rank")
             - arm_value(runs, comp["b"], s, key, "S_rank") for s in SEEDS
             if _arm_n_dir(runs, comp["a"], s, key) == _arm_n_dir(runs, comp["b"], s, key)]
        d = [x for x in d if np.isfinite(x)]
        ax.scatter(np.full(len(d), i), d, s=22, alpha=.75)
        if d:
            ax.hlines(np.mean(d), i - .25, i + .25, lw=2.2, color="k")
    ax.axhline(0, color="0.6", lw=.8)
    ax.set_xticks(range(len(shape)))
    ax.set_xticklabels([f"{c['name']}\n{c['purpose']}" for c in shape], fontsize=7.5)
    ax.set_ylabel("same-tier paired difference vs manual_smooth")
    ax.set_title("Pre-registered shape comparisons")
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "stage1_shape_deltas.pdf")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    width = 0.38
    for j, d in enumerate(("coverage_forward", "coverage_reverse")):
        v = [float(np.mean([runs[(arm, s) + key][d] for s in SEEDS])) for arm in SIM_ARMS]
        ax.bar(np.arange(len(SIM_ARMS)) + (j - .5) * width, v, width,
               label=d.replace("coverage_", ""))
    ax.set_xticks(range(len(SIM_ARMS)))
    ax.set_xticklabels(SIM_ARMS, rotation=28, ha="right", fontsize=8)
    ax.set_ylabel("contacts recruited / frozen support")
    ax.legend(frameon=False, fontsize=8); ax.set_title("Per-direction coverage")
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "stage1_coverage.pdf")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for i, arm in enumerate(SIM_ARMS):
        d = [r["delta"] for r in md if r["arm"] == arm and np.isfinite(r["delta"])]
        ax.scatter(np.full(len(d), i) + rng.uniform(-.12, .12, len(d)), d, s=22, alpha=.75)
        if d:
            ax.hlines(np.mean(d), i - .28, i + .28, lw=2.2, color="k")
    ax.axhline(0, color="crimson", ls="--", lw=1.2)
    ax.set_xticks(range(len(SIM_ARMS)))
    ax.set_xticklabels(SIM_ARMS, rotation=28, ha="right", fontsize=8)
    ax.set_ylabel("model minus coverage-matched axis-only")
    ax.set_title("Does any field beat pure geometry on its OWN contacts?")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "stage1_vs_matched_axis_only.pdf")); plt.close(fig)
    print(f"[stage1] wrote figures to {fig_dir}")


if __name__ == "__main__":
    main()
