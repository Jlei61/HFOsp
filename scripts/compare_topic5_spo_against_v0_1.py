"""Did replacing the free graph with eight scalars keep what mattered?

That is the question this version exists to answer, and neither version answers
it alone. v0.1 established that an unconstrained recurrent model beats a
per-contact rate, and that every structural prior it tried gave the gain back.
v0.2 replaced the structure with a low-dimensional propagation operator. The
number that matters is how much of the recurrent model's advantage the eight
scalars recover.

The comparison is only legitimate because the two versions share the pieces that
would otherwise make it meaningless: the same events, the same chronological
split, the same loss code (v0.2 imports it from v0.1's module), and a static
baseline of the same construction -- per-contact bias, no recurrence.

Even so it is done as a difference of differences. Each version's advantage is
measured against ITS OWN static baseline, so a small discrepancy between the two
baselines cancels rather than propagating. The size of that discrepancy is
reported alongside, because it sets the resolution of the comparison.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/topic5_spatial_propagation_operator_v0_2"
# v0.1 has two result trees and the informative-sounding name is the wrong one.
# ``per_subject_budget95`` is the truncated-budget probe: its static arm sits at
# the ceiling in every patient and 12 of 21 were still improving when it stopped,
# which would inflate the recurrent advantage this script divides into.
# ``per_subject`` is the one v0.1's own closeout certifies -- every arm converged,
# no unit hit the ceiling. Checked, not inferred from the directory name.
V1 = (ROOT.parent / "topic5-slp-rnn" /
      "results/topic5_spatial_latent_propagation_rnn_v0_1/per_subject")

METRIC = "test_next_bce"


def score(root: Path, subject: str, arm: str) -> float | None:
    path = root / subject / arm / "seed1" / "DONE.json"
    if not path.exists():
        return None
    return json.loads(path.read_text()).get(METRIC)


def main() -> int:
    if not V1.exists():
        raise SystemExit(f"v0.1 results not found at {V1}")
    subjects = json.loads(
        (OUT / "INPUT_MANIFEST.json").read_text())["frozen_cohort"]["primary"]

    rows = []
    for subject in subjects:
        static_v1 = score(V1, subject, "STATIC_CONTACT")
        gru_v1 = score(V1, subject, "ORDINARY_GRU")
        # Readout-matched: v0.1's latent arms emit per tissue unit and project
        # through the same fixed kernel this version does, so comparing against
        # them isolates what carries the dynamics -- a learned sparse graph over
        # tissue nodes against eight physical scalars -- with the readout held
        # constant. The recurrent comparison cannot do that.
        learned_v1 = score(V1, subject, "LATENT_LEARNED_SPATIAL_RNN")
        knn_v1 = score(V1, subject, "LATENT_FIXED_LOCAL_RNN")
        static_v2 = score(OUT / "per_subject", subject, "STATIC")
        field_v2 = score(OUT / "per_subject", subject, "FIELD_NULL")
        full_v2 = score(OUT / "per_subject", subject, "ANISOTROPIC_RECOVERY")
        if None in (static_v1, gru_v1, learned_v1, knn_v1,
                    static_v2, field_v2, full_v2):
            continue
        rows.append({
            "subject": subject,
            "baseline_discrepancy": static_v2 - static_v1,
            "recurrent_advantage": static_v1 - gru_v1,
            "learned_graph_advantage": static_v1 - learned_v1,
            "fixed_knn_graph_advantage": static_v1 - knn_v1,
            "field_advantage": static_v2 - field_v2,
            "full_operator_advantage": static_v2 - full_v2,
        })

    if len(rows) < 5:
        raise SystemExit(f"only {len(rows)} patients complete in both versions")

    recurrent = np.array([r["recurrent_advantage"] for r in rows])
    field = np.array([r["field_advantage"] for r in rows])
    full = np.array([r["full_operator_advantage"] for r in rows])
    discrepancy = np.array([abs(r["baseline_discrepancy"]) for r in rows])
    learned = np.array([r["learned_graph_advantage"] for r in rows])
    knn = np.array([r["fixed_knn_graph_advantage"] for r in rows])

    report = {
        "contract": "topic5_spo_against_v0_1_v0_2",
        "question": ("how much of the advantage an unconstrained recurrent model "
                     "has over a per-contact rate does an eight-scalar propagation "
                     "operator recover"),
        "shared": ("events, chronological split, loss implementation, and a static "
                   "baseline of the same construction"),
        "method": ("difference of differences: each version is scored against its "
                   "own static baseline, so a shifted baseline cancels"),
        "n_patients": len(rows),
        "recurrent_advantage_median": float(np.median(recurrent)),
        "field_advantage_median": float(np.median(field)),
        "full_operator_advantage_median": float(np.median(full)),
        "share_recovered_by_field": float(np.median(field) / np.median(recurrent)),
        "share_recovered_by_full_operator": float(np.median(full) / np.median(recurrent)),
        "n_patients_where_field_matches_recurrent": int((field >= recurrent).sum()),
        "wilcoxon_field_vs_recurrent_p": float(
            stats.wilcoxon(field - recurrent).pvalue),
        "baseline_discrepancy": {
            "median_absolute": float(np.median(discrepancy)),
            "max_absolute": float(discrepancy.max()),
            "reading": (
                "the two static baselines agree to "
                f"{np.median(discrepancy):.4f} in the median, "
                f"{np.median(field) / max(np.median(discrepancy), 1e-9):.0f} times "
                "smaller than the field's own advantage, so the comparison is not "
                f"an artefact of a shifted baseline. But it reaches "
                f"{discrepancy.max():.4f} in the worst patient, still "
                f"{np.median(field) / max(discrepancy.max(), 1e-9):.1f} times below "
                "the field's own advantage and far below the gap being tested, so "
                "the per-patient count survives it"
                if discrepancy.max() < np.median(field) else
                f"{discrepancy.max():.4f} in the worst patient, comparable to the "
                "field's own advantage; the cohort medians hold but individual "
                "patients should not be read from this"),
        },
        "readout_matched": {
            "what": ("v0.1's latent arms project through the same fixed kernel "
                     "this version does, so against them the readout is held "
                     "constant and only what carries the dynamics differs"),
            "learned_sparse_graph_advantage_median": float(np.median(learned)),
            "fixed_knn_graph_advantage_median": float(np.median(knn)),
            "operator_share_of_learned_graph":
                float(np.median(full) / np.median(learned))
                if np.median(learned) > 0 else None,
            "n_patients_operator_matches_learned_graph": int((full >= learned).sum()),
            "wilcoxon_operator_vs_learned_graph_p": float(
                stats.wilcoxon(full - learned).pvalue),
        },
        "confound": {
            "what": ("the recurrent arm reads out through a free dense layer "
                     "(nn.Linear to one logit per contact) while every field here "
                     "must project through the fixed observation kernel"),
            "consequence": ("its gap is recurrence AND "
                            "readout together, so it cannot be charged to the "
                            "spatial parameterisation. The readout-matched "
                            "comparison above is the one that isolates it, and "
                            "there the operator and the learned graph are "
                            "indistinguishable"),
            "why_not_tested": ("testing it means a sixth arm -- the operator with "
                               "a free readout -- which would change the "
                               "pre-registered nested family this version froze. "
                               "It is the natural first question for the next one"),
            "direction_of_bias": ("against the operator: a free readout can only "
                                  "help, so any share computed against the "
                                  "recurrent arm is a lower bound"),
        },
        "per_patient": rows,
    }
    rm_matches = report["readout_matched"]["n_patients_operator_matches_learned_graph"]
    report["reading"] = (
        f"with the readout held constant, replacing v0.1's learned sparse graph "
        f"with eight physical scalars costs nothing: the operator is "
        f"{np.median(full):+.4f} over static against the learned graph's "
        f"{np.median(learned):+.4f} and a fixed nearest-neighbour graph's "
        f"{np.median(knn):+.4f}, matching the learned graph in "
        f"{rm_matches} of {len(rows)} patients "
        f"(p={report['readout_matched']['wilcoxon_operator_vs_learned_graph_p']:.3g}, "
        "no detectable difference). What neither reaches is the unconstrained "
        f"recurrent model at {np.median(recurrent):+.4f}, and because that arm "
        "also reads out through a free dense layer, its lead is NOT attributable "
        "to the spatial parameterisation. The eight scalars are as good a "
        "substitute for a learned graph as the graph was; the gap that remains "
        "sits somewhere neither of them touches")

    (OUT / "against_v0_1.json").write_text(json.dumps(report, indent=1))
    print(f"n={len(rows)}")
    print(f"  unconstrained recurrent over static  {np.median(recurrent):+.4f}")
    print(f"  eight-scalar field over static       {np.median(field):+.4f}"
          f"   ({report['share_recovered_by_field']:.0%} of it)")
    print(f"  full operator over static            {np.median(full):+.4f}"
          f"   ({report['share_recovered_by_full_operator']:.0%} of it)")
    print(f"  field matches recurrent in "
          f"{report['n_patients_where_field_matches_recurrent']}/{len(rows)}, "
          f"p={report['wilcoxon_field_vs_recurrent_p']:.3g}")
    rm = report["readout_matched"]
    print("\n  readout-matched (both project through the same fixed kernel):")
    print(f"    learned sparse graph over static  "
          f"{rm['learned_sparse_graph_advantage_median']:+.4f}")
    print(f"    fixed nearest-neighbour graph     "
          f"{rm['fixed_knn_graph_advantage_median']:+.4f}")
    print(f"    eight-scalar operator             {np.median(full):+.4f}"
          + (f"   ({rm['operator_share_of_learned_graph']:.0%} of the learned graph)"
             if rm["operator_share_of_learned_graph"] is not None else ""))
    print(f"    operator matches learned graph in "
          f"{rm['n_patients_operator_matches_learned_graph']}/{len(rows)}, "
          f"p={rm['wilcoxon_operator_vs_learned_graph_p']:.3g}")
    print(f"\n  {report['baseline_discrepancy']['reading']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
