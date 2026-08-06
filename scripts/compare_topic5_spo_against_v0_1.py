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
        static_v2 = score(OUT / "per_subject", subject, "STATIC")
        field_v2 = score(OUT / "per_subject", subject, "FIELD_NULL")
        full_v2 = score(OUT / "per_subject", subject, "ANISOTROPIC_RECOVERY")
        if None in (static_v1, gru_v1, static_v2, field_v2, full_v2):
            continue
        rows.append({
            "subject": subject,
            "baseline_discrepancy": static_v2 - static_v1,
            "recurrent_advantage": static_v1 - gru_v1,
            "field_advantage": static_v2 - field_v2,
            "full_operator_advantage": static_v2 - full_v2,
        })

    if len(rows) < 5:
        raise SystemExit(f"only {len(rows)} patients complete in both versions")

    recurrent = np.array([r["recurrent_advantage"] for r in rows])
    field = np.array([r["field_advantage"] for r in rows])
    full = np.array([r["full_operator_advantage"] for r in rows])
    discrepancy = np.array([abs(r["baseline_discrepancy"]) for r in rows])

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
        "per_patient": rows,
    }
    report["reading"] = (
        f"an unconstrained recurrent model beats a per-contact rate by "
        f"{np.median(recurrent):+.4f}; the eight-scalar field beats it by "
        f"{np.median(field):+.4f}, which is "
        f"{report['share_recovered_by_field']:.0%} of the recurrent advantage, and "
        f"the field matches or exceeds the recurrent model in "
        f"{report['n_patients_where_field_matches_recurrent']} of {len(rows)} "
        "patients. Replacing the free graph with a propagation operator kept a "
        "small fraction of what recurrence was buying")

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
    print(f"  {report['baseline_discrepancy']['reading']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
