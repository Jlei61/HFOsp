"""Re-report leave-contact-out on absolute performance, not only on degradation.

The original analysis reported how much each architecture LOSES at a contact it
never scored, and justified omitting the raw score on the grounds that the
tissue field "sits lower everywhere" so raw numbers would mostly re-measure
that.  That reasoning does not hold: if the tissue field is worse everywhere
then that is the finding, not a reason to hide it -- and the gap at withheld
contacts turns out to be an order of magnitude larger than any baseline offset
could explain.

So both quantities are computed here and the absolute one leads.  Nothing is
retrained; this reads the metrics the original run already wrote.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/topic5_spatial_latent_propagation_rnn_v0_1"

TISSUE = "LATENT_LEARNED_SPATIAL_RNN"
CONTACT = "CONTACT_GRAPH_RNN"
# Lower is better on every metric here, so a POSITIVE delta always means the
# tissue field wins, matching the sign convention of the cohort statistics.
METRIC = "next_bce"


def paired(values: np.ndarray) -> dict:
    if len(values) < 3:
        return {"status": "INSUFFICIENT", "n": int(len(values))}
    rng = np.random.default_rng(20260806)
    boot = np.array([
        np.median(rng.choice(values, len(values), replace=True)) for _ in range(4000)
    ])
    return {
        "status": "COMPLETE",
        "n": int(len(values)),
        "median": float(np.median(values)),
        "bootstrap_95ci": [float(np.percentile(boot, 2.5)),
                           float(np.percentile(boot, 97.5))],
        "n_tissue_better": int((values > 0).sum()),
        "wilcoxon_two_sided_p": float(stats.wilcoxon(values).pvalue),
    }


def main() -> int:
    rows = list(csv.DictReader((OUT / "leave_contact_out_metrics.csv").open()))
    cells: dict[tuple[str, str], dict[str, dict]] = {}
    for r in rows:
        cells.setdefault((r["subject"], r["mode"]), {})[r["arm"]] = r

    per_patient: list[dict] = []
    for (subject, mode), arms in sorted(cells.items()):
        if TISSUE not in arms or CONTACT not in arms:
            continue
        t, c = arms[TISSUE], arms[CONTACT]
        t_held, c_held = float(t[f"heldout_{METRIC}"]), float(c[f"heldout_{METRIC}"])
        t_ret, c_ret = float(t[f"retained_{METRIC}"]), float(c[f"retained_{METRIC}"])
        per_patient.append({
            "subject": subject,
            "mode": mode,
            "n_holdout_contacts": int(t["n_holdout_contacts"]),
            "tissue_retained": t_ret,
            "tissue_heldout": t_held,
            "contact_retained": c_ret,
            "contact_heldout": c_held,
            # within-model degradation: how much worse the withheld contact is
            "tissue_degradation": t_held - t_ret,
            "contact_degradation": c_held - c_ret,
            # the two contrasts, both signed so that positive favours tissue
            "raw_heldout_tissue_minus_contact": c_held - t_held,
            "degradation_tissue_minus_contact": (c_held - c_ret) - (t_held - t_ret),
        })

    with (OUT / "leave_contact_out_patient_first.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_patient[0]))
        w.writeheader()
        w.writerows(per_patient)

    summary: dict = {
        "contract": "topic5_slp_loco_revised_v0_1",
        "retrained": False,
        "source": "leave_contact_out_metrics.csv (original run)",
        "metric": METRIC,
        "sign_convention": "positive favours the tissue field on every contrast",
        "geometry_status": "RETROSPECTIVE_TEST_INFORMED_GEOMETRY",
        "modes": {},
    }
    for mode in ("weak", "strong"):
        sel = [p for p in per_patient if p["mode"] == mode]
        if not sel:
            continue
        raw = np.array([p["raw_heldout_tissue_minus_contact"] for p in sel])
        deg = np.array([p["degradation_tissue_minus_contact"] for p in sel])
        summary["modes"][mode] = {
            "n_patients": len(sel),
            "absolute_heldout_tissue_minus_contact": paired(raw),
            "relative_degradation_tissue_minus_contact": paired(deg),
            "median_tissue_heldout": float(np.median([p["tissue_heldout"] for p in sel])),
            "median_contact_heldout": float(np.median([p["contact_heldout"] for p in sel])),
        }

    strong = summary["modes"].get("strong", {})
    absolute = strong.get("absolute_heldout_tissue_minus_contact", {})
    summary["verdict"] = {
        "unseen_contact_generalisation": "NOT_SUPPORTED",
        "why": (
            "In the strong condition -- the contact removed from the input as well -- "
            f"the tissue field is worse at the withheld contacts by "
            f"{-absolute.get('median', float('nan')):.4f} in the median, with "
            f"{absolute.get('n', 0) - absolute.get('n_tissue_better', 0)} of "
            f"{absolute.get('n', 0)} patients favouring the contact graph. That gap is "
            "several times the largest effect anywhere in the study, so it cannot be a "
            "baseline offset. The tissue field degrades less relative to its own "
            "baseline, but its own baseline at those contacts is far lower."
        ),
        "allowed_sentence": (
            "The tissue-field model degraded less relative to its own baseline, but its "
            "absolute performance at held-out contacts remained substantially below the "
            "contact graph."
        ),
        "forbidden_sentences": [
            "better unseen-contact prediction",
            "the tissue field predicts contacts it never trained on",
            "a field can infer that location from its neighbours while a per-contact "
            "node cannot",
        ],
        "note_on_contact_bias": (
            "Both arms did train with the per-contact bias disabled (train script "
            "forces use_contact_bias=False whenever a holdout is set), so the gap is "
            "not the tissue field being denied a parameter the contact graph kept."
        ),
    }
    (OUT / "leave_contact_out_revised.json").write_text(json.dumps(summary, indent=1))

    print(f"patients per mode: "
          f"{ {m: v['n_patients'] for m, v in summary['modes'].items()} }")
    for mode, v in summary["modes"].items():
        a, d = (v["absolute_heldout_tissue_minus_contact"],
                v["relative_degradation_tissue_minus_contact"])
        print(f"\n--- {mode} ---")
        print(f"  absolute  median {a['median']:+.4f}  tissue better "
              f"{a['n_tissue_better']}/{a['n']}  p={a['wilcoxon_two_sided_p']:.3g}")
        print(f"  degradation median {d['median']:+.4f}  tissue better "
              f"{d['n_tissue_better']}/{d['n']}  p={d['wilcoxon_two_sided_p']:.3g}")
    print(f"\nverdict: {summary['verdict']['unseen_contact_generalisation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
