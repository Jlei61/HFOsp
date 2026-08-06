"""Revised closeout: the same run, reported without the two claims that failed review.

Nothing is retrained.  Two analyses were redone from artefacts the original run
already wrote, and both reversed a headline:

- leave-contact-out now leads with the absolute score at the withheld contacts
  instead of the drop relative to each model's own baseline;
- flow ordering now uses the patient as the unit and carries an untrained-graph
  control on the same node positions.

The original files are left untouched; everything here is written beside them.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/topic5_spatial_latent_propagation_rnn_v0_1"

STATUS = {
    "ENGINEERING_EXECUTION": "PASS",
    "WITHIN_EVENT_HISTORY_VALUE": "SUPPORTED_BY_UNCONSTRAINED_GRU",
    "CONTACT_GRAPH_INCREMENT": "NOT_SUPPORTED",
    "TISSUE_FIELD_INCREMENT": "SMALL_AND_BELOW_UNCONSTRAINED_GRU",
    "LEARNED_TOPOLOGY_VALUE": "NOT_SUPPORTED",
    "EDGE_IDENTITY_RECOVERY": "FAIL",
    "DIRECTION_RECOVERY": "FAIL",
    "COARSE_REACHABILITY_ORDERING": "SUPPORTIVE_ONLY",
    "UNSEEN_CONTACT_GENERALIZATION": "NOT_SUPPORTED",
    "PATIENT_SPECIFIC_FLOW_ORDERING": "NOT_SEPARABLE_FROM_GEOMETRY",
    "GEOMETRY_STATUS": "RETROSPECTIVE_TEST_INFORMED_GEOMETRY",
}

WITHDRAWN = [
    {
        "was": "The advantage appears in the harder condition, where the contact is "
               "invisible to the model and a field can still infer that location from "
               "its neighbours while a per-contact node cannot.",
        "now": "The tissue-field model degraded less relative to its own baseline, but "
               "its absolute performance at held-out contacts remained substantially "
               "below the contact graph.",
        "evidence": "strong condition, tissue minus contact at the withheld contacts: "
                    "median -0.2843, 1/21 patients favour the tissue field, p=2.9e-06",
        "rerun_needed": False,
        "reason": "the original reported only the drop from each model's own baseline "
                  "and argued the raw score would 'mostly re-measure' the tissue "
                  "field sitting lower everywhere; that gap is 4.5x the largest effect "
                  "in the study, so it is the finding rather than an offset",
    },
    {
        "was": "The ordering is more reproducible within a patient than it is shared "
               "between patients, so it carries something patient-specific.",
        "now": "The within-versus-between gap is not larger than an untrained graph on "
               "the same node positions already produces, so it cannot be called "
               "patient-specific propagation structure.",
        "evidence": "learned Delta 0.1015 (17/21); untrained control on the same nodes "
                    "0.0591 (21/21); learned minus untrained 0.0396, 16/21, "
                    "exact Wilcoxon p=0.070, sign-flip p=0.119. |rho(flow, node axis "
                    "coordinate)| median 0.805",
        "rerun_needed": False,
        "reason": "the original pooled 21 within-patient and 70 between-patient pairs "
                  "into one Mann-Whitney, treating pairs as independent samples, and "
                  "had no control for node positions being patient-specific by "
                  "construction",
    },
    {
        "was": "Five arms share one task head so that every comparison is like for like.",
        "now": "Five arms share the same prediction task and loss; the output heads "
               "differ by arm.",
        "evidence": "ORDINARY_GRU reads out densely from a hidden vector; "
                    "CONTACT_GRAPH_RNN reads one scalar per contact node; the latent "
                    "arms emit per tissue unit and project through the fixed H",
        "rerun_needed": False,
        "reason": "the arms differ in state representation, recurrent parameterisation "
                  "and output mapping simultaneously, so this is not a factorial "
                  "decomposition of one factor at a time",
    },
    {
        "was": "All nodes share one GRU cell, so patients differ in the graph and not "
               "in the cell.",
        "now": "The cell is shared across the nodes of one patient. Across patients "
               "the cell, the contact bias, the emission, the logit scale and the "
               "STOP head are all fitted separately, so patient differences are not "
               "confined to the graph.",
        "evidence": "every nn.Module in SLPModel is instantiated per (patient, arm, seed)",
        "rerun_needed": False,
        "reason": "stated in a recap, not in the run itself; corrected in the module "
                  "docstring",
    },
    {
        "was": "LATENT_DENSE_RNN is a ceiling control.",
        "now": "LATENT_DENSE_RNN is a dense-adjacency control, and it is post hoc: it "
               "was added after the sparse arms had run.",
        "evidence": "it is not a ceiling in any sense that bounds the family -- it is "
                    "slightly worse than the sparse field (median -0.0001, 8/21)",
        "rerun_needed": False,
        "reason": "calling it a ceiling implies it upper-bounds what the "
                  "parameterisation can reach, which it does not",
    },
]

SURVIVING = [
    "an unconstrained recurrent model beats a static per-contact rate: median "
    "+0.0635, 21/21 patients, p=9.5e-07 -- within-event history carries stable "
    "predictive value",
    "the contact-node graph and both tissue-field arms lose to that unconstrained "
    "model on every patient (0/21 in each case)",
    "removing the sparsity constraint does not help and slightly hurts (median "
    "-0.0001, 8/21, p=0.243), so the wiring budget is not the limitation",
    "the recovery gate verdicts themselves: edge identity and direction of travel "
    "are not recoverable from rank observations under this observation contract, "
    "coarse ordering is",
    "every arm converged; no unit hit the epoch ceiling; the static baseline sits "
    "at its second-order optimum to within 2% of each patient's own effect size",
    "the geometry-shuffle control: real geometry shortens the connections it "
    "selects (0.94 against 1.01) and does not change prediction (+0.0008, p=0.078)",
]


def main() -> int:
    loco = json.loads((OUT / "leave_contact_out_revised.json").read_text())
    ordering = json.loads((OUT / "flow_ordering_revised.json").read_text())
    strong = loco["modes"]["strong"]

    add = []
    w = add.append
    w("# Topic 5 — Spatial Latent Propagation RNN v0.1 — revised closeout\n")
    w("Supersedes `CLOSEOUT_REPORT.md`. No model was retrained; two analyses were")
    w("redone from the artefacts the original run already wrote, and both reversed a")
    w("headline. The original files are left in place beside this one.\n")

    w("## 1. Frozen status\n")
    w("```text")
    for k, v in STATUS.items():
        w(f"{k}:\n{v}\n")
    w("```\n")
    w("The safe one-paragraph statement:\n")
    w("> Within-event rank history carries stable predictive value for the next")
    w("> recruitment step, but a tissue-field RNN observed through a fixed local")
    w("> electrode kernel does not reach the predictive power of an unconstrained")
    w("> recurrent model, and cannot recover interpretable edge-level or")
    w("> direction-level propagation structure from rank observations. The tissue")
    w("> field degrades less than the contact graph relative to its own baseline")
    w("> under contact holdout, but its absolute performance at those contacts")
    w("> remains substantially worse. The patient-specific coarse ordering is not")
    w("> separable from the fixed node geometry.\n")

    w("## 2. What was withdrawn, and what replaced it\n")
    for i, item in enumerate(WITHDRAWN, 1):
        w(f"**{i}. Withdrawn:** {item['was']}\n")
        w(f"**Replaced by:** {item['now']}\n")
        w(f"- evidence: {item['evidence']}")
        w(f"- why it was wrong: {item['reason']}")
        w(f"- re-run required: {'yes' if item['rerun_needed'] else 'no'}\n")

    w("## 3. What survives unchanged\n")
    for item in SURVIVING:
        w(f"- {item}")
    w("")

    w("## 4. Leave-contact-out, reported on absolute performance\n")
    w("Positive favours the tissue field on both contrasts.\n")
    for mode in ("weak", "strong"):
        m = loco["modes"][mode]
        a, d = (m["absolute_heldout_tissue_minus_contact"],
                m["relative_degradation_tissue_minus_contact"])
        label = ("the contact stayed visible in the sequence but scored nowhere"
                 if mode == "weak" else "the contact was removed from the input as well")
        w(f"- **{label}**")
        w(f"  - absolute at the withheld contacts: median {a['median']:+.4f}, "
          f"{a['n_tissue_better']}/{a['n']} patients favour the tissue field, "
          f"p={a['wilcoxon_two_sided_p']:.3g}")
        w(f"  - drop from each model's own baseline: median {d['median']:+.4f}, "
          f"{d['n_tissue_better']}/{d['n']}, p={d['wilcoxon_two_sided_p']:.3g}")
    w("")
    w(f"{loco['verdict']['why']}\n")
    w("Both arms trained with the per-contact bias disabled, so the gap is not the")
    w("tissue field being denied a parameter the contact graph kept.\n")

    w("## 5. Flow ordering, with the patient as the unit\n")
    for key, label in (("learned", "learned graph"),
                       ("untrained_control", "untrained graph, same node positions"),
                       ("learned_minus_untrained", "learned minus untrained, paired")):
        v = ordering[key]
        w(f"- **{label}**: median {v['median_delta']:+.4f}, "
          f"95% CI [{v['bootstrap_95ci'][0]:+.4f}, {v['bootstrap_95ci'][1]:+.4f}], "
          f"{v['n_positive']}/{v['n_patients']} positive, exact Wilcoxon "
          f"p={v['wilcoxon_exact_p']:.3g}, sign-flip p={v['sign_flip_p']:.3g}")
    g = ordering["geometry_only_descriptive"]
    w(f"- the ordering correlates with the node's own axis coordinate at "
      f"|rho|={g['median_abs_spearman_flow_vs_node_axis_coordinate']:.3f} in the "
      f"median, and that coordinate is fixed before training\n")
    w(f"**{ordering['reading']}**\n")
    w("The flow-ordering panel therefore leaves the overview figure. Its slot now")
    w("carries the dense-adjacency control, which does separate cleanly.\n")

    w("## 6. Conditions that apply to every number above\n")
    w("- the propagation plane was fitted on the whole recording, so this run is")
    w("  retrospective; it is not evidence that the geometry could have been known")
    w("  in advance, and every holdout analysis inherits that")
    w("- the model is a teacher-forced next-rank predictor; there is no free-running")
    w("  complete-event validation in this version")
    w("- the frozen primary matrix is five arms; the dense-adjacency arm is a post-hoc")
    w("  diagnostic added after the sparse arms had run")
    w("- topology freezing retains the top-k edges by opening probability, where k is")
    w("  the budgeted degree; it does not threshold at P(gate)>0.5")
    w("- arms share the prediction task and the loss, not the output head")
    w("- one seed carries coverage on all 21 patients; a second seed exists for the")
    w("  learned arm only, and carries the stability checks\n")

    w("## 7. What this run is for\n")
    w("A bounded negative and a model-identifiability audit. The useful result is the")
    w("gate: under this observation contract, an edge-level graph is less identifiable")
    w("than a coarse propagation operator would be. The next version should carry")
    w("fewer parameters that recover on synthetic data, not a better-regularised graph.\n")

    (OUT / "REVISED_CLOSEOUT_REPORT.md").write_text("\n".join(add))

    final = {
        "contract": "topic5_slp_revised_final_status_v0_1",
        "supersedes": "FINAL_STATUS.json",
        "retrained": False,
        "status": STATUS,
        "withdrawn_claims": WITHDRAWN,
        "surviving_claims": SURVIVING,
        "revised_artefacts": [
            "leave_contact_out_revised.json",
            "leave_contact_out_patient_first.csv",
            "flow_ordering_revised.json",
            "flow_ordering_geometry_control.csv",
            "REVISED_CLOSEOUT_REPORT.md",
            "REVISED_FIGURE_README.md",
            "figures/topic5_slp_rnn_v0_1_overview.png",
        ],
        "originals_left_untouched": [
            "CLOSEOUT_REPORT.md", "FINAL_STATUS.json",
            "leave_contact_out_summary.json", "flow_ordering.json",
            "cohort_statistics.json", "patient_prediction_metrics.csv",
        ],
    }
    (OUT / "REVISED_FINAL_STATUS.json").write_text(json.dumps(final, indent=1))

    print("wrote REVISED_CLOSEOUT_REPORT.md and REVISED_FINAL_STATUS.json")
    for k, v in STATUS.items():
        print(f"  {k:36s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
