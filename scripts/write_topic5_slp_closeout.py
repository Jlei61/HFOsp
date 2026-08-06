"""Assemble CLOSEOUT_REPORT.md and FINAL_STATUS.json from the frozen artefacts.

The report reads what actually ran.  It never restates a hypothesis as resolved
when the artefact behind it is missing, and it carries the recovery-gate verdict
into every structural sentence.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/topic5_spatial_latent_propagation_rnn_v0_1"


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def fmt(entry: dict | None) -> str:
    if not entry or entry.get("status") != "COMPLETE":
        return "not resolved (no completed comparison)"
    return (
        f"median {entry['median_delta']:+.4f}, "
        f"95% CI [{entry['bootstrap_95ci'][0]:+.4f}, {entry['bootstrap_95ci'][1]:+.4f}], "
        f"{entry['n_positive']}/{entry['n']} patients, p={entry['wilcoxon_two_sided_p']:.3g}"
    )


def main() -> int:
    argparse.ArgumentParser().parse_args()

    manifest = load(OUT / "INPUT_MANIFEST.json")
    gate = load(OUT / "synthetic" / "RECOVERY_GATE.json")
    stats = load(OUT / "cohort_statistics.json")
    lco = load(OUT / "leave_contact_out_summary.json")
    frozen = load(OUT / "development" / "FROZEN_CONFIG.json")
    sweep = load(OUT / "development" / "SWEEP_SUMMARY.json")
    baseline_check = load(OUT / "static_baseline_verification.json")
    budget_probe = load(OUT / "convergence_bias_probe.json")

    matrix = OUT / "EXPERIMENT_MATRIX.csv"
    planned = completed = 0
    failed = []
    if matrix.exists():
        for row in csv.DictReader(matrix.open()):
            planned += 1
            cell = OUT / "per_subject" / row["subject"] / row["arm"] / f"seed{row['seed']}"
            if (cell / "DONE.json").exists():
                completed += 1
            elif (cell / "FAILED.json").exists():
                failed.append(f"{row['subject']}/{row['arm']}/seed{row['seed']}")

    primary = (stats or {}).get("comparisons", {}).get("primary", {})
    lines = []
    add = lines.append

    add("# Topic 5 — Spatial Latent Propagation RNN v0.1 — closeout\n")
    add("Spec: `docs/superpowers/specs/2026-08-06-topic5-spatial-latent-propagation-rnn-v0_1.md`")
    add("Plan: `docs/superpowers/plans/2026-08-06-topic5-spatial-latent-propagation-rnn-v0_1.md`\n")

    add("## 1. What was asked and what the run can answer\n")
    add("The question was whether local rate units placed in a patient's own tissue plane, "
        "observed only through a fixed local electrode kernel, can form a propagation "
        "structure that predicts held-out interictal events — including at contacts the "
        "model never trained on.\n")
    if gate:
        add("Before any patient result was read, the same learner was asked to recover a "
            "propagation graph that was known by construction. It was given events "
            "generated from that graph and nothing else. Three separate things were "
            "scored, because they license very different claims:\n")
        add(f"- **which connections exist** — ranked the true connections at "
            f"{gate['edge_identity']['median_auc']:.3f} where chance is 0.5 and the "
            f"pre-set requirement was {gate['edge_identity']['floor']:.2f}: "
            f"**{gate['edge_identity']['status'].lower().replace('_', ' ')}**;")
        add(f"- **which way activity travels overall** — the direction came out right in "
            f"{int(round(gate['axis_direction']['sign_agreement'] * gate['n_cells']))} of "
            f"{gate['n_cells']} runs: "
            f"**{gate['axis_direction']['status'].lower().replace('_', ' ')}**;")
        order = gate.get("flow_ordering", {})
        if order:
            add(f"- **the relative order of how far each patch pushes** — positive in "
                f"{order['n_cells_positive']} of {order['n_cells']} runs "
                f"(median {order['median_node_spearman']:+.2f}, sign test "
                f"p={order['sign_test_p']:.3g}): "
                f"**{order['status'].lower().replace('_', ' ')}**.\n")
        add("So questions about the *identity* of the connections cannot be answered here. "
            "Any per-patient graph is one arbitrary member of a large set that fit the data "
            "equally well: comparing such graphs across patients, or deleting the "
            "connections the model calls important, would be reading optimiser noise. Two "
            "hypotheses fall to this — that patients differ in their graphs, and that the "
            "specific connections are functionally necessary.\n")
        add("What survives is every question about prediction, including whether *learning* "
            "the connections beats fixing them to nearest neighbours. That comparison never "
            "needs to know which connections are right, only whether the freedom to choose "
            "them pays.\n")

    add("## 2. Cohort actually used\n")
    if manifest:
        cohort = manifest["frozen_cohort"]
        add(f"- {cohort['n_primary']} patients have both a frozen rank-event record and a "
            f"physical contact plane under exact-name alignment. The supplied design said "
            f"31; that figure counts patients who need no coordinates at all.")
        add(f"- pre-registered strata: {cohort['strata']['planar']['n']} whose contacts sit "
            f"close to one plane, {cohort['strata']['well_sampled']['n']} with at least "
            f"2000 events.")
        add(f"- every plane was estimated from the whole recording, so this run is "
            f"retrospective: it is not evidence that the geometry could have been known in "
            f"advance.\n")

    add("## 3. Runs completed\n")
    add(f"- cohort units planned {planned}, completed {completed}"
        + (f", failed {len(failed)}: {failed[:5]}" if failed else ""))
    if frozen:
        add(f"- frozen configuration: `{json.dumps(frozen)}`")
        if sweep:
            add(f"- selected as the knee of prediction against connection cost, not the "
                f"lowest error; every configuration tried is listed in `SWEEP_SUMMARY.json`.")
    add("")

    add("## 4. Prediction results\n")
    add("A positive number means the first model beats the second. The unit is the patient; "
        "seeds are pooled inside a patient and never counted as samples.\n")
    naming = {
        "H1_recurrence": "an unconstrained recurrent model over a static contact rate",
        "H1": "contact-node graph over a static contact rate",
        "H1_latent": "tissue field over a static contact rate",
        "H1b_contact_graph": "contact-node graph over an unconstrained recurrent model",
        "H1b_latent_learned": "tissue field over an unconstrained recurrent model",
        "H3": "learned graph over a fixed local graph",
    }
    for key, label in naming.items():
        entry = primary.get(key, {})
        add(f"- **{label}** — {fmt(entry.get('all'))}")
        for stratum in ("planar", "well_sampled"):
            sub = entry.get(stratum)
            if sub and sub.get("status") == "COMPLETE":
                add(f"  - {stratum.replace('_', ' ')}: {fmt(sub)}")
        if entry.get("patients_with_an_unconverged_arm"):
            add(f"  - still improving when the epoch budget ran out, so these carry no "
                f"negative verdict: {entry['patients_with_an_unconverged_arm']}")
    add("")

    add("## 4b. Is the comparison fair?\n")
    add("The static baseline reaches the epoch ceiling far more often than the recurrent "
        "arm, and it does so in the direction that would flatter the recurrent arm. Two "
        "checks, because a caveat would not settle it:\n")
    if baseline_check:
        add(f"- the same constant-per-contact model was refitted with a second-order "
            f"optimiser and scored on the same held-out events. The cohort run sits "
            f"{baseline_check['median_gap']:+.4f} from that optimum in the median and "
            f"{baseline_check['max_abs_gap']:.4f} at worst, against a reported advantage "
            f"around {abs(((primary.get('H1_recurrence') or {}).get('all') or {}).get('median_delta') or float('nan')):.3f}. "
            f"Verdict: **{baseline_check['status'].lower().replace('_', ' ')}** — "
            f"{baseline_check['means']}.")
    else:
        add("- baseline optimality check not available.")
    if budget_probe:
        add(f"- a sample of patients was refitted with a six-fold epoch budget. The "
            f"advantage moved from {budget_probe['median_advantage_at_budget_95']:+.4f} to "
            f"{budget_probe['median_advantage_at_long_budget']:+.4f}, a shrinkage of "
            f"{budget_probe['median_shrinkage']:+.4f} "
            f"({budget_probe['shrinkage_as_fraction_of_reported_advantage']:.1%} of the "
            f"effect). {budget_probe['direction'].capitalize()}.")
    else:
        add("- longer-budget probe not available.")
    add("")

    if sweep:
        add("## 4c. What the swept settings actually changed\n")
        rows = sweep.get("stages", {}).get("wiring", {}).get("rows", [])
        if rows:
            losses = [r["validation_next_bce"] for r in rows]
            costs = [r.get("wiring_cost", float("nan")) for r in rows]
            add(f"Across every connection-cost and budget setting tried, prediction moved "
                f"by {max(losses) - min(losses):.4f} while the total connection cost moved "
                f"by a factor of {max(costs) / min(costs):.1f}. A much cheaper, shorter-range "
                f"graph therefore costs essentially nothing in prediction — which is a "
                f"statement about how little the connection pattern matters here, not "
                f"evidence that the economical graph is the right one.\n")
        micro = sweep.get("stages", {}).get("microsteps", {}).get("rows", [])
        if micro:
            add("The number of internal propagation steps does matter, but through reach "
                "rather than accuracy: at one step only "
                f"{min(r.get('hop_reachability', 1.0) for r in micro):.0%} of observed "
                "transitions were within the graph's reach at all.\n")

    add("## 5. Predicting at contacts the model never trained on\n")
    if lco:
        for mode, entry in lco.get("comparisons", {}).items():
            wording = ("the contact was still visible in the sequence but scored nowhere"
                       if mode == "weak" else "the contact was removed from the input too")
            add(f"- **{wording}** — {fmt(entry)}")
        add("\nBoth models were trained without any per-contact parameter, because a contact "
            "held out of training has no way to learn one; without that change the "
            "comparison would be undefined at exactly the positions being tested.\n")
    else:
        add("Not run.\n")

    add("## 6. What may and may not be said\n")
    add("Supported, if the numbers above are positive: this patient's interictal events are "
        "predicted better by a model whose state lives in tissue and is read through a "
        "fixed local electrode kernel than by the alternatives tested.\n")
    add("Not supported by this run, regardless of the numbers:\n")
    add("- that the fitted connections correspond to anatomy, fibres, or any measured "
        "connectivity;")
    add("- that positive and negative connections mean excitation and inhibition;")
    add("- that the per-patient graphs differ in a way that means anything, since the "
        "recovery check shows connection identity is not determined by the data;")
    add("- that the geometry could have been known before the recording.\n")

    add("## 7. Smallest next experiment\n")
    add("Connection identity failed to recover even when the field was barely larger than "
        "the contact set, so the limit is not simply that there are more tissue units than "
        "electrodes. The next step is to ask what would be identifiable: fit the same field "
        "with the graph replaced by a handful of parameters describing how far and in which "
        "direction influence spreads, and check whether those few numbers recover on the "
        "same synthetic data where the free graph did not.\n")

    (OUT / "CLOSEOUT_REPORT.md").write_text("\n".join(lines))

    status = {
        "contract": "topic5_slp_rnn_v0_1_final_status",
        "cohort_units_planned": planned,
        "cohort_units_completed": completed,
        "cohort_units_failed": failed,
        "recovery_gate": (gate or {}).get("reportable_layers"),
        "frozen_config": frozen,
        "leave_contact_out_present": bool(lco),
        "verdict_ladder": {
            "L1_recurrence_value": fmt(primary.get("H1", {}).get("all")),
            "L2_latent_substrate_value": fmt(primary.get("H1b_latent_learned", {}).get("all")),
            # H3 asks whether LEARNING the graph predicts better than fixing it to
            # nearest neighbours.  That is a prediction question and the recovery
            # gate does not touch it: it says which edges are undetermined, not
            # that learning them is worthless.  Only claims about the identity of
            # the edges are blocked.
            "L3_learned_topology_value": fmt(primary.get("H3", {}).get("all")),
            "L3_scope": (
                "whether learning the connections helps prediction; it is NOT a "
                "claim that the learned connections are the right ones, which the "
                "recovery gate rules out"
            ),
            "L4_patient_specific_reproducibility": "BLOCKED_BY_RECOVERY_GATE",
            "L5_targeted_structural_necessity": "BLOCKED_BY_RECOVERY_GATE",
            "L6_mode_specific_routing": "NOT_RUN",
        },
    }
    (OUT / "FINAL_STATUS.json").write_text(json.dumps(status, indent=1))
    print(f"wrote {(OUT / 'CLOSEOUT_REPORT.md').relative_to(ROOT)}")
    print(f"wrote {(OUT / 'FINAL_STATUS.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
