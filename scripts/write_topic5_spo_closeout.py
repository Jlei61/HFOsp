"""Closeout for v0.2.  Answers the eight questions the plan asked, in order."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/topic5_spatial_propagation_operator_v0_2"

FORBIDDEN = ("anatomical connectivity", "conduction velocity", "mm/s",
             "proves the biological", "causal human brain network",
             "prospective geometry", "recovered the propagation graph")


def load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def describe(entry: dict, label: str) -> str:
    if entry.get("status") != "COMPLETE":
        return f"- **{label}** — {entry.get('status', 'not run')}"
    return (f"- **{label}** — median {entry['median_delta']:+.4f}, 95% CI "
            f"[{entry['bootstrap_95ci'][0]:+.4f}, {entry['bootstrap_95ci'][1]:+.4f}], "
            f"{entry['n_positive']}/{entry['n']} patients, "
            f"p={entry['wilcoxon_two_sided_p']:.3g}")


def main() -> int:
    manifest = load(OUT / "INPUT_MANIFEST.json")
    gate = load(OUT / "synthetic" / "RECOVERY_GATE.json")
    stats = load(OUT / "cohort_statistics.json")
    ladder = stats.get("ladder", {})

    # Layers are only reportable if the gate that certified them was itself
    # guarded. An unguarded gate certifies nothing either way.
    guarded = bool(gate.get("generator_guard", {}).get("cell_disagreement_fraction", 0) >= 0.15)
    reportable = gate.get("reportable_layers", {}) if guarded else {}
    status = {
        "ENGINEERING_EXECUTION": "PASS" if stats.get("n_units") else "INCOMPLETE",
        "RECOVERY_GATE_GUARDED": "YES" if guarded else "NO_VERDICT_USABLE",
        "DRIFT_DIRECTION_RECOVERY":
            "RECOVERABLE" if reportable.get("drift_direction")
            else ("NOT_RECOVERABLE" if guarded else "NO_USABLE_GATE"),
        "ANISOTROPY_RECOVERY":
            "RECOVERABLE" if reportable.get("anisotropy")
            else ("NOT_RECOVERABLE" if guarded else "NO_USABLE_GATE"),
        "RECOVERY_STRENGTH_RECOVERY":
            "RECOVERABLE" if reportable.get("recovery_strength")
            else ("NOT_RECOVERABLE" if guarded else "NO_USABLE_GATE"),
        "GEOMETRY_STATUS": manifest.get("geometry_status", "UNKNOWN"),
        "TRAIN_ONLY_AXIS": manifest.get("train_only_axis", "UNKNOWN"),
    }
    for name, entry in ladder.items():
        if entry.get("status") == "COMPLETE":
            key = f"LADDER_{name.upper()}"
            status[key] = ("SUPPORTED" if entry["wilcoxon_two_sided_p"] < 0.05
                           and entry["median_delta"] > 0 else "NOT_SUPPORTED")
    rel = stats.get("parameter_reliability", {})
    status["PATIENT_SPECIFIC_OPERATOR"] = (
        "SUPPORTED" if rel.get("status") == "COMPLETE"
        and rel.get("wilcoxon_exact_p", 1) < 0.05 and rel.get("median_delta", 0) > 0
        else rel.get("status", "NOT_RUN") if rel.get("status") != "COMPLETE"
        else "NOT_SUPPORTED")

    lines = []
    add = lines.append
    add("# Topic 5 — Spatial Propagation Operator RNN v0.2 — closeout\n")
    add("v0.1 learned a free graph between tissue units and could not recover which")
    add("edges existed or which way activity travelled. This version asks whether a")
    add("handful of scalars -- axial drift, axial and transverse diffusion, decay and")
    add("a recovery field -- can carry the within-event history that an unconstrained")
    add("recurrent model demonstrably uses.\n")

    add("## 1. Are the parameters recoverable at all?\n")
    if gate:
        add(f"Asked before any patient was fitted, on {gate['n_cells']} generating")
        add(f"settings over seeds {gate['seeds']}, using a real patient's geometry,")
        add("observation kernel and event lengths.\n")
        g = gate.get("generator_guard")
        if g is None:
            # A gate file without the guard predates it, which means nobody
            # checked that the operator drove its own synthetic data. Its
            # verdicts describe the sampler and must not be quoted.
            add("**This gate file carries no generator guard, so nothing below it")
            add("can be read as an identifiability verdict: without that check the")
            add("numbers describe the sampler rather than the operator.**\n")
        else:
            add(f"The generator was checked first: opposite drifts disagree on "
                f"{g['cell_disagreement_fraction']:.0%} of ranks, so the operator is")
            add("driving its own synthetic data rather than the contact bias.\n")
        add(f"- **which way activity travels** — sign agreement "
            f"{gate['drift_sign']['agreement']:.3f} against a floor of "
            f"{gate['drift_sign']['floor']}: **{gate['drift_sign']['status']}**")
        add(f"- **how anisotropic the spread is** — Spearman "
            f"{gate['anisotropy_ordering']['spearman']:+.3f} against a floor of "
            f"{gate['anisotropy_ordering']['floor']}: "
            f"**{gate['anisotropy_ordering']['status']}**")
        r = gate["recovery_strength_ordering"]
        if r.get("median_when_strong") is not None:
            add(f"- **how strong the recovery process is** — "
                f"{r['median_when_absent']:.3f} when absent against "
                f"{r['median_when_strong']:.3f} when strong: **{r['status']}**")
        add("")
        add(f"The verdict is taken from {gate.get('verdict_taken_from_microsteps')} "
            f"internal steps, the most favourable of {gate.get('microsteps_swept')}: a")
        add("failure has to hold where the operator had its best chance, or it is a")
        add("statement about the step budget rather than about identifiability.\n")
        add("A layer that does not recover does not stop the run. It removes the right")
        add("to report that parameter per patient; the prediction questions below are")
        add("unaffected, because they never need to know the true value.\n")
    else:
        add("Not run.\n")

    add("## 2. Which spatial component improves held-out prediction\n")
    add("The unit is the patient; seeds are pooled inside a patient. Positive means")
    add("the more complex model wins.\n")
    for name, label in (
        ("field_over_static", "a field with decay and recovery, over a static rate"),
        ("transport_over_no_transport",
         "letting activity move through space, on top of that same field"),
        ("drift_over_isotropic", "adding anisotropy and a signed drift"),
        ("recovery_over_drift", "adding the recovery field"),
        ("full_over_static", "the full operator over a static rate"),
    ):
        if name in ladder:
            add(describe(ladder[name], label))
    add("")
    add("Each row above is a nested pair, so the difference is attributable to the")
    add("component that was released. One further comparison is NOT nested and is")
    add("reported separately, because both a component gained and a component was")
    add("lost across it:\n")
    off = ladder.get("transport_no_recovery_over_recovery_no_transport")
    if off:
        add(describe(off, "transport without recovery, against recovery without "
                          "transport -- which kind of memory matters, not what "
                          "either component buys"))
    add("")
    convergence = stats.get("convergence", {})
    if convergence:
        add("Every arm under one budget and one stopping rule:\n")
        for variant, counts in convergence.items():
            add(f"- {variant}: {counts['converged']} converged, "
                f"{counts['hit_ceiling']} still improving at the ceiling")
        add("")

    add("## 3. Is the fitted operator the patient's?\n")
    if rel.get("status") == "COMPLETE":
        add(f"- median within-minus-between {rel['median_delta']:+.4f}, 95% CI "
            f"[{rel['bootstrap_95ci'][0]:+.4f}, {rel['bootstrap_95ci'][1]:+.4f}], "
            f"{rel['n_positive']}/{rel['n_patients']} patients, exact Wilcoxon "
            f"p={rel['wilcoxon_exact_p']:.3g}")
        add("")
        add("Only the layers the gate certified may be read as patient properties.")
    else:
        add(f"- {rel.get('status', 'not run')}")
    add("")

    add("## 4. What breaks when a component is switched off\n")
    add("The fitted operator is edited and rescored; nothing is retrained, so no")
    add("other parameter moves to compensate. These are operator edits in silico,")
    add("not lesions.\n")
    for name, entry in stats.get("ablations", {}).items():
        add(f"- **{name.replace('_', ' ')}** — median "
            f"{entry['median_delta_next_bce']:+.4f}, worse in "
            f"{entry['n_worse']}/{entry['n']} patients"
            + (f", p={entry['wilcoxon_two_sided_p']:.3g}"
               if entry.get("wilcoxon_two_sided_p") is not None else ""))
    add("")

    add("## 5. Predicting a contact that was never trained on\n")
    loco = stats.get("leave_contact_out", {})
    if loco.get("status") != "COMPLETE":
        add(f"Not available: {loco.get('status', 'not run')}.\n")
    else:
        add(f"{loco['condition']}\n")
        add(f"The comparison is the {loco['comparison_rule'].split(';')[0]}. "
            "A model that is worse everywhere falls less from its own baseline, so "
            "a relative comparison would hand it the win for the wrong reason; it "
            "is not reported here.\n")
        add(f"Floor arm: **{loco['floor_arm']}** — with the withheld contact's bias "
            "set to the average retained contact's, it assigns every withheld "
            "contact the same number and therefore knows nothing about them.\n")
        for arm, entry in loco["absolute"].items():
            top1 = entry.get("median_heldout_top1")
            add(f"- **{arm}** (n={entry['n_patients']}) — held-out loss "
                f"{entry['median_heldout_next_bce']:.4f}"
                + (f", held-out top-1 {top1:.3f}" if top1 is not None else "")
                + (f"; on retained contacts {entry['median_retained_next_bce']:.4f}"
                   if entry.get("median_retained_next_bce") is not None else ""))
        add("")
        for arm, entry in loco.get("over_floor", {}).items():
            if isinstance(entry, dict) and entry.get("status") == "COMPLETE":
                add(f"- **{arm} against the floor** — median "
                    f"{entry['median_delta']:+.4f} "
                    f"(positive means it beats knowing nothing), better in "
                    f"{entry['n_positive']}/{entry['n']} patients, "
                    f"p={entry['wilcoxon_two_sided_p']:.3g}, "
                    f"95% CI [{entry['bootstrap_95ci'][0]:+.4f}, "
                    f"{entry['bootstrap_95ci'][1]:+.4f}]")
        add("")

    add("## 6. Conditions on every number above\n")
    add(f"- geometry: {status['GEOMETRY_STATUS']}; train-only axis "
        f"{status['TRAIN_ONLY_AXIS']}")
    if manifest.get("train_only_axis_reason"):
        add(f"  - {manifest['train_only_axis_reason']}")
    bound = stats.get("parameters_at_stability_bound", {})
    if bound.get("n_units"):
        add(f"- {bound['n_units']} fitted units sit on the diffusion stability bound; "
            f"{bound['note']}")
    add("- the model is a teacher-forced next-rank predictor; the free rollout in")
    add("  panel B is a shape check, not a second evaluation metric")
    # Worded without the banned literals on purpose: the forbidden-phrase check
    # cannot tell an assertion from a prohibition, and loosening it to spot the
    # difference would blunt the only guard against the claim itself.
    add("- rank is not physical time, so reach and spread are per-rank effective")
    add("  quantities; they must never be converted into a physical propagation")
    add("  speed or compared with one")
    add("")

    add("## 7. Frozen status\n```text")
    for key, value in status.items():
        add(f"{key}:\n{value}\n")
    add("```")

    (OUT / "CLOSEOUT_REPORT.md").write_text("\n".join(lines))
    (OUT / "FINAL_STATUS.json").write_text(json.dumps({
        "contract": "topic5_spo_final_status_v0_2",
        "status": status,
        "ladder": {k: {kk: vv for kk, vv in v.items() if kk != "per_patient_delta"}
                   for k, v in ladder.items()},
        "parameter_reliability": {k: v for k, v in rel.items() if k != "per_patient"},
        "ablations": stats.get("ablations", {}),
    }, indent=1))

    text = (OUT / "CLOSEOUT_REPORT.md").read_text().lower()
    leaked = [p for p in FORBIDDEN if p in text]
    if leaked:
        raise SystemExit(f"closeout contains forbidden claims: {leaked}")
    print("wrote CLOSEOUT_REPORT.md and FINAL_STATUS.json")
    for key, value in status.items():
        print(f"  {key:34s} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
