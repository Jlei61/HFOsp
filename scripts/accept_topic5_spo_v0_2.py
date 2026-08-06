"""Acceptance gate for v0.2.  Exits 0 only when the deliverables exist and cohere.

This checks that the work was DONE and reported honestly.  It never checks that
a scientific result came out a particular way -- a run where nothing recovers and
nothing improves prediction passes, provided it says so.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/topic5_spatial_propagation_operator_v0_2"

VARIANTS = ("STATIC", "FIELD_NULL", "ISOTROPIC_DIFFUSION",
            "ANISOTROPIC_DRIFT", "ANISOTROPIC_RECOVERY")


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    return ok


def main() -> int:
    results = []

    manifest_path = OUT / "INPUT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    cohort = manifest.get("frozen_cohort", {}).get("primary", [])
    results.append(check("input manifest with a frozen cohort", bool(cohort),
                         f"n={len(cohort)}"))

    # Geometry status must be recorded, whatever it is. A run that quietly drops
    # this is the one that later gets read as prospective.
    results.append(check(
        "geometry status and train-only-axis status both recorded",
        bool(manifest.get("geometry_status")) and bool(manifest.get("train_only_axis")),
        f"{manifest.get('geometry_status')} / {manifest.get('train_only_axis')}"))

    missing_cache = [s for s in cohort if not (OUT / "cache" / s / "grid.npz").exists()]
    results.append(check("grid and kernel cached for every patient", not missing_cache,
                         f"missing {missing_cache[:3]}" if missing_cache else ""))

    test = subprocess.run(
        [sys.executable, "-m", "pytest",
         # The holdout tests belong in the gate, not beside it: they guard the
         # bug class that produced a publishable-looking number rather than an
         # error, twice.
         str(ROOT / "tests/test_topic5_spatial_propagation_operator.py"),
         str(ROOT / "tests/test_topic5_spo_holdout.py"), "-q"],
        capture_output=True, text=True, cwd=ROOT)
    results.append(check("correctness suite green", test.returncode == 0,
                         test.stdout.strip().splitlines()[-1] if test.stdout else ""))

    gate_path = OUT / "synthetic" / "RECOVERY_GATE.json"
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else {}
    results.append(check(
        "recovery gate returned a verdict for every layer",
        all(gate.get(k, {}).get("status") for k in
            ("drift_sign", "anisotropy_ordering", "recovery_strength_ordering")),
        " ".join(f"{k.split('_')[0]}={gate.get(k, {}).get('status')}"
                 for k in ("drift_sign", "anisotropy_ordering",
                           "recovery_strength_ordering")) if gate else "missing"))

    # The guard that makes the gate mean anything: if the generator had not been
    # informative, every verdict above would describe the sampler.
    guard = gate.get("generator_guard", {})
    results.append(check(
        "generator was shown to be informative before the gate was read",
        guard.get("cell_disagreement_fraction", 0) >= 0.15,
        f"opposite drifts differ on "
        f"{guard.get('cell_disagreement_fraction', 0):.0%} of ranks"))

    expected = [(s, v) for s in cohort for v in VARIANTS]
    done = [(s, v) for s, v in expected
            if (OUT / "per_subject" / s / v / "seed1" / "DONE.json").exists()]
    results.append(check("every variant on every patient at the first seed",
                         len(done) == len(expected),
                         f"{len(done)}/{len(expected)}"))

    stragglers = sorted(
        p.parent.relative_to(OUT / "per_subject").as_posix()
        for p in (OUT / "per_subject").rglob("FAILED.json")
        if not (p.parent / "DONE.json").exists()
    ) if (OUT / "per_subject").exists() else []
    results.append(check("no unit left in a failed state", not stragglers,
                         f"{len(stragglers)}, e.g. {stragglers[:2]}" if stragglers else ""))

    stats_path = OUT / "cohort_statistics.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    results.append(check("nested ladder aggregated", bool(stats.get("ladder"))))
    results.append(check("operator ablations run",
                         (OUT / "operator_ablation.csv").exists()))
    results.append(check("per-patient parameter estimates written",
                         (OUT / "parameter_estimates.csv").exists()))

    figures = OUT / "figures"
    pngs = list(figures.glob("*.png")) if figures.exists() else []
    loco = stats.get("leave_contact_out", {})
    results.append(check(
        "leave-contact-out aggregated with a floor arm",
        loco.get("status") == "COMPLETE" and bool(loco.get("over_floor")),
        f"status={loco.get('status', 'not run')}, "
        f"arms={sorted(loco.get('absolute', {}))}"))
    # The failure this guards against did not raise: withholding a contact from
    # every split left it out of the targets too, so the evaluation scored an
    # empty question and returned a near-perfect number.
    results.append(check(
        "every leave-contact-out arm withheld the same contacts",
        loco.get("arms_withheld_the_same_contacts") is True
        if loco.get("status") == "COMPLETE" else False,
        "leave-contact-out has not run yet" if loco.get("status") != "COMPLETE"
        else f"arms disagree on {loco.get('patients_where_arms_disagree')}"))

    absolute = loco.get("absolute", {})
    empty = [a for a, e in absolute.items() if e["median_heldout_next_bce"] <= 1e-3]
    results.append(check(
        "held-out evaluation is not scoring an empty target set",
        bool(absolute) and not empty,
        "leave-contact-out has not run yet" if not absolute else
        f"{empty} sit at zero: the withheld contacts carry no positives"
        if empty else ""))

    results.append(check("figure rendered with a Chinese README",
                         bool(pngs) and (figures / "README.md").exists(),
                         f"{len(pngs)} png"))

    report = OUT / "CLOSEOUT_REPORT.md"
    results.append(check("closeout report written", report.exists()))

    print()
    if all(results):
        print("ACCEPTED: Topic 5 SPO-RNN v0.2 deliverables complete.")
        return 0
    print(f"NOT YET: {sum(1 for r in results if not r)} of {len(results)} checks failing.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
