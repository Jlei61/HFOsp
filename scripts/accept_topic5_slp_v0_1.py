"""Acceptance gate for Topic 5 SLP-RNN v0.1.

Exits 0 only when every deliverable the plan promised exists and is internally
consistent.  This is what "done" means for this run -- not a judgement call.

Run:  python scripts/accept_topic5_slp_v0_1.py
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/topic5_spatial_latent_propagation_rnn_v0_1"

# Phrases the spec forbids anywhere in the report, whatever the results say.
FORBIDDEN = (
    "anatomical connectivity",
    "synaptic connectome",
    "causal human brain network",
    "excitatory edge",
    "inhibitory edge",
    "proves the biological",
    "prospective geometry",
)


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    return ok


def main() -> int:
    results = []

    # 1. inputs frozen
    manifest_path = OUT / "INPUT_MANIFEST.json"
    ok = manifest_path.exists()
    manifest = json.loads(manifest_path.read_text()) if ok else {}
    results.append(check(
        "input manifest with a frozen cohort", ok and "frozen_cohort" in manifest,
        f"n={manifest.get('frozen_cohort', {}).get('n_primary')}" if ok else "missing",
    ))
    cohort = manifest.get("frozen_cohort", {}).get("primary", [])

    # 2. cache complete for the frozen cohort
    missing_cache = [s for s in cohort if not (OUT / "cache" / s / "events.npz").exists()]
    results.append(check("cache built for every cohort patient", not missing_cache,
                         f"missing {missing_cache[:3]}" if missing_cache else ""))

    # 3. tests green
    test = subprocess.run(
        [sys.executable, "-m", "pytest",
         str(ROOT / "tests/test_topic5_spatial_latent_rnn.py"), "-q"],
        capture_output=True, text=True, cwd=ROOT,
    )
    results.append(check("TDD suite green", test.returncode == 0,
                         test.stdout.strip().splitlines()[-1] if test.stdout else ""))

    # 4. recovery gate decided, whichever way
    gate_path = OUT / "synthetic" / "RECOVERY_GATE.json"
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else {}
    gate_ok = bool(gate.get("edge_identity", {}).get("status")
                   and gate.get("axis_direction", {}).get("status"))
    results.append(check("recovery gate returned a verdict", gate_ok,
                         f"edges={gate.get('edge_identity', {}).get('status')} "
                         f"axis={gate.get('axis_direction', {}).get('status')}"
                         if gate_ok else "missing"))

    # 5. exactly one frozen config
    frozen = OUT / "development" / "FROZEN_CONFIG.json"
    results.append(check("exactly one frozen config", frozen.exists()))

    # 6. cohort coverage recorded and reconciled
    matrix = OUT / "EXPERIMENT_MATRIX.csv"
    coverage_ok = False
    detail = "missing"
    if matrix.exists():
        rows = [r for r in matrix.read_text().strip().splitlines()[1:] if r]
        done_rows = [
            r for r in rows
            if (OUT / "per_subject" / r.split(",")[0] / r.split(",")[1]
                / f"seed{r.split(',')[2]}" / "DONE.json").exists()
        ]
        # "Some units ran" is not coverage.  The bar is every arm on every
        # cohort patient at the first seed, which is what the plan calls the
        # minimum publishable matrix; further seeds are a bonus.
        seed_one = {r.rsplit(",", 1)[0] for r in rows if r.endswith(",1")}
        seed_one_done = {r.rsplit(",", 1)[0] for r in done_rows if r.endswith(",1")}
        missing = sorted(seed_one - seed_one_done)
        summary_path = OUT / "patient_prediction_metrics.csv"
        coverage_ok = summary_path.exists() and not missing
        detail = (f"{len(done_rows)}/{len(rows)} units; seed 1 "
                  f"{len(seed_one_done)}/{len(seed_one)}"
                  + (f", missing e.g. {missing[:2]}" if missing else ""))
    results.append(check("cohort units run and aggregated", coverage_ok, detail))

    # 7. the mandatory guard arm is present for every patient that has any arm
    guard_ok = True
    guard_detail = ""
    if (OUT / "per_subject").exists():
        for subject_dir in (OUT / "per_subject").iterdir():
            arms = {a.name for a in subject_dir.iterdir() if a.is_dir()}
            if arms and "ORDINARY_GRU" not in arms:
                guard_ok = False
                guard_detail = f"{subject_dir.name} has no unconstrained-GRU arm"
                break
    results.append(check("unconstrained GRU present wherever any arm ran",
                         guard_ok, guard_detail))

    # 8. leave-contact-out reported
    lco = OUT / "leave_contact_out_metrics.csv"
    results.append(check("leave-contact-out results written", lco.exists()))

    # 9. figures with their Chinese README
    figures = OUT / "figures"
    figure_files = list(figures.glob("*.png")) if figures.exists() else []
    results.append(check("figures rendered with a README", bool(figure_files)
                         and (figures / "README.md").exists(),
                         f"{len(figure_files)} png"))

    # 10. closeout report, and no forbidden claim in it
    report = OUT / "CLOSEOUT_REPORT.md"
    if report.exists():
        text = report.read_text().lower()
        leaked = [p for p in FORBIDDEN if p in text]
        results.append(check("closeout report free of forbidden claims", not leaked,
                             f"leaked {leaked}" if leaked else ""))
    else:
        results.append(check("closeout report exists", False, "missing"))

    print()
    if all(results):
        print("ACCEPTED: Topic 5 SLP-RNN v0.1 deliverables complete.")
        return 0
    print(f"NOT YET: {sum(1 for r in results if not r)} of {len(results)} checks failing.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
