#!/usr/bin/env python3
"""Build a same-level Human–RNN–SNN observable table without rerunning the SNN."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def finite(rows: list[dict[str, str]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value): values.append(value)
    return np.asarray(values, float)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--snn-readout", type=Path, required=True)
    args = parser.parse_args()
    out_root = args.out_root.resolve()
    snn_path = args.snn_readout.resolve()
    snn = json.loads(snn_path.read_text())
    early = read_csv(out_root / "early_ictal_per_patient_model.csv")
    fields = read_csv(out_root / "model_field_patient_metrics.csv")
    influence = read_csv(out_root / "effective_influence_fit_seed.csv")
    lesions = read_csv(out_root / "matched_lesion_patient_metrics.csv")
    interictal = read_csv(out_root / "interictal_per_patient.csv")
    output: list[dict[str, Any]] = []

    human_early = [row for row in early if row["model"] == "EMPIRICAL_REFERENCE"
                   and row["endpoint"] == "canonical_full" and row["primary"] == "True"]
    output.extend([
        {"system": "Human", "observable": "opposite-direction interictal templates",
         "status": "available", "value": "frozen empirical A/B template labels",
         "denominator": len({row["subject"] for row in interictal}),
         "comparison_level": "contact-rank field",
         "boundary": "not an anatomical edge map", "source": "INPUT_MANIFEST.json"},
        {"system": "Human", "observable": "early-ictal canonical-field concordance",
         "status": "available", "value": float(np.nanmedian(finite(human_early, "all_contact_margin"))),
         "denominator": len(human_early), "comparison_level": "patient-level null-relative R3 field margin",
         "boundary": "association, not seizure prediction", "source": "early_ictal_per_patient_model.csv"},
        {"system": "Human", "observable": "matched perturbation",
         "status": "not available", "value": None, "denominator": 0,
         "comparison_level": "none", "boundary": "no human intervention in this analysis",
         "source": "not applicable"},
    ])

    models = sorted({row["model"] for row in fields if row["cell"] == "rnn"})
    for model in models:
        model_fields = [row for row in fields if row["model"] == model and row["cell"] == "rnn"]
        model_early = [row for row in early if row["model"] == model and row["cell"] == "rnn"
                       and row["endpoint"] == "canonical_full" and row["primary"] == "True"]
        model_influence = [row for row in influence if row["model"] == model and row["cell"] == "rnn"]
        model_interictal = [row for row in interictal if row["model"] == model and row["cell"] == "rnn"]
        model_lesion = [row for row in lesions if row["model"] == model and row["cell"] == "rnn"
                        and row["lesion"] == "connector_nodes" and row["all_inference_available"] == "True"]
        output.extend([
            {"system": f"RNN:{model}", "observable": "heldout interictal propagation",
             "status": "available", "value": float(np.nanmedian(finite(model_interictal, "rollout_spearman"))),
             "denominator": len(model_interictal), "comparison_level": "seed-removed free-rollout rank correlation",
             "boundary": "same-start model sufficiency, not connectome recovery",
             "source": "interictal_per_patient.csv"},
            {"system": f"RNN:{model}", "observable": "empirical interictal A/B field fidelity",
             "status": "available", "value": float(np.nanmedian(finite(model_fields, "matched_empirical_r"))),
             "denominator": len(model_fields), "comparison_level": "contact-rank field",
             "boundary": "shared patients support one-model A/B; non-collinear patients use separate fits",
             "source": "model_field_patient_metrics.csv"},
            {"system": f"RNN:{model}", "observable": "early-ictal canonical-field concordance",
             "status": "available", "value": float(np.nanmedian(finite(model_early, "all_contact_margin"))),
             "denominator": len(model_early), "comparison_level": "patient-level null-relative R3 field margin",
             "boundary": "target-free frozen external benchmark", "source": "early_ictal_per_patient_model.csv"},
            {"system": f"RNN:{model}", "observable": "open-loop effective reach lag 1/2/3",
             "status": "available" if model_influence else "missing", "value": (
                 [float(np.nanmedian(finite(model_influence, key))) for key in
                  ("lag1_reach_mm", "lag2_reach_mm", "lag3_reach_mm")] if model_influence else None),
             "denominator": len(model_influence), "comparison_level": "contact-space finite-pulse response (mm)",
             "boundary": "rank-step, not real-time dynamics", "source": "effective_influence_fit_seed.csv"},
            {"system": f"RNN:{model}", "observable": "connector matched-lesion specificity",
             "status": "available" if model_lesion else "not estimable", "value": (
                 float(np.nanmedian(finite(model_lesion, "specificity_contact_nll"))) if model_lesion else None),
             "denominator": len(model_lesion), "comparison_level": "heldout interictal ΔNLL beyond matched random lesion",
             "boundary": "in-model perturbation only", "source": "matched_lesion_patient_metrics.csv"},
        ])

    output.extend([
        {"system": "SNN:E1146", "observable": "opposite-source bidirectionality",
         "status": "available", "value": {"bidirectional": bool(snn["bidirectional"]),
                                             "clean_forward": int(snn["clean_forward"]),
                                             "clean_reverse": int(snn["clean_reverse"])},
         "denominator": int(snn["n_clean"]), "comparison_level": "virtual-contact event ranks",
         "boundary": "single subject-specific forward model", "source": str(snn_path)},
        {"system": "SNN:E1146", "observable": "spatial reach",
         "status": "available", "value": float(snn["inter_core_sheet"]),
         "denominator": 1, "comparison_level": "sheet distance (model units)",
         "boundary": "units are not directly equated to RNN millimetres", "source": str(snn_path)},
        {"system": "SNN:E1146", "observable": "early seizure-like recruitment field",
         "status": "not established", "value": None, "denominator": 0,
         "comparison_level": "not comparable", "boundary": (
             "existing SNN artifact supports bidirectional interictal readout but not an accepted closed-loop "
             "early-ictal recruitment field"), "source": str(snn_path)},
        {"system": "SNN:E1146", "observable": "matched perturbation",
         "status": "not comparable", "value": None, "denominator": 0,
         "comparison_level": "not comparable", "boundary": "no matched lesion contract shared with the RNN",
         "source": str(snn_path)},
    ])

    csv_path = out_root / "COMMON_OBSERVABLES.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        for row in output:
            row = dict(row)
            if isinstance(row["value"], (dict, list)): row["value"] = json.dumps(row["value"])
            writer.writerow(row)
    (out_root / "COMMON_OBSERVABLES.json").write_text(json.dumps({
        "contract": "topic5_human_rnn_snn_common_observables_v0_4",
        "edge_to_edge_comparison": False,
        "hidden_unit_to_neuron_comparison": False,
        "snn_rerun": False,
        "rows": output,
    }, indent=2))
    (out_root / "stage_h_scientific_drift_audit.json").write_text(json.dumps({
        "status": "ALIGNED",
        "comparison_level": "shared mesoscopic observables only",
        "snn_rerun": False,
        "explicitly_missing": ["accepted SNN early-ictal recruitment field", "cross-system edge mapping"],
        "not_claimed": ["RNN units are neurons", "RNN edges are synapses", "SNN validates the RNN architecture"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
