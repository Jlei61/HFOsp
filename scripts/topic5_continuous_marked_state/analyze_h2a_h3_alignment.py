#!/usr/bin/env python3
"""Post-hoc patient-level alignment of old H2a and new H3-S0 effects."""
from __future__ import annotations

import json
import os
from pathlib import Path

from scipy.stats import spearmanr

from src.topic5_continuous_marked_state import contract
from src.topic5_continuous_marked_state.exposure import EXPOSURE_REVISION


REVISION = "h2a_graph_vs_h3_exposure_patient_alignment_v1"
TAUS = (3.0, 5.0, 10.0, 20.0, 30.0)
GRAPH_METADATA = (
    contract.UPSTREAM_ROOT
    / "figures/revisions/20260821-0151/epi_prssm_core_evidence"
    / "epi_prssm_core_evidence_metadata.json"
)
GRAPH_CONTRASTS = (
    "degree_preserving_rewire@generator",
    "forward_only_shuffled@generator",
    "degree_preserving_rewire@decoder",
)


def main() -> None:
    graph = json.loads(GRAPH_METADATA.read_text())
    if (
        graph.get("status") != "EXPLORATORY_DEVELOPMENT"
        or graph.get("package_hash")
        != "8fd11957dceec1c2a81b4b87ca9687fa5d8ab93557f5bc20715e4b4f38048087"
    ):
        raise ValueError("unexpected or unpinned H2a graph evidence package")
    exposure = json.loads(
        (contract.RESULT_ROOT / "exposure_screen/EXPOSURE_SCREEN_SUMMARY.json")
        .read_text()
    )
    if (
        exposure.get("exposure_revision") != EXPOSURE_REVISION
        or exposure.get("n_runs") != 748
        or exposure.get("sealed_opened") is not False
    ):
        raise ValueError("stale, incomplete, or unsealed H3-S0 package")

    rows = []
    for graph_name in GRAPH_CONTRASTS:
        graph_values = (
            graph["panels"]["B_graph_path"]["contrasts"]
            [graph_name]["per_patient"]
        )
        for kind in ("load", "participation"):
            for tau in TAUS:
                cell = next(
                    row for row in exposure["by_tau"]
                    if row["exposure_kind"] == kind
                    and float(row["tau_minutes"]) == tau
                )
                for endpoint in ("mark_nll", "stop_nll"):
                    exposure_values = (
                        cell["endpoints"][endpoint]
                        ["patient_deltas_real_minus_placebo"]
                    )
                    common = sorted(set(graph_values) & set(exposure_values))
                    if len(common) != 34:
                        raise ValueError(
                            f"H2a/H3 patient mismatch for {graph_name}/{kind}/{tau}"
                        )
                    result = spearmanr(
                        [graph_values[subject] for subject in common],
                        [exposure_values[subject] for subject in common],
                    )
                    rows.append({
                        "graph_contrast": graph_name,
                        "exposure_kind": kind,
                        "tau_minutes": tau,
                        "h3_endpoint": endpoint,
                        "n_patients": len(common),
                        "spearman_rho": float(result.statistic),
                        "two_sided_p_descriptive_unadjusted": float(result.pvalue),
                        "same_direction_semantics": (
                            "Both deltas are lower-is-better; positive rho would "
                            "mean patients with stronger graph benefit also tend "
                            "to have stronger exposure benefit."
                        ),
                    })
    output = {
        "contract": contract.REVISION,
        "alignment_revision": REVISION,
        "h2a_graph_package_hash": graph["package_hash"],
        "h3_exposure_revision": EXPOSURE_REVISION,
        "n_cells": len(rows),
        "rows": rows,
        "sealed_opened": False,
        "claim_boundary": (
            "Post-hoc cross-line alignment. These correlations do not test "
            "mediation or causality and are not adjusted across the exploratory "
            "grid; absence of positive alignment keeps H2a and H3 as distinct "
            "evidence rather than disproving either result."
        ),
    }
    path = contract.RESULT_ROOT / "exposure_screen/H2A_H3_ALIGNMENT.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True))
    os.replace(temporary, path)
    print(json.dumps({
        "path": str(path), "n_cells": len(rows), "n_patients": 34
    }, sort_keys=True))


if __name__ == "__main__":
    main()
