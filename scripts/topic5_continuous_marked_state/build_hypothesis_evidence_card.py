#!/usr/bin/env python3
"""Build one machine-readable H1-H3 evidence ledger from accepted artifacts."""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path("results/epi_prssm/continuous_marked_state/r0_1")
OLD_ROOT = Path("results/epi_prssm/v0_1")
CORE = OLD_ROOT / (
    "figures/revisions/20260821-0151/epi_prssm_core_evidence/"
    "epi_prssm_core_evidence_metadata.json"
)
H2A = OLD_ROOT / "event_distribution/H2A_EVIDENCE_CARD.json"
BRIDGE = ROOT / "bridge/BRIDGE_E0_SUMMARY.json"
T1_SPECTRAL = ROOT / "regular_t1/REGULAR_T1_SUMMARY.json"
T1_RAW = ROOT / "regular_t1/raw_e0/REGULAR_T1_SUMMARY.json"
STATE16 = ROOT / "regular_t1/sensitivities/state16/STATE16_SENSITIVITY_SUMMARY.json"
CLOCK = ROOT / "exposure_clock_control/PHYSICAL_VS_EVENT_COUNT_CLOCK.json"
CLOCK_SYNTHETIC = ROOT / "exposure_clock_control/CLOCK_IDENTIFIABILITY_SYNTHETIC.json"
CLOCK_SEPARABILITY = ROOT / "exposure_clock_control/CLOCK_SEPARABILITY_STRATA.json"
FIXED_COUNT = ROOT / "exposure_event_count_grid/FIXED_MEMORY_CLOCK_GRID_SUMMARY.json"
H2B_REPORT = Path(
    "docs/archive/topic5/epi_prssm_h2b_h3_revision_technical_2026-08-20.md"
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _patient_summary(values: list[float], favourable: str = "negative") -> dict:
    array = np.asarray(values, dtype=float)
    if favourable == "negative":
        n_favourable = int(np.sum(array < 0))
    elif favourable == "positive":
        n_favourable = int(np.sum(array > 0))
    else:
        raise ValueError(favourable)
    return {
        "median": float(np.median(array)),
        "n_favourable": n_favourable,
        "n_patients": int(len(array)),
        "favourable_direction": favourable,
    }


def _t1_summary(path: Path) -> dict:
    rows = _read(path)["per_subject"]
    filtered = [
        row["contrasts"]["validation_filtered"]["joint_nll"]
        ["median_t1_minus_t0"] for row in rows
    ]
    split_start = [
        row["contrasts"]["validation_correction_off_from_split_start"]
        ["joint_nll"]["median_t1_minus_t0"] for row in rows
    ]
    post = {
        horizon: [
            row["contrasts"]["post_anchor_correction_off"][horizon]
            ["joint_nll"]["median_t1_minus_t0"] for row in rows
        ]
        for horizon in ("5", "10", "20")
    }
    swap = [
        row["state_swap"]["joint_nll"]["median_t1_wrong_minus_correct"]
        for row in rows
    ]
    return {
        "filtered_t1_minus_t0": _patient_summary(filtered),
        "correction_off_from_split_start_t1_minus_t0": _patient_summary(
            split_start
        ),
        "post_anchor_correction_off_t1_minus_t0": {
            horizon: _patient_summary(values) for horizon, values in post.items()
        },
        "wrong_time_minus_correct_state": _patient_summary(
            swap, favourable="positive"
        ),
    }


def main() -> None:
    core = _read(CORE)
    h2a = _read(H2A)
    bridge = _read(BRIDGE)
    clock = _read(CLOCK)
    clock_synthetic = _read(CLOCK_SYNTHETIC)
    clock_separability = _read(CLOCK_SEPARABILITY)
    fixed_count = _read(FIXED_COUNT)
    state16 = _read(STATE16)
    classifications = Counter(row["classification"] for row in bridge["per_subject"])
    graph = core["panels"]["B_graph_path"]["contrasts"]

    def _prov(entry, *, origin, source, note=""):
        """Stamp every leaf with where it came from.

        The first card mixed this run's Bridge/T1/H3 output with values
        re-exported from the previous line (H2a's four legs came from the v0_1
        figure metadata and evidence card; H2b's readout is transcribed from a
        markdown report) and carried only a flat top-level source_files hash
        list. A reader -- including the run's own summary -- naturally read all
        of it as this run's output. Origin is now attached to the entry itself.
        """
        if isinstance(entry, dict):
            out = dict(entry)
        else:
            out = {"value": entry}
        out["provenance"] = {
            "origin": origin,           # new_this_run | carried_forward
            "source": str(source),
            "note": note,
        }
        return out

    clock_cells = []
    for cell in clock["cells"]:
        endpoints = {}
        for endpoint in ("joint_nll", "timing_nll", "mark_nll",
                         "participation_nll", "rank_nll", "stop_nll"):
            row = cell["endpoints"][endpoint]
            # Both contrasts are exported. delta_vs_history is the CLEAN one --
            # the history baseline is bit-identical across arms, verified to
            # 1e-10 -- while delta_vs_placebo subtracts two controls whose delay
            # max(30 min, 3*tau) differs once tau exceeds 10 min. Shipping only
            # the placebo variant hid the cleaner companion from any machine
            # reader.
            endpoints[endpoint] = {
                "physical_minus_event_count_vs_placebo": row["delta_vs_placebo"],
                "physical_minus_event_count_vs_history": row["delta_vs_history"],
                "event_count_minus_current_event_vs_placebo": row[
                    "event_count_minus_current_event_delta_vs_placebo"
                ],
                "event_count_minus_current_event_vs_history": row[
                    "event_count_minus_current_event_delta_vs_history"
                ],
            }
        clock_cells.append({
            "exposure_kind": cell["exposure_kind"],
            "tau_minutes_rate_matched_label": cell["tau_minutes"],
            "median_memory_events": cell[
                "nominal_memory_events_median_across_patients"
            ],
            "endpoints": endpoints,
        })
    fixed_count_cells = []
    for cell in fixed_count["cells"]:
        fixed_count_cells.append({
            "exposure_kind": cell["exposure_kind"],
            "memory_events": cell["memory_events"],
            "rate_matched_tau_minutes_across_patients": cell[
                "rate_matched_tau_minutes_across_patients"
            ],
            "endpoints": {
                endpoint: {
                    "real_minus_placebo": cell["endpoints"][endpoint][
                        "real_minus_placebo"
                    ],
                    "distributed_minus_current_delta_vs_placebo": cell[
                        "endpoints"
                    ][endpoint]["distributed_minus_current_delta_vs_placebo"],
                    "physical_minus_count_delta_vs_placebo": cell[
                        "endpoints"
                    ][endpoint]["physical_minus_count_delta_vs_placebo"],
                }
                for endpoint in ("joint_nll", "timing_nll", "mark_nll",
                                 "participation_nll", "rank_nll", "stop_nll")
            },
        })

    def _derive_provenance_summary(node, path=""):
        """Collect the origin stamps that are actually on the entries.

        Typed twice, an index and its entries drift. This walks the finished
        card so the summary is a projection of the stamps, not a parallel claim
        about them.
        """
        found = []
        if isinstance(node, dict):
            stamp = node.get("provenance")
            if isinstance(stamp, dict) and "origin" in stamp:
                found.append((path.lstrip("."), stamp["origin"]))
            for key, value in node.items():
                if key != "provenance":
                    found.extend(_derive_provenance_summary(value, f"{path}.{key}"))
        return found

    card = {
        "contract": "continuous_marked_state_h1_h3_evidence_ledger_v1",
        "date": "2026-08-24",
        "partition": "development_only",
        "sealed_formal_partition_opened": False,
        "scientific_unit": "patient",
        "provenance_summary": {
            # filled in below by _derive_provenance_summary so the index can
            # never drift from the stamps on the entries themselves
            "new_this_run": [],
            "carried_forward": [],
            "note": (
                "This session produced new evidence for H1 (regular-observation "
                "T1) and H3a (clock identifiability and the fixed event-count "
                "grid) plus the Bridge-E0 patient-level screen. It produced NO "
                "new H2a, H2b or H3b analysis; those entries are re-exported or "
                "transcribed from the previous line and are marked "
                "origin=carried_forward on the entry itself."
            ),
        },
        "multiplicity_summary": {
            "clock_primary_grid": clock["multiplicity"]["primary_grid"],
            "clock_dataset_strata": clock["multiplicity"]["dataset_strata"],
            "fixed_event_count_primary": fixed_count["multiplicity"]["primary"],
            "fixed_event_count_control_and_sensitivity": fixed_count[
                "multiplicity"]["control_and_sensitivity"],
            "clock_separability_strata": clock_separability["multiplicity"],
            "reading_rule": (
                "Every sign-test p in this package is now shipped with a Holm "
                "and a Benjamini-Hochberg companion. Holm answers 'is this one "
                "cell significant on its own' and is very conservative here "
                "because the cells are strongly dependent and a 34-patient sign "
                "test has a coarse p lattice. BH is the appropriate lens for a "
                "grid that is deliberately scanned. Neither replaces the "
                "pre-registered reading: where the spec froze a primary window "
                "set before results were seen, the direction counts over that "
                "frozen set carry the claim."
            ),
        },
        "headline": (
            "H2a has the strongest development-level support. H1 is historical "
            "predictive memory but not yet a raw-informed autonomous physical-time "
            "state. H2b is suggestive but not resolved. H3a supports recent multi-IED "
            "termination/extent memory, with event count rather than physical time "
            "currently sufficient; the generator edge and H3b remain unresolved."
        ),
        "hypotheses": {
            "H1": {
                "status": "PARTIAL_DEVELOPMENT_SUPPORT_PHYSICAL_CLOCK_UNRESOLVED",
                "previous_event_model_open_loop": _prov(
                    core["panels"]["A_open_loop"], origin="carried_forward",
                    source=CORE,
                    note="event-model open-loop panel from the previous line; "
                         "not recomputed in this run"),
                "new_regular_observation_state8": {
                    "provenance": {"origin": "new_this_run",
                                   "source": f"{T1_SPECTRAL} + {T1_RAW}",
                                   "note": "trained and evaluated in this session"},
                    "spectral": _t1_summary(T1_SPECTRAL),
                    "raw_E0": _t1_summary(T1_RAW),
                    "interpretation": (
                        "Filtering has small mixed benefit, but H5/H10/H20 autonomous "
                        "rollout does not beat T0 at patient level. State16 does not "
                        "rescue this. This prototype therefore does not establish a "
                        "raw-informed autonomous state."
                    ),
                },
                "state16_sensitivity": {
                    "provenance": {"origin": "new_this_run",
                                   "source": str(STATE16),
                                   "note": "trained and evaluated in this session"},
                    "n_runs": state16["n_runs"],
                    "seeds": state16["seeds"],
                    "claim_boundary": state16["claim_boundary"],
                },
            },
            "H2a": {
                "status": "SUPPORTED_EXPLORATORY_DEVELOPMENT",
                "core_conclusion": core["core_conclusion"],
                "generator_graph_degree_rewire": _prov(
                    graph["degree_preserving_rewire@generator"],
                    origin="carried_forward", source=CORE,
                    note="re-exported from the previous line; no new H2a "
                         "analysis was run in this session"),
                "decoder_graph_degree_rewire": _prov(
                    graph["degree_preserving_rewire@decoder"],
                    origin="carried_forward", source=CORE,
                    note="re-exported from the previous line; no new H2a "
                         "analysis was run in this session. Sign and Wilcoxon "
                         "p-values are raw; the upstream H2A card carries the "
                         "Holm-corrected primary family for the endpoints it "
                         "pre-registered"),
                "state_swap": _prov(
                    core["panels"]["C_state_swap"], origin="carried_forward",
                    source=CORE, note="figure metadata from the previous line"),
                "same_prefix_continuation": _prov(
                    core["panels"]["D_prefix_branch"], origin="carried_forward",
                    source=CORE, note="figure metadata from the previous line"),
                "capacity_matched_card_status": h2a["status"],
                "claim_boundary": core["claim_boundary"],
                "final_model_requirement": (
                    "The development signal is predictive. The final sequential "
                    "decoder must replace the current tied-group approximation with "
                    "the exact unordered without-replacement subset likelihood before "
                    "formal-partition inference."
                ),
            },
            "H2b": {
                "status": "SUGGESTIVE_NOT_RESOLVED",
                "accepted_primary_readout": {
                    "n_patients": 27,
                    "n_seizures_all_eligible": 361,
                    "decoder_open_loop_median_sd": 0.446,
                    "n_positive_patients": 20,
                    "two_sided_sign_p": 0.019,
                    "high_observation_seizures": 203,
                    "high_observation_result": "weaker, p=0.12",
                    "coverage_gradient": "near zero",
                    "leave_one_patient_range": [0.421, 0.447],
                    "leave_one_seizure_range": [0.396, 0.449],
                },
                "limitations": [
                    "high-observation sensitivity is weaker than the full set",
                    "strict six-dimensional matching is feasible for 0/361 seizures",
                    "subtype contrasts have only 6/4 genuinely within-patient two-group patients",
                    "development partition only",
                ],
                "source": str(H2B_REPORT),
                "provenance": {
                    "origin": "carried_forward",
                    "source": str(H2B_REPORT),
                    "note": (
                        "These values are literals transcribed by hand from the "
                        "cited markdown; the package audit hashes that document "
                        "but cannot verify the transcription. Verified by "
                        "inspection on 2026-08-24 against its table rows "
                        "(+0.446, 361, 20/27, p=0.019, LOPO [0.421,0.447], "
                        "LOSO [0.396,0.449], 203 high-observability seizures). "
                        "No new H2b analysis was run in this session."
                    ),
                    "transcription_checked": True,
                },
            },
            "H3a": {
                "status": "RECENT_EVENT_COUNT_ACCUMULATION_SUPPORTED_GENERATOR_EDGE_UNRESOLVED",
                "clock_identifiability": {
                    "provenance": {
                        "origin": "new_this_run",
                        "source": f"{CLOCK} + {CLOCK_SYNTHETIC} + "
                                  f"{CLOCK_SEPARABILITY} + {FIXED_COUNT}",
                        "note": "all four analyses were run in this session",
                    },
                    "n_physical_runs": clock["n_physical_source_runs"],
                    "n_event_count_runs": clock["n_event_count_source_runs"],
                    "n_current_event_runs": clock["n_current_event_source_runs"],
                    "pairing_audit": clock["pairing_audit"],
                    "cells": clock_cells,
                    "real_timeline_synthetic_recovery": {
                        "median_exposure_correlation": clock_synthetic[
                            "median_physical_event_count_exposure_correlation"
                        ],
                        "aggregate": clock_synthetic["aggregate"],
                        "interpretation": (
                            "On the same 34 real irregular event timelines, the "
                            "contrast reverses in the correct direction when the "
                            "synthetic truth is switched between physical time and "
                            "event count. The human result is therefore not explained "
                            "by a completely blind clock comparator."
                        ),
                    },
                    "clock_separability_sensitivity": {
                        "correlation_threshold": clock_separability[
                            "clock_correlation_threshold"
                        ],
                        "n_more_separable": clock_separability["n_more_separable"],
                        "n_less_separable": clock_separability["n_less_separable"],
                        "human_rows": clock_separability["human_rows"],
                        "claim_boundary": clock_separability["claim_boundary"],
                    },
                    "fixed_event_count_grid": {
                        "n_event_count_runs": fixed_count[
                            "n_event_count_source_runs"
                        ],
                        "n_physical_runs": fixed_count["n_physical_source_runs"],
                        "memories_events": fixed_count["memories_events"],
                        "r2_primary_memories_events_pre_result_freeze": [50, 100, 200],
                        "fast_control_memory_events": 25,
                        "long_memory_sensitivity_events": 400,
                        "cells": fixed_count_cells,
                        "claim_boundary": fixed_count["claim_boundary"],
                    },
                },
                "interpretation": (
                    "Distributed recent-event exposure beats the single-current-IED "
                    "limit for mark, mainly STOP and in some participation-exposure "
                    "cells rank. Actual elapsed-time decay does not consistently beat "
                    "a matched event-count clock, so minute labels are not identified "
                    "physiological time constants. The screen is predictive and does "
                    "not yet implement exposure-to-persistent-generator causality. "
                    "Because the S0 innovation is not conditioned on a learned "
                    "pre-event z, an unobserved persistent state causing serially "
                    "similar IEDs remains a primary alternative explanation."
                ),
            },
            "H3b": {
                "status": "UNRESOLVED_NO_CURRENT_SUPPORT",
                "accepted_previous_case_crossover": {
                    "provenance": {
                        "origin": "carried_forward",
                        "source": str(H2B_REPORT),
                        "note": (
                            "Literals transcribed by hand from the cited "
                            "markdown; verified by inspection on 2026-08-24 "
                            "against its line 78 (21 patients, 327 seizures, "
                            "11/21 positive, sign p=1.0). No new H3b analysis "
                            "was run in this session."
                        ),
                        "transcription_checked": True,
                    },
                    "n_patients": 21,
                    "n_seizures": 327,
                    "n_favourable_patients": 11,
                    "two_sided_sign_p": 1.0,
                },
                "interpretation": (
                    "The previous case-crossover result is patient-level null and was "
                    "not a T2-specific frozen-state probe. It neither supports H3b nor "
                    "rules out the planned T2-specific analysis."
                ),
            },
        },
        "bridge_E0": {
            "provenance": {"origin": "new_this_run", "source": str(BRIDGE),
                           "note": "patient-level screen run in this session"},
            "n_subjects": bridge["n_subjects_complete"],
            "classification_counts": dict(sorted(classifications.items())),
            "claim_boundary": bridge["claim_boundary"],
        },
        "source_files": {
            str(path): _sha256(path)
            for path in (
                CORE, H2A, BRIDGE, T1_SPECTRAL, T1_RAW, STATE16, CLOCK,
                CLOCK_SYNTHETIC,
                CLOCK_SEPARABILITY,
                FIXED_COUNT,
                H2B_REPORT,
            )
        },
    }
    stamps = _derive_provenance_summary(card)
    card["provenance_summary"]["new_this_run"] = sorted(
        p.replace("hypotheses.", "") for p, o in stamps if o == "new_this_run")
    card["provenance_summary"]["carried_forward"] = sorted(
        p.replace("hypotheses.", "") for p, o in stamps if o == "carried_forward")
    unknown = sorted({o for _, o in stamps} - {"new_this_run", "carried_forward"})
    if unknown:
        raise ValueError(f"unrecognised provenance origin(s): {unknown}")

    output = ROOT / "manifests/HYPOTHESIS_EVIDENCE_CARD.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(card, indent=2, sort_keys=True))
    os.replace(temporary, output)
    print(json.dumps({"path": str(output), "hypotheses": list(card["hypotheses"])}))


if __name__ == "__main__":
    main()
