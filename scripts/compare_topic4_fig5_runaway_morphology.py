#!/usr/bin/env python3
"""Compare the frozen Qi/gK positive control with the data-driven Joint state."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = (
    ROOT / "results" / "topic4_sef_hfo" / "data_driven_zm_ictal_transition"
    / "runaway_morphology"
)


def _load(path):
    return json.loads(Path(path).read_text())


def _compact(payload):
    morphology = payload["runaway_morphology"]
    recruitment = morphology["full_field_recruitment"]
    contact = morphology["contact_oscillation"]
    population = morphology["population_rate_frequency"]
    return {
        "classification": morphology["classification"],
        "q05_active_neuron_fraction_20ms": recruitment[
            "q05_active_neuron_fraction_20ms"],
        "q05_recruited_spatial_fraction_1mm": recruitment[
            "q05_recruited_spatial_fraction_1mm"],
        "median_contact_high_envelope_duty": contact[
            "median_post_high_envelope_duty"],
        "contact_fraction_high_for_half_post_window": contact[
            "contact_fraction_high_for_half_post_window"],
        "median_contact_rms_ratio": contact[
            "median_band_rms_ratio_post_over_pre"],
        "population_centroid_pre_hz": population["spectral_centroid_pre_hz"],
        "population_centroid_post_hz": population["spectral_centroid_post_hz"],
        "population_centroid_shift_hz": population[
            "spectral_centroid_shift_hz"],
        "population_rate_pre_hz": population["median_rate_pre_hz"],
        "population_rate_post_hz": population["median_rate_post_hz"],
        "population_rate_ratio": population["median_rate_ratio_post_over_pre"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference", default=DEFAULT_DIR / "qigk_e1146_reference.json")
    parser.add_argument(
        "--data-driven", default=DEFAULT_DIR / "joint_seed_1801_post2s.json")
    parser.add_argument(
        "--out", default=DEFAULT_DIR / "runaway_morphology_comparison.json")
    args = parser.parse_args()
    reference = _compact(_load(args.reference))
    data_driven = _compact(_load(args.data_driven))
    if not reference["classification"]["all_checks_pass"]:
        verdict = "MORPHOLOGY_INSTRUMENT_NOT_VALIDATED_BY_QIGK_REFERENCE"
    elif data_driven["classification"]["all_checks_pass"]:
        verdict = "DATA_DRIVEN_SUSTAINED_HIGH_OSCILLATORY_STATE_CANARY_PASS"
    else:
        verdict = "DATA_DRIVEN_SUSTAINED_HIGH_OSCILLATORY_STATE_NOT_REACHED"
    payload = {
        "verdict": verdict,
        "reference": reference,
        "data_driven_joint": data_driven,
        "claim_boundary": (
            "A passing model-state morphology can support a runaway/ictal-like "
            "dynamical-state label only. It does not establish patient seizure "
            "waveform, frequency, duration, or clinical seizure reproduction."
        ),
        "next_action": (
            "Redraw Figure 5A from the data-driven trajectory only if both the "
            "Qi/gK positive control validates the instrument and the Joint state "
            "passes the same frozen morphology contract."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": verdict, "out": str(out)}), flush=True)


if __name__ == "__main__":
    main()
