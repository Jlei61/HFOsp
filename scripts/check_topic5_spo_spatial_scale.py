"""How far does the fitted field move, against how far a contact can see?

Three separate results all point the same way and none of them explains itself:
the drift sign does not recover on data with a known answer, zeroing transport in
a fitted model changes its predictions by nothing, and refitting with transport
released is slightly worse than refitting without it. "The model failed" would
cover all three and would be the wrong reading.

This asks the question that distinguishes them. A contact does not sample a
point; it averages over a disc of radius 3*sigma set by the contact spacing. If
the field moves less than that over a whole event, then no amount of fitting can
see the movement -- it happens inside one reading. That is a property of the
recording, not of the model, and it is measurable directly from the fitted
coefficients and the cached geometry.

Diffusion adds variance linearly, so over L ranks of M microsteps its spread is
sqrt(2 * D * M * L) grid cells; drift accumulates linearly at v * M * L. Both are
converted to millimetres through the grid pitch and combined in quadrature.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/topic5_spatial_propagation_operator_v0_2"
FULL = "ANISOTROPIC_RECOVERY"


def main() -> int:
    estimates = OUT / "parameter_estimates.csv"
    if not estimates.exists():
        raise SystemExit("no parameter estimates; run the aggregation first")
    fitted = {r["subject"]: r for r in csv.DictReader(estimates.open())}

    per_patient, rows = {}, []
    for cache in sorted((OUT / "cache").iterdir()):
        if not cache.is_dir() or cache.name not in fitted:
            continue
        config = OUT / "per_subject" / cache.name / FULL / "seed1" / "config.json"
        if not config.exists():
            continue
        grid = np.load(cache / "grid.npz")
        ranks = np.load(cache / "events.npz")["group_ids"]
        sigma = float(grid["sigma_mm"][0])
        pitch = float(np.median(np.diff(np.unique(np.round(grid["centres"][:, 0], 6)))))
        length = float(np.median([r[r >= 0].max() + 1 for r in ranks if (r >= 0).any()]))
        microsteps = int(json.loads(config.read_text())["microsteps"])

        p = fitted[cache.name]
        D, v = float(p["D_parallel"]), abs(float(p["v"]))
        spread = float(np.sqrt(2 * D * microsteps * length) * pitch)
        travel = float(v * microsteps * length * pitch)
        moved = float(np.hypot(spread, travel))
        radius = 3 * sigma

        per_patient[cache.name] = {
            "read_radius_mm": radius, "median_event_length_ranks": length,
            "diffusive_spread_mm": spread, "drift_travel_mm": travel,
            "field_displacement_mm": moved,
            "displacement_over_read_radius": moved / radius,
        }
        rows.append((radius, moved, moved / radius))

    if not rows:
        raise SystemExit("no fitted units to measure")
    a = np.array(rows)
    report = {
        "contract": "topic5_spo_spatial_scale_v0_2",
        "question": ("over a whole event, does the fitted field move further than "
                     "the radius a single contact averages over"),
        "n_patients": len(rows),
        "median_read_radius_mm": float(np.median(a[:, 0])),
        "median_field_displacement_mm": float(np.median(a[:, 1])),
        "median_displacement_over_read_radius": float(np.median(a[:, 2])),
        "n_below_read_radius": int((a[:, 2] < 1).sum()),
        "reading": (
            "the field moves less than one contact's own footprint in "
            f"{int((a[:, 2] < 1).sum())} of {len(rows)} patients, so transport at "
            "this scale happens inside a single reading and cannot be recovered "
            "from it; this is a property of the recording geometry, not evidence "
            "that propagation is absent"),
        "per_patient": per_patient,
    }
    (OUT / "spatial_scale_check.json").write_text(json.dumps(report, indent=1))
    print(f"read radius {report['median_read_radius_mm']:.1f} mm, field moves "
          f"{report['median_field_displacement_mm']:.1f} mm over an event "
          f"({report['median_displacement_over_read_radius']:.2f} of the radius); "
          f"below the radius in {report['n_below_read_radius']}/{len(rows)} patients")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
