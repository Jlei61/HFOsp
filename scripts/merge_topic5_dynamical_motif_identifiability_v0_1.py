#!/usr/bin/env python3
"""Merge the three identifiability shards into one map.

The shards partition the 60 synthetic cells; nothing is recomputed here.  The
merged numbers are the sensitivity statement the cohort nulls are read against:
how often the pipeline invents a motif when the generator has none, and how
often it recovers one that is really there.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/topic5_dynamical_motif_rnn_v0_1/toy_identifiability"


def band(frame: pd.DataFrame) -> dict:
    gain = pd.to_numeric(frame.unseen_gain, errors="coerce").dropna()
    return {
        "n_rows": int(len(frame)),
        "n_scored": int(len(gain)),
        "held_out_gain_positive": int((gain > 0).sum()),
        "median_held_out_gain": float(gain.median()) if len(gain) else None,
        "max_held_out_gain": float(gain.max()) if len(gain) else None,
        "p95_abs_held_out_gain": float(np.percentile(gain.abs(), 95)) if len(gain) else None,
    }


def main() -> None:
    grids = sorted(OUT.glob("IDENTIFIABILITY_GRID_?.csv"))
    table = pd.concat([pd.read_csv(p) for p in grids], ignore_index=True)
    table = table.sort_values(["cell_id", "sweep"]).reset_index(drop=True)
    table.to_csv(OUT / "IDENTIFIABILITY_GRID.csv", index=False)

    profiles = sorted(OUT.glob("IDENTIFIABILITY_PROFILE_?.csv"))
    if profiles:
        pd.concat([pd.read_csv(p) for p in profiles], ignore_index=True).to_csv(
            OUT / "IDENTIFIABILITY_PROFILE.csv", index=False)

    scored = table[table.sweep.notna()].copy()
    for column in ("truth_value", "selected_value", "unseen_gain", "calibration_gain"):
        scored[column] = pd.to_numeric(scored[column], errors="coerce")
    failures = sorted(set(table[table.sweep.isna()].cell_id))

    null = scored[scored.truth_value == 0]
    real = scored[(scored.truth_value != 0) & scored.is_target_motif.astype(str).isin(
        ["True", "true", "1", "1.0"])]
    ratio = (real.selected_value / real.truth_value).replace(
        [np.inf, -np.inf], np.nan).dropna()

    summary = {
        "contract": "topic5_dynamical_motif_identifiability_merged_v0_1",
        "n_shards": len(grids),
        "n_cells_planned": int(table.cell_id.nunique()),
        "n_cells_fit_failed": len(failures),
        "cells_fit_failed": failures,
        "n_scored_rows": int(len(scored)),
        "false_positive_band": band(null),
        "recovery_band": band(real),
        "recovery_sign_correct": int((np.sign(real.selected_value)
                                      == np.sign(real.truth_value)).sum()),
        "recovery_selected_exactly_zero": int((real.selected_value == 0).sum()),
        "median_value_ratio": float(ratio.median()) if len(ratio) else None,
        "by_cell_size": {
            size: band(scored[scored.size_label == size])
            for size in sorted(set(scored.size_label.dropna()))
        },
        "by_noise": {
            str(level): band(scored[scored.noise == level])
            for level in sorted(set(pd.to_numeric(scored.noise, errors="coerce").dropna()))
        },
        "real_cohort_beta_gain_range": [-0.0038, 0.0049],
        "note": ("The generator is a contact-level elliptical transport rule while the model "
                 "is a tissue-level diffusion operator, so a failure to recover mixes limited "
                 "precision with form mismatch; this round cannot separate them."),
    }
    (OUT / "IDENTIFIABILITY_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float) + "\n")
    print(json.dumps(summary, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
