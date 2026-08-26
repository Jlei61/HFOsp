#!/usr/bin/env python
"""Non-neural baselines 1-3 of scientific spec section 8.1 for one patient.

CPU only, no GPU: this script never imports CUDA. It reads Worker B's minute
spectra + window index, normalises with the TRAIN statistics, and scores

  1. patient mean  (predict 0 in normalised space)
  2. persistence   (predict the last context minute)
  3. low-capacity spectral feature-AR ridge (alpha chosen by K-fold CV inside train)

on exactly the validation windows the model is scored on.

Example
-------
    LD_LIBRARY_PATH=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib:$LD_LIBRARY_PATH \
    /home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
      scripts/topic5_raw_seeg_state/run_baselines.py --subject epilepsiae_1073
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.topic5_raw_seeg_state import baselines as B, contract  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subject", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--horizons", type=int, nargs="+", default=list(contract.HORIZONS_MIN))
    p.add_argument("--n-folds", type=int, default=B.RIDGE_N_FOLDS)
    return p


def as_single_batch(inputs, horizons):
    """Wrap the assembled arrays as one collated batch for the cheap baselines."""
    return [{
        "target": {int(h): inputs["target"][int(h)] for h in horizons},
        "target_mask": {int(h): inputs["target_mask"][int(h)] for h in horizons},
        "persistence": inputs["persistence"],
        "t_index": inputs["window_id"],
    }]


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    hs = [int(h) for h in args.horizons]
    out_dir = Path(args.out) if args.out else contract.subject_dir(args.subject)
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays = B.load_subject_spectral_arrays(args.subject)
    train = B.assemble_feature_ar_inputs(arrays, "train", hs)
    val = B.assemble_feature_ar_inputs(arrays, "validation", hs)

    mean_train = B.patient_mean_baseline(as_single_batch(train, hs), hs)
    B.assert_patient_mean_is_unit_on_train(mean_train)
    mean_val = B.patient_mean_baseline(as_single_batch(val, hs), hs)
    pers_val = B.persistence_baseline(as_single_batch(val, hs), hs)
    ar = B.spectral_feature_ar_baseline(train, val, hs, n_folds=args.n_folds)

    B.align_windows({
        "patient_mean": {h: mean_val[h]["window_ids"] for h in hs},
        "persistence": {h: pers_val[h]["window_ids"] for h in hs},
    })

    # contract.EVAL_SET_PRIMARY: the windows scoreable at EVERY horizon. Ridge
    # fitting is untouched -- only the scoring is restricted -- so the primary
    # and secondary numbers come from one and the same fitted baseline.
    common_ids = sorted(set.intersection(
        *[set(int(w) for w in mean_val[h]["window_ids"]) for h in hs])) if hs else []
    if common_ids:
        mean_c = B.patient_mean_baseline(as_single_batch(val, hs), hs, restrict_ids=common_ids)
        pers_c = B.persistence_baseline(as_single_batch(val, hs), hs, restrict_ids=common_ids)
        ar_c = B.spectral_feature_ar_baseline(train, val, hs, n_folds=args.n_folds,
                                              restrict_ids=common_ids)
        common_block = {
            str(h): {
                "patient_mean": mean_c[h]["mse"],
                "persistence": pers_c[h]["mse"],
                "feature_ar": ar_c["per_horizon"][h]["mse"],
                "n_elements": mean_c[h]["n_elements"],
                "n_windows": mean_c[h]["n_windows"],
            } for h in hs
        }
    else:
        common_block = {}

    payload = {
        "subject": args.subject,
        "horizons": hs,
        "n_train_windows": int(train["window_id"].size),
        "n_val_windows": int(val["window_id"].size),
        "patient_mean_train_mse": {str(h): mean_train[h]["mse"] for h in hs},
        "per_horizon": {
            str(h): {
                "patient_mean": mean_val[h]["mse"],
                "persistence": pers_val[h]["mse"],
                "feature_ar": ar["per_horizon"][h]["mse"],
                "feature_ar_full_context_only": ar["per_horizon"][h]["mse_full_context_only"],
                "feature_ar_frac_context_imputed": ar["per_horizon"][h]["frac_context_imputed"],
                "n_elements": mean_val[h]["n_elements"],
                "n_windows": mean_val[h]["n_windows"],
                "window_ids": mean_val[h]["window_ids"],
            } for h in hs
        },
        contract.EVAL_SET_PRIMARY: {
            "per_horizon": common_block,
            "n_common_windows": len(common_ids),
            "common_window_ids": common_ids,
            "empty": not bool(common_ids),
            "definition": ("validation windows scoreable at every horizon in "
                           f"{hs}; all arms share this exact index set"),
        },
        "ridge_alpha_grid": list(B.RIDGE_ALPHA_GRID),
        "ridge_alpha_per_freq_bin": {str(h): ar["alpha"][h] for h in hs},
        "code_revision": contract.code_revision(),
        "package_hash": contract.package_hash(contract.r0_1_source_files()),
        "note": ("normalised MSE is the fraction of the patient's own train "
                 "variance left unexplained; patient mean is 1.0 on train by "
                 "construction"),
    }
    contract.atomic_write_json(out_dir / "baseline_metrics.json", payload)
    np.savez_compressed(
        out_dir / "feature_ar_coefficients.npz",
        **{f"coef_h{h}": ar["coef"][h] for h in hs},
        **{f"intercept_h{h}": ar["intercept"][h] for h in hs},
    )
    print(json.dumps({str(h): payload["per_horizon"][str(h)]["feature_ar"] for h in hs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
