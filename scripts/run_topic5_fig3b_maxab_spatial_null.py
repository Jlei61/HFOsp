#!/usr/bin/env python3
"""Fig3-B shared-gradient maxAB similarity — matched spatial nulls per subject.

Replays two within-subject spatial nulls for the maxAB scaffold-similarity panel
ONLY: all-contact value shuffling and the primary within-shaft value shuffling.
Both use the exact successful seizure set and frozen fingerprint-verified
``shared_a/shared_b`` scorers from the observed two-dimensional trajectory;
``own_a/own_b`` are never a fallback. Not tested here (out of scope): onset
increment, signed A/B fixed-direction polarity, or multi-band effects.

Observed, per subject and per common window center t:

    O_s(t) = median over that subject's seizures of  max(|r_A(t)|, |r_B(t)|)

Spatial null (per subject; the FULL readout is replayed, never the finished
maxAB): for each seizure and window, independently permute the per-channel
robust-z energy values either across every contact or within each shaft. Contact
positions, support, shared templates and smoothing remain fixed; A/B, mirror
choice and maxAB are recomputed after every shuffle. Seizures/windows are never
pooled as independent samples. The output reports one-sided pointwise p plus
maxT and cluster correction across the overlapping 66-window time axis.

The support-weighted field is linear in the per-channel values with a
value-independent support gate, so a spatial permutation is a matmul; the
vectorised readout below is verified equal to the exact ``score()`` path to
machine precision (``--verify``). Feature = line-noise-notch-filtered 1-150 Hz
summed spectrogram log power, per-channel baseline robust-z (identical to the
observed Fig3-B readout; NO extra FFT-bin line mask).

Tier: per-subject time-resolved material for Fig3-B, not a formal cohort gate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.compute_topic5_signed_broadband_similarity import (  # noqa: E402
    _compute_shared_values,
    _load_frozen_shared,
    _nan,
    _shared_geometry_metadata,
    _shared_scorer,
)
from scripts.paper_figures.plot_fig3_peri_onset_field_similarity import (  # noqa: E402
    _load_peri_onset,
)
from scripts.plot_topic5_signed_broadband_similarity_timecourse import (  # noqa: E402
    _on_common_grid,
)
from src.propagation_skeleton_geometry import parse_shaft  # noqa: E402
from src.topic5_axis_alignment import (  # noqa: E402
    channel_shuffle,
    within_shaft_shuffle,
)
from src.topic5_template_axis_field import score_field_batch  # noqa: E402

try:
    from src.plot_style import FS_LABEL, FS_TICK, savefig_pub, style_panel  # noqa: E402
    _HAVE_PLOT_STYLE = True
except Exception:  # pragma: no cover - styling is best-effort
    _HAVE_PLOT_STYLE = False

OUT_DIR = _ROOT / "results/paper-ready-figure/fig3_peri_onset_field_similarity/spatial_null"
FIG_DIR = OUT_DIR / "figures"
PAPER_DIR = OUT_DIR.parent
PAPER_INDEX = PAPER_DIR / "fig3_peri_onset_subject_index.csv"
PAPER_INDEX_JSON = PAPER_DIR / "fig3_peri_onset_subject_index.json"
PAPER_MANIFEST_JSON = PAPER_DIR / "fig3_peri_onset_run_manifest.json"

# Locked Fig3-B window contract (docs/figure_style_guide.md 5a).
START_SEC, STOP_SEC = -120.0, 20.0
WINDOW_SEC, STEP_SEC = 10.0, 2.0
BAND = (1.0, 150.0)

COL_OBS = "#A35E48"   # maxAB rust, matches Fig3-B panel a
COL_WS = "#3E6D9C"    # within-shaft null (primary, stronger control)
COL_AC = "#9AA0A6"    # all-contact null (weaker reference)
SIG_ALPHA = 0.05

# Static descriptive text shared by compute + rebuild paths (single source of truth).
_SUMMARY_TEXT = {
    "tier": "single-subject material for Fig3-B; NOT a formal cohort statistic",
    "readout": "shared-gradient maxAB similarity = max(|r_A|, |r_B|); onset-increment / signed A/B / multiband NOT tested",
    "feature": "1-150 Hz summed spectrogram log power (notch-filtered input at 50/100/150/200 Hz; "
               "NO extra FFT-bin line mask), per-channel baseline robust-z",
    "nulls_def": {
        "all_contact": "channel_shuffle — permute per-channel energy values across EVERY matched contact. "
                       "WEAKEST spatial control: does NOT hold shaft-level implantation geometry / "
                       "same-shaft local smoothing fixed.",
        "within_shaft": "within_shaft_shuffle — permute values only WITHIN each electrode shaft (preserves "
                        "which shaft is hot / shaft geometry; the STRONGER, PRIMARY control). Power depends "
                        "on shaft sizes (see shaft_structure); singleton shafts cannot be shuffled, so a "
                        "subject whose similarity is carried entirely by which shaft is hot will not clear it.",
    },
    "null_construction": "within-subject; full readout replayed per shuffle (values -> make_field_record -> "
                         "support-weighted smoothing -> corr to A/B -> max|r|), median over seizures. "
                         "Each seizure x replicate draws one spatial mapping and reuses it across all 66 "
                         "windows, preserving null-trajectory temporal dependence. NOT a shuffle of the "
                         "finished maxAB.",
    "corrections": {
        "pointwise_p": "one-sided per window (1 + #{null >= obs}) / (R + 1); UNCORRECTED for the 66 "
                       "overlapping (10 s / 2 s) windows.",
        "maxt_p": "Nichols-Holmes single-step maxT across windows on standardized z (family-wise per-window "
                  "control, one-sided upper).",
        "cluster": "Maris-Oostenveld cluster permutation (cluster-forming = pointwise p<0.05 on standardized "
                   "z; cluster mass = sum z; null = max cluster mass per permutation). This is the "
                   "paper-facing 'significant time interval' verdict.",
    },
    "scope": "exploratory two-dimensional per-subject material; two WITHIN-subject spatial nulls, NOT a formal cohort spatial gate.",
}


# --------------------------------------------------------------------------
# Vectorised readout engine (exact-equivalent to compute script `score()`).
# --------------------------------------------------------------------------
def _pearson_cols(t: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Pearson r of vector `t` (n,) against each column of `M` (n, R)."""
    tc = t - t.mean()
    tnorm = float(np.sqrt((tc * tc).sum()))
    Mc = M - M.mean(axis=0, keepdims=True)
    Mnorm = np.sqrt((Mc * Mc).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = (tc @ Mc) / (tnorm * Mnorm)
    corr = np.asarray(corr, float)
    corr[(tnorm < 1e-12) | (Mnorm < 1e-12)] = np.nan
    return corr


def build_engine(ds_sid: str, names: list[str]) -> dict:
    """Build the exact frozen shared-field batch scorer in source-name order."""
    record, shared = _load_frozen_shared(ds_sid)
    target_names = [str(name) for name in record["interictal_field"]["contact_order"]]
    if len(names) != len(set(names)):
        raise ValueError(f"{ds_sid}: matched channel names are not unique")
    target_index = {name: index for index, name in enumerate(target_names)}
    missing = [name for name in names if name not in target_index]
    if missing:
        raise ValueError(f"{ds_sid}: names outside frozen contact order: {missing}")
    return {
        "record": record,
        "shared": shared,
        "target_names": target_names,
        "source_indices": np.asarray([target_index[name] for name in names], int),
        "n_source": len(names),
    }


def maxab_batch(engine: dict, vals_matrix: np.ndarray) -> np.ndarray:
    """Shared maxAB for each row of values, including mirror/maxAB selection."""
    values = np.asarray(vals_matrix, float)
    if values.ndim != 2 or values.shape[1] != engine["n_source"]:
        raise ValueError(
            f"activation matrix {values.shape} != (n_draw, {engine['n_source']})"
        )
    aligned = np.full((len(values), len(engine["target_names"])), np.nan, float)
    aligned[:, engine["source_indices"]] = values
    a = score_field_batch(engine["shared"]["shared_a"], aligned)["abs_r"]
    b = score_field_batch(engine["shared"]["shared_b"], aligned)["abs_r"]
    return np.nanmax(np.vstack([a, b]), axis=0)


# --------------------------------------------------------------------------
# Null models + time-dimension multiple-comparison corrections.
# --------------------------------------------------------------------------
# Two within-subject nulls, weakest -> strongest:
#   all_contact  : channel_shuffle       — permute values across EVERY contact.
#   within_shaft : within_shaft_shuffle  — permute values only WITHIN each shaft
#                  (preserves shaft-level implantation geometry / which shaft is hot).
NULL_MODELS = {
    "all_contact": lambda vals, names, rng: channel_shuffle(vals, rng),
    "within_shaft": lambda vals, names, rng: within_shaft_shuffle(vals, names, rng),
}
NULL_LABELS = {"all_contact": "all-contact shuffle", "within_shaft": "within-shaft shuffle"}
ALPHA_CLUSTER = 0.05
SPATIAL_NULL_ALGORITHM_VERSION = "fig3b_shared_gradient_matched_spatial_null_v2"
PERMUTATION_COUPLING_VERSION = (
    "per_seizure_per_replicate_fixed_mapping_across_all_66_windows_v1"
)


def _permutation_indices(
    names: list[str],
    rng: np.random.Generator,
    model: str,
    n_perm: int,
) -> np.ndarray:
    """Draw spatial mappings once for a seizure and reuse them across time.

    Row ``r`` is the contact-index mapping for permutation replicate ``r``.
    The caller must apply this same matrix to every window of the seizure so
    that each null trajectory retains the observed temporal dependence.
    """
    n_contacts = len(names)
    if model == "all_contact":
        return np.argsort(rng.random((n_perm, n_contacts)), axis=1)
    if model != "within_shaft":
        raise KeyError(model)
    indices = np.tile(np.arange(n_contacts, dtype=int), (n_perm, 1))
    shafts: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        shafts.setdefault(parse_shaft(name)[0], []).append(index)
    for members in shafts.values():
        if len(members) < 2:
            continue
        group = np.asarray(members, int)
        order = np.argsort(rng.random((n_perm, len(group))), axis=1)
        indices[:, group] = group[order]
    return indices


def _permutation_batch(
    values: np.ndarray,
    names: list[str],
    rng: np.random.Generator,
    model: str,
    n_perm: int,
) -> np.ndarray:
    """Apply freshly drawn mappings to one vector (single-window helper)."""
    vals = np.asarray(values, float)
    return vals[_permutation_indices(names, rng, model, n_perm)]


def _shaft_structure(names: list[str]) -> dict:
    sizes: dict[str, int] = {}
    for n in names:
        sizes[parse_shaft(n)[0]] = sizes.get(parse_shaft(n)[0], 0) + 1
    vals = sorted(sizes.values(), reverse=True)
    return {"n_contacts": len(names), "n_shafts": len(sizes),
            "shaft_sizes": vals, "n_singleton_shafts": int(sum(1 for v in vals if v == 1))}


def _standardize(obs: np.ndarray, null: np.ndarray):
    """obs:(W,), null:(R,W). Per-window z by the null mean/std (one-sided upper convention)."""
    mu = null.mean(axis=0)
    sd = null.std(axis=0, ddof=1)
    sd = np.where(sd < 1e-12, np.nan, sd)
    return (obs - mu) / sd, (null - mu) / sd


def _pointwise_p(obs: np.ndarray, null: np.ndarray) -> np.ndarray:
    R = null.shape[0]
    return np.array([(1 + int(np.sum(null[:, t] >= obs[t]))) / (R + 1) for t in range(obs.size)])


def _maxt_p(z_obs: np.ndarray, z_null: np.ndarray) -> np.ndarray:
    """Nichols-Holmes single-step maxT, one-sided upper. Corrected p per window.
    Degenerate windows (z_obs NaN — e.g. a within-shaft null with no within-shaft variation)
    cannot be significant -> p=1; NaN null entries are excluded from the per-permutation max."""
    M = np.max(np.where(np.isfinite(z_null), z_null, -np.inf), axis=1)   # (R,)
    R = M.size
    p = (1 + (M[None, :] >= z_obs[:, None]).sum(axis=1)) / (R + 1)
    return np.where(np.isfinite(z_obs), p, 1.0)


def _runs_above(z: np.ndarray, z_thr: float):
    above = z > z_thr
    out, i, W = [], 0, z.size
    while i < W:
        if above[i]:
            j = i
            while j < W and above[j]:
                j += 1
            out.append((i, j, float(np.nansum(z[i:j]))))
            i = j
        else:
            i += 1
    return out


def _cluster_correction(z_obs: np.ndarray, z_null: np.ndarray, alpha: float = SIG_ALPHA):
    """Maris-Oostenveld cluster permutation, one-sided upper. cluster-forming threshold =
    pooled (1-ALPHA_CLUSTER) null-z quantile; mass = sum of standardized z over the run;
    null = max cluster mass per permutation. Returns (sig_mask, clusters, z_thr)."""
    z_thr = (float(np.nanpercentile(z_null, 100 * (1 - ALPHA_CLUSTER)))
             if np.isfinite(z_null).any() else np.inf)   # inf -> no runs on a degenerate null
    R = z_null.shape[0]
    null_max = np.array([max((m for _, _, m in _runs_above(z_null[r], z_thr)), default=0.0)
                         for r in range(R)])
    sig = np.zeros(z_obs.size, bool)
    clusters = []
    for i0, i1, mass in _runs_above(z_obs, z_thr):
        p = (1 + int(np.sum(null_max >= mass))) / (R + 1)
        clusters.append({"start_idx": int(i0), "end_idx": int(i1), "mass": float(mass), "p": float(p)})
        if p < alpha:
            sig[i0:i1] = True
    return sig, clusters, z_thr


# --------------------------------------------------------------------------
# Per-subject observed + null.
# --------------------------------------------------------------------------
def _keep_window(lo: float) -> bool:
    return (_on_common_grid(lo, start_sec=START_SEC, step_sec=STEP_SEC)
            and (lo >= START_SEC - 1e-9) and (lo + WINDOW_SEC <= STOP_SEC + 1e-9))


def _seizure_args(ds_sid: str, seizure_idx: int) -> SimpleNamespace:
    return SimpleNamespace(
        subject=ds_sid, seizure_idx=int(seizure_idx), start_sec=START_SEC, stop_sec=STOP_SEC,
        band_lo=BAND[0], band_hi=BAND[1], spectral_win_sec=1.0, hop_sec=0.5,
        smooth_sec=WINDOW_SEC, frame_step_sec=STEP_SEC, onset_win_sec=10.0, chunk_ch=16)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _trajectory_source_provenance(ds_sid: str):
    """Resolve and fingerprint the exact canonical trajectory consumed by nulls."""
    if not PAPER_INDEX_JSON.exists():
        raise FileNotFoundError(f"canonical trajectory index missing: {PAPER_INDEX_JSON}")
    if not PAPER_MANIFEST_JSON.exists():
        raise FileNotFoundError(
            f"canonical trajectory manifest missing: {PAPER_MANIFEST_JSON}"
        )
    index_payload = json.loads(PAPER_INDEX_JSON.read_text())
    manifest_payload = json.loads(PAPER_MANIFEST_JSON.read_text())
    if (
        index_payload.get("run_complete") is not True
        or index_payload.get("canonical_run") is not True
        or not index_payload.get("run_id")
    ):
        raise RuntimeError("canonical trajectory index is incomplete or noncanonical")
    if (
        manifest_payload.get("run_complete") is not True
        or manifest_payload.get("canonical_run") is not True
        or manifest_payload.get("run_id") != index_payload.get("run_id")
    ):
        raise RuntimeError("canonical trajectory index/manifest run mismatch")
    matches = [
        row
        for row in index_payload.get("subjects", [])
        if row.get("subject") == ds_sid
        and row.get("status") in {"complete_ok", "partial_ok", "severely_partial"}
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"{ds_sid}: expected one generated record in canonical trajectory index, "
            f"found {len(matches)}"
        )
    source_value = matches[0].get("source_csv")
    if not source_value:
        raise RuntimeError(f"{ds_sid}: canonical index has no source_csv")
    source_path = Path(source_value)
    source_csv = source_path if source_path.is_absolute() else _ROOT / source_path
    if not source_csv.exists():
        raise FileNotFoundError(source_csv)
    source = _load_peri_onset(source_csv, ds_sid)
    source_sha256 = _sha256_file(source_csv)
    manifest_matches = [
        item
        for item in manifest_payload.get("artifacts", [])
        if item.get("subject") == ds_sid
        and item.get("role") == "source_per_seizure_csv"
        and item.get("path") == str(source_csv.relative_to(_ROOT))
    ]
    if (
        len(manifest_matches) != 1
        or manifest_matches[0].get("sha256") != source_sha256
        or int(manifest_matches[0].get("size_bytes", -1)) != source_csv.stat().st_size
    ):
        raise RuntimeError(f"{ds_sid}: source CSV does not match canonical trajectory manifest")
    seizure_ids = sorted(int(value) for value in source["seizure_idx"].unique())
    grid_columns = ["window_start_sec", "window_end_sec", "window_center_sec"]
    grid = (
        source[grid_columns]
        .drop_duplicates()
        .sort_values("window_start_sec")
        .astype(float)
        .to_dict(orient="records")
    )
    if len(grid) != 66:
        raise RuntimeError(f"{ds_sid}: expected one canonical 66-window grid")
    provenance = {
        "source_per_seizure_csv": str(source_csv.relative_to(_ROOT)),
        "source_per_seizure_csv_sha256": source_sha256,
        "trajectory_run_id": str(index_payload["run_id"]),
        "trajectory_manifest_sha256": _sha256_file(PAPER_MANIFEST_JSON),
        "source_seizure_ids": seizure_ids,
        "source_seizure_ids_sha256": _sha256_json(seizure_ids),
        "window_grid_sha256": _sha256_json(grid),
        "spatial_null_algorithm_version": SPATIAL_NULL_ALGORITHM_VERSION,
        "permutation_coupling_version": PERMUTATION_COUPLING_VERSION,
    }
    return source_csv, source, provenance


def compute_subject(ds_sid: str, n_perm: int, seed: int, verify: bool):
    """Return shared-matched observed/null trajectories for the exact figure seizures."""
    source_csv, source, trajectory_provenance = _trajectory_source_provenance(ds_sid)
    idxs = sorted(int(value) for value in source["seizure_idx"].unique())
    expected_obs = {
        (int(row.seizure_idx), float(row.window_start_sec)): float(row.maxAB_abs_corr)
        for row in source.itertuples(index=False)
    }
    rng = np.random.default_rng(seed)
    obs_by_win: dict[float, list[float]] = {}
    null_by_win = {key: {} for key in NULL_MODELS}   # model -> {lo -> [ (R,) per seizure ]}
    drops = []
    max_fid_err = 0.0
    max_source_err = 0.0
    n_seizures = 0
    shaft = None
    provenance = None

    for seizure_idx in idxs:
        try:
            _ds, _i, sw, offset, bl, field_record, names, starts, window_vals, _onset = \
                _compute_shared_values(_seizure_args(ds_sid, seizure_idx))
        except Exception as exc:  # fail-closed per seizure
            drops.append({"seizure_idx": int(seizure_idx), "reason": f"{type(exc).__name__}: {exc}"})
            continue
        engine = build_engine(ds_sid, names)
        n_seizures += 1
        current_provenance = {
            "field_contract": field_record["contract"],
            "field_plane": "shared",
            "field_scorers": ["shared_a", "shared_b"],
            "own_field_fallback": False,
            "field_fingerprint_sha256": field_record["interictal_field"]["fingerprint_sha256"],
            "axis_definition": field_record["axis_definition"],
            "axis_direction_convention": field_record["axis_direction_convention"],
            **_shared_geometry_metadata(field_record),
            **trajectory_provenance,
        }
        if provenance is None:
            provenance = current_provenance
        elif current_provenance != provenance:
            raise RuntimeError(f"{ds_sid}: mixed provenance across seizures")
        current_shaft = _shaft_structure(names)
        if shaft is None:
            shaft = current_shaft
        elif current_shaft != shaft:
            raise RuntimeError(f"{ds_sid}: mixed shaft structure across seizures")
        kept_windows = [
            (float(lo), np.asarray(vals, float))
            for lo, vals in zip(starts, window_vals)
            if _keep_window(float(lo))
        ]
        if len(kept_windows) != 66:
            raise RuntimeError(
                f"{ds_sid} seizure {seizure_idx}: expected 66 shared windows, "
                f"found {len(kept_windows)}"
            )
        permutation_indices = {
            key: _permutation_indices(names, rng, key, n_perm)
            for key in NULL_MODELS
        }
        for lo, vals in kept_windows:
            vals = np.asarray(vals, float)
            obs = float(maxab_batch(engine, vals[None, :])[0])
            if verify:
                max_fid_err = max(
                    max_fid_err, abs(obs - _exact_maxab(names, ds_sid, vals))
                )
            for key in NULL_MODELS:
                batch = vals[permutation_indices[key]]
                null_by_win[key].setdefault(float(lo), []).append(
                    maxab_batch(engine, batch)
                )
            source_key = (int(seizure_idx), float(lo))
            if source_key not in expected_obs:
                raise RuntimeError(f"{ds_sid}: source CSV missing {source_key}")
            source_err = abs(obs - expected_obs[source_key])
            max_source_err = max(max_source_err, source_err)
            if source_err > 1e-9:
                raise RuntimeError(
                    f"{ds_sid}: shared observed/source mismatch at {source_key}: {source_err}"
                )
            obs_by_win.setdefault(float(lo), []).append(obs)

    if drops or n_seizures != len(idxs):
        raise RuntimeError(
            f"{ds_sid}: source-matched seizure recompute incomplete "
            f"({n_seizures}/{len(idxs)}, drops={drops})"
        )

    los = sorted(obs_by_win)
    centers = np.array([lo + WINDOW_SEC / 2.0 for lo in los])
    n_sz = np.array([len(obs_by_win[lo]) for lo in los])
    obs = np.array([float(np.nanmedian(obs_by_win[lo])) for lo in los])
    obs_q25 = np.array([float(np.nanpercentile(obs_by_win[lo], 25)) for lo in los])
    obs_q75 = np.array([float(np.nanpercentile(obs_by_win[lo], 75)) for lo in los])
    # per null model: (R, W) = median over seizures per realization
    null_mats = {}
    for key in NULL_MODELS:
        M = np.empty((n_perm, len(los)))
        for j, lo in enumerate(los):
            M[:, j] = np.nanmedian(np.vstack(null_by_win[key][lo]), axis=0)
        null_mats[key] = M

    meta = {"n_perm": int(n_perm), "seed": int(seed), "n_seizures": int(n_seizures),
            "n_seizure_drops": len(drops), "seizure_drops": drops, "shaft_structure": shaft,
            "fidelity_max_abs_err": float(max_fid_err) if verify else None,
            "source_observed_max_abs_err": float(max_source_err),
            "provenance": provenance}
    rows, summary = _finalize(ds_sid, centers, n_sz, obs, obs_q25, obs_q75, null_mats, meta)
    null_npz = {"window_center_sec": centers, "n_seizures": n_sz,
                "obs": obs, "obs_q25": obs_q25, "obs_q75": obs_q75,
                "spatial_null_algorithm_version": np.asarray(
                    SPATIAL_NULL_ALGORITHM_VERSION
                ),
                "permutation_coupling_version": np.asarray(
                    PERMUTATION_COUPLING_VERSION
                ),
                "source_per_seizure_csv_sha256": np.asarray(
                    trajectory_provenance["source_per_seizure_csv_sha256"]
                ),
                "window_grid_sha256": np.asarray(
                    trajectory_provenance["window_grid_sha256"]
                ),
                **{f"{key}_null": null_mats[key] for key in NULL_MODELS}}
    return rows, summary, drops, null_npz


def _finalize(ds_sid, centers, n_sz, obs, obs_q25, obs_q75, null_mats, meta):
    """Given per-window observed + per-null (R,W) matrices, apply pointwise / maxT / cluster
    corrections and build (per_window rows, summary). Shared by compute and rebuild paths."""
    W = obs.size
    per_null = {}
    for key, M in null_mats.items():
        z_obs, z_null = _standardize(obs, M)
        cl_sig, clusters, z_thr = _cluster_correction(z_obs, z_null)
        per_null[key] = {"M": M, "pw": _pointwise_p(obs, M), "mt": _maxt_p(z_obs, z_null),
                         "cl_sig": cl_sig, "clusters": clusters, "z_thr": z_thr}

    rows = []
    for j in range(W):
        row = {"window_start_sec": float(centers[j] - WINDOW_SEC / 2.0),
               "window_end_sec": float(centers[j] + WINDOW_SEC / 2.0),
               "window_center_sec": float(centers[j]), "n_seizures": int(n_sz[j]),
               "obs_median_maxAB": float(obs[j]), "obs_q25": float(obs_q25[j]),
               "obs_q75": float(obs_q75[j])}
        for key, d in per_null.items():
            col = d["M"][:, j]
            row[f"{key}_null_median"] = float(np.nanmedian(col))
            row[f"{key}_null_p2.5"] = float(np.nanpercentile(col, 2.5))
            row[f"{key}_null_p97.5"] = float(np.nanpercentile(col, 97.5))
            row[f"{key}_pointwise_p"] = float(d["pw"][j])
            row[f"{key}_maxt_p"] = float(d["mt"][j])
            row[f"{key}_cluster_sig"] = bool(d["cl_sig"][j])
        rows.append(row)

    def _counts(key):
        d = per_null[key]
        sig_cl = [{"start_sec": float(centers[c["start_idx"]]),
                   "end_sec": float(centers[c["end_idx"] - 1]),
                   "n_windows": int(c["end_idx"] - c["start_idx"]), "p": c["p"]}
                  for c in d["clusters"] if c["p"] < SIG_ALPHA]
        return {"n_pointwise_p05": int(np.sum(d["pw"] < SIG_ALPHA)),
                "n_maxt_p05": int(np.sum(d["mt"] < SIG_ALPHA)),
                "n_cluster_sig_windows": int(np.sum(d["cl_sig"])),
                "n_clusters": len(d["clusters"]), "significant_clusters": sig_cl}

    summary = {
        "subject": ds_sid, **_SUMMARY_TEXT,
        "time_range_sec": [START_SEC, STOP_SEC], "window_sec": WINDOW_SEC, "step_sec": STEP_SEC,
        "n_perm": meta["n_perm"], "seed": meta["seed"], "n_seizures": meta["n_seizures"],
        "n_seizure_drops": meta["n_seizure_drops"], "seizure_drops": meta["seizure_drops"],
        "shaft_structure": meta["shaft_structure"], "n_windows": W,
        "obs_median_of_window_medians": float(np.nanmedian(obs)),
        "primary_null": "within_shaft",
        "nulls": {key: _counts(key) for key in per_null},
        "fidelity_max_abs_err": meta.get("fidelity_max_abs_err"),
        "source_observed_max_abs_err": meta.get("source_observed_max_abs_err"),
        **(meta.get("provenance") or {}),
    }
    return rows, summary


def _exact_maxab(names, ds_sid, vals) -> float:
    score = _shared_scorer(ds_sid, names)
    per, _best = score(np.asarray(vals, float))
    return max(_nan(per.get("A", {}).get("abs_corr")), _nan(per.get("B", {}).get("abs_corr")))


# --------------------------------------------------------------------------
# Figure + outputs.
# --------------------------------------------------------------------------
def _shade_runs(ax, x, mask, **kw):
    """axvspan over each contiguous True run of `mask` (window-centered)."""
    i, W = 0, len(mask)
    while i < W:
        if mask[i]:
            j = i
            while j < W and mask[j]:
                j += 1
            ax.axvspan(x[i] - STEP_SEC / 2, x[j - 1] + STEP_SEC / 2, **kw)
            i = j
        else:
            i += 1


def _plot(ds_sid: str, rows: list[dict], summary: dict, out_png: Path, out_pdf: Path) -> None:
    x = np.array([r["window_center_sec"] for r in rows], float)
    obs = np.array([r["obs_median_maxAB"] for r in rows], float)
    oq25 = np.array([r["obs_q25"] for r in rows], float)
    oq75 = np.array([r["obs_q75"] for r in rows], float)
    ws_med = np.array([r["within_shaft_null_median"] for r in rows], float)
    ws_lo = np.array([r["within_shaft_null_p2.5"] for r in rows], float)
    ws_hi = np.array([r["within_shaft_null_p97.5"] for r in rows], float)
    ac_med = np.array([r["all_contact_null_median"] for r in rows], float)
    ws_clu = np.array([r["within_shaft_cluster_sig"] for r in rows], bool)
    ws_maxt = np.array([r["within_shaft_maxt_p"] for r in rows], float) < SIG_ALPHA
    label = ds_sid.replace("epilepsiae_", "E").replace("yuquan_", "Y-")
    ws = summary["nulls"]["within_shaft"]
    ac = summary["nulls"]["all_contact"]

    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    # primary = within-shaft null band (stronger control)
    ax.fill_between(x, ws_lo, ws_hi, color=COL_WS, alpha=0.18, lw=0, zorder=1, label="within-shaft null 95%")
    ax.plot(x, ws_med, color=COL_WS, lw=1.3, ls="--", zorder=2, label="within-shaft null median")
    # weaker reference = all-contact null median only
    ax.plot(x, ac_med, color=COL_AC, lw=1.1, ls=":", zorder=2, label="all-contact null median")
    # observed
    ax.fill_between(x, oq25, oq75, color=COL_OBS, alpha=0.13, lw=0, zorder=3, label="observed IQR")
    ax.plot(x, obs, color=COL_OBS, lw=2.2, zorder=5, label="observed median")
    # cluster-corrected significant spans vs within-shaft (the "significant interval")
    if ws_clu.any():
        _shade_runs(ax, x, ws_clu, color=COL_OBS, alpha=0.11, lw=0, zorder=0)
        ax.axvspan(np.nan, np.nan, color=COL_OBS, alpha=0.11, lw=0, label="within-shaft cluster p<0.05")
    # maxT-significant windows vs within-shaft
    if ws_maxt.any():
        ax.scatter(x[ws_maxt], np.full(ws_maxt.sum(), 0.965), marker="v", s=16,
                   color=COL_WS, zorder=6, label="within-shaft maxT p<0.05")
    ax.axvline(0, color="0.30", ls="--", lw=0.9, zorder=0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_xlabel("window center from onset (s)", fontsize=FS_LABEL if _HAVE_PLOT_STYLE else 11)
    ax.set_ylabel("maxAB field similarity |r|", fontsize=FS_LABEL if _HAVE_PLOT_STYLE else 11)
    ax.set_title("shared-gradient maxAB vs spatial-shuffle nulls",
                 fontsize=(FS_LABEL if _HAVE_PLOT_STYLE else 11), pad=16)
    # subtitle: within-shaft (primary) corrected counts + all-contact pointwise for context
    ax.text(0.5, 1.015,
            f"{label} · {summary['n_seizures']} sz · R={summary['n_perm']} · within-shaft: "
            f"cluster {ws['n_cluster_sig_windows']} / maxT {ws['n_maxt_p05']} / pointwise "
            f"{ws['n_pointwise_p05']} win · all-contact pointwise {ac['n_pointwise_p05']}",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=7.3, color="0.35")
    ax.legend(frameon=False, loc="lower left", fontsize=6.8, handlelength=1.5, ncol=1,
              labelspacing=0.3)
    if _HAVE_PLOT_STYLE:
        ax.tick_params(labelsize=FS_TICK - 2)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.16, top=0.85)
    for out in (out_png, out_pdf):
        if _HAVE_PLOT_STYLE:
            savefig_pub(fig, out, dpi=300)
        else:
            fig.savefig(out, dpi=300)
    plt.close(fig)


def _write_readme() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (FIG_DIR / "README.md").write_text(
        "# Fig3-B shared-gradient maxAB — 匹配空间置换 null（两档）+ 时间维校正\n\n"
        "### `<subject>_maxab_spatial_null.png / .pdf`\n\n"
        "仅对通过二维几何门的 shared-gradient Fig3-B 病例叠加**两个被试内空间置换 null + 时间维多重比较校正**。"
        "两个 null 与主图保持同一批成功 seizure、同一 66-window grid、同一冻结 `shared_a/shared_b`、同一 fingerprint"
        " 和 maxAB 选择；每个 seizure×replicate 只抽一次空间映射，并将同一映射贯穿全部 66 个窗口，"
        "以保留 null trajectory 的时间依赖。每次 shuffle 都完整重跑 shared scorer，再对 seizure 取中位。两档强度：\n"
        "- **all-contact**（弱，灰点线=null 中位）：值在**所有触点**间打乱。\n"
        "- **within-shaft**（强，主对比，蓝带）：值只在**每根杆(shaft)内**打乱，保留'哪根杆热'的植入几何。\n\n"
        "图元：粗 rust=观测中位、浅 rust 带=观测 IQR；蓝虚线+蓝带=within-shaft null 中位+95%；灰点线=all-contact "
        "null 中位；浅 rust 竖带=within-shaft **cluster 校正显著区间**；蓝三角=within-shaft **maxT 校正显著窗**；"
        "0 s 虚线=onset。副标题给 within-shaft 的 cluster/maxT/pointwise 显著窗数 + all-contact pointwise。\n\n"
        "**三档显著性（都在 stats CSV，两个 null 各一套）**：pointwise（逐窗，未校正）< maxT（逐窗 FWER）< "
        "cluster（时间维、对持续抬升敏感，= paper-facing '显著区间'）。\n\n"
        "**关注点**：观测中位数是否**离开蓝色 within-shaft null 带**并形成 cluster 显著区间。⚠️within-shaft null "
        "的分辨力取决于每根杆的触点数（见 summary.shaft_structure）；单触点杆无法打乱，若相似度完全由'哪根杆热'"
        "解释，within-shaft null 就贴着观测、几乎无显著窗——这是诚实的强 null 结果。只检验 maxAB scaffold，"
        "不做 onset increment / signed A/B / 多频带。旧 own-plane null 已移入 `legacy_own_plane_spatial_null/`，"
        "不得与本目录结果混用。**单被试二维素材，非 formal cohort spatial gate。**\n",
        encoding="utf-8")


def _write_outputs(ds_sid, rows, summary, null_npz=None) -> dict:
    stats_csv = OUT_DIR / f"{ds_sid}_maxab_spatial_null_stats.csv"
    npz_fp = OUT_DIR / f"{ds_sid}_maxab_spatial_null_matrices.npz"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_png = FIG_DIR / f"{ds_sid}_maxab_spatial_null.png"
    out_pdf = FIG_DIR / f"{ds_sid}_maxab_spatial_null.pdf"
    summary_fp = OUT_DIR / f"{ds_sid}_maxab_spatial_null_summary.json"
    summary["outputs"] = {"figure_png": str(out_png.relative_to(_ROOT)),
                          "figure_pdf": str(out_pdf.relative_to(_ROOT)),
                          "stats_csv": str(stats_csv.relative_to(_ROOT)),
                          "null_matrices_npz": str(npz_fp.relative_to(_ROOT))}
    temp_paths = {}
    for key, path in (
        ("csv", stats_csv), ("png", out_png), ("pdf", out_pdf),
        ("summary", summary_fp),
    ):
        handle = tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.stem}.", suffix=path.suffix, delete=False
        )
        handle.close()
        temp_paths[key] = Path(handle.name)
    if null_npz is not None:
        handle = tempfile.NamedTemporaryFile(
            dir=npz_fp.parent, prefix=f".{npz_fp.stem}.", suffix=".npz", delete=False
        )
        handle.close()
        temp_paths["npz"] = Path(handle.name)
    try:
        with temp_paths["csv"].open("w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=list(rows[0].keys()),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        if null_npz is not None:
            np.savez_compressed(temp_paths["npz"], **null_npz)
        _plot(ds_sid, rows, summary, temp_paths["png"], temp_paths["pdf"])
        temp_paths["summary"].write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
        )
        os.replace(temp_paths["csv"], stats_csv)
        if null_npz is not None:
            os.replace(temp_paths["npz"], npz_fp)
        os.replace(temp_paths["png"], out_png)
        os.replace(temp_paths["pdf"], out_pdf)
        os.replace(temp_paths["summary"], summary_fp)
    finally:
        for path in temp_paths.values():
            path.unlink(missing_ok=True)
    _write_readme()
    return summary


def run_subject(ds_sid: str, n_perm: int, seed: int, verify: bool) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, summary, _drops, null_npz = compute_subject(ds_sid, n_perm, seed, verify)
    return _write_outputs(ds_sid, rows, summary, null_npz)


_MATCHED_SOURCE_PROVENANCE_KEYS = (
    "source_per_seizure_csv",
    "source_per_seizure_csv_sha256",
    "trajectory_run_id",
    "trajectory_manifest_sha256",
    "source_seizure_ids",
    "source_seizure_ids_sha256",
    "window_grid_sha256",
    "spatial_null_algorithm_version",
    "permutation_coupling_version",
)


def _validate_cached_summary(
    ds_sid: str,
    summary: dict,
    *,
    n_perm: int | None = None,
    seed: int | None = None,
) -> dict:
    """Fail closed unless a cached null matches the current trajectory exactly."""
    record, _shared = _load_frozen_shared(ds_sid)
    _source_csv, source, expected = _trajectory_source_provenance(ds_sid)
    checks = {
        "subject": ds_sid,
        "field_plane": "shared",
        "own_field_fallback": False,
        "geometry_2d_supported": True,
        "field_fingerprint_sha256": record["interictal_field"]["fingerprint_sha256"],
        **expected,
    }
    for key, value in checks.items():
        if summary.get(key) != value:
            raise ValueError(
                f"{ds_sid}: cached null provenance mismatch for {key}: "
                f"{summary.get(key)!r} != {value!r}"
            )
    if int(summary.get("n_seizures", -1)) != int(source["seizure_idx"].nunique()):
        raise ValueError(f"{ds_sid}: cached null seizure count is stale")
    if int(summary.get("n_windows", -1)) != 66:
        raise ValueError(f"{ds_sid}: cached null window grid is incomplete")
    if n_perm is not None and int(summary.get("n_perm", -1)) != int(n_perm):
        raise ValueError(f"{ds_sid}: cached null n_perm is stale")
    if seed is not None and int(summary.get("seed", -1)) != int(seed):
        raise ValueError(f"{ds_sid}: cached null seed is stale")
    outputs = summary.get("outputs") or {}
    for role in ("figure_png", "figure_pdf", "stats_csv", "null_matrices_npz"):
        relative_path = outputs.get(role)
        if not relative_path or not (_ROOT / relative_path).exists():
            raise FileNotFoundError(f"{ds_sid}: cached output missing for {role}")
    return expected


def rebuild_subject(ds_sid: str) -> dict:
    """Recompute corrections + re-render from the cached null-matrix .npz (obs + both null
    matrices), WITHOUT reloading raw data. Lets correction params / labels change cheaply."""
    npz_fp = OUT_DIR / f"{ds_sid}_maxab_spatial_null_matrices.npz"
    summ_fp = OUT_DIR / f"{ds_sid}_maxab_spatial_null_summary.json"
    if not (npz_fp.exists() and summ_fp.exists()):
        raise FileNotFoundError(f"{ds_sid}: cached .npz/summary missing; run once without --rebuild-from-stats")
    old = json.loads(summ_fp.read_text())
    expected = _validate_cached_summary(ds_sid, old)
    z = np.load(npz_fp, allow_pickle=False)
    for key in (
        "spatial_null_algorithm_version",
        "permutation_coupling_version",
        "source_per_seizure_csv_sha256",
        "window_grid_sha256",
    ):
        if key not in z.files or str(z[key].item()) != str(expected[key]):
            raise ValueError(f"{ds_sid}: cached null matrix provenance mismatch for {key}")
    null_mats = {key: z[f"{key}_null"] for key in NULL_MODELS}
    meta = {"n_perm": int(old["n_perm"]), "seed": int(old["seed"]), "n_seizures": int(old["n_seizures"]),
            "n_seizure_drops": int(old.get("n_seizure_drops", 0)), "seizure_drops": old.get("seizure_drops", []),
            "shaft_structure": old.get("shaft_structure"),
            "fidelity_max_abs_err": old.get("fidelity_max_abs_err"),
            "source_observed_max_abs_err": old.get("source_observed_max_abs_err"),
            "provenance": {
                key: old[key] for key in (
                    "field_contract", "field_plane", "field_scorers",
                    "own_field_fallback", "field_fingerprint_sha256",
                    "axis_definition", "axis_direction_convention",
                    "geometry_2d_supported", "geometry_quality_tier",
                    "minimum_axis_n_shafts", "minimum_axis_effective_rank",
                    *_MATCHED_SOURCE_PROVENANCE_KEYS,
                )
            }}
    rows, summary = _finalize(ds_sid, z["window_center_sec"], z["n_seizures"], z["obs"],
                              z["obs_q25"], z["obs_q75"], null_mats, meta)
    return _write_outputs(ds_sid, rows, summary, null_npz=None)


INDEX_CSV = OUT_DIR / "fig3b_maxab_spatial_null_index.csv"
INDEX_JSON = OUT_DIR / "fig3b_maxab_spatial_null_index.json"
MANIFEST_JSON = OUT_DIR / "fig3b_maxab_spatial_null_manifest.json"
INDEX_COLS = ["subject", "status", "drop_reason", "n_seizures", "n_windows",
              "n_shafts", "n_singleton_shafts",
              "field_plane", "field_fingerprint_sha256", "geometry_2d_supported",
              "ac_pointwise_sig", "ws_pointwise_sig", "ws_maxt_sig",
              "ws_cluster_sig_windows", "ws_n_sig_clusters",
              "obs_median_of_window_medians", "figure_png", "stats_csv"]


def _ok_subjects() -> list[str]:
    """Subjects that produced a current two-dimensional shared Fig3-B figure."""
    if not PAPER_INDEX.exists():
        raise FileNotFoundError(f"paper index missing: {PAPER_INDEX}")
    with PAPER_INDEX.open() as fh:
        return [
            row["subject"]
            for row in csv.DictReader(fh)
            if row["status"] in {"complete_ok", "partial_ok", "severely_partial"}
        ]


def _record_from_summary(summ: dict) -> dict:
    ws = summ["nulls"]["within_shaft"]
    ac = summ["nulls"]["all_contact"]
    sh = summ.get("shaft_structure") or {}
    return {"subject": summ["subject"], "status": "ok", "drop_reason": "",
            "n_seizures": summ["n_seizures"], "n_windows": summ["n_windows"],
            "n_shafts": sh.get("n_shafts", ""), "n_singleton_shafts": sh.get("n_singleton_shafts", ""),
            "field_plane": summ["field_plane"],
            "field_fingerprint_sha256": summ["field_fingerprint_sha256"],
            "geometry_2d_supported": summ["geometry_2d_supported"],
            "ac_pointwise_sig": ac["n_pointwise_p05"],
            "ws_pointwise_sig": ws["n_pointwise_p05"],
            "ws_maxt_sig": ws["n_maxt_p05"],
            "ws_cluster_sig_windows": ws["n_cluster_sig_windows"],
            "ws_n_sig_clusters": len(ws["significant_clusters"]),
            "obs_median_of_window_medians": summ["obs_median_of_window_medians"],
            "figure_png": summ["outputs"]["figure_png"],
            "stats_csv": summ["outputs"]["stats_csv"]}


def _sig_str(rec: dict) -> str:
    return (f"ws-cluster {rec['ws_cluster_sig_windows']}/maxT {rec['ws_maxt_sig']} · "
            f"ac-pw {rec['ac_pointwise_sig']} /{rec['n_windows']}")


def _write_cohort_index(records: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_tmp_handle = tempfile.NamedTemporaryFile(
        dir=OUT_DIR, prefix=".fig3b_null_index.", suffix=".csv", delete=False
    )
    csv_tmp_handle.close()
    json_tmp_handle = tempfile.NamedTemporaryFile(
        dir=OUT_DIR, prefix=".fig3b_null_index.", suffix=".json", delete=False
    )
    json_tmp_handle.close()
    manifest_tmp_handle = tempfile.NamedTemporaryFile(
        dir=OUT_DIR, prefix=".fig3b_null_manifest.", suffix=".json", delete=False
    )
    manifest_tmp_handle.close()
    csv_tmp = Path(csv_tmp_handle.name)
    json_tmp = Path(json_tmp_handle.name)
    manifest_tmp = Path(manifest_tmp_handle.name)
    n_ok = sum(1 for r in records if r["status"] == "ok")
    payload = {
        "figure": "Fig3-B shared-gradient maxAB — matched spatial-shuffle nulls (all-contact + within-shaft) "
                  "with maxT / cluster time correction",
        "generated_by": "scripts/run_topic5_fig3b_maxab_spatial_null.py",
        "run_complete": True,
        "field_plane": "shared",
        "own_field_fallback": False,
        "geometry_2d_required": True,
        "tier": "two-dimensional single-subject material for Fig3-B; NOT a formal cohort statistic",
        "primary_null": "within_shaft",
        "spatial_null_algorithm_version": SPATIAL_NULL_ALGORITHM_VERSION,
        "permutation_coupling_version": PERMUTATION_COUPLING_VERSION,
        "test": "two within-subject spatial nulls (all_contact channel-shuffle = weak; within_shaft = strong, "
                "primary); one mapping per seizure x replicate is fixed across all 66 windows; full readout "
                "replayed, median over seizures; per window pointwise / maxT / cluster one-sided p; maxAB "
                "scaffold only. Index columns: ws_* = within-shaft (primary), ac_* = all-contact.",
        "caveats": [
            "within-shaft (primary) is the stronger control but its power depends on shaft sizes "
            "(n_shafts / n_singleton_shafts); a subject whose similarity is carried by which-shaft-is-hot will "
            "not clear it — that is honest, not a failure. Neither null is a formal cohort spatial gate.",
            "ws_cluster_sig_windows = Maris-Oostenveld cluster-corrected (paper-facing 'significant interval'); "
            "ws_maxt_sig = Nichols-Holmes maxT; ws_pointwise_sig / ac_pointwise_sig = UNCORRECTED pointwise counts.",
            "per-subject material heterogeneity; NOT a cohort claim.",
        ],
        "n_subjects": len(records), "n_ok": n_ok, "n_drop": len(records) - n_ok,
        "subjects": records,
    }
    artifacts = []
    for record in records:
        if record["status"] != "ok":
            continue
        summary_path = OUT_DIR / f"{record['subject']}_maxab_spatial_null_summary.json"
        summary = json.loads(summary_path.read_text())
        _validate_cached_summary(record["subject"], summary)
        output_paths = [("summary_json", str(summary_path.relative_to(_ROOT)))]
        output_paths.extend(summary["outputs"].items())
        for role, relative_path in output_paths:
            artifact = _ROOT / relative_path
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            artifacts.append({
                "subject": record["subject"],
                "role": role,
                "path": relative_path,
                "size_bytes": artifact.stat().st_size,
                "sha256": digest,
            })
    manifest = {
        "contract": "fig3b_shared_gradient_matched_spatial_null_v2",
        "run_complete": True,
        "n_subjects": len(records),
        "n_ok": n_ok,
        "field_plane": "shared",
        "own_field_fallback": False,
        "geometry_2d_required": True,
        "spatial_null_algorithm_version": SPATIAL_NULL_ALGORITHM_VERSION,
        "permutation_coupling_version": PERMUTATION_COUPLING_VERSION,
        "trajectory_run_ids": sorted({
            json.loads(
                (OUT_DIR / f"{record['subject']}_maxab_spatial_null_summary.json").read_text()
            )["trajectory_run_id"]
            for record in records if record["status"] == "ok"
        }),
        "n_perm": sorted({
            json.loads(
                (OUT_DIR / f"{record['subject']}_maxab_spatial_null_summary.json").read_text()
            )["n_perm"]
            for record in records if record["status"] == "ok"
        }),
        "artifacts": artifacts,
    }
    try:
        with csv_tmp.open("w", newline="") as fh:
            w = csv.DictWriter(
                fh,
                fieldnames=INDEX_COLS,
                extrasaction="ignore",
                lineterminator="\n",
            )
            w.writeheader()
            for rec in records:
                w.writerow({c: rec.get(c, "") for c in INDEX_COLS})
        json_tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        manifest_tmp.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
        )
        os.replace(csv_tmp, INDEX_CSV)
        os.replace(json_tmp, INDEX_JSON)
        os.replace(manifest_tmp, MANIFEST_JSON)
    finally:
        csv_tmp.unlink(missing_ok=True)
        json_tmp.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", default="epilepsiae_1146")
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="batch: explicit subject list")
    ap.add_argument("--all-ok", action="store_true",
                    help="batch: every subject with an observed Fig3-B figure (paper index status==ok)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="batch: reuse a subject's summary JSON if present (resume without recompute)")
    ap.add_argument("--rebuild-from-stats", action="store_true",
                    help="batch: re-render figures + refresh summary text from cached stats CSVs (no recompute)")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify", action="store_true",
                    help="assert the vectorised readout matches exact score() to 1e-9")
    args = ap.parse_args()

    if args.all_ok:
        subjects = _ok_subjects()
    elif args.subjects:
        subjects = args.subjects
    else:
        subjects = [args.subject]

    if len(subjects) == 1 and not args.all_ok and not args.subjects:
        t0 = time.time()
        summ = run_subject(subjects[0], args.n_perm, args.seed, args.verify)
        print(json.dumps({"subject": summ["subject"], "n_seizures": summ["n_seizures"],
                          "n_windows": summ["n_windows"], "shaft_structure": summ["shaft_structure"],
                          "obs_median_of_window_medians": summ["obs_median_of_window_medians"],
                          "nulls": summ["nulls"], "fidelity_max_abs_err": summ["fidelity_max_abs_err"]},
                         ensure_ascii=False, indent=2))
        print(f"[{subjects[0]}] done in {time.time()-t0:.1f}s -> {summ['outputs']['figure_png']}")
        return

    # fail-closed batch
    print(f"processing {len(subjects)} subject(s)", flush=True)
    records = []
    for i, ds_sid in enumerate(subjects, 1):
        t0 = time.time()
        print(f"[{i}/{len(subjects)}] {ds_sid} ...", flush=True)
        summ_fp = OUT_DIR / f"{ds_sid}_maxab_spatial_null_summary.json"
        if args.rebuild_from_stats:
            try:
                rec = _record_from_summary(rebuild_subject(ds_sid))
                print(f"    rebuilt {_sig_str(rec)}", flush=True)
            except Exception as exc:
                print(
                    f"    stale existing output ({type(exc).__name__}: {exc}); recomputing",
                    flush=True,
                )
                try:
                    summ = run_subject(ds_sid, args.n_perm, args.seed, args.verify)
                    rec = _record_from_summary(summ)
                    print(f"    ok  {_sig_str(rec)} ({time.time()-t0:.0f}s)", flush=True)
                except Exception as rerun_exc:
                    rec = {
                        "subject": ds_sid,
                        "status": "drop",
                        "drop_reason": f"{type(rerun_exc).__name__}: {rerun_exc}",
                    }
                    print(f"    DROP {rec['drop_reason']}", flush=True)
        elif args.skip_existing and summ_fp.exists():
            try:
                existing = json.loads(summ_fp.read_text())
                if "nulls" not in existing:
                    raise ValueError("existing null has no correction results")
                _validate_cached_summary(
                    ds_sid,
                    existing,
                    n_perm=args.n_perm,
                    seed=args.seed,
                )
                rec = _record_from_summary(existing)
                print(f"    skip (exists) {_sig_str(rec)}", flush=True)
            except Exception as exc:
                print(
                    f"    stale existing output ({type(exc).__name__}: {exc}); recomputing",
                    flush=True,
                )
                try:
                    summ = run_subject(ds_sid, args.n_perm, args.seed, args.verify)
                    rec = _record_from_summary(summ)
                    print(f"    ok  {_sig_str(rec)} ({time.time()-t0:.0f}s)", flush=True)
                except Exception as rerun_exc:
                    rec = {
                        "subject": ds_sid,
                        "status": "drop",
                        "drop_reason": f"{type(rerun_exc).__name__}: {rerun_exc}",
                    }
                    print(f"    DROP {rec['drop_reason']}", flush=True)
        else:
            try:
                summ = run_subject(ds_sid, args.n_perm, args.seed, args.verify)
                rec = _record_from_summary(summ)
                print(f"    ok  {_sig_str(rec)} ({time.time()-t0:.0f}s)", flush=True)
            except Exception as exc:
                rec = {"subject": ds_sid, "status": "drop", "drop_reason": f"{type(exc).__name__}: {exc}"}
                print(f"    DROP {rec['drop_reason']}", flush=True)
        records.append(rec)
    n_ok = sum(1 for r in records if r["status"] == "ok")
    if args.all_ok:
        _write_cohort_index(records)
    print(f"\nDONE: {n_ok}/{len(records)} ok -> {INDEX_CSV}", flush=True)


if __name__ == "__main__":
    main()
