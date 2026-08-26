"""Minute contact x log-frequency power field -- the R0.1 prediction target.

Owner: Worker B.

Plain words: for every recorded minute we ask "how much power did each bipolar
contact carry in each of 12 logarithmically spaced frequency bands between 1
and 100 Hz".  That 2-D picture (contacts x bands) is the only thing the model
is asked to predict.  Nothing about spikes, seizures or clinical labels enters
it.

Three artifacts are produced per subject, all on the same minute grid as the
raw cache so ``minute_index`` means the same thing everywhere:

    spectral_target.zarr        (n_minutes, C, 12) float32   log10 band power
    broadband_log.zarr          (n_minutes, C)     float32   log10 1-100 Hz power
    saturation_fraction.zarr    (n_minutes, C)     float32   fraction at int16 rail
    artifact_mask.zarr          (n_minutes, C)     bool      True = artifact
    train_stats.json                                          train-only normalisation

Minutes that were never cached (never recorded, or outside the cache cap) are
NaN in the float arrays and True in the artifact mask, so an accidental read
produces a loud NaN rather than a quiet zero.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import welch

from . import contract
from .raw_cache import load_cache, load_cache_index

WINDOW_WELCH_NPERSEG = 512
"""Welch length for the 5 s diagnostic field (2 s -> 0.5 Hz resolution).

At 0.5 Hz resolution the lowest contract band [1.000, 1.468) Hz contains a
single FFT bin, so the 5 s field's band 0 is *under-resolved by construction*.
It is a diagnostic only; the model target is always the one-minute field.
"""

TARGET_CHUNK_MINUTES = 1440
MINUTES_PER_READ = 120
"""Minutes decoded per Welch pass; 120 x 15360 x C float32 ~= 0.7 GB at C=100."""

STD_FLOOR = 1e-3
"""Normalising std floor: a contact-band with no train variance would otherwise
divide by ~0 and turn a physiologically dead channel into a huge z-score."""


def _dir(subject: str, cache_dir: Optional[Path]) -> Path:
    """Sibling artifacts always live next to the raw cache they describe.

    ``contract.py`` has no path helper for these three (reported to the main
    agent), so they are derived from the cache directory rather than from the
    subject name -- otherwise a caller that redirects ``cache_path`` (a test, a
    scratch rebuild) would silently write its side arrays into the real cohort
    cache root.
    """
    return Path(cache_dir) if cache_dir is not None else contract.cache_dir(subject)


def artifact_mask_path(subject: str, cache_dir: Optional[Path] = None) -> Path:
    return _dir(subject, cache_dir) / "artifact_mask.zarr"


def broadband_log_path(subject: str, cache_dir: Optional[Path] = None) -> Path:
    return _dir(subject, cache_dir) / "broadband_log.zarr"


def saturation_path(subject: str, cache_dir: Optional[Path] = None) -> Path:
    return _dir(subject, cache_dir) / "saturation_fraction.zarr"


# --------------------------------------------------------------------------
# 1. Field estimators
# --------------------------------------------------------------------------


def _band_average(psd: np.ndarray, bands: List[np.ndarray]) -> np.ndarray:
    """(C, F) PSD -> (C, N_FREQ_BINS) log10 band-averaged power."""
    out = np.empty((psd.shape[0], contract.N_FREQ_BINS), dtype=np.float64)
    for i, sel in enumerate(bands):
        out[:, i] = psd[:, sel].mean(axis=1)
    return np.log10(out + contract.TARGET_LOG_EPS)


def _broadband_log(psd: np.ndarray, freqs: np.ndarray, bands: List[np.ndarray]) -> np.ndarray:
    """log10 of the 1-100 Hz integrated power, line-noise bins excluded.

    This is the scalar the artifact rule (``contract.ARTIFACT_ROBUST_Z``) tests.
    """
    keep = np.unique(np.concatenate(bands))
    df = float(freqs[1] - freqs[0])
    return np.log10(psd[:, keep].sum(axis=1) * df + contract.TARGET_LOG_EPS)


def _welch_field(
    x: np.ndarray, nperseg: int, noverlap: int
) -> Tuple[np.ndarray, np.ndarray]:
    """(T, C) float -> ((C, 12) log band power, (C,) log broadband power)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected (T, C), got shape {x.shape}")
    if x.shape[0] < nperseg:
        raise ValueError(f"need >= {nperseg} samples, got {x.shape[0]}")
    freqs, psd = welch(
        x,
        fs=contract.ANALYSIS_RATE_HZ,
        window=contract.TARGET_WELCH_WINDOW,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        axis=0,
    )
    psd = np.asarray(psd).T  # (C, F)
    bands = contract.band_indices(freqs)
    return _band_average(psd, bands), _broadband_log(psd, freqs, bands)


def minute_spectral_field(x_256hz: np.ndarray) -> np.ndarray:
    """(MINUTE_SAMPLES, C) -> (C, N_FREQ_BINS) log10 band-averaged power.

    Welch with the frozen contract parameters (8 s hann, 50 % overlap, 14
    segments per minute), band-averaged over ``contract.band_indices`` (which
    already drops the 50/100 Hz +-1 Hz neighbourhoods), then log10.
    """
    if x_256hz.shape[0] != contract.MINUTE_SAMPLES:
        raise ValueError(
            f"minute field needs exactly {contract.MINUTE_SAMPLES} samples, "
            f"got {x_256hz.shape[0]}"
        )
    return _welch_field(
        x_256hz, contract.TARGET_WELCH_NPERSEG, contract.TARGET_WELCH_NOVERLAP
    )[0].astype(np.float32)


def minute_spectral_field_with_broadband(
    x_256hz: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Same as ``minute_spectral_field`` plus the (C,) broadband log power."""
    bands, broad = _welch_field(
        x_256hz, contract.TARGET_WELCH_NPERSEG, contract.TARGET_WELCH_NOVERLAP
    )
    return bands.astype(np.float32), broad.astype(np.float32)


def window_spectral_field(x_256hz: np.ndarray) -> np.ndarray:
    """(WINDOW_SAMPLES, C) -> (C, N_FREQ_BINS), diagnostic only.

    ``nperseg=512`` (2 s) is the longest segment that still gives >1 Welch
    segment in a 5 s window.  Its 0.5 Hz resolution leaves band 0
    ([1.000, 1.468) Hz) with a single FFT bin -- under-resolved by construction.
    Do not use this for the model target.
    """
    if x_256hz.shape[0] != contract.WINDOW_SAMPLES:
        raise ValueError(
            f"window field needs exactly {contract.WINDOW_SAMPLES} samples, "
            f"got {x_256hz.shape[0]}"
        )
    return _welch_field(x_256hz, WINDOW_WELCH_NPERSEG, WINDOW_WELCH_NPERSEG // 2)[0].astype(
        np.float32
    )


# --------------------------------------------------------------------------
# 2. Per-subject target build
# --------------------------------------------------------------------------


def _zarr_float_array(path: Path, shape, chunks, dtype, overwrite: bool, fill=None):
    import zarr
    from zarr.codecs import BloscCodec, BloscShuffle

    kwargs = {}
    if fill is not None:
        kwargs["fill_value"] = fill
    return zarr.create_array(
        store=str(path),
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        compressors=[BloscCodec(cname="zstd", clevel=3, shuffle=BloscShuffle.shuffle)],
        overwrite=overwrite,
        **kwargs,
    )


def build_subject_targets(
    subject: str,
    cache_path: Optional[Path] = None,
    target_path: Optional[Path] = None,
    *,
    overwrite: bool = True,
    minutes_per_read: int = MINUTES_PER_READ,
    log=None,
) -> Dict[str, object]:
    """Welch the whole cached raw into the minute field + broadband + saturation."""
    t0 = time.time()
    say = log or (lambda *_a, **_k: None)
    cache_path = Path(cache_path) if cache_path else contract.raw_cache_path(subject)
    cache_dir = cache_path.parent
    target_path = Path(target_path) if target_path else contract.spectral_target_path(subject)

    arr, scale = load_cache(subject, cache_path)
    n_minutes, C = arr.shape[0] // contract.MINUTE_SAMPLES, arr.shape[1]
    if arr.shape[0] % contract.MINUTE_SAMPLES:
        raise ValueError(
            f"{subject}: raw cache length {arr.shape[0]} is not a whole number of "
            f"{contract.MINUTE_SAMPLES}-sample minutes"
        )
    cached = load_cache_index(cache_dir, n_minutes)["cached"]

    tgt = _zarr_float_array(
        target_path, (n_minutes, C, contract.N_FREQ_BINS),
        (min(TARGET_CHUNK_MINUTES, n_minutes), C, contract.N_FREQ_BINS),
        "float32", overwrite, fill=np.nan,
    )
    bb = _zarr_float_array(
        broadband_log_path(subject, cache_dir), (n_minutes, C),
        (min(TARGET_CHUNK_MINUTES, n_minutes), C), "float32", overwrite, fill=np.nan,
    )
    sat = _zarr_float_array(
        saturation_path(subject, cache_dir), (n_minutes, C),
        (min(TARGET_CHUNK_MINUTES, n_minutes), C), "float32", overwrite, fill=np.nan,
    )

    runs: List[List[int]] = []
    for m in np.flatnonzero(cached):
        if runs and m == runs[-1][-1] + 1 and len(runs[-1]) < minutes_per_read:
            runs[-1].append(int(m))
        else:
            runs.append([int(m)])

    n_nonfinite = 0
    for run in runs:
        lo = run[0] * contract.MINUTE_SAMPLES
        blk = np.asarray(arr[lo:lo + len(run) * contract.MINUTE_SAMPLES, :], dtype=np.int16)
        rail = (np.abs(blk.astype(np.int32)) >= 32767).reshape(
            len(run), contract.MINUTE_SAMPLES, C
        ).mean(axis=1)
        uv = blk.astype(np.float32) * scale[None, :]
        del blk
        for j, m in enumerate(run):
            seg = uv[j * contract.MINUTE_SAMPLES:(j + 1) * contract.MINUTE_SAMPLES, :]
            f, b = minute_spectral_field_with_broadband(seg)
            n_nonfinite += int((~np.isfinite(f)).sum())
            tgt[m, :, :] = f
            bb[m, :] = b
        sat[run[0]:run[0] + len(run), :] = rail.astype(np.float32)

    out = {
        "subject": subject,
        "contract_version": contract.CONTRACT_VERSION,
        "n_minutes": int(n_minutes),
        "n_contacts": int(C),
        "n_minutes_cached": int(cached.sum()),
        "n_nonfinite_target_values": int(n_nonfinite),
        "target_path": str(target_path),
        "wall_seconds": float(time.time() - t0),
    }
    say(f"{subject}: spectral target {n_minutes}x{C}x{contract.N_FREQ_BINS} "
        f"in {out['wall_seconds']:.0f}s")
    return out


# --------------------------------------------------------------------------
# 3. Train-only statistics
# --------------------------------------------------------------------------


def compute_train_stats(
    subject: str,
    cache_path: Optional[Path] = None,
    target_path: Optional[Path] = None,
    stats_path: Optional[Path] = None,
    *,
    raw_scale_minutes: int = 60,
    log=None,
) -> Dict[str, object]:
    """Normalisation + artifact thresholds, estimated on TRAIN minutes ONLY.

    Hard invalidity condition #5 says normalisation may not see validation or
    sealed data, so every statistic here is computed from the rows whose
    ``cache_index.split == 'train'`` and nothing else.  Two different robust
    scales are produced and they are NOT interchangeable:

      * ``target_mean`` / ``target_std``  -- per contact x band, on the log10
        band power.  ``(log_power - mean) / std`` is the normalised target, so
        the "patient mean" baseline is exactly 0 with normalised MSE 1.0.
      * ``raw_center_uv`` / ``raw_scale_uv`` -- per contact, on the *time
        domain* microvolt signal.  This is what normalises the encoder's ``raw``
        input.
      * ``broadband_log_median`` / ``broadband_log_mad`` -- per contact, on the
        broadband log power.  This is what the ``ARTIFACT_ROBUST_Z`` rule tests.
    """
    import zarr

    say = log or (lambda *_a, **_k: None)
    cache_path = Path(cache_path) if cache_path else contract.raw_cache_path(subject)
    cache_dir = cache_path.parent
    target_path = Path(target_path) if target_path else contract.spectral_target_path(subject)
    stats_path = Path(stats_path) if stats_path else contract.subject_stats_path(subject)

    tgt = zarr.open_array(str(target_path), mode="r")
    index = load_cache_index(cache_dir, int(tgt.shape[0]))
    cached, split = index["cached"], index["split"]
    train = np.flatnonzero(cached & (split == "train"))
    if train.size == 0:
        raise ValueError(f"{subject}: no cached train minutes; cannot normalise")

    bbz = zarr.open_array(str(broadband_log_path(subject, cache_dir)), mode="r")
    C = int(tgt.shape[1])

    rows = np.asarray(tgt[train, :, :], dtype=np.float64)          # (n_train, C, 12)
    mean = np.nanmean(rows, axis=0)
    std = np.nanstd(rows, axis=0)
    std = np.where(np.isfinite(std) & (std > STD_FLOOR), std, STD_FLOOR)
    mean = np.where(np.isfinite(mean), mean, 0.0)

    bb = np.asarray(bbz[train, :], dtype=np.float64)               # (n_train, C)
    bb_med = np.nanmedian(bb, axis=0)
    bb_mad = np.nanmedian(np.abs(bb - bb_med[None, :]), axis=0)
    bb_sigma = 1.4826 * bb_mad
    bb_sigma = np.where(np.isfinite(bb_sigma) & (bb_sigma > 1e-6), bb_sigma, np.nan)

    arr, scale = load_cache(subject, cache_path)
    take = train
    if take.size > raw_scale_minutes:
        take = take[np.linspace(0, take.size - 1, raw_scale_minutes).round().astype(int)]
    chunks = []
    for m in take:
        lo = int(m) * contract.MINUTE_SAMPLES
        chunks.append(np.asarray(arr[lo:lo + contract.MINUTE_SAMPLES, :], dtype=np.float32))
    sample = np.concatenate(chunks, axis=0).astype(np.float64) * scale[None, :]
    raw_center = np.median(sample, axis=0)
    raw_mad = np.median(np.abs(sample - raw_center[None, :]), axis=0)
    raw_sigma = 1.4826 * raw_mad
    raw_sigma = np.where(np.isfinite(raw_sigma) & (raw_sigma > 1e-9), raw_sigma, 1.0)

    stats = {
        "subject": subject,
        "contract_version": contract.CONTRACT_VERSION,
        "code_revision": contract.code_revision(),
        "n_contacts": C,
        "n_freq_bins": int(contract.N_FREQ_BINS),
        "n_train_minutes": int(train.size),
        "n_train_minutes_used_for_raw_scale": int(take.size),
        "n_cached_minutes": int(cached.sum()),
        "n_validation_minutes": int((cached & (split == "validation")).sum()),
        "freq_edges_hz": [float(v) for v in contract.FREQ_EDGES],
        "target_mean": mean.astype(np.float32).tolist(),
        "target_std": std.astype(np.float32).tolist(),
        "broadband_log_median": bb_med.astype(np.float32).tolist(),
        "broadband_log_mad": bb_mad.astype(np.float32).tolist(),
        "broadband_log_robust_sigma": bb_sigma.astype(np.float32).tolist(),
        "raw_center_uv": raw_center.astype(np.float32).tolist(),
        "raw_mad_uv": raw_mad.astype(np.float32).tolist(),
        "raw_scale_uv": raw_sigma.astype(np.float32).tolist(),
        "int16_scale_uv": scale.astype(np.float32).tolist(),
        "artifact_robust_z": float(contract.ARTIFACT_ROBUST_Z),
        "artifact_saturation_fraction": float(contract.ARTIFACT_SATURATION_FRACTION),
        "minute_min_valid_contact_fraction": float(contract.MINUTE_MIN_VALID_CONTACT_FRACTION),
        "target_std_floor": float(STD_FLOOR),
    }
    contract.atomic_write_json(stats_path, stats)
    say(f"{subject}: train stats from {train.size} train minutes -> {stats_path}")
    return stats


# --------------------------------------------------------------------------
# 4. Artifact mask
# --------------------------------------------------------------------------


def refine_train_stats_with_artifacts(
    subject: str,
    target_path: Optional[Path] = None,
    stats_path: Optional[Path] = None,
    *,
    log=None,
) -> Dict[str, object]:
    """Second pass: re-standardise on the TRAIN minutes the model actually sees.

    ``compute_train_stats`` has to run before the artifact mask exists, because
    the artifact rule is defined against its robust broadband median/MAD. That
    leaves ``target_mean`` / ``target_std`` estimated over every train minute,
    artifacts included -- and on epilepsiae_620 the 1.35 % of contact-minutes the
    artifact rule rejects carry 87 % of the variance (their mean z-squared is
    64.9 against 0.124 for the rest), inflating the standard deviation by a
    median factor of 3.0 and up to 9.3.

    Two things break if that is left alone. The stated reading of the metric --
    "normalised MSE is the fraction of the patient's own train variance left
    unexplained, and the patient-mean baseline is 1.0 by construction" -- is
    simply false; the mean baseline lands at 0.12. And every arm's error gets
    divided by the same inflated number, squeezing the differences between the
    model, persistence and the baselines into the third decimal place, which
    would make the horizon curve look flat for a reason that has nothing to do
    with the brain.

    So the standardisation is recomputed over the artifact-clean train
    contact-minutes, which is exactly the population that is trained on and
    scored. The first-pass values are kept under ``*_all_train`` for audit, and
    the inflation factor is recorded.
    """
    import zarr

    say = log or (lambda *_a, **_k: None)
    target_path = Path(target_path) if target_path else contract.spectral_target_path(subject)
    stats_path = Path(stats_path) if stats_path else contract.subject_stats_path(subject)
    cache_dir = target_path.parent

    payload = json.loads(Path(stats_path).read_text())
    if payload.get("standardisation_basis") == "artifact_clean_train_minutes":
        say(f"{subject}: train stats already artifact-refined; nothing to do")
        return payload

    tgt = zarr.open_array(str(target_path), mode="r")
    mask = zarr.open_array(str(artifact_mask_path(subject, cache_dir)), mode="r")
    index = load_cache_index(cache_dir, int(tgt.shape[0]))
    train = np.flatnonzero(index["cached"] & (index["split"] == "train"))
    if train.size == 0:
        raise ValueError(f"{subject}: no cached train minutes")

    rows = np.asarray(tgt[train, :, :], dtype=np.float64)
    bad = np.asarray(mask[train, :], dtype=bool)             # True == artifact
    clean = np.where(~bad[:, :, None], rows, np.nan)
    n_clean = int(np.isfinite(clean).sum())
    if n_clean < 100:
        raise ValueError(f"{subject}: only {n_clean} clean train elements; refusing to "
                         "standardise on them")

    with np.errstate(invalid="ignore"):
        mean = np.nanmean(clean, axis=0)
        std = np.nanstd(clean, axis=0)
    std = np.where(np.isfinite(std) & (std > STD_FLOOR), std, STD_FLOOR)
    mean = np.where(np.isfinite(mean), mean, 0.0)

    old_std = np.asarray(payload["target_std"], dtype=np.float64)
    ratio = old_std / np.where(std > STD_FLOOR, std, np.nan)
    z = (clean - mean) / std
    unit = float(np.nanmean(z ** 2))

    payload["target_mean_all_train"] = payload["target_mean"]
    payload["target_std_all_train"] = payload["target_std"]
    payload["target_mean"] = mean.tolist()
    payload["target_std"] = std.tolist()
    payload["standardisation_basis"] = "artifact_clean_train_minutes"
    payload["standardisation_note"] = (
        "target_mean/target_std are estimated over TRAIN contact-minutes that "
        "survive the artifact rule, i.e. exactly the population that is trained "
        "on and scored. The first-pass values over all train minutes are kept as "
        "*_all_train; they are inflated by the artifact tail and must not be used "
        "to normalise."
    )
    payload["standardisation_audit"] = {
        "n_clean_elements": n_clean,
        "artifact_fraction_train": float(bad.mean()),
        "mean_z_squared_on_clean_train": unit,
        "std_inflation_median": float(np.nanmedian(ratio)),
        "std_inflation_p95": float(np.nanpercentile(ratio, 95)),
        "std_inflation_max": float(np.nanmax(ratio)),
    }
    if not (0.98 <= unit <= 1.02):
        raise ValueError(
            f"{subject}: re-standardised clean train mean z^2 is {unit:.4f}, not ~1.0")
    contract.atomic_write_json(stats_path, payload)
    say(f"{subject}: re-standardised on {n_clean} clean train elements; "
        f"mean z^2 = {unit:.4f}; std inflation removed "
        f"(median {payload['standardisation_audit']['std_inflation_median']:.2f}x, "
        f"max {payload['standardisation_audit']['std_inflation_max']:.2f}x)")
    return payload


def artifact_mask(
    subject: str,
    cache_path: Optional[Path] = None,
    stats_path: Optional[Path] = None,
    mask_path: Optional[Path] = None,
    *,
    overwrite: bool = True,
    log=None,
) -> Dict[str, object]:
    """(n_minutes, C) bool, True = artifact.  Written to ``artifact_mask.zarr``.

    A contact-minute is an artifact when EITHER its broadband log power sits
    more than ``ARTIFACT_ROBUST_Z`` robust SD from that contact's TRAIN median,
    OR more than ``ARTIFACT_SATURATION_FRACTION`` of its samples sit at the
    int16 rail.  Uncached minutes are marked artifact too so nothing downstream
    can read a filled zero as signal.
    """
    import zarr

    say = log or (lambda *_a, **_k: None)
    cache_path = Path(cache_path) if cache_path else contract.raw_cache_path(subject)
    cache_dir = cache_path.parent
    stats_path = Path(stats_path) if stats_path else contract.subject_stats_path(subject)
    mask_path = Path(mask_path) if mask_path else artifact_mask_path(subject, cache_dir)

    stats = json.loads(Path(stats_path).read_text())
    bb = np.asarray(zarr.open_array(str(broadband_log_path(subject, cache_dir)), mode="r")[:], dtype=np.float64)
    cached = load_cache_index(cache_dir, int(bb.shape[0]))["cached"]
    sat = np.asarray(zarr.open_array(str(saturation_path(subject, cache_dir)), mode="r")[:], dtype=np.float64)

    med = np.asarray(stats["broadband_log_median"], dtype=np.float64)[None, :]
    sigma = np.asarray(stats["broadband_log_robust_sigma"], dtype=np.float64)[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.abs(bb - med) / sigma
    bad_z = np.isfinite(z) & (z > contract.ARTIFACT_ROBUST_Z)
    bad_sat = np.isfinite(sat) & (sat > contract.ARTIFACT_SATURATION_FRACTION)
    bad = bad_z | bad_sat | ~np.isfinite(bb)
    bad[~cached, :] = True

    import zarr as _z
    from zarr.codecs import BloscCodec, BloscShuffle

    out = _z.create_array(
        store=str(mask_path), shape=bad.shape,
        chunks=(min(TARGET_CHUNK_MINUTES, bad.shape[0]), bad.shape[1]),
        dtype="bool",
        compressors=[BloscCodec(cname="zstd", clevel=3, shuffle=BloscShuffle.shuffle)],
        overwrite=overwrite,
    )
    out[:] = bad

    n_cached_cells = int(cached.sum() * bad.shape[1])
    n_bad_cached = int(bad[cached, :].sum())
    valid_frac = (~bad[cached, :]).mean(axis=1) if cached.any() else np.zeros(0)
    summary = {
        "subject": subject,
        "artifact_mask_path": str(mask_path),
        "n_minutes": int(bad.shape[0]),
        "n_contacts": int(bad.shape[1]),
        "artifact_rate_cached": float(n_bad_cached / max(n_cached_cells, 1)),
        "artifact_rate_robust_z": float(bad_z[cached, :].sum() / max(n_cached_cells, 1)),
        "artifact_rate_saturation": float(bad_sat[cached, :].sum() / max(n_cached_cells, 1)),
        "n_minutes_below_contact_fraction": int(
            (valid_frac < contract.MINUTE_MIN_VALID_CONTACT_FRACTION).sum()
        ),
        "n_minutes_cached": int(cached.sum()),
    }
    say(f"{subject}: artifact rate {summary['artifact_rate_cached']:.4f} "
        f"({summary['n_minutes_below_contact_fraction']} minutes below the "
        f"{contract.MINUTE_MIN_VALID_CONTACT_FRACTION:.0%} contact floor)")
    return summary
