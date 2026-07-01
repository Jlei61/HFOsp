# src/topic5_v2_band_scan.py
from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np, yaml
_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CFG = _ROOT / "config/topic5_v2_phase1.yaml"
def load_phase1_config(path=None) -> dict:
    with open(path or _DEFAULT_CFG) as fh:
        return yaml.safe_load(fh)


def line_noise_bin_mask(freqs, harmonics_hz, halfwidth_hz):
    freqs = np.asarray(freqs, float); m = np.zeros(freqs.shape, bool)
    for h in harmonics_hz: m |= np.abs(freqs - float(h)) <= float(halfwidth_hz)
    return m
def band_bin_selection(freqs, lo, hi, line_mask, half_open=False):
    freqs = np.asarray(freqs, float)
    in_band = (freqs >= float(lo)) & ((freqs < float(hi)) if half_open else (freqs <= float(hi)))
    n_band = int(in_band.sum())
    band_mask = in_band & ~np.asarray(line_mask, bool)
    return band_mask, float(band_mask.sum()) / max(n_band, 1), n_band


def masked_band_power_trace(signal, fs, lo, hi, spec_win_sec, spec_hop_sec,
                            harmonics_hz, halfwidth_hz, fs512_hi_safe, half_open=False):
    from src.topic5_ictal_recruitment import _spectrogram_on_hop
    nyq=float(fs)/2.0
    if hi>=nyq: raise ValueError(f"band hi {hi} >= Nyquist {nyq} for fs={fs}")
    f,t,Sxx=_spectrogram_on_hop(signal, fs, spec_win_sec, spec_hop_sec)
    lm=line_noise_bin_mask(f, harmonics_hz, halfwidth_hz)
    bmask,eff,n_band=band_bin_selection(f, lo, hi, lm, half_open=half_open)
    if not bmask.any(): raise ValueError(f"no bins in ({lo},{hi}) after line mask")
    power=Sxx[:,bmask,:].sum(axis=1)
    return {"logpower":np.log(np.maximum(power,1e-30)),"t":t,"eff_frac":eff,
            "fs_edge_flag":bool(float(fs)<=512.0 and float(hi)>float(fs512_hi_safe)),"n_band_bins":n_band}
def robust_z_with_flags(logpower, baseline_idx, hop_sec, min_baseline_valid_sec):
    from src.topic5_ictal_recruitment import baseline_robust_z
    z=baseline_robust_z(logpower, baseline_idx, hop_sec=hop_sec, min_baseline_valid_sec=min_baseline_valid_sec)
    return z, np.all(~np.isfinite(z), axis=1)
def channel_artifact_flags(logpower, z, sat_abs_z, sat_frac, flatline_mad_eps):
    z=np.asarray(z,float); flat=np.all(~np.isfinite(z),axis=1)
    with np.errstate(invalid="ignore"):
        sat=np.nanmean(np.abs(z)>float(sat_abs_z), axis=1) > float(sat_frac)
    sat=np.where(np.isfinite(sat), sat, False)
    bad=flat|sat
    return {"flatline":flat, "saturation":sat, "bad_channel":bad}


def contact_alignment(vals_by_name, rank_a_by_name, rank_b_by_name, oriented_template="a"):
    from scipy.stats import spearmanr, pearsonr
    def _one(rank_by):
        names=[n for n in vals_by_name if n in rank_by
               and np.isfinite(vals_by_name[n]) and np.isfinite(rank_by[n])]
        if len(names)<4: return None
        v=np.array([vals_by_name[n] for n in names]); r=np.array([rank_by[n] for n in names])
        if np.std(v)==0 or np.std(r)==0: return None
        return {"sp":float(spearmanr(v,r).statistic),"pe":float(pearsonr(v,r)[0]),"n":len(names)}
    a,b=_one(rank_a_by_name),_one(rank_b_by_name)
    def g(o,k,d=float("nan")): return o[k] if o else d
    # tie-break: on an exact |sp| tie, prefer the larger raw signed value —
    # a post-hoc cherry-picker reports whichever template looks more
    # confirmatory, not whichever was evaluated first.
    posthoc=max([o for o in (a,b) if o], key=lambda o:(abs(o["sp"]),o["sp"]), default=None)
    return {"signed_pearson_a":g(a,"pe"),"signed_spearman_a":g(a,"sp"),
            "signed_pearson_b":g(b,"pe"),"signed_spearman_b":g(b,"sp"),
            "align_signed_oriented":(g(a,"sp") if oriented_template=="a" else g(b,"sp")),
            "align_signed_posthoc_max":(posthoc["sp"] if posthoc else float("nan")),
            "align_abs_maxab_contact":max([abs(o["sp"]) for o in (a,b) if o], default=float("nan")),
            "n_contacts_a":g(a,"n",0),"n_contacts_b":g(b,"n",0)}


def _nearest_distance_bins(orphan_names, coord_by_name, min_group):
    # Tier-2 fallback: greedily bin leftover (shaft-too-small) contacts by
    # coordinate distance into groups of exactly min_group, nearest-neighbour
    # first; tie-break by original order for determinism. Contacts left over
    # after the last full bin (too few remaining) are returned separately.
    pos = {n: i for i, n in enumerate(orphan_names)}
    remaining = list(orphan_names)
    bins = []
    while len(remaining) >= min_group:
        seed = remaining[0]
        seed_xy = np.asarray(coord_by_name[seed], float)
        others = sorted(remaining[1:], key=lambda n: (
            float(np.linalg.norm(np.asarray(coord_by_name[n], float) - seed_xy)), pos[n]))
        group = [seed] + others[:min_group - 1]
        bins.append(group)
        remaining = [n for n in remaining if n not in group]
    return bins, remaining


def spatial_constrained_permute(names, values_by_name, shaft_by_name, coord_by_name,
                                 rng, mode, min_group):
    """Spatially-constrained permutation null (Gate A spatial null, issue #10).

    Permutes per-contact VALUES (contacts keep identity; values move) inside a
    spatial constraint, at three tiers of strength (strongest first):
      1. within-shaft  - a shaft's contacts are permuted among themselves
         whenever that shaft has >= min_group FINITE values (mode="within_shaft",
         the only mode wired to config nulls.spatial today).
      2. distance-bin fallback - contacts whose native shaft is too small are
         pooled by nearest coordinate distance into bins of size min_group.
      3. subject-wide fallback - contacts distance-binning still can't reach
         min_group for are pooled with each other (not with tier-1/2 contacts;
         those groups stay isolated) and permuted as one leftover pool (weakest
         null; flags subject_wide_weak).
    Non-finite values never move (nothing sensible to permute into them); they
    pass through unchanged.

    Reported spatial_null_strength is the WEAKEST tier actually invoked for this
    call: a call that is mostly within-shaft but needs even one subject-wide
    fallback contact cannot claim within_shaft_strong (P1-c: only
    within_shaft_strong supports a formal Gate A pass; the rest are
    descriptive/sensitivity only).
    """
    if mode != "within_shaft":
        raise ValueError(f"unsupported spatial null mode: {mode!r}")

    names = list(names)
    is_finite = {n: bool(np.isfinite(float(values_by_name[n]))) for n in names}

    shaft_groups = {}
    for n in names:
        shaft_groups.setdefault(shaft_by_name[n], []).append(n)
    n_singleton_groups = sum(1 for members in shaft_groups.values() if len(members) == 1)

    def shuffled_map(group):
        vals = rng.permutation([values_by_name[n] for n in group])
        return {n: float(v) for n, v in zip(group, vals)}

    perm_values = dict(values_by_name)
    n_effectively_permutable = 0
    used_distance_bin = used_subject_wide = False
    leftover_finite = []
    for members in shaft_groups.values():
        finite_members = [n for n in members if is_finite[n]]
        if len(finite_members) >= min_group:
            perm_values.update(shuffled_map(finite_members))
            n_effectively_permutable += len(finite_members)
        else:
            leftover_finite.extend(finite_members)

    if leftover_finite:
        bins, remainder = _nearest_distance_bins(leftover_finite, coord_by_name, min_group)
        if bins:
            used_distance_bin = True
            for group in bins:
                perm_values.update(shuffled_map(group))
                n_effectively_permutable += len(group)
        if remainder:
            used_subject_wide = True
            if len(remainder) >= 2:
                perm_values.update(shuffled_map(remainder))
                n_effectively_permutable += len(remainder)

    spatial_null_strength = ("subject_wide_weak" if used_subject_wide else
                              "distance_bin_fallback" if used_distance_bin else
                              "within_shaft_strong")
    return perm_values, {"spatial_null_strength": spatial_null_strength,
                          "n_effectively_permutable": n_effectively_permutable,
                          "n_singleton_groups": n_singleton_groups}


def common_field_residual(band_vals_by_name, common_field_vals_by_name):
    """OLS residual of a band's per-contact field on a common field (Gate B raw material, issue #14).

    Aligns on names present & finite in BOTH inputs, fits a degree-1 line
    band ~ slope*cf + intercept over those shared points, and returns the
    per-contact residual. Fewer than 3 shared finite points can't support a
    meaningful line fit, so this returns {} rather than fabricate a residual.

    The caller (Task 10b) supplies common_field_vals_by_name two ways -- the
    all-band 1-250 field and a leave-one-band-out field -- and calls this
    once per choice; this function is agnostic to which one it was given.
    """
    names = [n for n in band_vals_by_name if n in common_field_vals_by_name
             and np.isfinite(band_vals_by_name[n]) and np.isfinite(common_field_vals_by_name[n])]
    if len(names) < 3:
        return {}
    band = np.array([band_vals_by_name[n] for n in names], float)
    cf = np.array([common_field_vals_by_name[n] for n in names], float)
    slope, intercept = np.polyfit(cf, band, 1)
    return {n: float(band_vals_by_name[n] - (slope * common_field_vals_by_name[n] + intercept))
            for n in names}


def aperiodic_corrected_excess_power(freqs, psd_ch, lo, hi, line_mask,
                                      fit_lo=1, fit_hi=200, min_r2=0.5, half_open=False):
    """Guarded log-log 1/f (aperiodic) fit + band excess power (Gate C raw material, issue #15).

    Fits log10(psd) ~ slope*log10(freq) + offset by OLS over the closed range
    [fit_lo, fit_hi], excluding line-mask bins and non-positive/non-finite psd.
    `ok` requires the fit_r2 (coefficient of determination of that log-log fit)
    to clear `min_r2` AND at least ~10 valid fit points; a bad/failed fit
    reports ok=False and excess_power=nan rather than fabricate a number.

    Over the BAND bins (same half_open convention + line/non-positive
    exclusions as band_bin_selection), the aperiodic floor is
    10**(slope*log10(f)+offset); excess_power sums max(psd - floor, 0) per
    bin -- power ABOVE the fitted 1/f floor, clipped at zero so a below-floor
    bin can't cancel an above-floor bin elsewhere in the band.

    Named excess_power, not "oscillatory_power": this is a single non-iterative
    fit (no FOOOF-style peak removal before fitting the aperiodic), so a bump
    surviving the floor is evidence of excess power, not proof it is
    oscillatory.
    """
    freqs = np.asarray(freqs, float)
    psd_ch = np.asarray(psd_ch, float)
    line_mask = np.asarray(line_mask, bool)
    finite_pos = np.isfinite(psd_ch) & (psd_ch > 0) & (freqs > 0)

    fit_mask, _, _ = band_bin_selection(freqs, fit_lo, fit_hi, line_mask, half_open=False)
    fit_valid = fit_mask & finite_pos
    n_fit = int(fit_valid.sum())
    if n_fit < 10:
        return {"excess_power": float("nan"), "fit_r2": float("nan"),
                "slope": float("nan"), "offset": float("nan"), "ok": False}

    x = np.log10(freqs[fit_valid]); y = np.log10(psd_ch[fit_valid])
    slope, offset = np.polyfit(x, y, 1)
    ss_res = np.sum((y - (slope * x + offset)) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    fit_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    ok = bool(fit_r2 >= min_r2 and n_fit >= 10)

    band_mask, _, _ = band_bin_selection(freqs, lo, hi, line_mask, half_open=half_open)
    band_valid = band_mask & finite_pos
    if not ok:
        excess_power = float("nan")
    elif not band_valid.any():
        excess_power = float("nan")  # zero real bins in band != measured-flat spectrum; exclude (review T11)
    else:
        f_band, psd_band = freqs[band_valid], psd_ch[band_valid]
        aperiodic_pred = 10.0 ** (slope * np.log10(f_band) + offset)
        excess_power = float(np.sum(np.maximum(psd_band - aperiodic_pred, 0.0)))

    return {"excess_power": excess_power, "fit_r2": float(fit_r2),
            "slope": float(slope), "offset": float(offset), "ok": ok}


def confound_residual_rank(rank_by_name, covariate_maps, overfit_min_ratio=3):
    """Residualize the G_HFO rank against confound covariates (Patch H raw material, Task 12).

    `single`: for EACH covariate independently, OLS-residualizes rank on that one
    covariate via `common_field_residual` (identical present-and-finite alignment +
    degree-1 polyfit + <3-shared-points -> {} guard as the band/common-field case;
    here rank plays "band" and the covariate plays "common field"). Always computed,
    one entry per covariate in `covariate_maps`, regardless of the guard below --
    downstream alignment-to-timing-geometry claims want the single-covariate view
    even when the joint fit is overfitting-guarded away.

    `combined`: OLS-residualizes rank on ALL covariates JOINTLY (multiple regression,
    design matrix [1, cov_1, cov_2, ...] solved by `np.linalg.lstsq`), aligned on names
    finite across rank AND EVERY covariate (a stricter alignment than `single`, which
    only requires the one covariate in play to be finite). Guarded against overfitting:
    with p = len(covariate_maps) covariates the fit has p+1 free parameters, so
    `combined` is only attempted when the jointly-aligned n_contacts >=
    overfit_min_ratio*p + 3; otherwise returns None rather than fit an
    under-determined regression.
    """
    single = {cov: common_field_residual(rank_by_name, cov_map)
              for cov, cov_map in covariate_maps.items()}

    cov_names = list(covariate_maps)
    names = [n for n in rank_by_name if np.isfinite(rank_by_name[n])
             and all(n in covariate_maps[c] and np.isfinite(covariate_maps[c][n]) for c in cov_names)]
    n_contacts = len(names)
    if n_contacts < overfit_min_ratio * len(cov_names) + 3:
        return {"single": single, "combined": None}

    y = np.array([rank_by_name[n] for n in names], float)
    X = np.column_stack([np.ones(n_contacts)] +
                         [np.array([covariate_maps[c][n] for n in names], float) for c in cov_names])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    combined = {n: float(v) for n, v in zip(names, y - X @ beta)}
    return {"single": single, "combined": combined}


def rebuild_typical_rank(events_bool, event_lag, agg="mean"):
    """Rebuild a per-channel typical timing rank from an event x channel table (Patch F, Task 9).

    events_bool : (n_events, n_ch) bool -- per-event channel PARTICIPATION.
    event_lag   : (n_events, n_ch) float -- per-event per-channel lag/timing value
                  (NaN where not participating).
    agg='mean'  : for each channel, aggregate its lags over the events it PARTICIPATES in
                  (nanmean over events), then dense-rank channels by aggregated lag
                  (argsort-of-argsort; earliest lag -> smallest rank).

    Phantom-rank discipline (AGENTS.md lagPatRank): a channel that NEVER participates
    (``events_bool[:, c].sum() == 0``) receives NaN rank, never a fabricated finite rank
    -- this masking is the whole point of the null. Finite rank <=> the channel
    participates in >=1 event AND has a finite aggregated lag; the non-participation
    columns are dropped BEFORE the cross-channel argsort so they can never be ranked.
    """
    events_bool = np.asarray(events_bool, bool)
    event_lag = np.asarray(event_lag, float)
    n_ch = events_bool.shape[1]
    if agg != "mean":
        raise ValueError(f"unsupported agg: {agg!r}")
    participates = events_bool.sum(axis=0) > 0
    lag_where_part = np.where(events_bool, event_lag, np.nan)     # only participating lags count
    agg_lag = np.full(n_ch, np.nan, dtype=float)
    if participates.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN participating col -> NaN (kept out of rank)
            agg_lag[participates] = np.nanmean(lag_where_part[:, participates], axis=0)
    rank = np.full(n_ch, np.nan, dtype=float)
    rankable = participates & np.isfinite(agg_lag)
    idx = np.where(rankable)[0]
    if idx.size:
        rank[idx] = np.argsort(np.argsort(agg_lag[idx])).astype(float)
    return rank


def _order_null_one_template(events_bool, event_lag, rng):
    """One template's HFO-rate-preserving order null: WITHIN each event, permute the lag
    VALUES among that event's participating channels (participation is untouched -- every
    channel keeps its exact participation COUNT but gets a shuffled lag), then rebuild the
    typical rank. Destroys timing ORDER while preserving participation/HFO-rate topography."""
    events_bool = np.asarray(events_bool, bool)
    event_lag = np.asarray(event_lag, float)
    permuted = event_lag.copy()
    for e in range(events_bool.shape[0]):
        idx = np.where(events_bool[e])[0]
        if idx.size > 1:
            permuted[e, idx] = rng.permutation(event_lag[e, idx])   # shuffle lags among participants
    return rebuild_typical_rank(events_bool, permuted)


def order_null_rank_pair(events_a, lag_a, events_b, lag_b, rng):
    """HFO-rate-preserving order-null PAIR (A/B) (Patch F, Task 9).

    Rebuild EACH template independently from its own event table under a within-event lag
    permutation that preserves each channel's participation count, then rebuild the typical
    rank. Returns ``(rank_a_null, rank_b_null)``.

    ``np.isfinite(rank_a_null)`` equals the participation mask ``events_a.sum(0) > 0`` (same
    for B): the null PRESERVES HFO-rate/participation topography and DESTROYS only the timing
    order -- separating "interictal timing GEOMETRY carries information" from "alignment just
    tracks HFO-rich contacts". If ``events_b is None`` -> ``rank_b_null = None`` (downstream
    A/B-max uses A only, flagged).
    """
    rank_a_null = _order_null_one_template(events_a, lag_a, rng)
    rank_b_null = (None if events_b is None
                   else _order_null_one_template(events_b, lag_b, rng))
    return rank_a_null, rank_b_null
