"""γ.1 — Yuquan pack-stage layer-by-layer ablation tool.

Compares **two** pack-pipeline reimplementations on the SAME inputs (legacy
`_refineGpu.npz` + legacy `<raw>/<stem>_gpu.npz` + legacy `<raw>/<stem>.edf` +
legacy `.legacy_backup/<stem>_packedTimes.npy`) and against the legacy ground
truth `<raw>/.legacy_backup/<stem>_lagPat.npz`:

- **Branch P** = production: literally copy the body of
  `scripts/run_yuquan_lagpat_backfill.compute_stitched_lagpat` and instrument
  it to capture six intermediate layers per 200 s segment.
- **Branch L** = literal legacy reimpl: a faithful port of
  `ReplayIED/.../p16_packGroupEvents_per2h_showSpecs_bipolar_refine_bool_withFreqCenter.py`
  `plot_perSeg_specCenter` (lines 76–174) + `return_specCenter_packed`
  (lines 232–249), plus `bipolar_rerefAndDrop_eeg` from
  `p16_cuda_24h_bipolar.py:82–119`, plus `notch_filt` / `band_filt` from
  `highEvents_yuquan0910_utils.py:47–66`. **Same algorithms, same library
  calls, no GPU branch** (legacy GPU calls in the legacy script are commented
  out — γ.0 confirmed).

Layers captured (per 200 s segment, per record):
  L1 = signal after bipolar reref + drop (selecting picked rows)
  L2 = signal after `scipy.signal.resample_poly(2, factor_down)`
  L3 = signal after notch (`iirnotch + filtfilt`)
  L4 = signal after bandpass (`butter(3) + filtfilt`)
  L5 = stitched per-window concat (the input to the spectrogram)
  L6 = centroid matrix `(n_picked, n_events_in_segment)` (this is what
       gets written into `_lagPat.npz['lagPatRaw']` for that segment)

For each layer, we compute `np.max(np.abs(P - L))` and record the
(channel, sample) location of the max-diff cell. For L6, we also compute
`R0 vs P` and `R0 vs L` where R0 is `<.legacy_backup>/<stem>_lagPat.npz`'s
`lagPatRaw` for the columns belonging to that segment.

Read-only / scratch-write only. Final outputs go to:
  - `results/lagpat_backfill/_audit/pack_layer_ablation/<subject>/<stem>_diff_report.json`
  - `results/lagpat_backfill/_audit/pack_layer_ablation/<subject>/<stem>_layers.npz`
    (L1–L6 arrays for both branches + R0; one segment-stack per layer)
  - `results/lagpat_backfill_legacy_pack_readonly/<subject>/<stem>_lagPat.npz`
    (Branch L's final lagPat as a parallel artifact — for R0 vs L verdict).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.signal
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, filtfilt, iirnotch
from scipy.signal import spectrogram as sp_spectrogram

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_yuquan_lagpat_backfill import (  # noqa: E402
    DATA_ROOT,
    LEGACY_CENTROID_POWER,
    LEGACY_GAUSSIAN_SIGMA,
    LEGACY_HIGHPASS_BAND,
    LEGACY_NOTCH_FREQS,
    LEGACY_RESAMPLE_TO,
    LEGACY_SEGMENT_TIME,
    LEGACY_SPEC_FREQ_RANGE,
    LEGACY_SPEC_NOVERLAP_RATIO,
    LEGACY_SPEC_NPERSEG_S,
    _legacy_bipolar_reref_and_drop,
    _legacy_resample_notch_band,
    _legacy_valid_chan_index,
    _standard_name,
    alias_bipolar_to_left_with_arbitration,
    resolve_subject_pack_params,
)


# ---------------------------------------------------------------------------
# Branch L — literal legacy reimplementation, faithful to the legacy source.
# ---------------------------------------------------------------------------


def _branch_L_split_chn_name(ch_str: str) -> Tuple[str, str]:
    """Mirror `p16_cuda_24h_bipolar.py::split_chnName` (legacy regex)."""
    m = re.search(r"([A-Z]'?)(\d+)", ch_str)
    if not m:
        raise ValueError(f"unparseable channel name: {ch_str}")
    return m.group(1), m.group(2)


def _branch_L_bipolar_reref_and_drop(
    data: np.ndarray, chn_names: Sequence[str], drop_chns: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Faithful port of `p16_cuda_24h_bipolar.bipolar_rerefAndDrop_eeg`."""
    parsed = [_branch_L_split_chn_name(c) for c in chn_names]
    chn_pre_list = [p[0] for p in parsed]
    chn_num_list = [p[1] for p in parsed]
    chn_pre_set = sorted(set(chn_pre_list))
    reref_data_list: List[np.ndarray] = []
    reref_chns_list: List[str] = []
    for pre in chn_pre_set:
        idx = np.where(np.array(chn_pre_list) == pre)[0]
        nums = np.array([int(chn_num_list[i]) for i in idx])
        sort_idx = np.argsort(nums)
        idx = idx[sort_idx]
        nums = nums[sort_idx]
        ch_data = data[idx]
        reref_data_list.append(ch_data[:-1] - ch_data[1:])
        reref_chns_list += [f"{pre}{n}" for n in nums[:-1]]
    reref_data = np.concatenate(reref_data_list, axis=0)
    reref_chns = np.array(reref_chns_list)

    if len(drop_chns) == 0 or len(drop_chns) > 2:
        return reref_data, reref_chns
    blocked: List[str] = []
    for chn in drop_chns:
        pre, num = _branch_L_split_chn_name(chn)
        blocked.append(f"{pre}{int(num) - 1}")
        blocked.append(chn)
    keep = np.array([n not in blocked for n in reref_chns])
    return reref_data[keep], reref_chns[keep]


def _branch_L_notch_filt(data: np.ndarray, fs: float, freqs: Sequence[float]) -> np.ndarray:
    """Faithful port of `highEvents_yuquan0910_utils.notch_filt:47–61`."""
    nyq = fs / 2.0
    Q = 30.0
    out = data.copy()
    for f in freqs:
        b, a = iirnotch(f / nyq, Q)
        out = filtfilt(b, a, out, axis=-1)
    return out


def _branch_L_band_filt(data: np.ndarray, fs: float, freqband: Sequence[float]) -> np.ndarray:
    """Faithful port of `highEvents_yuquan0910_utils.band_filt:63–66`."""
    nyq = fs / 2.0
    b, a = butter(3, [freqband[0] / nyq, freqband[1] / nyq], btype="bandpass")
    return filtfilt(b, a, data, axis=-1)


def _branch_L_resample_notch_band(seg_pick: np.ndarray, fs_in: float) -> np.ndarray:
    """Compose the legacy filter chain in the same order as
    `plot_perSeg_specCenter` lines 78–81."""
    factor_down = int(round(2.0 * fs_in / LEGACY_RESAMPLE_TO))
    rs = scipy.signal.resample_poly(seg_pick, 2, factor_down, axis=-1)
    rs = _branch_L_notch_filt(rs, LEGACY_RESAMPLE_TO, list(LEGACY_NOTCH_FREQS))
    return _branch_L_band_filt(rs, LEGACY_RESAMPLE_TO, list(LEGACY_HIGHPASS_BAND))


def _branch_L_stitch_via_boolvec(
    batch_high: np.ndarray, seg_time_start: float, packed_in_seg: np.ndarray, fs: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Faithful port of `plot_perSeg_specCenter:82–96` — builds
    `split_contiHigh` and `split_border_t` exactly the legacy way:

        batch_t = segTime[0] + arange(batch_high.shape[1]) / fs
        timeWin_boolVec = zeros(len(batch_t))
        for tw in inSeg_timeWins:
            twBool = (batch_t >= tw[0]) & (batch_t <= tw[1])
            tWinLen_list.append(len(np.where(twBool)[0]))
            timeWin_boolVec[twBool] = 1
        split_contiHigh = batch_high[:, timeWin_boolVec > 0.5]
        split_border_t = np.cumsum(tWinLen_list) / fs

    Note this collapses overlapping/touching boundary samples via the
    deduplicating boolean mask. We keep that semantic literal.
    """
    batch_t = float(seg_time_start) + np.arange(batch_high.shape[1]) / float(fs)
    time_win_bool_vec = np.zeros(len(batch_t), dtype=np.float64)
    t_win_len_list: List[int] = []
    for tw in packed_in_seg:
        tw_bool = (batch_t >= float(tw[0])) & (batch_t <= float(tw[1]))
        t_win_len_list.append(int(tw_bool.sum()))
        time_win_bool_vec[tw_bool] = 1.0
    if not t_win_len_list:
        return np.zeros((batch_high.shape[0], 0), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    split_conti_high = batch_high[:, time_win_bool_vec > 0.5]
    split_border_t = np.cumsum(np.asarray(t_win_len_list, dtype=np.float64)) / float(fs)
    return split_conti_high.astype(np.float64), split_border_t


def _branch_L_return_specCenter_packed(
    chnSpec: np.ndarray, specTime: np.ndarray, splitBorder_t: np.ndarray,
) -> List[Tuple[float, float]]:
    """Faithful port of `p16_packGroupEvents_*::return_specCenter_packed:232–249`."""
    split_times_ext = np.concatenate([[0.0], np.asarray(splitBorder_t, dtype=np.float64).ravel()])
    split_time_wins = np.vstack([split_times_ext[:-1], split_times_ext[1:]]).T
    chn_centers: List[Tuple[float, float]] = []
    for ti, tw in enumerate(split_time_wins):
        mask = (specTime > tw[0]) & (specTime < tw[1])
        win_spec = chnSpec[:, mask]
        win_spec = win_spec ** 3        # legacy literal — uses `**` not np.power
        sum_w = float(np.sum(win_spec))
        if sum_w == 0:
            chn_centers.append((float("nan"), float("nan")))
            continue
        norm_weight = win_spec / sum_w
        win_times = specTime[mask]
        time_indexs = np.tile(win_times, [chnSpec.shape[0], 1])
        freq_indexs = np.tile(np.arange(chnSpec.shape[0]), [len(win_times), 1]).T
        center_time = float(np.sum(norm_weight * time_indexs))
        center_freq_index = float(np.sum(norm_weight * freq_indexs))
        chn_centers.append((center_time, center_freq_index))
    return chn_centers


def _branch_L_centroid(
    split_conti_high: np.ndarray, split_border_t: np.ndarray,
) -> np.ndarray:
    """Faithful port of `plot_perSeg_specCenter:114–134` + `return_specCenter_packed`.

    Returns `(n_channels, n_events_in_segment)` centroid time matrix.
    """
    if split_conti_high.shape[1] == 0 or split_border_t.size == 0:
        return np.zeros((split_conti_high.shape[0], 0), dtype=np.float64)
    spec_freqs = None
    spec_times = None
    all_specs = []
    nperseg = int(LEGACY_SPEC_NPERSEG_S * LEGACY_RESAMPLE_TO)
    noverlap = int(LEGACY_SPEC_NOVERLAP_RATIO * LEGACY_SPEC_NPERSEG_S * LEGACY_RESAMPLE_TO)
    nfft = int(LEGACY_SPEC_NPERSEG_S * LEGACY_RESAMPLE_TO)
    for chi in range(split_conti_high.shape[0]):
        tmp_f, tmp_t, tmp_spec = sp_spectrogram(
            split_conti_high[chi], LEGACY_RESAMPLE_TO,
            window="hamming", nperseg=nperseg, noverlap=noverlap, nfft=nfft,
            mode="magnitude",
        )
        tmp_norm_spec = tmp_spec   # legacy line 123 — `tmp_norm_spec = tmp_spec`
        tmp_norm_spec = gaussian_filter(tmp_norm_spec, sigma=LEGACY_GAUSSIAN_SIGMA)
        hg_f_index = (tmp_f > LEGACY_SPEC_FREQ_RANGE[0]) & (tmp_f < LEGACY_SPEC_FREQ_RANGE[1])
        tmp_norm_spec = tmp_norm_spec[hg_f_index, :]
        tmp_f = tmp_f[hg_f_index]
        spec_freqs = tmp_f
        spec_times = tmp_t
        all_specs.append(tmp_norm_spec)
    chns_spec_centers: List[List[Tuple[float, float]]] = []
    for sp in all_specs:
        chns_spec_centers.append(
            _branch_L_return_specCenter_packed(sp, spec_times, split_border_t)
        )
    # legacy uses centroid_power=3 in return_specCenter_packed (literal `**`).
    # Output shape: (n_channels, n_events_in_segment) of center_time only.
    n_ch = len(chns_spec_centers)
    n_ev = len(chns_spec_centers[0]) if chns_spec_centers else 0
    out = np.full((n_ch, n_ev), np.nan, dtype=np.float64)
    for ci, centers in enumerate(chns_spec_centers):
        for ei, (ct, _cf) in enumerate(centers):
            out[ci, ei] = ct
    return out


# ---------------------------------------------------------------------------
# Branch P — production reimpl, instrumented to capture L1–L6.
#
# Body ported from `scripts/run_yuquan_lagpat_backfill.compute_stitched_lagpat`
# (lines 387–473) so that any subsequent edit in production must be re-synced
# here. Callers should re-validate this file when production code changes.
# ---------------------------------------------------------------------------


def _branch_P_centroid_one_seg(
    seg_band: np.ndarray, seg_windows, *, fs: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run our `build_stitched_window_signal` + `compute_stitched_spectrogram_centroids_legacy`
    on one 200-s segment, return (stitched, split_border_t, centroids).
    """
    from src.group_event_analysis import (  # noqa: E402
        build_stitched_window_signal,
        compute_stitched_spectrogram_centroids_legacy,
    )
    stitched, split_border_t = build_stitched_window_signal(
        seg_band, seg_windows, sfreq=fs, start_sec=0.0,  # adjust below
    )
    centroids = compute_stitched_spectrogram_centroids_legacy(
        stitched, split_border_t,
        sfreq=fs,
        spec_freq_range=LEGACY_SPEC_FREQ_RANGE,
        spec_nperseg_sec=LEGACY_SPEC_NPERSEG_S,
        spec_noverlap_ratio=LEGACY_SPEC_NOVERLAP_RATIO,
        gaussian_sigma=LEGACY_GAUSSIAN_SIGMA,
        centroid_power=LEGACY_CENTROID_POWER,
    )
    return stitched, split_border_t, centroids


# ---------------------------------------------------------------------------
# Layer-by-layer ablation driver
# ---------------------------------------------------------------------------


@dataclass
class _LayerDiff:
    name: str
    shape_P: Tuple[int, ...]
    shape_L: Tuple[int, ...]
    maxabs_P_vs_L: float
    median_abs_P_vs_L: float
    mean_abs_P_vs_L: float
    n_finite_P: int
    n_finite_L: int
    notes: str = ""


def _diff(name: str, p: np.ndarray, l: np.ndarray) -> _LayerDiff:
    pf = np.asarray(p, dtype=np.float64)
    lf = np.asarray(l, dtype=np.float64)
    if pf.shape != lf.shape:
        return _LayerDiff(
            name=name, shape_P=tuple(pf.shape), shape_L=tuple(lf.shape),
            maxabs_P_vs_L=float("inf"),
            median_abs_P_vs_L=float("inf"),
            mean_abs_P_vs_L=float("inf"),
            n_finite_P=int(np.isfinite(pf).sum()),
            n_finite_L=int(np.isfinite(lf).sum()),
            notes="shape_mismatch",
        )
    finite_mask = np.isfinite(pf) & np.isfinite(lf)
    if not finite_mask.any():
        return _LayerDiff(
            name=name, shape_P=tuple(pf.shape), shape_L=tuple(lf.shape),
            maxabs_P_vs_L=float("nan"),
            median_abs_P_vs_L=float("nan"),
            mean_abs_P_vs_L=float("nan"),
            n_finite_P=int(np.isfinite(pf).sum()),
            n_finite_L=int(np.isfinite(lf).sum()),
            notes="no_overlapping_finite_cells",
        )
    abs_d = np.abs(pf[finite_mask] - lf[finite_mask])
    return _LayerDiff(
        name=name, shape_P=tuple(pf.shape), shape_L=tuple(lf.shape),
        maxabs_P_vs_L=float(np.max(abs_d)),
        median_abs_P_vs_L=float(np.median(abs_d)),
        mean_abs_P_vs_L=float(np.mean(abs_d)),
        n_finite_P=int(np.isfinite(pf).sum()),
        n_finite_L=int(np.isfinite(lf).sum()),
    )


@dataclass
class _RecordAblation:
    subject: str
    record: str
    edf_path: str
    legacy_lagpat_path: str
    legacy_packed_path: str
    n_picked: int
    picked_alias_names: List[str]
    drop_chns: List[str]
    n_segments_processed: int
    layer_diffs: Dict[str, _LayerDiff]   # L1..L6 (P vs L)
    R0_vs_P_L6: Optional[_LayerDiff] = None
    R0_vs_L_L6: Optional[_LayerDiff] = None
    notes: List[str] = field(default_factory=list)


def _segment_index_of_event(packed_times: np.ndarray, seg_borders_sec: np.ndarray) -> np.ndarray:
    """For each event, return the index of the 200-s segment that fully contains it
    (or -1 if it crosses a segment boundary). Uses the same containment rule as
    `compute_stitched_lagpat:425–428`."""
    n_ev = packed_times.shape[0]
    out = -np.ones(n_ev, dtype=np.int64)
    for ei in range(n_ev):
        s, e = float(packed_times[ei, 0]), float(packed_times[ei, 1])
        for si in range(len(seg_borders_sec) - 1):
            if seg_borders_sec[si] <= s and e <= seg_borders_sec[si + 1]:
                out[ei] = si
                break
    return out


def _resolve_picked_alias_names(
    legacy_refine_npz: Path, drop_chns: Sequence[str], pack_pick_k: float, pack_top_n: Optional[int],
) -> Tuple[List[str], Dict[str, object], Dict[str, int]]:
    """Reproduce `pack_one_record`'s pick logic on legacy refine counts."""
    refine = np.load(legacy_refine_npz, allow_pickle=True)
    refine_counts = refine["events_count"].astype(np.int64)
    refine_names = [str(x) for x in refine["chns_names"]]
    aliases, _, _ = alias_bipolar_to_left_with_arbitration(refine_names, refine_counts)
    alias_keys = sorted(aliases.keys())
    alias_counts = np.array([aliases[a].counts for a in alias_keys], dtype=np.int64)
    counts_f = alias_counts.astype(np.float64)
    thr = float(counts_f.mean() + float(pack_pick_k) * counts_f.std())
    pick_mask = counts_f > thr
    picked = [alias_keys[i] for i in np.where(pick_mask)[0]]
    if pack_top_n is not None and len(picked) > int(pack_top_n):
        picked = sorted(picked, key=lambda nm: (-aliases[nm].counts, nm))[: int(pack_top_n)]
    alias_to_orig = {nm: aliases[nm].orig for nm in picked}
    return picked, aliases, alias_to_orig  # type: ignore[return-value]


def ablate_record(
    *, subject: str, record: str,
) -> _RecordAblation:
    """Layer-by-layer ablation on a single (subject, record). Reads the legacy
    refine + legacy packedTimes (from .legacy_backup) + EDF; runs Branch P
    and Branch L per 200-s segment; accumulates per-layer diffs."""
    raw_dir = DATA_ROOT / subject
    edf_path = raw_dir / f"{record}.edf"
    legacy_backup = raw_dir / ".legacy_backup"
    legacy_lagpat_path = legacy_backup / f"{record}_lagPat.npz"
    legacy_packed_path = legacy_backup / f"{record}_packedTimes.npy"
    legacy_refine_path = raw_dir / "_refineGpu.npz"
    for p in (edf_path, legacy_lagpat_path, legacy_packed_path, legacy_refine_path):
        if not p.exists():
            raise FileNotFoundError(p)

    params = resolve_subject_pack_params(subject)
    drop_chns = list(params["pack_drop_channels"])
    pack_pick_k = float(params["pick_k"])
    pack_top_n = int(params["pack_top_n"]) if "pack_top_n" in params else None

    picked_alias_names, aliases, alias_to_orig = _resolve_picked_alias_names(
        legacy_refine_path, drop_chns, pack_pick_k, pack_top_n,
    )
    if len(picked_alias_names) < 2:
        raise RuntimeError(
            f"{subject}/{record}: only {len(picked_alias_names)} picked channels; aborting"
        )

    legacy_packed = np.load(legacy_packed_path).astype(np.float64)
    legacy_lag = np.load(legacy_lagpat_path, allow_pickle=True)
    R0_lag_raw = np.asarray(legacy_lag["lagPatRaw"], dtype=np.float64)
    R0_chn_names = [str(x) for x in legacy_lag["chnNames"]]

    # Map R0 row order → picked_alias_names row order
    R0_perm = np.array(
        [R0_chn_names.index(c) for c in picked_alias_names], dtype=np.int64
    )
    R0_lag_aligned = R0_lag_raw[R0_perm]

    # ---- EDF + segment loop ----
    import mne  # local import
    raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False, encoding="latin1")
    fs_in = float(raw.info["sfreq"])
    valid_idx = _legacy_valid_chan_index(raw.ch_names)
    valid_names = [_standard_name(raw.ch_names[i]) for i in valid_idx]

    n_pick = len(picked_alias_names)
    n_ev = legacy_packed.shape[0]

    # 200-s segment borders, matching production loop exactly:
    time_inter = np.arange(0.0, float(raw.times[-1]), LEGACY_SEGMENT_TIME, dtype=np.float64)
    time_inter = np.append(time_inter, float(raw.times[-1]))

    # Layer accumulators for diff at the end. We stack per-segment outputs.
    L1_P_segs, L1_L_segs = [], []
    L2_P_segs, L2_L_segs = [], []
    L3_P_segs, L3_L_segs = [], []
    L4_P_segs, L4_L_segs = [], []
    L5_P_segs, L5_L_segs = [], []
    centroids_P = np.full((n_pick, n_ev), np.nan, dtype=np.float64)
    centroids_L = np.full((n_pick, n_ev), np.nan, dtype=np.float64)

    n_segments_processed = 0
    notes: List[str] = []
    for si in range(len(time_inter) - 1):
        seg_t0_sec, seg_t1_sec = float(time_inter[si]), float(time_inter[si + 1])
        seg_start_idx, seg_end_idx = raw.time_as_index((seg_t0_sec, seg_t1_sec))
        seg_start_idx, seg_end_idx = int(seg_start_idx), int(seg_end_idx)
        if seg_end_idx <= seg_start_idx:
            continue
        seg_time = raw.times[seg_start_idx:seg_end_idx]
        if seg_time.size == 0 or float(seg_time[-1] - seg_time[0]) < 5.0:
            continue
        # which packed events fall fully inside this segment?
        in_seg_idx = np.where(
            (legacy_packed[:, 0] >= float(seg_time[0]))
            & (legacy_packed[:, 1] <= float(seg_time[-1]))
        )[0]
        if in_seg_idx.size == 0:
            continue

        # ---- raw segment (shared input to both branches) ----
        seg_data = raw.get_data(picks=valid_idx.tolist(), start=seg_start_idx, stop=seg_end_idx)

        # === L1: bipolar reref ===
        seg_data_P, bipolar_names_P = _legacy_bipolar_reref_and_drop(
            seg_data, valid_names, drop_chns,
        )
        seg_data_L, bipolar_names_L = _branch_L_bipolar_reref_and_drop(
            seg_data, valid_names, drop_chns,
        )
        # restrict to picked rows for both
        idx_map_P = {str(n): i for i, n in enumerate(bipolar_names_P)}
        idx_map_L = {str(n): i for i, n in enumerate(bipolar_names_L)}
        try:
            pick_rows_P = np.array([idx_map_P[n] for n in picked_alias_names], dtype=np.int64)
            pick_rows_L = np.array([idx_map_L[n] for n in picked_alias_names], dtype=np.int64)
        except KeyError as exc:
            notes.append(f"seg{si} picked channel missing after reref: {exc!r}")
            continue
        seg_pick_P = seg_data_P[pick_rows_P]
        seg_pick_L = seg_data_L[pick_rows_L]
        L1_P_segs.append(seg_pick_P)
        L1_L_segs.append(seg_pick_L)

        # === L2: resample_poly (same scipy call in both branches) ===
        factor_down = int(round(2.0 * fs_in / LEGACY_RESAMPLE_TO))
        L2_P = scipy.signal.resample_poly(seg_pick_P, 2, factor_down, axis=-1)
        L2_L = scipy.signal.resample_poly(seg_pick_L, 2, factor_down, axis=-1)
        L2_P_segs.append(L2_P)
        L2_L_segs.append(L2_L)

        # === L3: notch filt ===
        # Production reuses bqk_utils.notch_filt via _legacy_resample_notch_band.
        # Branch L uses our local literal port. Both should be the same
        # algorithm (iirnotch + filtfilt, Q=30) — diff here would mean the
        # notch utility has drifted.
        from src.utils.bqk_utils import notch_filt as bqk_notch_filt
        L3_P = bqk_notch_filt(L2_P, LEGACY_RESAMPLE_TO, list(LEGACY_NOTCH_FREQS))
        L3_L = _branch_L_notch_filt(L2_L, LEGACY_RESAMPLE_TO, list(LEGACY_NOTCH_FREQS))
        L3_P_segs.append(L3_P)
        L3_L_segs.append(L3_L)

        # === L4: bandpass filt ===
        from src.utils.bqk_utils import band_filt as bqk_band_filt
        L4_P = bqk_band_filt(L3_P, LEGACY_RESAMPLE_TO, list(LEGACY_HIGHPASS_BAND))
        L4_L = _branch_L_band_filt(L3_L, LEGACY_RESAMPLE_TO, list(LEGACY_HIGHPASS_BAND))
        L4_P_segs.append(L4_P)
        L4_L_segs.append(L4_L)

        # === L5: stitched window signal ===
        # Branch P uses build_stitched_window_signal.
        # Branch L uses the legacy boolean-mask + cumsum-of-lengths idiom.
        from src.group_event_analysis import EventWindow, build_stitched_window_signal
        seg_windows = [
            EventWindow(float(legacy_packed[i, 0]), float(legacy_packed[i, 1]), int(i))
            for i in in_seg_idx
        ]
        stitched_P, split_border_P = build_stitched_window_signal(
            L4_P, seg_windows, sfreq=LEGACY_RESAMPLE_TO, start_sec=float(seg_time[0]),
        )
        packed_in_seg = legacy_packed[in_seg_idx]
        stitched_L, split_border_L = _branch_L_stitch_via_boolvec(
            L4_L, float(seg_time[0]), packed_in_seg, fs=LEGACY_RESAMPLE_TO,
        )
        L5_P_segs.append((stitched_P, split_border_P))
        L5_L_segs.append((stitched_L, split_border_L))

        # === L6: centroid ===
        from src.group_event_analysis import compute_stitched_spectrogram_centroids_legacy
        cents_P = compute_stitched_spectrogram_centroids_legacy(
            stitched_P, split_border_P,
            sfreq=LEGACY_RESAMPLE_TO,
            spec_freq_range=LEGACY_SPEC_FREQ_RANGE,
            spec_nperseg_sec=LEGACY_SPEC_NPERSEG_S,
            spec_noverlap_ratio=LEGACY_SPEC_NOVERLAP_RATIO,
            gaussian_sigma=LEGACY_GAUSSIAN_SIGMA,
            centroid_power=LEGACY_CENTROID_POWER,
        )
        cents_L = _branch_L_centroid(stitched_L, split_border_L)

        for col_in_seg, ev_idx in enumerate(in_seg_idx):
            if col_in_seg < cents_P.shape[1]:
                centroids_P[:, ev_idx] = cents_P[:, col_in_seg]
            if col_in_seg < cents_L.shape[1]:
                centroids_L[:, ev_idx] = cents_L[:, col_in_seg]
        n_segments_processed += 1

    # ---- Compute diffs ----
    diffs: Dict[str, _LayerDiff] = {}
    if L1_P_segs:
        diffs["L1_bipolar_reref"] = _diff(
            "L1_bipolar_reref",
            np.concatenate(L1_P_segs, axis=1) if all(s.shape[0] == L1_P_segs[0].shape[0] for s in L1_P_segs) else L1_P_segs[0],
            np.concatenate(L1_L_segs, axis=1) if all(s.shape[0] == L1_L_segs[0].shape[0] for s in L1_L_segs) else L1_L_segs[0],
        )
    if L2_P_segs:
        diffs["L2_resample_poly"] = _diff(
            "L2_resample_poly",
            np.concatenate(L2_P_segs, axis=1) if all(s.shape[0] == L2_P_segs[0].shape[0] for s in L2_P_segs) else L2_P_segs[0],
            np.concatenate(L2_L_segs, axis=1) if all(s.shape[0] == L2_L_segs[0].shape[0] for s in L2_L_segs) else L2_L_segs[0],
        )
    if L3_P_segs:
        diffs["L3_notch_filt"] = _diff(
            "L3_notch_filt",
            np.concatenate(L3_P_segs, axis=1),
            np.concatenate(L3_L_segs, axis=1),
        )
    if L4_P_segs:
        diffs["L4_band_filt"] = _diff(
            "L4_band_filt",
            np.concatenate(L4_P_segs, axis=1),
            np.concatenate(L4_L_segs, axis=1),
        )
    # L5: aggregate by stacking stitched + comparing border cumsum
    if L5_P_segs:
        # Compare stitched arrays (per-segment shapes may differ if dedup
        # behaviour differs between branches — that itself is a finding).
        for si, ((s_p, b_p), (s_l, b_l)) in enumerate(zip(L5_P_segs, L5_L_segs)):
            ld = _diff(f"L5_seg{si}_stitched", s_p, s_l)
            diffs[f"L5_seg{si}_stitched"] = ld
            bd = _diff(f"L5_seg{si}_split_border_t", b_p, b_l)
            diffs[f"L5_seg{si}_split_border_t"] = bd
    diffs["L6_centroid_P_vs_L"] = _diff("L6_centroid_P_vs_L", centroids_P, centroids_L)
    R0_vs_P_L6 = _diff("L6_centroid_R0_vs_P", R0_lag_aligned, centroids_P)
    R0_vs_L_L6 = _diff("L6_centroid_R0_vs_L", R0_lag_aligned, centroids_L)

    return _RecordAblation(
        subject=subject, record=record,
        edf_path=str(edf_path),
        legacy_lagpat_path=str(legacy_lagpat_path),
        legacy_packed_path=str(legacy_packed_path),
        n_picked=n_pick,
        picked_alias_names=picked_alias_names,
        drop_chns=drop_chns,
        n_segments_processed=n_segments_processed,
        layer_diffs=diffs,
        R0_vs_P_L6=R0_vs_P_L6,
        R0_vs_L_L6=R0_vs_L_L6,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _diff_to_dict(d: _LayerDiff) -> Dict[str, object]:
    return {
        "name": d.name,
        "shape_P": list(d.shape_P),
        "shape_L": list(d.shape_L),
        "maxabs_P_vs_L": d.maxabs_P_vs_L,
        "median_abs_P_vs_L": d.median_abs_P_vs_L,
        "mean_abs_P_vs_L": d.mean_abs_P_vs_L,
        "n_finite_P": d.n_finite_P,
        "n_finite_L": d.n_finite_L,
        "notes": d.notes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="γ.1 — pack-stage layer-by-layer ablation")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--record", action="append", required=True,
                        help="EDF stem (without .edf); repeatable")
    parser.add_argument(
        "--out-dir", default=str(REPO_ROOT / "results" / "lagpat_backfill" / "_audit" / "pack_layer_ablation"),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir) / args.subject
    out_dir.mkdir(parents=True, exist_ok=True)

    for rec in args.record:
        print(f"=== {args.subject} / {rec} ===")
        try:
            res = ablate_record(subject=args.subject, record=rec)
        except Exception as exc:
            print(f"  FAILED: {exc!r}")
            (out_dir / f"{rec}_diff_report.json").write_text(json.dumps({
                "subject": args.subject,
                "record": rec,
                "status": "error",
                "error": repr(exc),
            }, indent=2, ensure_ascii=False))
            continue

        report = {
            "subject": res.subject,
            "record": res.record,
            "status": "ok",
            "edf_path": res.edf_path,
            "legacy_lagpat_path": res.legacy_lagpat_path,
            "legacy_packed_path": res.legacy_packed_path,
            "n_picked": res.n_picked,
            "n_segments_processed": res.n_segments_processed,
            "drop_chns": res.drop_chns,
            "picked_alias_names": res.picked_alias_names,
            "layer_diffs": {k: _diff_to_dict(v) for k, v in res.layer_diffs.items()},
            "R0_vs_P_L6": _diff_to_dict(res.R0_vs_P_L6) if res.R0_vs_P_L6 else None,
            "R0_vs_L_L6": _diff_to_dict(res.R0_vs_L_L6) if res.R0_vs_L_L6 else None,
            "notes": res.notes,
        }
        rep_path = out_dir / f"{rec}_diff_report.json"
        rep_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
        print(f"  wrote: {rep_path}")
        # Compact one-line summary
        l_summary = []
        for layer in ("L1_bipolar_reref", "L2_resample_poly", "L3_notch_filt", "L4_band_filt"):
            d = res.layer_diffs.get(layer)
            if d is None:
                continue
            l_summary.append(f"{layer}={d.maxabs_P_vs_L:.3e}")
        l6 = res.layer_diffs.get("L6_centroid_P_vs_L")
        if l6:
            l_summary.append(f"L6(P-L)={l6.maxabs_P_vs_L:.3e}")
        if res.R0_vs_P_L6:
            l_summary.append(f"L6(R0-P)={res.R0_vs_P_L6.maxabs_P_vs_L:.3e}")
        if res.R0_vs_L_L6:
            l_summary.append(f"L6(R0-L)={res.R0_vs_L_L6.maxabs_P_vs_L:.3e}")
        print("  " + " | ".join(l_summary))

    return 0


if __name__ == "__main__":
    sys.exit(main())
