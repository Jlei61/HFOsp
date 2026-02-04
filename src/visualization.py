"""
SEEG Visualization Module for HFO Analysis

This module provides visualization functions for:
- Multi-channel SEEG waveform plots (monopolar, bipolar, CAR)
- Segmented time series viewing (100s segments)
- Channel selection (all, list, shaft)
- HFO event overlays
- Preprocessing comparison plots

Author: HFOsp Team
Date: 2026-01-14
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings
from matplotlib.patches import Rectangle
from typing import List, Optional, Union, Tuple, Dict
from pathlib import Path
from dataclasses import dataclass

# Try to import preprocessing module for type hints
try:
    from .preprocessing import PreprocessingResult, ElectrodeParser
except ImportError:
    try:
        from preprocessing import PreprocessingResult, ElectrodeParser
    except ImportError:
        PreprocessingResult = None
        ElectrodeParser = None


# =============================================================================
# Color Schemes
# =============================================================================

SEEG_COLORS = {
    'default': '#1a1a2e',      # Dark blue-black
    'highlight': '#e94560',    # Red-pink
    'secondary': '#0f3460',    # Navy
    'background': '#fafafa',   # Light gray
    'grid': '#e0e0e0',         # Grid lines
    'shaft_colors': [          # Per-shaft colors
        # Deprecated: replaced by TABLEAU_20 below for better readability.
        '#4E79A7', '#A0CBE8', '#F28E2B', '#FFBE7D',
        '#59A14F', '#8CD17D', '#B6992D', '#F1CE63',
        '#499894', '#86BCB6', '#79706E', '#BAB0AC'
    ]
}

# Tableau 20 (explicit list; stable across matplotlib versions)
tableau_20 = [
    "#4E79A7", "#A0CBE8",  # 蓝 / 浅蓝
    "#F28E2B", "#FFBE7D",  # 橙 / 浅橙
    "#59A14F", "#8CD17D",  # 绿 / 浅绿
    "#B6992D", "#F1CE63",  # 黄褐 / 浅黄
    "#499894", "#86BCB6",  # 青 / 浅青
    "#E15759", "#FF9D9A",  # 红 / 浅红
    "#79706E", "#BAB0AC",  # 灰 / 浅灰
    "#D37295", "#FABFD2",  # 粉红 / 浅粉
    "#B07AA1", "#D4A6C8",  # 紫 / 浅紫
    "#9D7660", "#D7B5A6"   # 褐 / 浅褐
]

# Waveform colors must avoid red-ish tones to not clash with HFO overlays.
_TABLEAU_REDISH = {"#E15759", "#FF9D9A"}
tableau_20_no_red = [c for c in tableau_20 if c not in _TABLEAU_REDISH]

@dataclass(frozen=True)
class GroupVizConfig:
    """
    Configuration for group-event visualization.
    """

    event_line_color: str = "#d62728"  # red
    event_line_alpha: float = 0.8
    dot_size: float = 10.0
    tie_tol_ms: float = 2.0


# =============================================================================
# Main SEEG Plotting Functions
# =============================================================================

def plot_seeg_segment(
    data: np.ndarray,
    sfreq: float,
    ch_names: List[str],
    start_sec: float = 0,
    duration_sec: float = 100,
    channels: Union[str, List[str], List[int]] = 'all',
    title: Optional[str] = None,
    reference_type: str = 'unknown',
    figsize: Optional[Tuple[float, float]] = None,
    color_by_shaft: bool = True,
    show_scale_bar: bool = True,
    amplitude_scale: Optional[float] = None,
    events: Optional[List[Dict]] = None,
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot SEEG segment with stacked channels.
    
    Parameters
    ----------
    data : np.ndarray
        Shape (n_channels, n_samples). Can be monopolar, bipolar, or CAR referenced.
    sfreq : float
        Sampling frequency in Hz.
    ch_names : List[str]
        Channel names. For pre-bipolar data, uses original naming (A1 = A1-A2).
    start_sec : float
        Start time in seconds (default 0).
    duration_sec : float
        Duration to plot in seconds (default 100s).
    channels : str, list
        Channel selection:
        - 'all': All channels
        - List of strings: Specific channel names ['A1', 'A2', 'K3']
        - List of ints: Channel indices [0, 1, 5]
    title : str, optional
        Plot title. Auto-generated if None.
    reference_type : str
        Reference type for labeling: 'monopolar', 'bipolar', 'car', 'pre_bipolar'
    figsize : tuple, optional
        Figure size. Auto-calculated based on channel count if None.
    color_by_shaft : bool
        Whether to color channels by electrode shaft (default True).
    show_scale_bar : bool
        Whether to show amplitude/time scale bar (default True).
    amplitude_scale : float, optional
        Fixed amplitude scale in µV. Auto-calculated if None.
    events : list of dict, optional
        HFO events to overlay. Each dict: {'start': float, 'end': float, 'channel': str}
    ax : matplotlib.Axes, optional
        Existing axes to plot on. Creates new figure if None.
    
    Returns
    -------
    fig : matplotlib.Figure
        The figure object.
    """
    # Parse channel selection
    ch_indices, ch_labels = _parse_channel_selection(channels, ch_names)
    n_channels = len(ch_indices)
    
    if n_channels == 0:
        raise ValueError("No channels selected for plotting")
    
    # Extract time segment
    start_sample = int(start_sec * sfreq)
    end_sample = int((start_sec + duration_sec) * sfreq)
    end_sample = min(end_sample, data.shape[1])
    
    segment_data = data[ch_indices, start_sample:end_sample]
    time_vec = np.arange(start_sample, end_sample) / sfreq
    
    # Auto-calculate amplitude scale if not provided
    if amplitude_scale is None:
        # Use robust percentile-based scaling
        p5, p95 = np.percentile(segment_data, [5, 95])
        amplitude_scale = max(abs(p5), abs(p95)) * 1.2
    
    # Calculate channel spacing
    spacing = amplitude_scale * 2
    
    # Create figure if needed
    if ax is None:
        if figsize is None:
            height = max(6, min(24, n_channels * 0.3))
            figsize = (16, height)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Get colors for each channel
    if color_by_shaft and ElectrodeParser is not None:
        channel_colors = _get_shaft_colors(ch_labels)
    else:
        channel_colors = [SEEG_COLORS['default']] * n_channels
    
    # Plot each channel
    yticks = []
    yticklabels = []
    
    for i, (ch_idx, ch_name) in enumerate(zip(ch_indices, ch_labels)):
        y_offset = -i * spacing
        y_data = segment_data[i, :] + y_offset
        
        ax.plot(time_vec, y_data, linewidth=0.5, color=channel_colors[i], alpha=0.8)
        
        yticks.append(y_offset)
        yticklabels.append(ch_name)
    
    # Overlay events if provided
    if events:
        _overlay_events(ax, events, ch_labels, yticks, spacing, start_sec, start_sec + duration_sec)
    
    # Configure axes
    ax.set_xlim(time_vec[0], time_vec[-1])
    ax.set_ylim(-n_channels * spacing + spacing * 0.5, spacing * 0.5)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Channels', fontsize=12)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=8)
    ax.grid(True, axis='x', linestyle='--', alpha=0.3, color=SEEG_COLORS['grid'])
    
    # Title
    if title is None:
        ref_label = _get_reference_label(reference_type)
        title = f'SEEG Recording ({ref_label}) - {start_sec:.1f}s to {start_sec + duration_sec:.1f}s'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Scale bar
    if show_scale_bar:
        _add_scale_bar(ax, time_vec, spacing, amplitude_scale)
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_seeg_segments_browser(
    data: np.ndarray,
    sfreq: float,
    ch_names: List[str],
    segment_duration: float = 100,
    channels: Union[str, List[str]] = 'all',
    reference_type: str = 'unknown',
    save_dir: Optional[Path] = None,
    prefix: str = 'seeg'
) -> List[plt.Figure]:
    """
    Create multiple segment plots for browsing through long recordings.
    
    Parameters
    ----------
    data : np.ndarray
        Full recording data (n_channels, n_samples).
    sfreq : float
        Sampling frequency.
    ch_names : List[str]
        Channel names.
    segment_duration : float
        Duration of each segment in seconds (default 100s).
    channels : str or list
        Channel selection.
    reference_type : str
        Reference type for labeling.
    save_dir : Path, optional
        Directory to save figures. If None, figures are returned but not saved.
    prefix : str
        Filename prefix for saved figures.
    
    Returns
    -------
    figs : list of matplotlib.Figure
        List of figure objects.
    """
    total_duration = data.shape[1] / sfreq
    n_segments = int(np.ceil(total_duration / segment_duration))
    
    figs = []
    for seg_idx in range(n_segments):
        start_sec = seg_idx * segment_duration
        
        fig = plot_seeg_segment(
            data=data,
            sfreq=sfreq,
            ch_names=ch_names,
            start_sec=start_sec,
            duration_sec=segment_duration,
            channels=channels,
            reference_type=reference_type,
            title=f'Segment {seg_idx + 1}/{n_segments} ({start_sec:.0f}s - {start_sec + segment_duration:.0f}s)'
        )
        
        if save_dir is not None:
            save_path = Path(save_dir) / f'{prefix}_seg{seg_idx + 1:03d}.png'
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f'Saved: {save_path}')
        
        figs.append(fig)
    
    return figs


def plot_preprocessing_comparison(
    data_before: np.ndarray,
    data_after: np.ndarray,
    sfreq: float,
    ch_names_before: List[str],
    ch_names_after: List[str],
    start_sec: float = 0,
    duration_sec: float = 10,
    title: str = 'Preprocessing Comparison'
) -> plt.Figure:
    """
    Plot before/after comparison of preprocessing.
    
    Parameters
    ----------
    data_before : np.ndarray
        Data before preprocessing.
    data_after : np.ndarray
        Data after preprocessing.
    sfreq : float
        Sampling frequency.
    ch_names_before, ch_names_after : list
        Channel names before and after.
    start_sec : float
        Start time.
    duration_sec : float
        Duration to plot.
    title : str
        Plot title.
    
    Returns
    -------
    fig : matplotlib.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Before
    plot_seeg_segment(
        data_before, sfreq, ch_names_before,
        start_sec=start_sec, duration_sec=duration_sec,
        title='Before Preprocessing',
        ax=axes[0]
    )
    
    # After
    plot_seeg_segment(
        data_after, sfreq, ch_names_after,
        start_sec=start_sec, duration_sec=duration_sec,
        title='After Preprocessing',
        ax=axes[1]
    )
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_shaft_channels(
    data: np.ndarray,
    sfreq: float,
    ch_names: List[str],
    shaft: str,
    start_sec: float = 0,
    duration_sec: float = 100,
    reference_type: str = 'unknown',
    **kwargs
) -> plt.Figure:
    """
    Plot only channels from a specific electrode shaft.
    
    Parameters
    ----------
    data : np.ndarray
        Full recording data.
    sfreq : float
        Sampling frequency.
    ch_names : List[str]
        All channel names.
    shaft : str
        Shaft prefix (e.g., 'A', 'K', "A'").
    start_sec : float
        Start time.
    duration_sec : float
        Duration to plot.
    reference_type : str
        Reference type.
    **kwargs
        Additional arguments passed to plot_seeg_segment.
    
    Returns
    -------
    fig : matplotlib.Figure
    """
    # Find channels belonging to this shaft
    shaft_channels = []
    for ch in ch_names:
        if ElectrodeParser is not None:
            prefix, _ = ElectrodeParser.parse(ch)
            if prefix and prefix.upper() == shaft.upper():
                shaft_channels.append(ch)
        elif ch.startswith(shaft):
            shaft_channels.append(ch)
    
    if not shaft_channels:
        raise ValueError(f"No channels found for shaft '{shaft}'")
    
    return plot_seeg_segment(
        data=data,
        sfreq=sfreq,
        ch_names=ch_names,
        start_sec=start_sec,
        duration_sec=duration_sec,
        channels=shaft_channels,
        reference_type=reference_type,
        title=f'Shaft {shaft} ({len(shaft_channels)} channels)',
        **kwargs
    )


# =============================================================================
# Helper Functions
# =============================================================================

def _parse_channel_selection(
    channels: Union[str, List[str], List[int]],
    ch_names: List[str]
) -> Tuple[List[int], List[str]]:
    """Parse channel selection into indices and names."""
    if channels == 'all':
        return list(range(len(ch_names))), list(ch_names)
    
    if isinstance(channels, list):
        if len(channels) == 0:
            return [], []
        
        if isinstance(channels[0], int):
            # Index-based selection
            indices = [i for i in channels if 0 <= i < len(ch_names)]
            labels = [ch_names[i] for i in indices]
            return indices, labels
        
        else:
            # Name-based selection
            indices = []
            labels = []
            for ch in channels:
                if ch in ch_names:
                    indices.append(ch_names.index(ch))
                    labels.append(ch)
            return indices, labels
    
    raise ValueError(f"Invalid channel selection: {channels}")


def _get_shaft_colors(ch_names: List[str]) -> List[str]:
    """Assign colors based on electrode shaft."""
    colors = []
    shaft_color_map = {}
    color_idx = 0
    
    for ch in ch_names:
        prefix = _extract_shaft_prefix(ch)
        
        if prefix not in shaft_color_map:
            palette = tableau_20_no_red if len(tableau_20_no_red) > 0 else SEEG_COLORS["shaft_colors"]
            shaft_color_map[prefix] = palette[color_idx % len(palette)]
            color_idx += 1
        
        colors.append(shaft_color_map[prefix])
    
    return colors


def _extract_shaft_prefix(ch_name: str) -> str:
    """
    Extract shaft prefix from monopolar or bipolar names.
    Examples: "A1" -> "A", "A1-A2" -> "A", "A'1-A'2" -> "A'".
    """
    name = ch_name
    if "-" in name:
        name = name.split("-", 1)[0].strip()

    if ElectrodeParser is not None:
        prefix, _ = ElectrodeParser.parse(name)
        if prefix:
            return prefix

    # Fallback: take leading letters or first char
    for i, c in enumerate(name):
        if c.isdigit():
            return name[:i] if i > 0 else (name[:1] if name else "X")
    return name[:1] if name else "X"


def _get_reference_label(reference_type: str) -> str:
    """Get human-readable reference label."""
    labels = {
        'monopolar': 'Monopolar',
        'bipolar': 'Bipolar',
        'car': 'CAR',
        'pre_bipolar': 'Pre-Bipolar',
        'unknown': 'Unknown Ref'
    }
    return labels.get(reference_type, reference_type)


def _add_scale_bar(ax, time_vec, spacing, amplitude_scale):
    """Add time/amplitude scale bar."""
    # Position in bottom right
    x_pos = time_vec[-1] - (time_vec[-1] - time_vec[0]) * 0.08
    y_pos = ax.get_ylim()[0] + spacing * 0.5
    
    # Time bar (1 second)
    time_bar = 1.0
    ax.plot([x_pos, x_pos + time_bar], [y_pos, y_pos], 'k-', linewidth=2)
    ax.text(x_pos + time_bar/2, y_pos - spacing * 0.15, '1 s', 
            ha='center', va='top', fontsize=10)
    
    # Amplitude bar
    amp_bar = amplitude_scale
    ax.plot([x_pos, x_pos], [y_pos, y_pos + amp_bar], 'k-', linewidth=2)
    ax.text(x_pos - (time_vec[-1] - time_vec[0]) * 0.01, y_pos + amp_bar/2, 
            f'{amp_bar:.0f} µV', ha='right', va='center', fontsize=10, rotation=90)


def _overlay_events(ax, events, ch_labels, yticks, spacing, t_start, t_end):
    """Overlay HFO events on plot."""
    for event in events:
        evt_start = event.get('start', 0)
        evt_end = event.get('end', evt_start + 0.1)
        evt_ch = event.get('channel', None)
        evt_color = event.get('color', 'red')
        evt_alpha = float(event.get('alpha', 0.25))
        evt_edgecolor = event.get('edgecolor', evt_color)
        evt_linewidth = float(event.get('linewidth', 1.0))
        evt_style = event.get('style', 'tick')  # 'tick'|'box'|'span'
        
        # Skip events outside time window
        if evt_end < t_start or evt_start > t_end:
            continue
        
        # If channel specified, highlight only that channel
        if evt_ch and evt_ch in ch_labels:
            ch_idx = ch_labels.index(evt_ch)
            y_center = yticks[ch_idx]
            if evt_style == 'box':
                # Small box around the baseline (do NOT cover whole channel row)
                height = spacing * 0.25
                rect = Rectangle(
                    (evt_start, y_center - height / 2),
                    evt_end - evt_start,
                    height,
                    facecolor=evt_color,
                    alpha=evt_alpha,
                    edgecolor=evt_edgecolor,
                    linewidth=evt_linewidth,
                    zorder=3,
                )
                ax.add_patch(rect)
            else:
                # Default: thin tick line slightly above waveform baseline.
                y = y_center + spacing * 0.18
                ax.plot(
                    [evt_start, evt_end],
                    [y, y],
                    color=evt_color,
                    alpha=evt_alpha,
                    linewidth=max(1.0, evt_linewidth),
                    solid_capstyle='butt',
                    zorder=3,
                )
        else:
            # Highlight across all channels
            # Only use full-span highlight when explicitly requested.
            if evt_style == 'span':
                ax.axvspan(evt_start, evt_end, alpha=evt_alpha, color=evt_color, zorder=1)


def detections_to_events(
    det_result,
    color: str = "#e94560",
    alpha: float = 0.25,
    max_events_per_channel: Optional[int] = None,
    style: str = "tick",
    linewidth: float = 2.0,
) -> List[Dict]:
    """
    Convert HFODetectionResult -> event dicts usable by plot_seeg_segment(..., events=...).

    Notes
    -----
    - det_result.events_by_channel is a list of (n_events, 2) arrays in seconds.
    - For performance on dense detections, you can cap events per channel.
    """
    if det_result is None:
        return []
    if not hasattr(det_result, "events_by_channel") or not hasattr(det_result, "ch_names"):
        raise ValueError("det_result must look like HFODetectionResult (events_by_channel + ch_names).")

    events: List[Dict] = []
    for ch_name, ev in zip(list(det_result.ch_names), list(det_result.events_by_channel)):
        if ev is None:
            continue
        ev = np.asarray(ev)
        if ev.size == 0:
            continue
        if ev.ndim != 2 or ev.shape[1] != 2:
            raise ValueError(f"Invalid event array for channel {ch_name}: shape={ev.shape}")

        if max_events_per_channel is not None and ev.shape[0] > max_events_per_channel:
            ev = ev[:max_events_per_channel, :]

        for s, e in ev:
            events.append(
                {
                    "start": float(s),
                    "end": float(e),
                    "channel": ch_name,
                    "color": color,
                    "alpha": alpha,
                    "style": style,
                    "linewidth": linewidth,
                }
            )
    return events


def plot_group_events_band_raster(
    *,
    cache_npz_path: str,
    packed_times_path: str,
    channel_order: Optional[List[str]] = None,
    event_indices: Optional[List[int]] = None,
    max_events: int = 30,
    mode: str = "envelope",  # 'envelope'|'bandpassed_abs'|'bandpassed'
    plot_style: str = "imshow",  # 'imshow'|'trace'
    robust_scale: bool = True,
    cmap: str = "gray",
    x_axis: str = "event_index",  # 'seconds'|'samples'|'event_index'
    downsample_ms: Optional[float] = 5.0,
    value_transform: str = "log1p",  # 'none'|'log1p'
    trace_normalize: str = "p99",  # 'none'|'mad'|'p99'
    trace_scale_percentile: float = 99.0,
    trace_lw: float = 0.6,
    trace_alpha: float = 0.9,
    trace_spacing: float = 6.0,
    show_event_boundaries: bool = True,
    boundary_lw: float = 0.3,
    boundary_alpha: float = 0.4,
    figsize: Tuple[float, float] = (12, 8),
) -> plt.Figure:
    """
    Figure 1 (paper-like): 80-250Hz (or cached band) bandpassed raster.

    X-axis: concatenated group-event windows (packedTimes order)
    Y-axis: channels (typically bipolar/aliased naming)

    Requires cache created by:
      precompute_envelope_cache(..., save_bandpass=True)
    """
    from .group_event_analysis import load_envelope_cache

    meta = load_envelope_cache(cache_npz_path)
    env = meta.get("env", None)
    x_band = meta.get("x_band", None)
    if mode not in ("envelope", "bandpassed_abs", "bandpassed"):
        raise ValueError("mode must be 'envelope', 'bandpassed_abs', or 'bandpassed'")
    if mode in ("bandpassed_abs", "bandpassed") and x_band is None:
        raise ValueError("Cache does not contain x_band. Regenerate with save_bandpass=True or use mode='envelope'.")
    if plot_style not in ("imshow", "trace"):
        raise ValueError("plot_style must be 'imshow' or 'trace'")

    sfreq = float(meta["sfreq"])
    ch_names = list(meta["ch_names"])

    packed = np.load(packed_times_path, allow_pickle=True)
    if event_indices is None:
        event_indices = list(range(min(int(max_events), packed.shape[0])))
    else:
        event_indices = [int(i) for i in event_indices][: int(max_events)]

    if channel_order is None:
        channel_order = ch_names
    # Map order
    idx_map = {c: i for i, c in enumerate(ch_names)}
    rows = [idx_map[c] for c in channel_order if c in idx_map]
    labels = [c for c in channel_order if c in idx_map]
    if not rows:
        raise ValueError("No channels matched channel_order.")

    # Build concatenated matrix: (n_ch, total_samples)
    segs = []
    seg_lens = []
    for ei in event_indices:
        s, e = float(packed[ei, 0]), float(packed[ei, 1])
        i0 = int(round(s * sfreq))
        i1 = int(round(e * sfreq))
        if mode == "envelope":
            seg = np.asarray(env[rows, i0:i1])
        elif mode == "bandpassed_abs":
            seg = np.abs(np.asarray(x_band[rows, i0:i1]))
        else:
            seg = np.asarray(x_band[rows, i0:i1])
        # Optional downsample for readability (reduce high-frequency texture)
        if downsample_ms is not None and downsample_ms > 0:
            bin_n = max(1, int(round((downsample_ms * 1e-3) * sfreq)))
            n = seg.shape[1] // bin_n * bin_n
            if n >= bin_n:
                seg = seg[:, :n].reshape(seg.shape[0], -1, bin_n).mean(axis=-1)
        segs.append(seg)
        seg_lens.append(seg.shape[1])
    X = np.concatenate(segs, axis=1) if segs else np.zeros((len(rows), 0))

    # Robust normalization per channel (mainly for imshow; for trace we do a separate, waveform-safe normalization)
    Xn = X.astype(np.float64, copy=False)
    if value_transform not in ("none", "log1p"):
        raise ValueError("value_transform must be 'none' or 'log1p'")
    if value_transform == "log1p" and mode == "bandpassed":
        # bandpassed traces are signed; log1p would silently clip negatives and distort the waveform
        value_transform = "none"
    if plot_style == "trace" and value_transform == "log1p":
        # Trace view should look like a waveform, not a compressed heatmap.
        value_transform = "none"
    if value_transform == "log1p":
        Xn = np.log1p(np.maximum(Xn, 0.0))
    if robust_scale and Xn.size > 0:
        med = np.median(Xn, axis=1, keepdims=True)
        mad = np.median(np.abs(Xn - med), axis=1, keepdims=True) + 1e-9
        Xn = (Xn - med) / mad
        # clip for visualization stability
        Xn = np.clip(Xn, -5.0, 5.0)

    fig, ax = plt.subplots(figsize=figsize)
    # x-axis scaling
    if x_axis not in ("seconds", "samples", "event_index"):
        raise ValueError("x_axis must be 'seconds', 'samples', or 'event_index'")
    # If downsampled, adjust effective sfreq for x-axis scaling
    eff_sfreq = sfreq
    if downsample_ms is not None and downsample_ms > 0:
        bin_n = max(1, int(round((downsample_ms * 1e-3) * sfreq)))
        eff_sfreq = sfreq / float(bin_n)

    if x_axis == "seconds":
        xs = np.arange(Xn.shape[1], dtype=np.float64) / eff_sfreq
        xlabel = "Concatenated event time (s)"
    elif x_axis == "samples":
        xs = np.arange(Xn.shape[1], dtype=np.float64)
        xlabel = "Concatenated event samples"
    else:
        xs = np.arange(Xn.shape[1], dtype=np.float64) / eff_sfreq
        xlabel = "Event index (packedTimes order)"

    if plot_style == "imshow":
        extent = (float(xs[0]) if xs.size else 0.0, float(xs[-1]) if xs.size else 0.0, float(len(labels)), 0.0)
        ax.imshow(
            Xn,
            aspect="auto",
            cmap=cmap,
            interpolation="nearest",
            origin="upper",
            extent=extent,
        )
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_ylabel("Channels")
    else:
        # Trace view: plot bandpassed waveform (or envelope) as stacked lines
        # Normalize per channel to avoid one channel dominating, without destroying waveform shape.
        if trace_normalize not in ("none", "mad", "p99"):
            raise ValueError("trace_normalize must be 'none', 'mad', or 'p99'")
        Y = X.astype(np.float64, copy=False)  # IMPORTANT: use pre-transform data for waveform fidelity
        if Y.size > 0:
            # Center each channel to remove DC drift in the concatenated timeline.
            Y = Y - np.median(Y, axis=1, keepdims=True)
            if trace_normalize == "mad":
                med = np.median(Y, axis=1, keepdims=True)
                mad = np.median(np.abs(Y - med), axis=1, keepdims=True) + 1e-9
                Y = Y / mad
            elif trace_normalize == "p99":
                p = float(trace_scale_percentile)
                p = min(max(p, 50.0), 100.0)
                scale = np.nanpercentile(np.abs(Y), p, axis=1, keepdims=True) + 1e-9
                Y = Y / scale
        for ci, lab in enumerate(labels):
            y = Y[ci]
            offset = float(ci) * float(trace_spacing)
            ax.plot(xs, y + offset, color="k", linewidth=float(trace_lw), alpha=float(trace_alpha))
        ax.set_yticks([float(i) * float(trace_spacing) for i in range(len(labels))])
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_ylabel("Channels (stacked)")

    # event boundaries
    if show_event_boundaries:
        cur = 0.0
        for n in seg_lens:
            xline = cur if x_axis == "samples" else cur / eff_sfreq
            ax.axvline(xline, color="k", linewidth=float(boundary_lw), alpha=float(boundary_alpha))
            cur += float(n)

    if x_axis == "event_index" and seg_lens:
        centers = []
        cur = 0.0
        for n in seg_lens:
            centers.append(cur + 0.5 * float(n))
            cur += float(n)
        centers = np.asarray(centers, dtype=np.float64) / eff_sfreq
        labels_evt = [str(int(i) + 1) for i in event_indices]
        ax.set_xticks(centers)
        ax.set_xticklabels(labels_evt, fontsize=8)

    # style polish: remove useless whitespace + frame clutter
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.0)

    ax.set_title(f"{mode} ({plot_style}) — concatenated events", fontweight="bold")
    ax.set_xlabel(xlabel)
    plt.tight_layout()
    return fig


def build_event_spans_from_packed_times(
    packed_times_path: str,
    *,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
    max_events: Optional[int] = None,
    color: str = "#e94560",
    alpha: float = 0.15,
    style: str = "span",
) -> List[Dict]:
    """
    Build event spans (for plot_seeg_segment events=...) from packedTimes.

    Parameters
    ----------
    packed_times_path : str
        Path to *_packedTimes.npy
    start_sec, end_sec : float, optional
        Only include events overlapping [start_sec, end_sec]
    max_events : int, optional
        Limit number of events for visualization
    color, alpha, style : str/float
        Event drawing style for plot_seeg_segment
    """
    packed = np.load(packed_times_path, allow_pickle=True)
    spans: List[Dict] = []
    count = 0
    for s, e in packed:
        s = float(s)
        e = float(e)
        if start_sec is not None and e < float(start_sec):
            continue
        if end_sec is not None and s > float(end_sec):
            continue
        spans.append(
            {
                "start": s,
                "end": e,
                "color": color,
                "alpha": float(alpha),
                "style": style,
            }
        )
        count += 1
        if max_events is not None and count >= int(max_events):
            break
    return spans


def plot_bandpassed_segment_from_cache(
    *,
    cache_npz_path: str,
    packed_times_path: str,
    start_sec: float = 0.0,
    duration_sec: float = 10.0,
    channels: Union[str, List[str], List[int]] = "all",
    mode: str = "bandpassed",  # "bandpassed"|"envelope"
    color_by_shaft: bool = True,
    event_alpha: float = 0.15,
    max_events: Optional[int] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """
    Plot a bandpassed waveform (or envelope) segment from *_envCache.npz with event shading.
    """
    from .group_event_analysis import load_envelope_cache

    meta = load_envelope_cache(cache_npz_path)
    sfreq = float(meta["sfreq"])
    ch_names = list(meta["ch_names"])
    x_band = meta.get("x_band", None)
    env = meta.get("env", None)

    if mode not in ("bandpassed", "envelope"):
        raise ValueError("mode must be 'bandpassed' or 'envelope'")
    if mode == "bandpassed" and x_band is None:
        raise ValueError("Cache does not contain x_band. Regenerate with save_bandpass=True.")
    if mode == "envelope" and env is None:
        raise ValueError("Cache does not contain env.")

    data = x_band if mode == "bandpassed" else env
    events = build_event_spans_from_packed_times(
        packed_times_path,
        start_sec=float(start_sec),
        end_sec=float(start_sec + duration_sec),
        max_events=max_events,
        alpha=float(event_alpha),
        style="span",
    )
    return plot_seeg_segment(
        data=data,
        sfreq=sfreq,
        ch_names=ch_names,
        start_sec=float(start_sec),
        duration_sec=float(duration_sec),
        channels=channels,
        title=title,
        reference_type="bipolar",
        color_by_shaft=bool(color_by_shaft),
        events=events,
        figsize=figsize,
    )


def plot_channel_waveform_and_envelope_from_cache(
    *,
    cache_npz_path: str,
    packed_times_path: str,
    channel: str,
    start_sec: float = 0.0,
    duration_sec: float = 2.0,
    event_alpha: float = 0.15,
    max_events: Optional[int] = None,
    figsize: Tuple[float, float] = (12, 4),
    waveform_color: str = "#111111",
    envelope_color: str = "#1f77b4",
) -> plt.Figure:
    """
    Two-panel view:
      Left  - bandpassed waveform (x_band)
      Right - envelope (env)
    Both panels overlay group-event shading.
    """
    from .group_event_analysis import load_envelope_cache

    meta = load_envelope_cache(cache_npz_path)
    sfreq = float(meta["sfreq"])
    ch_names = list(meta["ch_names"])
    x_band = meta.get("x_band", None)
    env = meta.get("env", None)

    if x_band is None or env is None:
        raise ValueError("Cache must contain both x_band and env.")
    if channel not in ch_names:
        raise ValueError(f"Channel '{channel}' not found in cache.")

    idx = ch_names.index(channel)
    s0 = int(round(float(start_sec) * sfreq))
    s1 = int(round(float(start_sec + duration_sec) * sfreq))
    s1 = min(s1, x_band.shape[1])

    t = np.arange(s0, s1) / sfreq
    x = x_band[idx, s0:s1]
    e = env[idx, s0:s1]

    events = build_event_spans_from_packed_times(
        packed_times_path,
        start_sec=float(start_sec),
        end_sec=float(start_sec + duration_sec),
        max_events=max_events,
        alpha=float(event_alpha),
        style="span",
    )

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)
    ax0, ax1 = axes

    ax0.plot(t, x, color=waveform_color, linewidth=0.8)
    ax0.set_title(f"{channel} bandpassed waveform", fontweight="bold")
    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel("Amplitude")

    ax1.plot(t, e, color=envelope_color, linewidth=0.8)
    ax1.set_title(f"{channel} envelope", fontweight="bold")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Envelope")

    for ev in events:
        ax0.axvspan(ev["start"], ev["end"], color=ev["color"], alpha=ev["alpha"])
        ax1.axvspan(ev["start"], ev["end"], color=ev["color"], alpha=ev["alpha"])

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    return fig

def plot_paper_fig1_bandpassed_traces(
    *,
    cache_npz_path: str,
    packed_times_path: str,
    channel_order: Optional[List[str]] = None,
    event_indices: Optional[List[int]] = None,
    max_events: int = 30,
    downsample_ms: Optional[float] = None,
    figsize: Tuple[float, float] = (16, 10),
) -> plt.Figure:
    """
    Paper-like Fig1:
      - bandpassed waveform traces (80–250Hz in ripple cache)
      - channels stacked (y)
      - concatenated packedTimes windows (x)

    This is a thin wrapper around `plot_group_events_band_raster` with "paper defaults".
    """
    fig = plot_group_events_band_raster(
        cache_npz_path=cache_npz_path,
        packed_times_path=packed_times_path,
        channel_order=channel_order,
        event_indices=event_indices,
        max_events=int(max_events),
        mode="bandpassed",
        plot_style="trace",
        robust_scale=False,  # waveform view should not use MAD z-scoring by default
        value_transform="none",
        downsample_ms=downsample_ms,
        x_axis="event_index",
        trace_normalize="p99",
        trace_scale_percentile=99.0,
        trace_lw=0.55,
        trace_alpha=0.9,
        trace_spacing=3.2,
        show_event_boundaries=True,
        boundary_lw=0.3,
        boundary_alpha=0.35,
        figsize=figsize,
    )
    ax = fig.axes[0]
    # Typography + frame polish: close to typical paper Fig1 style
    ax.set_title("80–250 Hz bandpassed traces (concatenated events)", fontweight="bold")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.margins(x=0.0)
    fig.tight_layout()
    return fig


def plot_raw_segment_from_cache(
    *,
    raw_cache_npz_path: str,
    packed_times_path: str,
    start_sec: float = 0.0,
    duration_sec: float = 10.0,
    channels: Union[str, List[str], List[int]] = "all",
    color_by_shaft: bool = True,
    event_alpha: float = 0.15,
    max_events: Optional[int] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """
    Plot raw waveform segment from *_rawCache_*.npz with group-event shading.
    """
    from .preprocessing import load_raw_cache

    meta = load_raw_cache(raw_cache_npz_path)
    data = np.asarray(meta["data"])
    sfreq = float(np.asarray(meta["sfreq"]).ravel()[0])
    ch_names = [str(x) for x in meta["ch_names"]]

    events = build_event_spans_from_packed_times(
        packed_times_path,
        start_sec=float(start_sec),
        end_sec=float(start_sec + duration_sec),
        max_events=max_events,
        alpha=float(event_alpha),
        style="span",
    )

    return plot_seeg_segment(
        data=data,
        sfreq=sfreq,
        ch_names=ch_names,
        start_sec=float(start_sec),
        duration_sec=float(duration_sec),
        channels=channels,
        title=title,
        reference_type=str(np.asarray(meta.get("reference_type", ["unknown"])).ravel()[0]),
        color_by_shaft=bool(color_by_shaft),
        events=events,
        figsize=figsize,
    )


def plot_raw_segment_from_edf(
    *,
    edf_path: str,
    packed_times_path: str,
    start_sec: float = 0.0,
    duration_sec: float = 10.0,
    channels: Union[str, List[str], List[int]] = "all",
    reference: str = "none",
    color_by_shaft: bool = True,
    event_alpha: float = 0.15,
    max_events: Optional[int] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """
    Plot raw EDF waveform segment with group-event shading.

    This uses SEEGPreprocessor(reference='none') to read raw data.
    """
    try:
        from .preprocessing import SEEGPreprocessor
    except Exception as exc:
        raise ImportError("SEEGPreprocessor unavailable; ensure dependencies (mne) are installed.") from exc

    pre = SEEGPreprocessor(reference=reference, crop_seconds=float(start_sec + duration_sec))
    res = pre.run(str(edf_path))
    events = build_event_spans_from_packed_times(
        packed_times_path,
        start_sec=float(start_sec),
        end_sec=float(start_sec + duration_sec),
        max_events=max_events,
        alpha=float(event_alpha),
        style="span",
    )
    return plot_seeg_segment(
        data=res.data,
        sfreq=res.sfreq,
        ch_names=res.ch_names,
        start_sec=float(start_sec),
        duration_sec=float(duration_sec),
        channels=channels,
        title=title,
        reference_type=res.reference_type,
        color_by_shaft=bool(color_by_shaft),
        events=events,
        figsize=figsize,
    )

def plot_group_event_tf_propagation_from_cache(
    *,
    tfr_tile_cache_npz_path: str,
    group_analysis_npz_path: Optional[str] = None,
    channel_order: Optional[List[str]] = None,
    event_indices: Optional[List[int]] = None,
    show_centroids: bool = True,
    show_paths: bool = True,
    centroid_marker_size: float = 25.0,
    centroid_marker_color: str = "black",
    centroid_line_width: float = 1.2,
    centroid_line_alpha: float = 0.6,
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    vmin_percentile: float = 5.0,
    vmax_percentile: float = 99.5,
    mask_percentile: Optional[float] = None,
    smooth_sigma: Optional[Tuple[float, float]] = (0.8, 2.0),
    time_downsample: int = 1,
    freq_downsample: int = 1,
    plot_window_sec: Optional[float] = None,
    low_color: Optional[str] = "#1f4b99",
    low_color_percentile: float = 70.0,
    scale_bar_sec: Optional[float] = 0.05,
    scale_bar_label: Optional[str] = None,
    freq_scale_bar_hz: Optional[float] = 50.0,
    base_fontsize: float = 12.0,
    interpolation: str = "bicubic",
    figsize: Tuple[float, float] = (14, 8),
) -> plt.Figure:
    """
    Plot multi-channel multi-event TF propagation from a cached TF tile cube.
    """
    from .group_event_analysis import load_group_analysis_results

    tfr = np.load(str(tfr_tile_cache_npz_path), allow_pickle=True)
    freqs_hz = np.asarray(tfr["freqs_hz"], dtype=np.float64)
    time_axis = np.asarray(tfr["time_axis"], dtype=np.float64)
    power_db = np.asarray(tfr["power_db"], dtype=np.float64)
    cache_event_indices = np.asarray(tfr.get("event_indices", np.arange(power_db.shape[1])), dtype=np.int64)
    cache_ch_names = [str(x) for x in np.asarray(tfr.get("channel_names", []), dtype=object).tolist()]
    window_sec = float(np.asarray(tfr.get("window_sec", np.array([time_axis[-1] if time_axis.size else 0.0]))).ravel()[0])

    if channel_order is None:
        channel_order = cache_ch_names
    channel_order = [str(c) for c in channel_order]
    ch_idx = {n: i for i, n in enumerate(cache_ch_names)}
    rows = [ch_idx[c] for c in channel_order if c in ch_idx]
    labels = [c for c in channel_order if c in ch_idx]
    if not rows:
        raise ValueError("No channels matched in TF tile cache.")

    if event_indices is None:
        pick = np.arange(cache_event_indices.shape[0])
        event_labels = cache_event_indices
    else:
        idx_map = {int(e): i for i, e in enumerate(cache_event_indices.tolist())}
        pick = [idx_map[int(e)] for e in event_indices if int(e) in idx_map]
        event_labels = np.asarray([int(e) for e in event_indices if int(e) in idx_map], dtype=np.int64)
    if len(pick) == 0:
        raise ValueError("No matching events found in TF tile cache.")

    power_db = power_db[np.asarray(rows, dtype=int)][:, np.asarray(pick, dtype=int)]
    n_ch = power_db.shape[0]
    n_events = power_db.shape[1]

    Z = power_db[np.isfinite(power_db)]
    if vmin is None or vmax is None:
        if Z.size > 0:
            p99 = float(np.nanpercentile(np.abs(Z), float(vmax_percentile)))
            if vmax is None:
                vmax = p99
            if vmin is None:
                vmin = -p99 * 0.2
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmin, vmax = -5.0, 10.0
        else:
            vmin, vmax = -5.0, 10.0

    norm = None
    low_thresh = None
    use_low_color = low_color is not None
    if use_low_color:
        Zpos = Z[Z > 0]
        if Zpos.size > 0:
            low_thresh = float(np.nanpercentile(Zpos, float(low_color_percentile)))
        else:
            low_thresh = float(vmin)
        norm = matplotlib.colors.Normalize(vmin=float(low_thresh), vmax=float(vmax))
    elif float(vmin) < 0.0 < float(vmax):
        norm = matplotlib.colors.TwoSlopeNorm(vmin=float(vmin), vcenter=0.0, vmax=float(vmax))

    mask_thresh = None
    if mask_percentile is not None and Z.size > 0:
        mask_thresh = float(np.nanpercentile(np.abs(Z), float(mask_percentile)))

    time_mask = None
    plot_win = None
    if plot_window_sec is not None and time_axis.size > 0:
        plot_win = float(plot_window_sec)
        center_t = 0.5 * float(window_sec)
        half = 0.5 * plot_win
        time_mask = (time_axis >= (center_t - half)) & (time_axis <= (center_t + half))
        if not np.any(time_mask):
            time_mask = None
            plot_win = None

    fig, ax = plt.subplots(figsize=figsize)
    fmin = float(np.nanmin(freqs_hz)) if freqs_hz.size else 0.0
    fmax = float(np.nanmax(freqs_hz)) if freqs_hz.size else 1.0
    freq_span = max(1e-9, fmax - fmin)

    for ci in range(n_ch):
        y0 = float(ci)
        y1 = float(ci) + 0.95
        for ei in range(n_events):
            tile = power_db[ci, ei]
            if not np.isfinite(tile).any():
                continue
            if smooth_sigma is not None:
                from scipy.ndimage import gaussian_filter

                tile_smooth = gaussian_filter(tile, sigma=tuple(smooth_sigma))
            else:
                tile_smooth = tile
            tile_display = np.array(tile_smooth, copy=True)
            tile_display[~np.isfinite(tile_display)] = np.nan
            if time_downsample and tile_display.shape[1] >= int(time_downsample):
                ds = int(time_downsample)
                new_len = tile_display.shape[1] // ds
                if new_len > 0:
                    trimmed = tile_display[:, : new_len * ds]
                    tile_display = trimmed.reshape(trimmed.shape[0], new_len, ds).mean(axis=2)
            if freq_downsample and tile_display.shape[0] >= int(freq_downsample):
                fs = int(freq_downsample)
                new_len_f = tile_display.shape[0] // fs
                if new_len_f > 0:
                    trimmed_f = tile_display[: new_len_f * fs, :]
                    tile_display = trimmed_f.reshape(new_len_f, fs, trimmed_f.shape[1]).mean(axis=1)
            if time_mask is not None:
                tile_display = tile_display[:, time_mask[: tile_display.shape[1]]]
            if mask_thresh is not None:
                tile_display[np.abs(tile_display) < mask_thresh] = np.nan
            if use_low_color and norm is not None:
                tile_display = np.where(
                    np.isfinite(tile_display),
                    tile_display,
                    np.nan,
                )
                tile_display = np.where(tile_display < float(norm.vmin), float(norm.vmin) - 1e-6, tile_display)
            if plot_win is None:
                x0 = float(ei) * float(window_sec)
                x1 = x0 + float(window_sec)
            else:
                x0 = float(ei) * float(plot_win)
                x1 = x0 + float(plot_win)
            if use_low_color:
                cmap_obj = plt.get_cmap(str(cmap)).copy()
                cmap_obj.set_under(str(low_color))
            else:
                cmap_obj = str(cmap)
            ax.imshow(
                tile_display,
                origin="lower",
                aspect="auto",
                interpolation=str(interpolation),
                cmap=cmap_obj,
                norm=norm,
                vmin=float(vmin) if norm is None else None,
                vmax=float(vmax) if norm is None else None,
                extent=(x0, x1, y0, y1),
            )

    for ei in range(n_events + 1):
        step = float(window_sec) if plot_win is None else float(plot_win)
        x = float(ei) * step
        ax.axvline(x, color="white", linewidth=1.5, linestyle="-", zorder=2)

    if use_low_color:
        cbar_cmap = plt.get_cmap(str(cmap)).copy()
        cbar_cmap.set_under(str(low_color))
    else:
        cbar_cmap = str(cmap)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cbar_cmap)
    cbar = fig.colorbar(
        mappable,
        ax=ax,
        fraction=0.015,
        pad=0.02,
        extend="min" if use_low_color else "neither",
    )
    cbar.set_label("Power (dB)", rotation=270, labelpad=15, fontsize=base_fontsize)
    cbar.ax.tick_params(labelsize=base_fontsize)
    if use_low_color and low_thresh is not None:
        cbar.ax.text(
            0.5,
            -0.06,
            f"<{low_thresh:.2f} dB",
            transform=cbar.ax.transAxes,
            ha="center",
            va="top",
            fontsize=base_fontsize - 2,
        )

    ax.set_ylim(0, n_ch)
    ax.set_yticks([i + 0.5 for i in range(n_ch)])
    ax.set_yticklabels(labels, fontsize=base_fontsize, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Time (s) - Concatenated Events", fontsize=base_fontsize)
    ax.set_title("Normalized Spectrogram (80-250 Hz)", fontweight="bold", fontsize=base_fontsize + 2)
    ax.tick_params(axis="x", length=0, labelsize=base_fontsize)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if scale_bar_sec is not None and float(scale_bar_sec) > 0:
        step = float(window_sec) if plot_win is None else float(plot_win)
        total_time = float(step) * float(n_events)
        bar_len = min(float(scale_bar_sec), total_time * 0.5)
        x_end = total_time - 0.02 * total_time
        x_start = x_end - bar_len
        y = float(n_ch) - 0.2
        ax.plot([x_start, x_end], [y, y], color="black", linewidth=4.0, solid_capstyle="butt")
        label = scale_bar_label or f"{bar_len * 1000:.0f} ms"
        ax.text(
            (x_start + x_end) * 0.5,
            min(float(n_ch) - 0.02, y + 0.15),
            label,
            ha="center",
            va="top",
            fontsize=base_fontsize,
            color="black",
            fontweight="bold",
        )

    if freq_scale_bar_hz is not None and float(freq_scale_bar_hz) > 0:
        bar_hz = min(float(freq_scale_bar_hz), freq_span * 0.5)
        step = float(window_sec) if plot_win is None else float(plot_win)
        total_time = float(step) * float(n_events)
        x = total_time - 0.05 * total_time
        y_bottom = float(n_ch) - 0.25
        y_top = y_bottom - (bar_hz / freq_span) * 0.95
        ax.plot([x, x], [y_top, y_bottom], color="black", linewidth=4.0, solid_capstyle="butt")
        ax.text(
            x - 0.01 * total_time,
            (y_top + y_bottom) * 0.5,
            f"{bar_hz:.0f} Hz",
            ha="right",
            va="center",
            fontsize=base_fontsize,
            color="black",
            fontweight="bold",
        )

    if show_centroids and group_analysis_npz_path is not None:
        ga = load_group_analysis_results(group_analysis_npz_path)
        tf_centroid_time = np.asarray(ga.get("tf_centroid_time", np.zeros((0, 0))), dtype=np.float64)
        tf_centroid_freq = np.asarray(ga.get("tf_centroid_freq", np.zeros((0, 0))), dtype=np.float64)
        events_bool = np.asarray(ga.get("events_bool", np.zeros((0, 0))), dtype=bool)
        ga_ch_names = [str(x) for x in ga.get("ch_names", [])]
        ga_idx = {n: i for i, n in enumerate(ga_ch_names)}

        for ei, ev in enumerate(event_labels):
            xs = []
            ys = []
            for ci, ch in enumerate(labels):
                gi = ga_idx.get(ch, None)
                if gi is None or ev >= tf_centroid_time.shape[1]:
                    continue
                if events_bool.size > 0 and not bool(events_bool[gi, ev]):
                    continue
                t_c = float(tf_centroid_time[gi, ev])
                f_c = float(tf_centroid_freq[gi, ev])
                if not np.isfinite(t_c) or not np.isfinite(f_c):
                    continue
                if plot_win is None:
                    x = float(ei) * float(window_sec) + t_c
                else:
                    center_t = 0.5 * float(window_sec)
                    half = 0.5 * float(plot_win)
                    t_plot = t_c - (center_t - half)
                    if t_plot < 0.0 or t_plot > plot_win:
                        continue
                    x = float(ei) * float(plot_win) + t_plot
                rel_f = (f_c - fmin) / freq_span
                y = float(ci) + rel_f * 0.95
                if 0.0 <= y <= float(n_ch):
                    xs.append(x)
                    ys.append(y)
            if xs:
                ax.scatter(
                    xs,
                    ys,
                    s=float(centroid_marker_size),
                    facecolors="none",
                    edgecolors=centroid_marker_color,
                    linewidth=2.5,
                    zorder=10,
                )
                if show_paths and len(xs) > 1:
                    ax.plot(
                        xs,
                        ys,
                        color=centroid_marker_color,
                        linewidth=float(centroid_line_width),
                        alpha=float(centroid_line_alpha),
                        zorder=9,
                    )

    plt.tight_layout()
    return fig


def plot_lag_heatmaps(
    *,
    energy: np.ndarray,
    lag_ms: np.ndarray,
    rank: np.ndarray,
    ch_names: List[str],
    event_ids: List[int],
    cmap_energy: str = "viridis",
    cmap_lag: str = "viridis",
    cmap_rank: str = "magma",
    figsize: Tuple[float, float] = (14, 4),
) -> Tuple[plt.Figure, plt.Figure, plt.Figure]:
    """
    Plot 3 heatmaps (channels × events):
      1) energy (envelope energy per window)
      2) rank (0=earliest), -1 masked
      3) lag (ms, aligned to first centroid), NaN masked
    """
    energy = np.asarray(energy, dtype=np.float64)
    lag_ms = np.asarray(lag_ms, dtype=np.float64)
    rank = np.asarray(rank, dtype=np.float64)

    def _imshow(mat, title, cmap, vmin=None, vmax=None, bad_color=None):
        fig, ax = plt.subplots(figsize=figsize)
        if isinstance(cmap, str):
            cmap_obj = plt.get_cmap(cmap).copy()
        else:
            cmap_obj = cmap
        if bad_color is not None and hasattr(cmap_obj, "set_bad"):
            cmap_obj.set_bad(bad_color)
        im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap_obj, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Channels")
        ax.set_xlabel("Event id")
        ax.set_yticks(np.arange(len(ch_names)))
        ax.set_yticklabels(ch_names, fontsize=8)
        # keep x ticks sparse
        if len(event_ids) <= 50:
            ax.set_xticks(np.arange(len(event_ids)))
            ax.set_xticklabels([str(e) for e in event_ids], fontsize=7, rotation=90)
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        plt.tight_layout()
        return fig

    # energy: robust clip for readability
    e = energy.copy()
    if np.isfinite(e).any():
        vmax = np.nanpercentile(e, 99)
    else:
        vmax = None
    fig1 = _imshow(e, "Event envelope energy (normalized; channels × events)", cmap_energy, vmin=0.0, vmax=vmax)

    r = rank.copy()
    r[~np.isfinite(r)] = np.nan
    r[r < 0] = np.nan
    if np.isfinite(r).any():
        vmax_r = int(np.nanmax(r))
    else:
        vmax_r = None
    fig2 = _imshow(
        r,
        "Centroid order (rank; 0=earliest)",
        cmap_rank,
        vmin=0,
        vmax=vmax_r,
        bad_color="#d9d9d9",
    )

    l = lag_ms.copy()
    l[~np.isfinite(l)] = np.nan
    # clamp to physiological window for readability
    if np.isfinite(l).any():
        v = np.nanpercentile(np.abs(l), 99)
        v = max(v, 1.0)
    else:
        v = None
    if v is not None and np.nanmin(l) >= 0:
        vmin = 0.0
    else:
        vmin = -v if v is not None else None
    fig3 = _imshow(l, "Lag (ms) aligned to first centroid", cmap_lag, vmin=vmin, vmax=v)
    return fig1, fig2, fig3


def plot_lag_heatmaps_from_group_analysis(
    *,
    group_analysis_npz: str,
    env_cache_npz: str,
    packed_times_npy: str,
    channel_names: Optional[List[str]] = None,
    max_events: int = 100,
    cmap_energy: str = "viridis",
    cmap_lag: str = "RdYlBu_r",
    cmap_rank: str = "magma",
    figsize: Tuple[float, float] = (14, 4),
) -> Tuple[plt.Figure, plt.Figure, plt.Figure]:
    """
    Plot lag/rank/energy heatmaps directly from *_groupAnalysis.npz + *_envCache.npz.
    
    This is the HIGH-LEVEL convenience function: it reads pre-computed results and plots.
    NO computation is done here (following the architecture refactor).
    
    Parameters
    ----------
    group_analysis_npz : str
        Path to *_groupAnalysis.npz (contains lag_raw, lag_rank, events_bool, etc.)
    env_cache_npz : str
        Path to *_envCache.npz (contains envelope for energy calculation)
    packed_times_npy : str
        Path to *_packedTimes.npy (event windows)
    channel_names : List[str], optional
        Subset of channels to plot. If None, use all channels from group analysis.
    max_events : int
        Maximum number of events to plot (default 100)
    cmap_energy, cmap_lag, cmap_rank : str
        Colormaps for each heatmap
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig_energy, fig_rank, fig_lag : plt.Figure
        Three matplotlib figures
    """
    from .group_event_analysis import load_group_analysis_results, load_envelope_cache
    
    # Load pre-computed results
    ga = load_group_analysis_results(group_analysis_npz)
    cache = load_envelope_cache(env_cache_npz)
    packed = np.load(packed_times_npy, allow_pickle=True)
    
    # Select channels
    if channel_names is None:
        channel_names = ga['ch_names']
        ch_indices = list(range(ga['n_channels']))
    else:
        ch_name_to_idx = {n: i for i, n in enumerate(ga['ch_names'])}
        ch_indices = [ch_name_to_idx[ch] for ch in channel_names if ch in ch_name_to_idx]
        channel_names = [ch for ch in channel_names if ch in ch_name_to_idx]
    
    if len(ch_indices) == 0:
        raise ValueError("No matching channels found")
    
    # Limit events
    n_events = min(max_events, ga['n_events'])
    
    # Extract matrices
    lag_raw = ga['lag_raw'][ch_indices, :n_events]
    lag_rank = ga['lag_rank'][ch_indices, :n_events]
    events_bool = ga['events_bool'][ch_indices, :n_events]
    
    # Compute energy from envelope cache
    sf = cache['sfreq']
    cache_name_to_idx = {n: i for i, n in enumerate(cache['ch_names'])}
    cache_ch_indices = [cache_name_to_idx.get(ch) for ch in channel_names]
    cache_ch_indices = [i for i in cache_ch_indices if i is not None]
    
    energy = np.full((len(ch_indices), n_events), np.nan, dtype=np.float64)
    for ei in range(n_events):
        if ei >= len(packed):
            break
        s, e = float(packed[ei, 0]), float(packed[ei, 1])
        i0, i1 = int(round(s * sf)), int(round(e * sf))
        i0, i1 = max(0, i0), min(cache['env'].shape[1], i1)
        if i1 <= i0:
            continue
        for ci, cache_ci in enumerate(cache_ch_indices):
            if ci < len(ch_indices) and events_bool[ci, ei]:
                env_seg = cache['env'][cache_ci, i0:i1].astype(np.float64)
                energy[ci, ei] = float(np.sum(env_seg ** 2))

    # Normalize energy per event (column-wise), keep NaNs for non-participants.
    for ei in range(n_events):
        col = energy[:, ei]
        if not np.isfinite(col).any():
            continue
        total = float(np.nansum(col))
        if total > 0:
            energy[:, ei] = col / total
    
    # Plot
    return plot_lag_heatmaps(
        energy=energy,
        lag_ms=lag_raw * 1000.0,
        rank=lag_rank,
        ch_names=channel_names,
        event_ids=list(range(n_events)),
        cmap_energy=cmap_energy,
        cmap_lag=cmap_lag,
        cmap_rank=cmap_rank,
        figsize=figsize,
    )


def plot_coactivation_heatmap_from_group_analysis(
    *,
    group_analysis_npz: str,
    metric: str = "time_ratio",
    channel_names: Optional[List[str]] = None,
    use_all_channels: bool = True,
    cmap: str = "mako",
    figsize: Tuple[float, float] = (8, 7),
    show_values: bool = False,
) -> plt.Figure:
    """
    Plot channel×channel co-activation heatmap from *_groupAnalysis.npz.

    Parameters
    ----------
    group_analysis_npz : str
        Path to *_groupAnalysis.npz (contains co-activation matrices).
    metric : str
        'time_ratio' (absolute centroid time alignment) or
        'rank_ratio' (relative centroid rank alignment) or
        'event_ratio' (co-active event ratio).
    channel_names : list of str, optional
        Subset of channels to plot. If None, use all.
    cmap : str
        Seaborn colormap.
    figsize : tuple
        Figure size.
    show_values : bool
        Whether to annotate each cell with its value.
    """
    from .group_event_analysis import load_group_analysis_results
    import seaborn as sns

    ga = load_group_analysis_results(group_analysis_npz)

    metric = str(metric).lower().strip()
    use_all = bool(use_all_channels) and ("coact_all_time_ratio" in ga)
    if metric == "time_ratio":
        key = "coact_all_time_ratio" if use_all else "coact_time_ratio"
        mat = ga.get(key)
        title = "Co-activation (absolute time ratio)"
        vmin, vmax = 0.0, 1.0
    elif metric == "rank_ratio":
        key = "coact_all_rank_ratio" if use_all else "coact_rank_ratio"
        mat = ga.get(key)
        title = "Co-activation (centroid rank ratio)"
        vmin, vmax = 0.0, 1.0
    elif metric == "event_ratio":
        key = "coact_all_event_ratio" if use_all else "coact_event_ratio"
        mat = ga.get(key)
        title = "Co-activation (event ratio)"
        vmin, vmax = 0.0, 1.0
    else:
        raise ValueError("metric must be 'time_ratio', 'rank_ratio', or 'event_ratio'.")

    if mat is None:
        raise ValueError("Co-activation matrix not found in groupAnalysis results.")

    if use_all and "coact_all_ch_names" in ga:
        all_names = [str(x) for x in ga["coact_all_ch_names"]]
    else:
        all_names = [str(x) for x in ga["ch_names"]]
    if channel_names is None:
        channel_names = all_names
        idx = list(range(len(all_names)))
    else:
        name_to_idx = {n: i for i, n in enumerate(all_names)}
        idx = [name_to_idx[n] for n in channel_names if n in name_to_idx]
        channel_names = [n for n in channel_names if n in name_to_idx]

    if len(idx) == 0:
        raise ValueError("No matching channels found for co-activation heatmap.")

    mat = np.asarray(mat, dtype=np.float64)
    sub = mat[np.ix_(idx, idx)]

    # Drop channels whose row/col are entirely empty (no finite, non-zero values).
    has_signal = np.any(np.isfinite(sub) & (sub > 0.0), axis=0) | np.any(
        np.isfinite(sub) & (sub > 0.0), axis=1
    )
    if np.any(has_signal):
        sub = sub[np.ix_(has_signal, has_signal)]
        channel_names = [n for n, keep in zip(channel_names, has_signal) if keep]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        sub,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        square=True,
        annot=bool(show_values),
        fmt=".2f",
        cbar_kws={"label": "Co-activation strength"},
    )
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Channel")
    ticks = np.arange(len(channel_names)) + 0.5
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(channel_names, rotation=90, fontsize=7)
    ax.set_yticklabels(channel_names, rotation=0, fontsize=7)
    plt.tight_layout()
    return fig


def plot_lag_statistics(
    *,
    group_analysis_npz: str,
    patient_id: str = "",
    record_id: str = "",
    figsize: Tuple[float, float] = (15, 4),
) -> plt.Figure:
    """
    Plot statistical analysis from *_groupAnalysis.npz:
      1. Lag distribution (ms)
      2. Rank distribution (0=earliest)
      3. Channel participation rate (%)
    
    This reads pre-computed lag/rank/events_bool and plots statistics.
    NO computation is done here.
    
    Parameters
    ----------
    group_analysis_npz : str
        Path to *_groupAnalysis.npz
    patient_id, record_id : str
        Optional identifiers for plot title
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : plt.Figure
    """
    from .group_event_analysis import load_group_analysis_results
    
    ga = load_group_analysis_results(group_analysis_npz)
    lag_raw = ga['lag_raw']
    lag_rank = ga['lag_rank']
    events_bool = ga['events_bool']
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # 1. Lag distribution
    valid_lags = lag_raw[events_bool]
    valid_lags = valid_lags[np.isfinite(valid_lags)]
    if len(valid_lags) > 0:
        axes[0].hist(valid_lags * 1000, bins=50, edgecolor='white', alpha=0.8, color='steelblue')
        axes[0].set_xlabel('Relative Lag (ms)', fontsize=11)
        axes[0].set_ylabel('Count', fontsize=11)
        axes[0].set_title(f'Lag Distribution (n={len(valid_lags)})', fontweight='bold')
        axes[0].axvline(0, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
        axes[0].grid(axis='y', alpha=0.3)
    
    # 2. Rank distribution
    valid_ranks = lag_rank[events_bool]
    valid_ranks = valid_ranks[valid_ranks >= 0]
    if len(valid_ranks) > 0:
        max_rank = int(valid_ranks.max())
        axes[1].hist(valid_ranks, bins=np.arange(-0.5, max_rank + 1.5, 1), 
                     edgecolor='white', alpha=0.8, color='mediumseagreen')
        axes[1].set_xlabel('Rank (0=earliest)', fontsize=11)
        axes[1].set_ylabel('Count', fontsize=11)
        axes[1].set_title('Rank Distribution (0=source-like)', fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
    
    # 3. Channel participation rate
    participation = np.mean(events_bool, axis=1) * 100
    axes[2].barh(range(len(ga['ch_names'])), participation, color='coral', alpha=0.8)
    axes[2].set_xlabel('Participation Rate (%)', fontsize=11)
    axes[2].set_ylabel('Channel Index', fontsize=11)
    axes[2].set_title('Channel Participation', fontweight='bold')
    axes[2].invert_yaxis()
    axes[2].grid(axis='x', alpha=0.3)
    
    title_str = 'Lag/Rank/Participation Analysis'
    if patient_id or record_id:
        title_str = f'{patient_id}/{record_id} — {title_str}'
    plt.suptitle(title_str, fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_tf_centroid_statistics(
    *,
    group_analysis_npz: str,
    patient_id: str = "",
    record_id: str = "",
    figsize: Tuple[float, float] = (12, 4),
) -> plt.Figure:
    """
    Plot TF centroid statistics from *_groupAnalysis.npz:
      1. TF frequency centroid distribution (Hz)
      2. TF time centroid distribution (ms within event window)
    
    This reads pre-computed tf_centroid_time and tf_centroid_freq.
    NO computation is done here.
    
    Parameters
    ----------
    group_analysis_npz : str
        Path to *_groupAnalysis.npz
    patient_id, record_id : str
        Optional identifiers for plot title
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : plt.Figure or None
        None if TF centroids are not available in the file
    """
    from .group_event_analysis import load_group_analysis_results
    
    ga = load_group_analysis_results(group_analysis_npz)
    
    if 'tf_centroid_time' not in ga or 'tf_centroid_freq' not in ga:
        warnings.warn("TF centroids not found in groupAnalysis results")
        return None
    
    tf_time = ga['tf_centroid_time']
    tf_freq = ga['tf_centroid_freq']
    mask = ga['events_bool']
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 1. Frequency centroid distribution
    valid_freq = tf_freq[mask]
    valid_freq = valid_freq[np.isfinite(valid_freq)]
    if len(valid_freq) > 0:
        axes[0].hist(valid_freq, bins=30, edgecolor='white', alpha=0.8, color='mediumorchid')
        mean_freq = np.mean(valid_freq)
        axes[0].axvline(mean_freq, color='red', linestyle='--', linewidth=1.5, 
                       label=f'mean={mean_freq:.1f}Hz')
        axes[0].set_xlabel('TF Frequency Centroid (Hz)', fontsize=11)
        axes[0].set_ylabel('Count', fontsize=11)
        axes[0].set_title('TF Frequency Centroid Distribution', fontweight='bold')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
    
    # 2. Time centroid distribution
    valid_time = tf_time[mask]
    valid_time = valid_time[np.isfinite(valid_time)]
    if len(valid_time) > 0:
        axes[1].hist(valid_time * 1000, bins=30, edgecolor='white', alpha=0.8, color='darkorange')
        mean_time = np.mean(valid_time) * 1000
        axes[1].axvline(mean_time, color='red', linestyle='--', linewidth=1.5, 
                       label=f'mean={mean_time:.1f}ms')
        axes[1].set_xlabel('TF Time Centroid (ms)', fontsize=11)
        axes[1].set_ylabel('Count', fontsize=11)
        axes[1].set_title('TF Time Centroid Distribution', fontweight='bold')
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
    
    title_str = 'TF Centroid Analysis'
    if patient_id or record_id:
        title_str = f'{patient_id}/{record_id} — {title_str}'
    plt.suptitle(title_str, fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_channel_lag_distributions(
    *,
    group_analysis_npz: str,
    channel_order: Optional[List[str]] = None,
    bins: int = 40,
    figsize: Tuple[float, float] = (6, 10),
    color: str = "#7f7f7f",
    centroid_color: str = "#d62728",
    title: str = "Per-channel lag distributions",
) -> plt.Figure:
    """
    Plot per-channel lag distributions (ridge-like), with a red centroid path
    from top to bottom (channel order).
    """
    from .group_event_analysis import load_group_analysis_results

    ga = load_group_analysis_results(group_analysis_npz)
    lag_raw = np.asarray(ga["lag_raw"], dtype=np.float64) * 1000.0  # ms
    events_bool = np.asarray(ga["events_bool"], dtype=bool)
    ch_names = [str(x) for x in ga["ch_names"]]

    if channel_order is None:
        channel_order = ch_names
    idx_map = {c: i for i, c in enumerate(ch_names)}
    rows = [idx_map[c] for c in channel_order if c in idx_map]
    labels = [c for c in channel_order if c in idx_map]
    if not rows:
        raise ValueError("No channels matched channel_order.")

    # Determine global range for hist bins
    vals_all = []
    for r in rows:
        v = lag_raw[r][events_bool[r]]
        v = v[np.isfinite(v)]
        if v.size:
            vals_all.append(v)
    if not vals_all:
        raise ValueError("No valid lag values found for selected channels.")
    v_all = np.concatenate(vals_all)
    vmin, vmax = float(np.nanmin(v_all)), float(np.nanmax(v_all))
    if vmin == vmax:
        vmin -= 1.0
        vmax += 1.0
    bins_edges = np.linspace(vmin, vmax, int(bins) + 1)
    bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2.0

    fig, ax = plt.subplots(figsize=figsize)
    centroid_path_x = []
    centroid_path_y = []

    for i, (r, lbl) in enumerate(zip(rows, labels)):
        v = lag_raw[r][events_bool[r]]
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        hist, _ = np.histogram(v, bins=bins_edges)
        if hist.max() > 0:
            hist = hist / hist.max()
        y0 = len(labels) - 1 - i
        ax.fill_between(bin_centers, y0, y0 + hist, color=color, alpha=0.6, linewidth=0.0)

        centroid = float(np.mean(v))
        centroid_path_x.append(centroid)
        centroid_path_y.append(y0 + 0.5)

    if len(centroid_path_x) >= 2:
        ax.plot(centroid_path_x, centroid_path_y, color=centroid_color, linewidth=2.0, marker="o")

    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(list(reversed(labels)))
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel("Channels")
    ax.set_title(title, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def plot_channel_centroid_distributions(
    *,
    group_analysis_npz: str,
    channel_order: Optional[List[str]] = None,
    bins: int = 40,
    figsize: Tuple[float, float] = (6, 10),
    color: str = "#7f7f7f",
    centroid_color: str = "#d62728",
    title: str = "Per-channel centroid time distributions",
) -> plt.Figure:
    """
    Plot per-channel centroid time distributions (ridge-like), with a red centroid path.
    Uses centroid_time (seconds) from *_groupAnalysis.npz.
    """
    from .group_event_analysis import load_group_analysis_results

    ga = load_group_analysis_results(group_analysis_npz)
    centroid_time = np.asarray(ga["centroid_time"], dtype=np.float64) * 1000.0  # ms
    events_bool = np.asarray(ga["events_bool"], dtype=bool)
    ch_names = [str(x) for x in ga["ch_names"]]

    if channel_order is None:
        channel_order = ch_names
    idx_map = {c: i for i, c in enumerate(ch_names)}
    rows = [idx_map[c] for c in channel_order if c in idx_map]
    labels = [c for c in channel_order if c in idx_map]
    if not rows:
        raise ValueError("No channels matched channel_order.")

    vals_all = []
    for r in rows:
        v = centroid_time[r][events_bool[r]]
        v = v[np.isfinite(v)]
        if v.size:
            vals_all.append(v)
    if not vals_all:
        raise ValueError("No valid centroid values found for selected channels.")
    v_all = np.concatenate(vals_all)
    vmin, vmax = float(np.nanmin(v_all)), float(np.nanmax(v_all))
    if vmin == vmax:
        vmin -= 1.0
        vmax += 1.0
    bins_edges = np.linspace(vmin, vmax, int(bins) + 1)
    bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2.0

    fig, ax = plt.subplots(figsize=figsize)
    centroid_path_x = []
    centroid_path_y = []

    for i, (r, lbl) in enumerate(zip(rows, labels)):
        v = centroid_time[r][events_bool[r]]
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        hist, _ = np.histogram(v, bins=bins_edges)
        if hist.max() > 0:
            hist = hist / hist.max()
        y0 = len(labels) - 1 - i
        ax.fill_between(bin_centers, y0, y0 + hist, color=color, alpha=0.6, linewidth=0.0)

        centroid = float(np.mean(v))
        centroid_path_x.append(centroid)
        centroid_path_y.append(y0 + 0.5)

    if len(centroid_path_x) >= 2:
        ax.plot(centroid_path_x, centroid_path_y, color=centroid_color, linewidth=2.0, marker="o")

    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(list(reversed(labels)))
    ax.set_xlabel("Centroid time (ms)")
    ax.set_ylabel("Channels")
    ax.set_title(title, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig

def plot_event_counts(
    det_result,
    top_k: Optional[int] = 30,
    figsize: Tuple[float, float] = (14, 5),
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot detected event counts per channel (quick artifact triage).
    """
    if det_result is None:
        raise ValueError("det_result is required.")
    if not hasattr(det_result, "events_count") or not hasattr(det_result, "ch_names"):
        raise ValueError("det_result must look like HFODetectionResult (events_count + ch_names).")

    counts = np.asarray(det_result.events_count).astype(np.int64)
    ch_names = list(det_result.ch_names)
    if counts.shape[0] != len(ch_names):
        raise ValueError("events_count length must match ch_names length.")

    order = np.argsort(counts)[::-1]
    if top_k is not None:
        order = order[: int(top_k)]

    fig, ax = plt.subplots(figsize=figsize)
    xs = np.arange(len(order))
    ax.bar(xs, counts[order], color=SEEG_COLORS["secondary"], alpha=0.75)
    ax.set_xticks(xs)
    ax.set_xticklabels([ch_names[i] for i in order], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Detected events")
    if title is None:
        title = f"HFO detections per channel (top {len(order)})"
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_raw_filtered_envelope(
    data: np.ndarray,
    sfreq: float,
    ch_names: List[str],
    channels: Union[List[str], List[int]],
    start_sec: float = 0.0,
    duration_sec: float = 2.0,
    show_fast_ripple: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    band: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize for selected channels:
    - raw waveform
    - ripple-band filtered waveform + envelope
    - (optional) fast-ripple-band filtered waveform + envelope

    This is a debugging view to answer: "Is the detector firing on real band-limited bursts?"
    """
    # Lazy import to keep visualization module lightweight.
    try:
        from src.utils import bqk_utils as bqk
    except Exception:
        from .utils import bqk_utils as bqk  # type: ignore

    ch_indices, ch_labels = _parse_channel_selection(channels, ch_names)
    if len(ch_indices) == 0:
        raise ValueError("No channels selected")

    s0 = int(round(start_sec * sfreq))
    s1 = int(round((start_sec + duration_sec) * sfreq))
    s1 = min(s1, data.shape[1])
    x = data[ch_indices, s0:s1]
    t = np.arange(s0, s1) / sfreq

    # Ripple (RP)
    rp_band = (80.0, 250.0)
    x_rp = bqk.band_filt(x, sfreq, rp_band)
    env_rp = bqk.return_hil_enve_norm(x, sfreq, rp_band)

    # Fast-ripple (FR)
    show_fr = bool(show_fast_ripple)
    if band is not None:
        band_l = band.lower().strip()
        if band_l in ("ripple", "rp", "ripples"):
            show_fr = False
        elif band_l in ("fast_ripple", "fast-ripple", "fr", "fast"):
            show_fr = True
        else:
            raise ValueError(f"Unknown band='{band}'. Use 'ripple' or 'fast_ripple'.")

    if show_fr:
        nyq = sfreq / 2.0
        fr_band = (250.0, 500.0)
        if fr_band[1] >= nyq:
            if nyq - 1.0 <= fr_band[0]:
                warnings.warn(
                    f"Fast-ripple band requires sfreq > 2*{fr_band[0]}Hz; "
                    f"sfreq={sfreq}Hz too low. Skipping FR plots."
                )
                show_fr = False
            else:
                adj_high = max(fr_band[0] + 1.0, nyq - 1.0)
                warnings.warn(
                    f"Fast-ripple high {fr_band[1]}Hz >= Nyquist {nyq}Hz; "
                    f"clipping to {adj_high}Hz."
                )
                fr_band = (fr_band[0], adj_high)

    if show_fr:
        x_fr = bqk.band_filt(x, sfreq, fr_band)
        env_fr = bqk.return_hil_enve_norm(x, sfreq, fr_band)
    else:
        x_fr, env_fr = None, None

    n_ch = len(ch_indices)
    n_cols = 3 if show_fr else 2
    if figsize is None:
        figsize = (16, max(6, 2.2 * n_ch))

    fig, axes = plt.subplots(n_ch, n_cols, figsize=figsize, sharex=True)
    if n_ch == 1:
        axes = np.array([axes])

    for i, lbl in enumerate(ch_labels):
        ax0 = axes[i, 0]
        ax0.plot(t, x[i], color="#333333", linewidth=0.7)
        ax0.set_title(f"{lbl} raw", fontweight="bold")

        ax1 = axes[i, 1]
        ax1.plot(t, x_rp[i], color=tableau_20[0], linewidth=0.7, label="RP filt")
        ax1.plot(t, env_rp[i], color=tableau_20[2], linewidth=0.7, alpha=0.9, label="RP env")
        ax1.set_title("Ripple (80-250Hz)", fontweight="bold")
        ax1.legend(fontsize=8, loc="upper right")

        if show_fr:
            ax2 = axes[i, 2]
            ax2.plot(t, x_fr[i], color=tableau_20[4], linewidth=0.7, label="FR filt")
            ax2.plot(t, env_fr[i], color=tableau_20[6], linewidth=0.7, alpha=0.9, label="FR env")
            ax2.set_title("Fast Ripple (250-500Hz)", fontweight="bold")
            ax2.legend(fontsize=8, loc="upper right")

        for j in range(n_cols):
            axes[i, j].grid(alpha=0.2)

    for j in range(n_cols):
        axes[-1, j].set_xlabel("Time (s)")

    plt.tight_layout()
    return fig


# =============================================================================
# HFO Event Verification Card (Visualization only)
# =============================================================================

def plot_hfo_event_verification_from_tfr_cache(
    *,
    raw_cache_npz_path: str,
    env_cache_npz_path: str,
    tfr_npz_path: str,
    target_channel: Optional[str] = None,
    freq_band_label: Tuple[float, float] = (80.0, 250.0),
    cmap_tf: str = "jet",
    figsize: Tuple[float, float] = (12, 4),
    tf_vmin: Optional[float] = None,
    tf_vmax: Optional[float] = None,
    tf_map: str = "auto",  # 'auto'|'power_db'|'power_db_smooth'|'power_z'|'power_pctl'|'power_db_baseline'
    baseline_view: str = "none",  # legacy, ignored
    baseline_window_sec: Tuple[float, float] = (0.5, 0.2),
    baseline_mode_vis: str = "none",  # 'none'|'db_ratio'
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot a single HFO event "verification card" from PRECOMPUTED caches.

    Architecture
    ------------
    This function is visualization-only:
    - Raw trace: read from *_rawCache_*.npz
    - Ripple-band trace: read from *_envCache_*.npz (requires x_band in the cache)
    - Time-frequency (wavelet TFR): read from *_hfoVerify_wavelet_*.npz

    All heavy computation (wavelet transform) must be done upstream and saved to `tfr_npz_path`.
    """
    from .preprocessing import load_raw_cache
    from .group_event_analysis import load_envelope_cache

    raw = load_raw_cache(str(raw_cache_npz_path))
    envc = load_envelope_cache(str(env_cache_npz_path))
    tfr = np.load(str(tfr_npz_path), allow_pickle=True)

    tfr_target = str(np.asarray(tfr["target_channel"]).ravel()[0])
    if target_channel is None:
        target_channel = tfr_target
    if str(target_channel) != tfr_target:
        raise ValueError(f"target_channel='{target_channel}' != cached '{tfr_target}'. Use the cached channel.")

    sf_raw = float(np.asarray(raw["sfreq"]).ravel()[0])
    sf_env = float(envc["sfreq"])
    if abs(sf_raw - sf_env) > 1e-6:
        warnings.warn(f"raw sfreq={sf_raw}Hz != envCache sfreq={sf_env}Hz; plotting may be misaligned.")

    raw_ch_names = [str(x) for x in raw["ch_names"]]
    env_ch_names = [str(x) for x in envc["ch_names"]]
    if target_channel not in raw_ch_names:
        raise ValueError(f"Channel '{target_channel}' not found in raw cache.")
    if target_channel not in env_ch_names:
        raise ValueError(f"Channel '{target_channel}' not found in env cache.")
    if envc.get("x_band", None) is None:
        raise ValueError(
            "env cache does not contain x_band (bandpassed signal). "
            "Regenerate envCache with save_bandpass=True."
        )

    plot_start = float(np.asarray(tfr["plot_start"]).ravel()[0])
    plot_end = float(np.asarray(tfr["plot_end"]).ravel()[0])
    event_start = float(np.asarray(tfr["event_start"]).ravel()[0])
    event_end = float(np.asarray(tfr["event_end"]).ravel()[0])

    # Slice raw segment
    start_sec_raw = float(np.asarray(raw.get("start_sec", np.array([0.0]))).ravel()[0])
    data_raw = np.asarray(raw["data"])
    i0r = int(round((plot_start - start_sec_raw) * sf_raw))
    i1r = int(round((plot_end - start_sec_raw) * sf_raw))
    i0r = max(0, i0r)
    i1r = min(int(data_raw.shape[1]), i1r)
    if i1r <= i0r:
        raise ValueError("Empty raw slice for the requested plot window.")
    ridx = raw_ch_names.index(target_channel)
    raw_seg = np.asarray(data_raw[ridx, i0r:i1r], dtype=np.float64)
    t_raw = (np.arange(raw_seg.shape[0], dtype=np.float64) / sf_raw) + plot_start

    # Slice ripple bandpassed segment from env cache
    start_sec_env = float(envc.get("start_sec", 0.0))
    x_band = np.asarray(envc["x_band"])
    i0e = int(round((plot_start - start_sec_env) * sf_env))
    i1e = int(round((plot_end - start_sec_env) * sf_env))
    i0e = max(0, i0e)
    i1e = min(int(x_band.shape[1]), i1e)
    if i1e <= i0e:
        raise ValueError("Empty x_band slice for the requested plot window.")
    eidx = env_ch_names.index(target_channel)
    ripple_seg = np.asarray(x_band[eidx, i0e:i1e], dtype=np.float64)
    t_ripple = (np.arange(ripple_seg.shape[0], dtype=np.float64) / sf_env) + plot_start

    # TFR
    freqs = np.asarray(tfr["freqs_hz"], dtype=np.float64)
    t_sec = np.asarray(tfr["t_sec"], dtype=np.float64)
    power_db = np.asarray(tfr["power_db"], dtype=np.float64)
    power_db_smooth = None
    power_z = None
    power_pctl = None
    if "power_db_smooth" in tfr:
        s = np.asarray(tfr["power_db_smooth"], dtype=np.float64)
        if s.ndim == 2 and s.shape == power_db.shape and s.size > 0:
            power_db_smooth = s
    if "power_z" in tfr:
        z = np.asarray(tfr["power_z"], dtype=np.float64)
        if z.ndim == 2 and z.shape == power_db.shape and z.size > 0:
            power_z = z
    if "power_pctl" in tfr:
        p = np.asarray(tfr["power_pctl"], dtype=np.float64)
        if p.ndim == 2 and p.shape == power_db.shape and p.size > 0:
            power_pctl = p
    if power_db.ndim != 2 or power_db.shape[0] != freqs.shape[0] or power_db.shape[1] != t_sec.shape[0]:
        raise ValueError("Invalid TFR shapes in cache.")

    tf_map = str(tf_map).lower().strip()
    if tf_map not in (
        "auto",
        "power_db",
        "power_db_smooth",
        "power_z",
        "power_pctl",
        "power_db_baseline",
    ):
        raise ValueError(
            "tf_map must be 'auto', 'power_db', 'power_db_smooth', "
            "'power_z', 'power_pctl', or 'power_db_baseline'."
        )
    if tf_map == "power_db_smooth" and power_db_smooth is None:
        warnings.warn("power_db_smooth not found in TFR cache; falling back to power_db.")
        tf_map = "power_db"
    if tf_map == "power_pctl" and power_pctl is None:
        warnings.warn("power_pctl not found in TFR cache; falling back to power_db.")
        tf_map = "power_db"
    if tf_map == "power_z" and power_z is None:
        warnings.warn("power_z not found in TFR cache; falling back to power_db.")
        tf_map = "power_db"
    power_db_baseline = None
    if "power_db_baseline" in tfr:
        pb = np.asarray(tfr["power_db_baseline"], dtype=np.float64)
        if pb.ndim == 2 and pb.shape == power_db.shape and pb.size > 0:
            power_db_baseline = pb
    if tf_map == "power_db_baseline" and power_db_baseline is None:
        warnings.warn("power_db_baseline not found in TFR cache; falling back to power_db.")
        tf_map = "power_db"
    if tf_map == "auto":
        if power_db_baseline is not None:
            tf_map = "power_db_baseline"
        elif power_db_smooth is not None:
            tf_map = "power_db_smooth"
        elif power_pctl is not None:
            tf_map = "power_pctl"
        elif power_z is not None:
            tf_map = "power_z"
        else:
            tf_map = "power_db"
    if tf_map == "power_db_smooth":
        tf_data = power_db_smooth
    elif tf_map == "power_pctl":
        tf_data = power_pctl
    elif tf_map == "power_z":
        tf_data = power_z
    elif tf_map == "power_db_baseline":
        tf_data = power_db_baseline
    else:
        tf_data = power_db

    # Legacy baseline_view is ignored in new pipeline.

    # Auto scale for nicer cards (avoid one extreme pixel dominating)
    if tf_vmin is None or tf_vmax is None:
        finite = tf_data[np.isfinite(tf_data)]
        if finite.size:
            if tf_map == "power_pctl":
                p2, p98 = 0.0, 1.0
            elif tf_map == "power_db_baseline":
                p2, p98 = -2.0, 8.0
            else:
                p2, p98 = np.percentile(finite, [2, 98])
            if tf_vmin is None:
                tf_vmin = float(p2)
            if tf_vmax is None:
                tf_vmax = float(p98)
        else:
            tf_vmin = tf_vmin if tf_vmin is not None else 0.0
            tf_vmax = tf_vmax if tf_vmax is not None else 1.0

    # Layout: Left stacked traces, Right TFR heatmap
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.15)

    gs_left = gs[0].subgridspec(2, 1, hspace=0.05)
    ax_raw = fig.add_subplot(gs_left[0])
    ax_rip = fig.add_subplot(gs_left[1], sharex=ax_raw)

    ax_raw.plot(t_raw, raw_seg, color="k", lw=1.0)
    ax_raw.set_ylabel(f"{target_channel}\nRaw", fontsize=10)
    ax_raw.tick_params(labelbottom=False)

    ax_rip.plot(t_ripple, ripple_seg, color="#1f77b4", lw=1.0)
    ax_rip.set_ylabel("Ripple", fontsize=10)
    ax_rip.set_xlabel("Time (s)", fontsize=10)

    # Event shading on traces
    for ax in (ax_raw, ax_rip):
        ax.axvspan(event_start, event_end, color="#d62728", alpha=0.3, lw=0, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
    ax_rip.spines["bottom"].set_visible(True)

    ax_tf = fig.add_subplot(gs[1])
    # Avoid white margins by padding data to display range if needed.
    band_name = str(np.asarray(tfr.get("band", ["custom"])).ravel()[0])
    if band_name == "fast_ripple":
        display_max = 500.0
    else:
        display_max = 250.0
    display_min = 1.0

    freqs_plot = freqs
    tf_plot = tf_data
    if freqs_plot.size:
        if display_min < float(freqs_plot.min()):
            freqs_plot = np.concatenate([[display_min], freqs_plot])
            tf_plot = np.vstack([tf_plot[:1, :], tf_plot])
        if display_max > float(freqs_plot.max()):
            freqs_plot = np.concatenate([freqs_plot, [display_max]])
            tf_plot = np.vstack([tf_plot, tf_plot[-1:, :]])

    im = ax_tf.pcolormesh(
        t_sec,
        freqs_plot,
        tf_plot,
        cmap=str(cmap_tf),
        shading="auto",
        rasterized=True,
        vmin=float(tf_vmin),
        vmax=float(tf_vmax),
    )
    ax_tf.set_xlabel("Time (s)", fontsize=10)
    ax_tf.set_ylabel("Frequency (Hz)", fontsize=10)
    if tf_map == "power_pctl":
        title_suffix = "Wavelet pctl"
    elif tf_map == "power_z":
        title_suffix = "Wavelet z"
    elif tf_map == "power_db_smooth":
        title_suffix = "Wavelet dB (smooth)"
    elif tf_map in ("db_ratio", "db_ratio_smooth"):
        title_suffix = "Wavelet dB (baseline)"
    else:
        title_suffix = "Wavelet dB"
    ax_tf.set_title(f"Time-Frequency ({title_suffix})", fontsize=10, fontweight="bold")
    ax_tf.axvline(event_start, color="w", linestyle="--", alpha=0.5, lw=1)
    ax_tf.axvline(event_end, color="w", linestyle="--", alpha=0.5, lw=1)

    cb = plt.colorbar(im, ax=ax_tf, fraction=0.046, pad=0.04)
    if tf_map == "power_pctl":
        cb.set_label("Event pctl (0-1)", fontsize=8)
    elif tf_map in ("power_db", "power_db_smooth"):
        cb.set_label("Power (dB)", fontsize=8)
    elif tf_map in ("db_ratio", "db_ratio_smooth"):
        cb.set_label("Power (dB vs baseline)", fontsize=8)
    else:
        cb.set_label("Robust z", fontsize=8)

    # Display range: 1-250 (ripple) or 1-500 (fast ripple)
    ax_tf.set_ylim(display_min, display_max)

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
    return fig


# =============================================================================
# Convenience function for PreprocessingResult
# =============================================================================

def plot_from_result(
    result,  # PreprocessingResult
    start_sec: float = 0,
    duration_sec: float = 100,
    channels: Union[str, List[str]] = 'all',
    **kwargs
) -> plt.Figure:
    """
    Plot directly from PreprocessingResult.
    
    Parameters
    ----------
    result : PreprocessingResult
        Result from SEEGPreprocessor.run()
    start_sec : float
        Start time.
    duration_sec : float
        Duration to plot.
    channels : str or list
        Channel selection.
    **kwargs
        Additional arguments to plot_seeg_segment.
    
    Returns
    -------
    fig : matplotlib.Figure
    """
    return plot_seeg_segment(
        data=result.data,
        sfreq=result.sfreq,
        ch_names=result.ch_names,
        start_sec=start_sec,
        duration_sec=duration_sec,
        channels=channels,
        reference_type=result.reference_type,
        **kwargs
    )


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/Users/leijiaxin/Desktop/Pys/HFOsp')
    from src.preprocessing import SEEGPreprocessor
    
    print("=" * 70)
    print("Testing SEEG Visualization Module")
    print("=" * 70)
    
    edf_path = '/mnt/yuquan_data/yuquan_24h_edf/chengshuai/FC10477Q.edf'
    
    # Load data (explicit reference; no guessing)
    preprocessor = SEEGPreprocessor(
        reference='bipolar',
        crop_seconds=120  # 2 minutes for testing
    )
    result = preprocessor.run(edf_path)
    
    print(f"\nData loaded: {result.data.shape}")
    print(f"Reference type: {result.reference_type}")
    
    # Test 1: Plot all channels
    print("\n1. Plotting all channels (0-100s)...")
    fig1 = plot_from_result(result, start_sec=0, duration_sec=100)
    fig1.savefig('/Users/leijiaxin/Desktop/Pys/HFOsp/results/test_all_channels.png', 
                 dpi=150, bbox_inches='tight')
    print("   Saved: results/test_all_channels.png")
    plt.close(fig1)
    
    # Test 2: Plot specific channels
    print("\n2. Plotting specific channels...")
    fig2 = plot_from_result(
        result, 
        channels=['A1-A2', 'A2-A3', 'A3-A4', 'K1-K2', 'K2-K3', 'K3-K4'],
        start_sec=0,
        duration_sec=30
    )
    fig2.savefig('/Users/leijiaxin/Desktop/Pys/HFOsp/results/test_selected_channels.png',
                 dpi=150, bbox_inches='tight')
    print("   Saved: results/test_selected_channels.png")
    plt.close(fig2)
    
    # Test 3: Plot single shaft
    print("\n3. Plotting shaft K...")
    fig3 = plot_shaft_channels(
        result.data, result.sfreq, result.ch_names,
        shaft='K',
        start_sec=0,
        duration_sec=30,
        reference_type=result.reference_type
    )
    fig3.savefig('/Users/leijiaxin/Desktop/Pys/HFOsp/results/test_shaft_K.png',
                 dpi=150, bbox_inches='tight')
    print("   Saved: results/test_shaft_K.png")
    plt.close(fig3)
    
    print("\n" + "=" * 70)
    print("Visualization tests complete!")
    print("=" * 70)
