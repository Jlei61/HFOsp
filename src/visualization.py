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
    x_axis: str = "seconds",  # 'seconds'|'samples'
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
    if x_axis not in ("seconds", "samples"):
        raise ValueError("x_axis must be 'seconds' or 'samples'")
    # If downsampled, adjust effective sfreq for x-axis scaling
    eff_sfreq = sfreq
    if downsample_ms is not None and downsample_ms > 0:
        bin_n = max(1, int(round((downsample_ms * 1e-3) * sfreq)))
        eff_sfreq = sfreq / float(bin_n)

    if x_axis == "seconds":
        xs = np.arange(Xn.shape[1], dtype=np.float64) / eff_sfreq
        xlabel = "Concatenated event time (s)"
    else:
        xs = np.arange(Xn.shape[1], dtype=np.float64)
        xlabel = "Concatenated event samples"

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
        x_axis="seconds",
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

def plot_group_events_tf_centroids_per_channel(
    *,
    cache_npz_path: str,
    packed_times_path: str,
    detections_npz_path: Optional[str] = None,
    channel_order: Optional[List[str]] = None,
    event_indices: Optional[List[int]] = None,
    max_events: int = 30,
    freq_band: Tuple[float, float] = (80.0, 250.0),
    nperseg: int = 256,  # legacy, unused for wavelet
    noverlap: int = 192,  # legacy, unused for wavelet
    power_log1p: bool = True,
    mask_by_detections: bool = False,
    centroid_power: str = "power2",  # 'power'|'power2'
    centroid_min_total_power: float = 0.0,
    centroid_marker_size: float = 18.0,
    centroid_marker_color: str = "crimson",
    centroid_marker_edge: str = "white",
    centroid_marker_edge_lw: float = 0.6,
    show_colorbar: bool = True,
    cmap: str = "Blues",
    font_size: int = 14,
    title_font_size: int = 16,
    y_label_font_size: int = 12,
    tick_font_size: int = 12,
    hspace: float = 0.02,
    figsize: Tuple[float, float] = (12, 8),
    tf_n_freqs: int = 180,
    tf_n_cycles: float = 4.0,
    tf_n_cycles_mode: str = "linear",
    tf_n_cycles_min: float = 3.0,
    tf_n_cycles_max: float = 10.0,
    tf_freq_scale: str = "log",
    baseline_n_select: int = 10,
    baseline_min_distance_sec: float = 2.0,
    baseline_window_sec: float = 2.0,
) -> plt.Figure:
    """
    Per-channel spectrogram view (what you actually want for Fig2):
      - For each channel: Wavelet TF power (within freq_band) over concatenated packedTimes windows.
      - Overlay a TF centroid (t,f) per event window (from groupAnalysis if available).

    Notes:
      - This is NOT a (channels×events) heatmap. It's a real per-channel TF transform.
      - Centroid is computed on wavelet+baseline log TF maps (2D centroid).
    """
    from .group_event_analysis import load_envelope_cache, load_group_analysis_results
    from .utils import bqk_utils

    meta = load_envelope_cache(cache_npz_path)
    x_band = meta.get("x_band", None)
    if x_band is None:
        raise ValueError("Cache does not contain x_band. Regenerate with save_bandpass=True.")

    sfreq = float(meta["sfreq"])
    ch_names = [str(x) for x in meta["ch_names"]]
    packed = np.load(packed_times_path, allow_pickle=True)

    if event_indices is None:
        event_indices = list(range(min(int(max_events), packed.shape[0])))
    else:
        event_indices = [int(i) for i in event_indices][: int(max_events)]

    if channel_order is None:
        channel_order = ch_names
    idx_map = {c: i for i, c in enumerate(ch_names)}
    rows = [idx_map[c] for c in channel_order if c in idx_map]
    labels = [c for c in channel_order if c in idx_map]
    if not rows:
        raise ValueError("No channels matched channel_order.")

    # detection mask (channel,event): used only to decide whether to compute/plot a centroid dot
    det_mask = None
    if mask_by_detections and detections_npz_path is not None:
        gpu = np.load(detections_npz_path, allow_pickle=True)
        det_names = [str(x) for x in gpu["chns_names"].tolist()]
        name_to_i = {n: i for i, n in enumerate(det_names)}
        det_mask = np.zeros((len(labels), len(event_indices)), dtype=bool)
        for ci, ch in enumerate(labels):
            if ch not in name_to_i:
                continue
            ev = np.asarray(gpu["whole_dets"][name_to_i[ch]])
            if ev.size == 0:
                continue
            for ej, eidx in enumerate(event_indices):
                s, e = float(packed[eidx, 0]), float(packed[eidx, 1])
                det_mask[ci, ej] = bool(np.any((ev[:, 1] > s) & (ev[:, 0] < e)))

    # Build concatenated signal per channel + event boundaries (seconds in concatenated timeline)
    boundaries_s = [0.0]
    segs_idx = []
    cur_s = 0.0
    for eidx in event_indices:
        s, e = float(packed[eidx, 0]), float(packed[eidx, 1])
        i0 = int(round(s * sfreq))
        i1 = int(round(e * sfreq))
        segs_idx.append((i0, i1))
        cur_s += float(i1 - i0) / sfreq
        boundaries_s.append(cur_s)

    n_ch = len(rows)
    fig, axes = plt.subplots(
        n_ch,
        1,
        figsize=figsize,
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": float(hspace)},
    )
    if n_ch == 1:
        axes = [axes]

    # Pass 1: compute per-channel STFT + centroids; choose a shared vmax for a meaningful colorbar.
    per = []
    v99s = []
    for ci, (row, lab) in enumerate(zip(rows, labels)):
        x_cat = np.concatenate([np.asarray(x_band[row, i0:i1]) for (i0, i1) in segs_idx], axis=0)
        nwin = int(x_cat.shape[0])
        nps = min(int(nperseg), max(16, nwin))
        nov = min(int(noverlap), max(0, nps - 1))
        f, t, Z = stft(x_cat, fs=sfreq, nperseg=nps, noverlap=nov, boundary=None)
        P = (np.abs(Z) ** 2).astype(np.float64)
        band_mask = (f >= float(freq_band[0])) & (f <= float(freq_band[1]))
        f_band = f[band_mask]
        P_band = P[band_mask, :]

        if power_log1p:
            P_show = np.log1p(P_band)
        else:
            P_show = P_band

        if np.isfinite(P_show).any():
            v99s.append(float(np.nanpercentile(P_show, 99)))

        # TF centroids per event
        pts = []
        for ej in range(len(event_indices)):
            if det_mask is not None and not det_mask[ci, ej]:
                continue
            t0, t1 = float(boundaries_s[ej]), float(boundaries_s[ej + 1])
            cols = (t >= t0) & (t < t1)
            if not np.any(cols):
                continue
            W = P_band[:, cols]
            if centroid_power == "power2":
                W = W**2
            elif centroid_power != "power":
                raise ValueError("centroid_power must be 'power' or 'power2'")
            denom = float(np.sum(W))
            # Lower threshold: power^2 on small signals can be very small
            if denom <= 1e-30:
                continue
            if float(centroid_min_total_power) > 0.0 and denom < float(centroid_min_total_power):
                continue
            t_sel = t[cols]
            # 2D centroid: (t,f)
            t_c = float(np.sum(W * t_sel[None, :]) / denom)
            f_c = float(np.sum(W * f_band[:, None]) / denom)
            pts.append((t_c, f_c))

        per.append({"label": lab, "t": t, "f_band": f_band, "P_show": P_show, "pts": pts})

    shared_vmax = max(v99s) if v99s else None
    if shared_vmax is not None:
        shared_vmax = max(float(shared_vmax), 1e-12)

    # Pass 2: plot
    last_im = None
    for ci, (ax, item) in enumerate(zip(axes, per)):
        last_im = ax.pcolormesh(
            item["t"],
            item["f_band"],
            item["P_show"],
            shading="auto",
            cmap=cmap,
            vmin=0.0,
            vmax=shared_vmax,
        )

        # style: no top/right spines, minimal clutter
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="both", labelsize=int(tick_font_size))
        # x ticks only on bottom axis
        if ci != (len(axes) - 1):
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        # channel label on left
        ax.set_ylabel(item["label"], fontsize=int(y_label_font_size), rotation=0, labelpad=22, va="center")
        ax.set_yticks([])  # avoid dense y ticks; label is enough

        # vertical boundaries
        for b in boundaries_s:
            ax.axvline(float(b), color="k", linewidth=0.3, alpha=0.25)

        # centroid dots
        if item["pts"]:
            xs = [p[0] for p in item["pts"]]
            ys = [p[1] for p in item["pts"]]
            ax.scatter(
                xs,
                ys,
                s=float(centroid_marker_size),
                c=centroid_marker_color,
                alpha=0.95,
                linewidths=float(centroid_marker_edge_lw),
                edgecolors=centroid_marker_edge,
                zorder=5,
            )

        # remove useless whitespace at both ends
        ax.set_xlim(float(boundaries_s[0]), float(boundaries_s[-1]))
        ax.margins(x=0.0)

    axes[-1].set_xlabel("Concatenated event time (s)", fontsize=int(font_size))
    axes[0].set_title(
        f"Per-channel STFT + TF centroids ({freq_band[0]:.0f}-{freq_band[1]:.0f} Hz)",
        fontsize=int(title_font_size),
        fontweight="bold",
    )

    if show_colorbar and last_im is not None:
        cb = fig.colorbar(last_im, ax=axes, fraction=0.018, pad=0.01)
        cb.set_label("log1p(power)" if power_log1p else "power", fontsize=int(font_size))
        cb.ax.tick_params(labelsize=int(tick_font_size))

    fig.subplots_adjust(hspace=float(hspace))
    return fig


def plot_group_events_tf_centroid_paths(
    *,
    cache_npz_path: str,
    packed_times_path: str,
    detections_npz_path: Optional[str] = None,
    channel_order: Optional[List[str]] = None,
    event_indices: Optional[List[int]] = None,
    max_events: int = 30,
    freq_band: Tuple[float, float] = (80.0, 250.0),
    nperseg: int = 128,
    noverlap: int = 96,
    cfg: GroupVizConfig = GroupVizConfig(),
    x_axis: str = "seconds",  # 'seconds'|'samples'
    background_mode: str = "blob",  # 'tp'|'blob'
    blob_sigma_ms: float = 20.0,
    power_gamma: float = 1.0,
    power_threshold: float = 0.05,
    smooth_sigma_ms: Optional[float] = 8.0,
    normalize_0_1: bool = True,
    show_colorbar: bool = True,
    colorbar_label: str = "Normalized TF power",
    figsize: Tuple[float, float] = (12, 8),
) -> plt.Figure:
    """
    Figure 2 (paper-like): normalized TF-derived background + centroid dots + per-event path.

    Background:
      - Compute STFT power on bandpassed signal for each event, sum across frequencies in freq_band
        to get a time-power curve (still TF-derived).
      - Paste the curve into a concatenated timeline (same x-axis as Figure 1).

    Overlay:
      - Red dot per (channel,event): time centroid of TF-derived time-power.
      - Connect dots within each event in ascending centroid time order ("priority path").

    Note:
      We use bandpassed signal from cache (x_band). If you want to mask only channels/events that
      have detections, provide detections_npz_path and we'll use overlap with packedTimes.
    """
    from scipy.signal import stft
    from .group_event_analysis import load_envelope_cache

    meta = load_envelope_cache(cache_npz_path)
    x_band = meta.get("x_band", None)
    if x_band is None:
        raise ValueError("Cache does not contain x_band. Regenerate with save_bandpass=True.")

    sfreq = float(meta["sfreq"])
    ch_names = list(meta["ch_names"])

    packed = np.load(packed_times_path, allow_pickle=True)
    if event_indices is None:
        event_indices = list(range(min(int(max_events), packed.shape[0])))
    else:
        event_indices = [int(i) for i in event_indices][: int(max_events)]

    if channel_order is None:
        channel_order = ch_names
    idx_map = {c: i for i, c in enumerate(ch_names)}
    rows = [idx_map[c] for c in channel_order if c in idx_map]
    labels = [c for c in channel_order if c in idx_map]
    if not rows:
        raise ValueError("No channels matched channel_order.")

    # Optional detection mask (channel,event)
    det_mask = None
    if detections_npz_path is not None:
        gpu = np.load(detections_npz_path, allow_pickle=True)
        det_names = [str(x) for x in gpu["chns_names"].tolist()]
        name_to_i = {n: i for i, n in enumerate(det_names)}
        det_mask = np.zeros((len(labels), len(event_indices)), dtype=bool)
        for ci, ch in enumerate(labels):
            if ch not in name_to_i:
                continue
            ev = np.asarray(gpu["whole_dets"][name_to_i[ch]])
            if ev.size == 0:
                continue
            for ej, eidx in enumerate(event_indices):
                s, e = float(packed[eidx, 0]), float(packed[eidx, 1])
                det_mask[ci, ej] = bool(np.any((ev[:, 1] > s) & (ev[:, 0] < e)))

    # Build background and centroids
    bg_rows: List[np.ndarray] = []
    centroids: np.ndarray = np.full((len(rows), len(event_indices)), np.nan, dtype=np.float64)

    cur = 0
    boundaries = [0]
    for ej, eidx in enumerate(event_indices):
        s, e = float(packed[eidx, 0]), float(packed[eidx, 1])
        n = int(round((e - s) * sfreq))
        cur += n
        boundaries.append(cur)

    total_len = boundaries[-1]
    bg = np.zeros((len(rows), total_len), dtype=np.float32)

    for ej, eidx in enumerate(event_indices):
        s, e = float(packed[eidx, 0]), float(packed[eidx, 1])
        i0 = int(round(s * sfreq))
        i1 = int(round(e * sfreq))
        seg = x_band[rows, i0:i1]
        # STFT per channel, get TF power summed in freq_band -> time power curve
        for ci in range(seg.shape[0]):
            if det_mask is not None and not det_mask[ci, ej]:
                continue
            # Guard against very short windows
            nwin = int(seg.shape[1])
            nps = min(int(nperseg), max(8, nwin))
            nov = min(int(noverlap), max(0, nps - 1))
            f, t, Z = stft(seg[ci], fs=sfreq, nperseg=nps, noverlap=nov, boundary=None)
            P = (np.abs(Z) ** 2).astype(np.float64)
            band_mask = (f >= freq_band[0]) & (f <= freq_band[1])
            tp = np.sum(P[band_mask, :], axis=0)
            if tp.size == 0:
                continue
            # normalize
            tp = tp / (np.max(tp) + 1e-12)
            if power_gamma != 1.0:
                tp = np.power(tp, float(power_gamma))
            # resample tp to event samples by simple linear interp
            t_abs = t  # seconds within event
            x_samples = np.linspace(0, (i1 - i0) / sfreq, i1 - i0, endpoint=False)
            tp_s = np.interp(x_samples, t_abs, tp).astype(np.float32)
            # Optional threshold to avoid filling whole window with low-level noise
            tp_s = np.where(tp_s >= float(power_threshold), tp_s, 0.0).astype(np.float32)

            # time centroid on tp_s
            w = (tp_s.astype(np.float64) ** 2)
            denom = np.sum(w)
            if denom <= 1e-12:
                continue
            c = float(np.sum(x_samples * w) / denom)  # seconds within event
            centroids[ci, ej] = c

            if background_mode == "tp":
                bg[ci, boundaries[ej] : boundaries[ej + 1]] = tp_s
            elif background_mode == "blob":
                # Draw a localized blob around centroid (closer to paper visuals)
                sigma = max(1e-3, float(blob_sigma_ms) * 1e-3)
                g = np.exp(-0.5 * ((x_samples - c) / sigma) ** 2).astype(np.float32)
                amp = float(np.max(tp_s)) if tp_s.size > 0 else 0.0
                bg[ci, boundaries[ej] : boundaries[ej + 1]] = np.maximum(
                    bg[ci, boundaries[ej] : boundaries[ej + 1]],
                    (amp * g).astype(np.float32),
                )
            else:
                raise ValueError("background_mode must be 'tp' or 'blob'")

    # Optional smoothing (paper figures are almost never "pixel-noisy")
    if smooth_sigma_ms is not None and float(smooth_sigma_ms) > 0 and bg.size > 0:
        try:
            from scipy.ndimage import gaussian_filter1d

            sigma_samp = float(smooth_sigma_ms) * 1e-3 * float(sfreq)
            sigma_samp = max(0.5, sigma_samp)
            bg = gaussian_filter1d(bg, sigma=sigma_samp, axis=1, mode="nearest")
        except Exception:
            pass

    # Normalize to [0,1] for an interpretable paper-like colorbar.
    if bool(normalize_0_1) and bg.size > 0:
        vmax = float(np.nanmax(bg)) if np.isfinite(bg).any() else 0.0
        if vmax > 1e-12:
            bg = (bg / vmax).astype(np.float32, copy=False)
        bg = np.clip(bg, 0.0, 1.0).astype(np.float32, copy=False)

    fig, ax = plt.subplots(figsize=figsize)
    if x_axis not in ("seconds", "samples"):
        raise ValueError("x_axis must be 'seconds' or 'samples'")
    if x_axis == "seconds":
        extent = (0.0, float(total_len) / sfreq, float(len(labels)), 0.0)
        xlabel = "Concatenated event time (s)"
    else:
        extent = (0.0, float(total_len), float(len(labels)), 0.0)
        xlabel = "Concatenated event samples"

    im = ax.imshow(
        bg,
        aspect="auto",
        cmap="Blues",
        interpolation="nearest",
        origin="upper",
        vmin=0.0,
        vmax=1.0 if bool(normalize_0_1) else None,
        extent=extent,
    )
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Normalized spectrogram (TF-derived) + centroid paths", fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Channels")

    # vertical boundaries
    for b in boundaries:
        ax.axvline(b if x_axis == "samples" else b / sfreq, color="k", linewidth=0.3, alpha=0.4)

    # overlay centroids and per-event path
    for ej in range(len(event_indices)):
        xs = []
        ys = []
        for ci in range(len(rows)):
            c = centroids[ci, ej]
            if not np.isfinite(c):
                continue
            xpix = boundaries[ej] + int(round(c * sfreq))
            xplot = xpix if x_axis == "samples" else xpix / sfreq
            xs.append(xpix)
            ys.append(ci)
            ax.scatter([xplot], [ci], s=cfg.dot_size, c=cfg.event_line_color, alpha=cfg.event_line_alpha, linewidths=0)
        if len(xs) >= 2:
            # sort by x (earliest first) and connect
            order = np.argsort(xs)
            xs2 = [xs[i] for i in order]
            ys2 = [ys[i] for i in order]
            if x_axis == "seconds":
                xs2 = [x / sfreq for x in xs2]
            ax.plot(xs2, ys2, color=cfg.event_line_color, alpha=cfg.event_line_alpha, linewidth=1.0)

    # paper-like frame cleanup
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.0)

    if show_colorbar:
        cb = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
        cb.set_label(str(colorbar_label))
        cb.ax.tick_params(labelsize=10)

    plt.tight_layout()
    return fig


def plot_paper_fig2_normalized_spectrogram(
    *,
    cache_npz_path: str,
    packed_times_path: str,
    detections_npz_path: Optional[str] = None,
    group_analysis_npz_path: Optional[str] = None,
    channel_order: Optional[List[str]] = None,
    event_indices: Optional[List[int]] = None,
    max_events: int = 30,
    freq_band: Tuple[float, float] = (80.0, 250.0),
    nperseg: int = 128,
    noverlap: int = 96,
    centroid_marker_size: float = 30.0,
    centroid_marker_color: str = "#d62728",
    centroid_edge_color: str = "black",
    path_line_width: float = 1.2,
    path_line_alpha: float = 0.85,
    show_colorbar: bool = True,
    cmap: str = "Blues",
    figsize: Tuple[float, float] = (16, 10),
) -> plt.Figure:
    """
    Paper-like Fig2: Per-channel STFT spectrogram (y=frequency) + TF centroids + cross-channel paths.

    **REFACTORED 2026-01-16**: If group_analysis_npz_path is provided, reads pre-computed TF centroids
    from *_groupAnalysis.npz instead of computing them (following new architecture).

    Structure (matching the paper screenshot):
      - Each channel gets its own subplot (vertically stacked)
      - Each subplot has y-axis = frequency (within freq_band)
      - Background is the STFT power spectrogram (computed on-the-fly for visualization)
      - Red dots mark the TF centroid (time, frequency) per event per channel
      - Red lines connect centroids across channels within each event (sorted by time)
    
    Parameters
    ----------
    cache_npz_path : str
        Path to *_envCache.npz (contains bandpassed signal)
    packed_times_path : str
        Path to *_packedTimes.npy
    detections_npz_path : str, optional
        Path to *_gpu.npz (for events_bool mask). Only used if group_analysis_npz_path is None.
    group_analysis_npz_path : str, optional
        Path to *_groupAnalysis.npz (contains pre-computed TF centroids).
        If provided, reads centroids from here instead of computing.
    channel_order, event_indices, max_events : ...
        Channel and event selection
    freq_band : tuple
        Frequency band for STFT display
    ...
    
    Returns
    -------
    fig : plt.Figure
    """
    from scipy.signal import stft
    from .group_event_analysis import load_envelope_cache, load_group_analysis_results

    meta = load_envelope_cache(cache_npz_path)
    x_band = meta.get("x_band", None)
    if x_band is None:
        raise ValueError("Cache does not contain x_band. Regenerate with save_bandpass=True.")

    sfreq = float(meta["sfreq"])
    ch_names = [str(x) for x in meta["ch_names"]]
    packed = np.load(packed_times_path, allow_pickle=True)

    if event_indices is None:
        event_indices = list(range(min(int(max_events), packed.shape[0])))
    else:
        event_indices = [int(i) for i in event_indices][: int(max_events)]

    if channel_order is None:
        channel_order = ch_names
    idx_map = {c: i for i, c in enumerate(ch_names)}
    rows = [idx_map[c] for c in channel_order if c in idx_map]
    labels = [c for c in channel_order if c in idx_map]
    if not rows:
        raise ValueError("No channels matched channel_order.")

    # Build concatenated signal per channel + event boundaries
    boundaries_s = [0.0]
    segs_idx = []
    event_times = []
    for eidx in event_indices:
        s, e = float(packed[eidx, 0]), float(packed[eidx, 1])
        i0 = int(round(s * sfreq))
        i1 = int(round(e * sfreq))
        segs_idx.append((i0, i1))
        event_times.append((s, e))
        boundaries_s.append(boundaries_s[-1] + float(i1 - i0) / sfreq)

    n_ch = len(rows)
    n_ev = len(event_indices)

    use_precomputed = group_analysis_npz_path is not None
    ga = None
    tf_centroid_time = None
    tf_centroid_freq = None
    ga_events_bool = None
    ga_rows = None
    if use_precomputed:
        ga = load_group_analysis_results(group_analysis_npz_path)
        tf_centroid_time = ga.get("tf_centroid_time")
        tf_centroid_freq = ga.get("tf_centroid_freq")
        ga_events_bool = ga.get("events_bool")
        if tf_centroid_time is None or tf_centroid_freq is None:
            warnings.warn("TF centroids not in groupAnalysis, falling back to on-the-fly computation")
            use_precomputed = False
        else:
            ga_idx_map = {c: i for i, c in enumerate(ga["ch_names"])}
            ga_rows = [ga_idx_map.get(lbl) for lbl in labels]

    if not use_precomputed:
        events_bool = np.zeros((n_ch, n_ev), dtype=bool)
        if detections_npz_path is not None:
            gpu = np.load(detections_npz_path, allow_pickle=True)
            det_names = [str(x) for x in gpu["chns_names"].tolist()]
            name_to_i = {n: i for i, n in enumerate(det_names)}
            for ci, ch in enumerate(labels):
                if ch not in name_to_i:
                    continue
                det_times = np.asarray(gpu["whole_dets"][name_to_i[ch]])
                if det_times.size == 0:
                    continue
                for ej, (ev_s, ev_e) in enumerate(event_times):
                    if np.any((det_times[:, 1] > ev_s) & (det_times[:, 0] < ev_e)):
                        events_bool[ci, ej] = True
        else:
            events_bool[:, :] = True

    # Create figure
    fig, axes = plt.subplots(
        n_ch, 1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"hspace": 0.02},
    )
    if n_ch == 1:
        axes = [axes]

    all_centroids = [[None for _ in range(n_ev)] for _ in range(n_ch)]

    # Wavelet params (prefer groupAnalysis metadata)
    if use_precomputed and ga is not None and "tf_freqs_hz" in ga and "tf_n_cycles_vec" in ga:
        freqs_hz = np.asarray(ga["tf_freqs_hz"], dtype=np.float64)
        n_cycles_vec = np.asarray(ga["tf_n_cycles_vec"], dtype=np.float64)
        pool_starts = np.asarray(ga.get("baseline_pool_starts", np.zeros((0,))), dtype=np.float64)
        pool_indices = np.asarray(ga.get("baseline_pool_indices", np.zeros((0,))), dtype=np.int64)
        if "baseline_params" in ga:
            bp = ga["baseline_params"].tolist()[0]
            baseline_window_sec = float(bp.get("window_sec", baseline_window_sec))
        if "tf_baseline_n_select" in ga:
            baseline_n_select = int(np.asarray(ga["tf_baseline_n_select"]).ravel()[0])
        if "tf_baseline_min_distance_sec" in ga:
            baseline_min_distance_sec = float(np.asarray(ga["tf_baseline_min_distance_sec"]).ravel()[0])
    else:
        freqs_hz = bqk_utils.get_wavelet_freqs(freq_band[0], freq_band[1], int(tf_n_freqs), scale=tf_freq_scale)
        mode = str(tf_n_cycles_mode).lower().strip()
        if mode == "fixed":
            n_cycles_vec = np.full(freqs_hz.shape[0], float(tf_n_cycles), dtype=np.float64)
        elif mode == "linear":
            n_cycles_vec = np.linspace(float(tf_n_cycles_min), float(tf_n_cycles_max), freqs_hz.shape[0]).astype(np.float64)
        else:
            n_cycles_vec = (freqs_hz / 2.0).astype(np.float64)
        pool_starts = np.zeros((0,), dtype=np.float64)
        pool_indices = np.zeros((0,), dtype=np.int64)

    def _compute_wavelet_power(x: np.ndarray) -> np.ndarray:
        from scipy.signal import fftconvolve
        x = np.asarray(x, dtype=np.float64)
        power = np.zeros((freqs_hz.shape[0], x.shape[0]), dtype=np.float64)
        for fi, f0 in enumerate(freqs_hz):
            c = float(n_cycles_vec[fi])
            sigma_t = c / (2.0 * np.pi * float(f0))
            half_support = int(np.ceil(3.0 * sigma_t * sfreq))
            tt = (np.arange(-half_support, half_support + 1, dtype=np.float64) / sfreq)
            w = np.exp(2j * np.pi * float(f0) * tt) * np.exp(-(tt**2) / (2.0 * sigma_t**2))
            wn = np.sqrt(np.sum(np.abs(w) ** 2)) + 1e-30
            w = w / wn
            y = fftconvolve(x, w, mode="same")
            power[fi, :] = (np.abs(y) ** 2).astype(np.float64, copy=False)
        return power

    # Pass 1: compute per-channel wavelet TF maps + centroids
    per_channel_data = []
    v99_list = []

    for ci, row in enumerate(rows):
        t_concat = []
        p_concat = []

        for ej, (i0, i1) in enumerate(segs_idx):
            seg = np.asarray(x_band[row, i0:i1], dtype=np.float64)
            if seg.size == 0:
                continue
            power = _compute_wavelet_power(seg)

            # Dynamic baseline (per event)
            if pool_starts.size > 0 and pool_indices.size > 0:
                event_time = 0.5 * (event_times[ej][0] + event_times[ej][1])
                sel = np.argsort(np.abs(pool_starts - event_time))
                sel = sel[np.abs(pool_starts[sel] - event_time) >= float(baseline_min_distance_sec)]
                sel = sel[: int(baseline_n_select)]
            else:
                sel = np.array([], dtype=np.int64)

            if sel.size > 0:
                spectra = []
                win_b = max(1, int(round(float(baseline_window_sec) * sfreq)))
                for pj in pool_indices[sel]:
                    s0 = int(pj)
                    s1 = min(x_band.shape[1], s0 + win_b)
                    if s1 <= s0:
                        continue
                    bseg = np.asarray(x_band[row, s0:s1], dtype=np.float64)
                    bpow = _compute_wavelet_power(bseg)
                    spectra.append(np.mean(bpow, axis=1))
                if spectra:
                    base = np.median(np.stack(spectra, axis=1), axis=1, keepdims=True) + 1e-30
                    power_db = 10.0 * np.log10(power / base)
                else:
                    power_db = 10.0 * np.log10(power + 1e-30)
            else:
                power_db = 10.0 * np.log10(power + 1e-30)

            if power_log1p:
                P_show = np.log1p(np.maximum(power_db, 0.0))
            else:
                P_show = power_db

            t_rel = (np.arange(seg.shape[0], dtype=np.float64) / sfreq) + boundaries_s[ej]
            t_concat.append(t_rel)
            p_concat.append(P_show)

            # TF centroids: read from groupAnalysis or compute on-the-fly
            if use_precomputed:
                ga_ci = ga_rows[ci]
                if ga_ci is not None and ga_events_bool is not None:
                    global_ej = event_indices[ej]
                    if ga_events_bool[ga_ci, global_ej]:
                        t_c_abs = float(tf_centroid_time[ga_ci, global_ej])
                        f_c = float(tf_centroid_freq[ga_ci, global_ej])
                        if np.isfinite(t_c_abs) and np.isfinite(f_c):
                            t_c_concat = boundaries_s[ej] + t_c_abs
                            all_centroids[ci][ej] = (t_c_concat, f_c)
            else:
                if events_bool[ci, ej]:
                    W = np.maximum(power_db, 0.0)
                    denom = float(np.sum(W))
                    if denom > 1e-20:
                        t_rel0 = np.arange(seg.shape[0], dtype=np.float64) / sfreq
                        t_c = float(np.sum(W * t_rel0[None, :]) / denom)
                        f_c = float(np.sum(W * freqs_hz[:, None]) / denom)
                        all_centroids[ci][ej] = (boundaries_s[ej] + t_c, f_c)

        if not t_concat or not p_concat:
            t = np.array([0.0], dtype=np.float64)
            P_show = np.zeros((freqs_hz.shape[0], 1), dtype=np.float64)
        else:
            t = np.concatenate(t_concat, axis=0)
            P_show = np.concatenate(p_concat, axis=1)
        if np.isfinite(P_show).any():
            v99_list.append(float(np.nanpercentile(P_show, 99)))

        per_channel_data.append({
            "t": t,
            "f_band": freqs_hz,
            "P_show": P_show,
            "label": labels[ci],
        })

    shared_vmax = max(v99_list) if v99_list else 1.0
    shared_vmax = max(shared_vmax, 1e-12)

    # Pass 2: plot
    last_im = None
    for ci, (ax, data) in enumerate(zip(axes, per_channel_data)):
        last_im = ax.pcolormesh(
            data["t"],
            data["f_band"],
            data["P_show"],
            shading="auto",
            cmap=cmap,
            vmin=0.0,
            vmax=shared_vmax,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="both", labelsize=10)

        if ci != n_ch - 1:
            ax.tick_params(axis="x", bottom=False, labelbottom=False)

        ax.set_ylabel(data["label"], fontsize=11, rotation=0, labelpad=25, va="center")
        ax.set_yticks([])

        for b in boundaries_s[1:-1]:
            ax.axvline(b, color="k", linewidth=0.4, linestyle="--", alpha=0.4)

        # Centroid dots
        for ej in range(n_ev):
            pt = all_centroids[ci][ej]
            if pt is not None:
                ax.scatter(
                    [pt[0]], [pt[1]],
                    s=float(centroid_marker_size),
                    c=centroid_marker_color,
                    alpha=0.95,
                    linewidths=0.5,
                    edgecolors=centroid_edge_color,
                    zorder=5,
                )

        ax.set_xlim(boundaries_s[0], boundaries_s[-1])
        ax.margins(x=0.0)

    # Pass 3: cross-channel paths (top-to-bottom channel order)
    for ej in range(n_ev):
        event_pts = []
        for ci in range(n_ch):
            pt = all_centroids[ci][ej]
            if pt is not None:
                event_pts.append((pt[0], pt[1], ci, axes[ci]))
        if len(event_pts) < 2:
            continue
        # keep channel order; do not sort by time
        from matplotlib.patches import ConnectionPatch
        for k in range(len(event_pts) - 1):
            t1, f1, ci1, ax1 = event_pts[k]
            t2, f2, ci2, ax2 = event_pts[k + 1]
            con = ConnectionPatch(
                xyA=(t1, f1), coordsA=ax1.transData,
                xyB=(t2, f2), coordsB=ax2.transData,
                color=centroid_marker_color,
                linewidth=float(path_line_width),
                alpha=float(path_line_alpha),
                zorder=4,
            )
            fig.add_artist(con)

    axes[0].set_title(
        f"Normalized Spectrogram ({freq_band[0]:.0f}–{freq_band[1]:.0f} Hz)",
        fontsize=14,
        fontweight="bold",
    )
    axes[-1].set_xlabel("Time (s)", fontsize=12)

    if show_colorbar and last_im is not None:
        cb = fig.colorbar(last_im, ax=axes, fraction=0.018, pad=0.015)
        cb.set_label("log1p(power)", fontsize=11)
        cb.ax.tick_params(labelsize=10)

    fig.subplots_adjust(hspace=0.02)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            plt.tight_layout()
        except Exception:
            pass
    return fig


def plot_lag_heatmaps(
    *,
    energy: np.ndarray,
    lag_ms: np.ndarray,
    rank: np.ndarray,
    ch_names: List[str],
    event_ids: List[int],
    cmap_energy: str = "viridis",
    cmap_lag: str = "RdYlBu_r",
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

    def _imshow(mat, title, cmap, vmin=None, vmax=None):
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
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
    fig1 = _imshow(e, "Event envelope energy (channels × events)", cmap_energy, vmin=0.0, vmax=vmax)

    r = rank.copy()
    r[~np.isfinite(r)] = np.nan
    fig2 = _imshow(r, "Centroid order (rank; 0=earliest)", cmap_rank)

    l = lag_ms.copy()
    l[~np.isfinite(l)] = np.nan
    # clamp to physiological window for readability
    if np.isfinite(l).any():
        v = np.nanpercentile(np.abs(l), 99)
        v = max(v, 1.0)
    else:
        v = None
    fig3 = _imshow(l, "Lag (ms) aligned to first centroid", cmap_lag, vmin=-v if v is not None else None, vmax=v)
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
