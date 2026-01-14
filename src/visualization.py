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
from matplotlib.patches import Rectangle
from typing import List, Optional, Union, Tuple, Dict
from pathlib import Path

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
        if ElectrodeParser is not None:
            prefix, _ = ElectrodeParser.parse(ch)
        else:
            # Simple fallback: first letter
            prefix = ch[0] if ch else 'X'
        
        if prefix not in shaft_color_map:
            palette = tableau_20_no_red if len(tableau_20_no_red) > 0 else SEEG_COLORS["shaft_colors"]
            shaft_color_map[prefix] = palette[color_idx % len(palette)]
            color_idx += 1
        
        colors.append(shaft_color_map[prefix])
    
    return colors


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
    if show_fast_ripple:
        fr_band = (250.0, 500.0)
        x_fr = bqk.band_filt(x, sfreq, fr_band)
        env_fr = bqk.return_hil_enve_norm(x, sfreq, fr_band)
    else:
        x_fr, env_fr = None, None

    n_ch = len(ch_indices)
    n_cols = 3 if show_fast_ripple else 2
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

        if show_fast_ripple:
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
    
    edf_path = '/Volumes/Elements/yuquan_24h_edf/chengshuai/FC10477Q.edf'
    
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
