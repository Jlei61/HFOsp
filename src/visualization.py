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
        '#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
        '#ff7f00', '#ffff33', '#a65628', '#f781bf',
        '#999999', '#66c2a5', '#fc8d62', '#8da0cb'
    ]
}


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
            shaft_color_map[prefix] = SEEG_COLORS['shaft_colors'][color_idx % len(SEEG_COLORS['shaft_colors'])]
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
        
        # Skip events outside time window
        if evt_end < t_start or evt_start > t_end:
            continue
        
        # If channel specified, highlight only that channel
        if evt_ch and evt_ch in ch_labels:
            ch_idx = ch_labels.index(evt_ch)
            y_center = yticks[ch_idx]
            height = spacing * 0.8
            
            rect = Rectangle(
                (evt_start, y_center - height/2),
                evt_end - evt_start,
                height,
                facecolor='red',
                alpha=0.3,
                edgecolor='red',
                linewidth=1
            )
            ax.add_patch(rect)
        else:
            # Highlight across all channels
            ax.axvspan(evt_start, evt_end, alpha=0.2, color='red')


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
