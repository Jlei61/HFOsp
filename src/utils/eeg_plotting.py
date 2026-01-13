"""
Standard EEG Plotting Utilities

This module contains standardized plotting functions for EEG data visualization.
All functions in this module follow consistent styling and are optimized for clarity.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_eeg_segment(data, sfreq, ch_names, start_sec=0, duration_sec=10, title="EEG Data Segment"):
    """
    根据Numpy数组绘制EEG数据片段，并按脑区分组。
    """
    start_sample = int(start_sec * sfreq)
    end_sample = int((start_sec + duration_sec) * sfreq)
    
    segment_data = data[:, start_sample:end_sample]
    time_vec = np.arange(start_sample, end_sample) / sfreq
    
    left_channels = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1']
    middle_channels = ['Fz', 'Cz', 'Pz'] 
    right_channels = ['Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']
    colors = {'left': '#1f77b4', 'middle': '#2ca02c', 'right': '#ff7f0e'}
    
    fig, ax = plt.subplots(figsize=(18, 12))
    offset, yticks, yticklabels = 0, [], []
    CHANNEL_SPACING = 150
    
    def plot_channel_group(std_ch_names, color):
        nonlocal offset
        for ch_name in std_ch_names:
            if ch_name in ch_names:
                ch_idx = ch_names.index(ch_name)
                y_data = segment_data[ch_idx, :] * 1e6 + offset
                ax.plot(time_vec, y_data, linewidth=0.8, color=color)
                yticks.append(offset)
                yticklabels.append(ch_name)
                offset -= CHANNEL_SPACING
    
    plot_channel_group(left_channels, colors['left'])
    plot_channel_group(middle_channels, colors['middle'])
    plot_channel_group(right_channels, colors['right'])
    
    ax.set_ylim(offset + 50, 50)
    ax.set_xlim(time_vec[0], time_vec[-1])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Channels (Amplitude in μV)')
    ax.set_title(title)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    x_scale_bar = time_vec[-1] - 1.2
    y_scale_bar = offset + CHANNEL_SPACING * 0.5
    ax.plot([x_scale_bar, x_scale_bar + 1], [y_scale_bar, y_scale_bar], 'k-', linewidth=2)
    ax.plot([x_scale_bar, x_scale_bar], [y_scale_bar, y_scale_bar + 100], 'k-', linewidth=2)
    ax.text(x_scale_bar + 0.5, y_scale_bar - CHANNEL_SPACING * 0.2, '1 s', ha='center')
    ax.text(x_scale_bar - 0.05, y_scale_bar + 50, '100 μV', va='center', ha='right', rotation=90)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig



def plot_eeg_with_annotations(raw, annotations, start_sec=0, duration_sec=30, 
                             title="EEG Data with Annotations", save_path=None):
    """
    Plot EEG data with annotations/detections marked
    
    Parameters:
    -----------
    raw : mne.io.Raw
        Raw EEG data object
    annotations : list of dict
        List of annotation dictionaries with 'start_time' and 'duration' keys
    start_sec : float
        Start time in seconds
    duration_sec : float
        Duration to plot in seconds
    title : str
        Plot title
    save_path : str or Path, optional
        If provided, save the plot to this path
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object
    """
    # First create the standard EEG plot
    fig = plot_eeg_segment(raw, start_sec, duration_sec, title)
    ax = fig.gca()
    
    # Add annotations as vertical shaded regions
    end_sec = start_sec + duration_sec
    for ann in annotations:
        ann_start = ann.get('start_time', 0)
        ann_duration = ann.get('duration', 0.5)
        ann_end = ann_start + ann_duration
        
        # Only show annotations that overlap with the current time window
        if ann_end >= start_sec and ann_start <= end_sec:
            ax.axvspan(ann_start, ann_end, alpha=0.3, color='red', label='IED Detection')
    
    # Add legend if we have annotations
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles[:1], labels[:1], loc='upper right')  # Only show one instance in legend
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    return fig 