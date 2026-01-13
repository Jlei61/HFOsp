import matplotlib as mpb
import matplotlib.pyplot as plt
from matplotlib import colors, cm
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np

def visualize_SC_and_Gain(SC, Gr):
    '''Visualize Structural Connectivity and Gain matrices
    
    Parameters
    ----------
    SC : numpy.ndarray
        Structural Connectivity matrix
    Gr : pandas.DataFrame
        Gain matrix
        
    Returns
    -------
    None
        Displays the visualization
    '''
    plt.figure(figsize=(18, 14))  # Increased overall figure width
    plt.subplot(121)
    norm = colors.LogNorm(1e-7, SC.max())
    im = plt.imshow(SC,norm=norm,cmap=cm.Blues)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.gca().set_title('Strcutural Connectivity', fontsize=24.0)
    plt.ylabel('#Regions', fontsize=22.0)
    plt.xlabel('#Regions', fontsize=22.0)

    plt.subplot(122)
    cols = [(1, 0, 0), (1, 1, 1), (0, 0, 1)]  # R -> W -> B
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom1', cols, N=10)

    im = plt.imshow(Gr.values,cmap=custom_cmap, vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.03, pad=0.06)  # Slightly larger colorbar
    plt.gca().set_title('Gain Matrix', fontsize=24.0)
    plt.xticks(fontsize=16, rotation=0)
    plt.yticks(fontsize=16, rotation=0)
    plt.xlabel('#Regions', fontsize=22.0)
    plt.ylabel('#Sensors', fontsize=22.0)

    plt.tight_layout()
    plt.show()

def plot_source(Sim_source, times, region_names, ez_idx, pz_idx, nc=100):
    """
    Plot source brain activity with different colors for ez, pz, and hz regions.
    
    Parameters:
    -----------
    Sim_source : array
        Simulated source data with shape (n_regions, n_timepoints)
    times : array
        Time points for the simulation
    region_names : array
        Names of brain regions
    ez_idx : list
        Indices of epileptogenic zones
    pz_idx : list
        Indices of propagation zones
    nc : int, optional
        Number of channels to plot (default: 100)
    
    Returns:
    --------
    fig : matplotlib figure
        The figure object containing the plot
    """
    fig = plt.figure(figsize=(24,24))
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False

    # cmap
    cmp = [
        "#D80032",  # Red
        "#64CCC5",  # Cyan
        "#053B50",  # Deep Sky Blue
        "#7F00FF"   # Blue-Violet
    ]

    # Create horizontal spacing between channels
    offset = 2
    for i in range(0, nc):
        y_offset = (nc - i) * offset  # Reverse order to plot from top to bottom
        if i in ez_idx:
            plt.plot(times, Sim_source[i,:] + y_offset, cmp[0], lw=3)
        elif i in pz_idx:
            plt.plot(times, Sim_source[i,:] + y_offset, cmp[1], lw=3)
        else:
            plt.plot(times, Sim_source[i,:] + y_offset, cmp[2], lw=0.5)

    # Add dummy lines for legend
    plt.plot([], [], cmp[0], lw=3, label='ez')
    plt.plot([], [], cmp[1], lw=3, label='pz')
    plt.plot([], [], cmp[2], lw=0.5, label='hz')

    # Set y-ticks at the center of each channel's position
    y_positions = [(nc - i) * offset for i in range(nc)]
    plt.yticks(y_positions, region_names[:nc], fontsize=24)

    plt.xticks(np.arange(0,11), fontsize=24)
    plt.title("Source brain activity", fontsize=24)
    plt.xlabel('Time(s)', fontsize=25)
    plt.ylabel('Brain Regions', fontsize=25)
    plt.ylim([0, (nc + 1) * offset])  # Add some padding at top and bottom
    plt.xlim([0, 10])
    plt.legend(fontsize=24, loc='upper right')
    plt.tight_layout()
    
    return fig

def plot_simEEG(Sim_seeg, times, Chn_names, ez_idx, nc=10, T=100.0):
    """
    Plot simulated EEG activity with different colors for ez and hz regions.
    
    Parameters:
    -----------
    Sim_seeg : array
        Simulated SEEG data with shape (n_channels, n_timepoints)
    times : array
        Time points for the simulation
    Chn_names : array
        Names of EEG channels
    ez_idx : list
        Indices of epileptogenic zones
    nc : int, optional
        Number of channels to plot (default: 10)
    T : float, optional
        Total time duration (default: 100.0)
    
    Returns:
    --------
    fig : matplotlib figure
        The figure object containing the plot
    """
    fig = plt.figure(figsize=(24,13))
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False

    # cmap
    cmp = [
        "#D80032",  # Red
        "#64CCC5",  # Cyan
        "#053B50",  # Deep Sky Blue
        "#7F00FF"   # Blue-Violet
    ]

    # Calculate offset based on the mean of the signal
    offset = 5
    
    for i in range(0, nc):
        y_offset = offset * i
        if i in ez_idx:
            plt.plot(times, Sim_seeg[i,:] + y_offset, cmp[0], lw=0.5)
        else:
            plt.plot(times, Sim_seeg[i,:] + y_offset, cmp[2], lw=0.5)
            
    # Adjusting yticks to align with the mean of each signal
    tick_positions = []
    for i in range(nc):
        # Use the mean of the signal plus the offset to position the ytick
        mean_position = np.mean(Sim_seeg[i,:]) + (offset * i)
        tick_positions.append(mean_position)

    plt.yticks(tick_positions, Chn_names[:nc], fontsize=24)

    plt.xticks(np.arange(0,11,1), fontsize=24)
    plt.title("Simulation EEG", fontsize=24)
    plt.xlabel('Time(s)', fontsize=25)
    plt.ylabel('Electrodes(#)', fontsize=25)
    plt.ylim([min(np.min(Sim_seeg), 0), offset*nc + np.max(Sim_seeg)])
    plt.xlim([0, 10])
    
    plt.tight_layout()
    plt.close()
    return fig