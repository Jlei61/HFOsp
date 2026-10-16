"""Original Fig1B painters; contact count, labels and canvas are input parameters."""
import numpy as np
import matplotlib.pyplot as plt
from analysis.event_spectrogram import compute_smoothed_magnitude_spectrogram

def draw_spectrum(fig, events, positions, colorbar, title_baseline, *, large=False, halfwidth=.15, channel_names=None, display_label="E1146"):
    import matplotlib.patheffects as pe
    from matplotlib.ticker import FuncFormatter
    width, height = fig.get_size_inches()
    nc,nf=events[0]["specs"].shape[:2]
    axes = []
    for i, (event, pos) in enumerate(zip(events, positions)):
        ax = fig.add_axes([pos[0]/width, pos[1]/height, pos[2]/width, pos[3]/height])
        times = event["times"]
        edges = np.r_[times[0]-.005, (times[:-1]+times[1:])/2, times[-1]+.005]
        # Preserve contact order top-to-bottom; frequency increases upward within
        # each channel, as in the original spectrum coordinate convention.
        values = event["specs"][:, ::-1, :].reshape(nc*nf, -1)
        im = ax.pcolormesh(edges*1000, np.arange(nc*nf+1), values, cmap="coolwarm", vmin=0, vmax=1, rasterized=True)
        for row in range(1, nc): ax.axhline(row*nf, color=".75", lw=.4, ls="--")
        ax.axvline(0, color="white", lw=.6, alpha=.75)
        x = event["centers"][:, 0]*1000
        y = np.arange(nc)*nf + nf-event["centers"][:, 1]-.5
        line, = ax.plot(x, y, color="#151515", lw=1.45, zorder=6)
        line.set_path_effects([pe.Stroke(linewidth=2.65, foreground="white"), pe.Normal()])
        ax.scatter(x, y, s=18, facecolor="white", edgecolor="#151515", lw=.9, zorder=7)
        ax.set(xlim=(-halfwidth*1000, halfwidth*1000), ylim=(nc*nf, 0),
               xticks=[-halfwidth*1000, halfwidth*1000] if large else [-halfwidth*1000, 0, halfwidth*1000],
               yticks=(np.arange(nc)+.5)*nf)
        ax.set_yticklabels(channel_names if i == 0 else [])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: "0" if abs(value)<1e-9 else f"{value/1000:.2f}"))
        ax.tick_params(axis="y", length=0, labelsize=15 if large else 8.5, pad=3)
        ax.tick_params(axis="x", labelsize=17.5 if large else 9, pad=3, length=4 if large else 3)
        ax.get_xticklabels()[0].set_ha("left"); ax.get_xticklabels()[-1].set_ha("right")
        ax.spines[["top","right"]].set_visible(False)
        if i == 1: ax.set_xlabel("Time (s)", fontsize=21.25 if large else 12, labelpad=6)
        axes.append(ax)
        np.testing.assert_array_equal(line.get_xdata(), event["centers"][:,0]*1000)
    x,y,w,h = colorbar
    cax=fig.add_axes([x/width,y/height,w/width,h/height]);axes.append(cax)
    cb=fig.colorbar(im,cax=cax,ticks=[0,1]);cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=16.25 if large else 9,length=0,pad=3)
    fig.text(positions[0][0]/width,title_baseline/height,display_label,ha="left",va="baseline",
             fontsize=25 if large else 12,fontweight="bold")
    return axes

def _mean_spectrograms(
    snippets: np.ndarray,
    fs: float,
    freq_max: float = 240.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the legacy raw and baseline-normalized mean spectrograms."""
    win = int(round(0.18 * fs))
    overlap = int(round(0.16 * fs))
    specs = []
    spec_t = spec_f = None
    for row in snippets:
        freqs, times, magnitude = compute_smoothed_magnitude_spectrogram(
            row,
            fs,
            window="hann",
            window_sec=win / float(fs),
            overlap_sec=overlap / float(fs),
            freq_range_hz=(0.0, float(freq_max)),
            gaussian_sigma=1.5,
        )
        specs.append(magnitude)
        spec_t = times
        spec_f = freqs
    if spec_t is None or spec_f is None:
        raise ValueError("no HFO spectrograms were computed")
    mean_spec = np.mean(np.asarray(specs), axis=0)
    raw_spec = np.log(np.maximum(mean_spec, np.finfo(float).tiny))
    baseline = mean_spec[:, spec_t <= 0.15]
    if baseline.shape[1] == 0:
        raise ValueError("legacy baseline t<=0.15 s contains no spectrogram bins")
    baseline_mean = np.mean(baseline, axis=1, keepdims=True)
    norm_spec = mean_spec / np.maximum(baseline_mean, np.finfo(float).tiny) - 1.0
    return raw_spec, norm_spec, spec_t, spec_f

def _full_extent_edges(centers: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Return monotonic pcolormesh edges with exact requested outer bounds."""
    values = np.asarray(centers, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.diff(values) > 0):
        raise ValueError("centers must be a strictly increasing 1D array with >=2 values")
    if not lower < values[0] or not values[-1] < upper:
        raise ValueError(f"outer bounds {lower, upper} must contain center range")
    edges = np.empty(values.size + 1, dtype=np.float64)
    edges[0] = float(lower)
    edges[-1] = float(upper)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    if not np.all(np.diff(edges) > 0):
        raise ValueError("derived pcolormesh edges are not increasing")
    return edges

def _frequency_edges(freqs: np.ndarray) -> np.ndarray:
    values = np.asarray(freqs, dtype=np.float64)
    step = float(np.median(np.diff(values)))
    edges = np.empty(values.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = max(0.0, values[0] - 0.5 * step)
    edges[-1] = values[-1] + 0.5 * step
    return edges

def draw_hfo(fig, snippets, fs, axes):
    raw,norm,t,f=_mean_spectrograms(snippets,fs)
    duration=snippets.shape[1]/fs
    te=_full_extent_edges(t,0.,duration);fe=_frequency_edges(f)
    a,b,c=axes
    time = np.arange(snippets.shape[1])/fs
    a.plot(time, snippets.T, color="black", lw=.28, alpha=.23)
    a.plot(time, snippets.mean(0), color="#FFD000", lw=1.05)
    a.set_title(f"HFO n = {len(snippets)}", color="red", fontsize=12.5, pad=4)
    lim = np.nanpercentile(raw, [1., 99.])
    norm_lim = float(np.nanpercentile(np.abs(norm), 99.))
    b.pcolormesh(te, fe, raw, cmap="coolwarm", vmin=lim[0], vmax=lim[1], rasterized=True)
    c.pcolormesh(te, fe, norm, cmap="coolwarm", vmin=-norm_lim, vmax=norm_lim, rasterized=True)
    for ax, name in ((b, "raw Spec"), (c, "normalized Spec")):
        ax.set_title(name, fontsize=10.5, pad=3)
        ax.set_ylabel("Freq (Hz)", fontsize=9.5, labelpad=5)
        ax.set(ylim=(0, 240), yticks=[0, 100, 200])
    c.set_xlabel("Time (s)", fontsize=10, labelpad=5)
    for ax in axes:
        ax.set(xlim=(0, .6), xticks=[0, .25, .5])
        ax.set_xticklabels(["0.00", "0.25", "0.50"])
        ax.tick_params(labelsize=9, length=2.5, pad=2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_linewidth(.55)
    return axes
