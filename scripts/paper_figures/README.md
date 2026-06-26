# Paper Figure Scripts

This directory contains scripts used to generate manuscript-facing figures.
Keep final figure scripts here rather than scattering one-off plotting code in
the top-level `scripts/` directory.

## Fig1 HFO Group Event Demo

Formal entry point:

```bash
python scripts/paper_figures/plot_fig1_hfo_group_event_legacy_style.py
```

Default output:

```text
results/paper-ready-figure/fig1_hfo_group_event_demo/figures/
```

The script rebuilds the old ReplayIED-style panel from Yuquan Y1 using the
legacy artifact chain:

- EDF: `/mnt/yuquan_data/yuquan_24h_edf/chengshuai/FC10477Q.edf`
- HFO detections: `/mnt/yuquan_data/yuquan_24h_edf/chengshuai/FC10477Q_gpu.npz`
- packed group events: `/mnt/yuquan_data/yuquan_24h_edf/chengshuai/FC10477Q_packedTimes.npy`

Current accepted visual contract:

- fixed clean group-event examples: packed event indices `22,237,1458`
- compact panel width
- left panel: 80-250 Hz stacked bipolar traces
- right panel: legacy-style normalized spectrogram with spec-center trajectories
- channel names shown only on the left
- E9/K10 display rows excluded from the default channel candidate set

`plot_fig1_hfo_group_event_prototype.py` is retained as an earlier prototype
that used current pipeline TF-cache outputs. It is not the accepted manuscript
entry point for this panel.
