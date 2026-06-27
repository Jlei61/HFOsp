# Paper Figure Scripts

This directory contains scripts used to generate manuscript-facing figures.
Keep final figure scripts here rather than scattering one-off plotting code in
the top-level `scripts/` directory.

## Fig1-A HFO Group Event Demo

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

- manuscript panel label: `Fig1-A`
- fixed clean group-event examples: packed event indices `22,237,1458`
- compact panel width
- left panel: 80-250 Hz stacked bipolar traces
- right panel: legacy-style normalized spectrogram with spec-center trajectories
- both x axes start at `0`; spectrogram image extent is clamped to the full
  concatenated event duration to avoid leading/trailing blank x margins
- channel names shown only on the left
- E9/K10 display rows excluded from the default channel candidate set

`plot_fig1_hfo_group_event_prototype.py` is retained as an earlier prototype
that used current pipeline TF-cache outputs. It is not the accepted manuscript
entry point for this panel.

## Fig3-A Field Concordance Cohort Statistic

Formal entry point:

```bash
python scripts/paper_figures/plot_fig3_field_concordance_cohort_stat.py
```

Default output:

```text
results/paper-ready-figure/fig3_field_concordance_cohort_stat/figures/
```

The script is plotting-only and consumes the Topic 5 field-concordance
axis-alignment artifacts:

- `results/topic5_ictal_recruitment/axis_alignment/axis_alignment_broadband_max_ab_B1000.json`
- `results/topic5_ictal_recruitment/axis_alignment/axis_alignment_hfa_max_ab_B1000.json`

Current visual contract:

- manuscript panel label: `Fig3-A`
- Data-vs-Null violin + box + subject dots, not a per-subject board
- two comparisons: `Broadband maxAB` and `HFA maxAB`
- strict maxAB only; no broad fallback
- maxAB evaluable n is 19 because `yuquan_xuxinyi` is broad-only
- no background grid lines
- Null is the matched channel-shuffle median
- interpretation: shared coarse field axis, not pointwise directional replay

## Fig5-A Core Model Stage-3 Brake-Off

Formal entry point:

```bash
python scripts/paper_figures/plot_fig5_core_model_s3_brakeoff.py
```

Default output:

```text
results/paper-ready-figure/fig5_core_model_s3_brakeoff/figures/
```

The script is plotting-only and consumes the existing Topic 4 SNN artifacts:

- `results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/readout_s3_brakeoff.json`
- `results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/per_event/rep_s3_brakeoff_neg.npz`
- `results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/per_event/rep_s3_brakeoff_pos.npz`

Current accepted visual contract:

- manuscript panel label: `A`
- one-row SNN simulation standard: mechanism schematic, tempA source event, tempB source event, fused virtual-SEEG readout
- mechanism panel explicitly marks the E->E long-axis range
- readout event windows are shaded by direction: forward and reverse propagation events use different colors
- the compatibility path `results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/figures/core_model_s3_brakeoff.png` is also regenerated

## Subject-specific SNN (E1146) — Fig4-A/B

Fig4A formal entry point:

```bash
python scripts/paper_figures/plot_fig_subject_snn.py \
  --twoend-tag epilepsiae_1146_twoend_equal_tsrc_s3 \
  --source-tag epilepsiae_1146_source_tsrc_s1 \
  --sink-tag epilepsiae_1146_sink_tsrc_s1 \
  --fig-name fig_subject_snn_epilepsiae_1146
```

Fig4B KMeans verification (4 blocks: clustered heatmap | per-channel rank | cluster profile | model-vs-real
similarity matrix with ★ perm-p). Rank heatmap/distribution/profile use the mature canonical plotters;
heatmap rank colorbar is horizontal under the x-label; the 2×2 matrix uses stars only (no numbers), aspect=equal.

```bash
python scripts/paper_figures/plot_fig_subject_snn_kmeans2.py \
  --tag epilepsiae_1146_twoend_equal_tsrc_s3 \
  --fig-name fig_subject_snn_epilepsiae_1146
```

**LOCKED pattern**: each subject-SNN case = Fig4A (4-col readout) + Fig4B (4-block KMeans+matrix) as the
two main figures; Fig4C/Fig4D are optional supplements.

Default output: `results/paper-ready-figure/fig_subject_snn_epilepsiae_1146/figures/` (png + pdf + metadata + README).

Same one-row-four-column SNN standard as Fig5, but the substrate is placed on the
**patient's real electrode layout** (E1146 ICL strip). The two low-V_th cores are the
**earliest-3 electrodes of each interictal template** (`template_source` placement in
`src/sef_hfo_subject_placement.py`), real-geometry plane-fit (cores ~13 mm apart naturally,
no core-anchoring), stage3 params (m17.5/std1.0). Consumes
`results/topic4_sef_hfo/field_swap_subject_snn/{readout,figdata}_<tag>.npz/json` from
`scripts/run_sef_hfo_subject_snn.py`.

Fig4B consumes the same spontaneous twoend seed3 readout and runs plotting-level
`KMeans k=2` on clean directional events only. It is a readout clustering verification
panel, not a new simulation or cohort statistic.

Accepted contract / honesty:
- mechanism panel shows both template-source cores overlapping the electrode interictal-onset
  (earliest) region + the E->E long-axis band.
- readout = spontaneous twoend run; spontaneous bidirectionality is seed-dependent
  (seed3 balanced 6/8; seed1/2 reverse-dominant 1/8,1/7).
- Fig4B clean events: n=14, C0/C1 = 6/8, direction purity = 1.00,
  within-cluster tau = 0.939, shared-overlap corr = -0.69.
- `k_dir=2` sparse-electrode readout (load-bearing relaxation vs the standard k_dir=3).
- E958 (sparse subdural grid) is a negative case (events too local to read direction).

### Fig4C — real-vs-model interictal template consistency (E1146)

```bash
python scripts/paper_figures/plot_fig_subject_snn_realvsmodel.py
```

A = real E1146 interictal templates (t_a/t_b per-channel `typical_rank`); B = subject-SNN model
templates (forward/reverse cluster mean rank from the same twoend readout). Consistency by per-channel
Spearman: **model forward vs real t_a ρ=+0.87 (n=7); model reverse vs real t_b ρ=+0.62 (n=11)**;
cross terms negative → the model readout reproduces the real interictal template order (swap structure
preserved). Single-subject, read-out-level (k_dir=2, seed3); not a mechanism/cohort claim. B is empty
at SCL rows because the model events read out on the ICL contacts.

### Fig4D — model-vs-data template similarity statistic (E1146)

```bash
python scripts/paper_figures/plot_fig_subject_snn_similarity.py
```

2x2 Spearman similarity matrix (model fwd/rev × data t_a/t_b) + channel-shuffle permutation null
(B=10000). forward~t_a ρ+0.87 p=0.016; reverse~t_b ρ+0.62 p=0.023; combined swap-consistency
S=mean(diag)-mean(off-diag)=+1.72, perm p=0.0024 (n_ch=7). CAVEAT: cores placed at data
template-source channels -> endpoint match partly built in; the permutation tests full-channel-order
alignment beyond chance, NOT construction specificity. Specificity needs the core-location/axis-rotation
null (re-sim with mislocated cores; §3D).
