# Topic 4 rev10-D4: uniform forced-source route map

## Question

The frozen continuous Node field spontaneously preserves patient-supported mode
B across networks but not mode A. Static spatial edges, local adaptation, local
inhibitory resource and source-specific E->E depression did not restore shared
mode A. D4 introduces no mechanism. It asks whether the existing substrate can
express mode A when initiation location is externally controlled.

```text
forced A capacity + spontaneous A absent -> nucleation/accessibility gap
forced A capacity absent                 -> route capacity gap
```

## Intervention contract

Use a `5 x 5` Cartesian grid with coordinates `{2,6,10,14,18} mm` on both sheet
axes. At every point force the nearest `0.5%` of E neurons to spike once at
`100 ms`. Source selection cannot use contacts, shafts, patient labels, Node
field values, Gaussian components or peaks. The Node anchor remains
`v62_density_t050`; all static edge coefficients and dynamic states are off.

For each network, run one sham and reset the same dynamics RNG before every
forced source. The pre-trigger E spike raster must be bit-identical. Remove the
injected packet frame from primary readout, subtract the sham envelope, and use
the fixed `100-250 ms` paired-response window. A source response is clean only
when it has a returned time-locked event, recruits both shafts, and is inside
the frozen patient-training support.

## Frozen execution

- network seeds: `1201-1203`, reused from D3 for paired localization rather
  than treated as fresh confirmation;
- simulation: 400 ms, forced packet at 100 ms, latency at most 40 ms;
- detector: common absolute active-fraction threshold `0.0195703125`;
- source count: 25 per network, plus one sham;
- launcher: `systemd-run --user -> nohup`, measured-RSS sentinel, 180 s waits,
  one numeric thread per worker and completion notification.

## Decision

`FORCED_MODE_A_ROUTE_CAPACITY_OBSERVED` requires the same uniform source point
to produce clean A in at least `2/3` networks. A single-network or source-varying
A response is descriptive heterogeneity, not shared capacity. Mode B support is
reported as an internal positive control.

This diagnostic can identify the next mechanism class but cannot establish
spontaneous patient-mode recovery, patient generalization, a clinical waveform,
a core, or an ictal lifecycle. The accepted Fig.4 pair is replaced only after a
positive mechanism is frozen and confirmed on new networks.
