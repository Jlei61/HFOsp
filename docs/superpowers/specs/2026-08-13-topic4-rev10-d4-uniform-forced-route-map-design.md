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

## Outcome

All three network workers completed through the managed launcher. The sentinel
used `2.33 GiB` peak RSS with zero swap, and the two remaining workers ran in
parallel. All 75 forced responses had bit-identical pre-trigger spike rasters
relative to their network-matched sham, and none entered runaway.

One source, fixed a priori at `(18,6) mm`, produced clean mode A in `3/3`
networks. Per-network patient direction probabilities were strongly A-like
(`P(B)=0.00059, 0.00155, 0.000096`), and class-conditional OOD distances were
`41.25, 35.85, 25.13`, all below the frozen A threshold `47.22`. Responses were
returned, recruited both shafts and included `6, 5, 9` contacts. Their onset was
`138-140 ms`, about 38-40 ms after the packet; downstream positive spike mass
was about 72k and `r90` about 19 mm.

Mode B provided a spatially separate positive control: `(2,14)`, `(6,14)` and
`(2,18) mm` each produced clean B in `3/3` networks. The selected A source had
mean Node field `h=2.6e-6`, while the strongest B source `(2,18)` had mean
`h=0.844`. Thus mode A route capacity exists outside the current high-h Node
support; the observed spontaneous asymmetry is primarily consistent with a
nucleation/accessibility gap rather than total absence of an A propagation
route.

The formal status is:

```text
REV10D4_UNIFORM_FORCED_MODE_A_ROUTE_CAPACITY_OBSERVED
```

This does not yet prove that physiological spontaneous fluctuations can access
the route, because the fixed packet recruits `0.5%` of E neurons. The next step
is a pre-frozen packet-dose confirmation at the selected A source and a matched
B source on fresh network seeds. Only after a stable minimum packet is found may
a smooth, observation-invariant accessibility mechanism be designed. No source
point is added as a Gaussian core, contact anchor or direct fitted field peak.
