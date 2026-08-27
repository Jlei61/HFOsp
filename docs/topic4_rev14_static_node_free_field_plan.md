# Topic 4 rev14 execution plan

## Phase 0: zero-simulation reference (complete)

1. Implemented a dedicated training-only producer for `exact_off` seeds
   2311--2313.
2. Freeze original event indices, causal-family masks, input hashes and contact
   order.
3. Produced per-network and equal-network legacy diagnostics plus the frozen
   `J14_v1` reference, OOD/shaft support, natural KMeans diagnostics and a
   Fig.4 bundle.
4. Verified the frozen counts: isolated families 26/31/29 and isolated readable
   families 24/26/22.
5. Passed zero-simulation controls for SCL censoring, single-mode collapse,
   duplicated-root rejection, overlap-compound inflation and zero events.
6. Archive the static-baseline diagnosis before defining new candidates.

## Phase 1: historical artifact inventory

1. Inventory Stage-Z/AG/AK/AL worker NPZ files.
2. Rescore only artifacts that contain complete family intervals, source onset
   maps and fixed-contact ranks.
3. Do not infer corrected families from detector fragments.
4. Keep `exact_off` as the unique primary comparator. Other historical fields
   are descriptive benchmarks and cannot replace it post hoc.
5. Rescore stages within their original network pools and compare paired
   differences to stage anchors; do not mix raw scores across seed pools.

## Phase 2: Fourier implementation and parity

1. Implement paired-phase whole-sheet Fourier evaluation, physical-sheet
   centered-RMS dose normalization and the frozen spectral roughness surrogate.
2. Reuse the frozen signed-depth and mass-projection mapping. Zero Fourier
   coefficients must reconstruct the uniform fixed-mass field; `exact_off`
   remains a separate nonselectable benchmark. Reconstruct signed depth from
   the frozen quantile contract rather than from `delta Vtheta / h`.
3. Add Fourier-to-spline storage parity with whole-sheet `h` correlation at
   least 0.995.
4. Test that basis values are independent of contact coordinates and contact
   count.
5. Test mass, bounds and deterministic hashes. For the nonselectable
   `exact_off` benchmark, engine parity means that its reconstructed `h` and
   full `Vtheta` arrays pass through bitwise unchanged. It does not mean that
   the historical dual mean-dispersion mapping must satisfy the rev14 original
   signed-depth formula; that mapping-family difference is recorded as a
   diagnostic and is never used to reject a Fourier candidate.
6. Keep EE, E-to-I and Z/M off in the worker and manifest.

## Phase 3: bounded M3 canary

1. Freeze the training-only patient-support sidecar, including classifier-label
   parity, synchronized joint-null distributions and block-audit hashes. Do not
   load held-out or model artifacts.
2. Freeze eight observation-free orthogonalized Sobol directions.
3. Generate both signs at RMS 0.8 and 1.4: 32 selectable fields.
4. Add uniform and Stage-AK `exact_off` as nonselectable benchmarks.
5. Prewarm one network cache, then launch 34 seed-2321 runs with
   `systemd-run --user` plus `nohup`.
6. Use one numerical thread per worker. Derive worker count from measured peak
   RSS, reserve at least 48 GiB and retain at least 40 GiB free disk.
7. Monitor at 600-s intervals; do not continuously poll.
8. Aggregate and rank with the frozen training-only `J14` producer alone;
   patient-support is an absolute acceptance diagnostic and cannot reorder the
   canary fields.

## Phase 4: replication and refinement

1. Freeze the best eight selectable M3 fields without looking at KMeans or
   patient held-out.
2. Run seeds 2322--2323 with common random numbers.
3. Define a usable M3 anchor as positive paired `J14_v1` improvement on at
   least two of seeds 2321--2323, both confidence-adjusted mode supports at
   least six, and no runaway or numerical failure.
4. If a usable M3 anchor exists, run a bounded full-M3 CMA-ES: population 16,
   three generations and two CRN fit networks, with no restart under the
   frozen primary budget.
5. Release the M4 shell only around a usable frozen M3 anchor. Without such an
   anchor, a shell-only result is invalid and any M4 experiment must search M3
   and M4 jointly.
6. The optional M4-shell extension uses eight Sobol shell directions and both
   signs on one canary seed, the best four on two replication seeds, then
   population 8 x four generations x two CRN fit networks, with no restart:
   16 + 8 + 64 runs.
7. Do not add Gaussian cores, contact-centered basis functions or direct
   18 x 18 coefficient optimization.

## Phase 5: fresh-network selection

1. Freeze at most two candidates and the selection rule.
2. Run each candidate plus `exact_off` on four fresh selection network/noise
   seeds using `J14_v1` only.
3. Score networks independently; use equal-network aggregation and paired
   differences.
4. Freeze one candidate only if improvement is not driven by one network or by
   overlap inflation. Do not inspect natural KMeans during selection.
5. Run the winner and `exact_off` on six additional confirmation networks.
6. Once frozen, apply the predeclared natural-KMeans acceptance contract. At
   least four of six networks must have both clusters, >=6 events per cluster,
   minority fraction >=0.20, seed AMI >=0.90, patient balanced alignment >=0.70
   and positive contact-split signed margin.
7. Require weakest-mode improvement in at least four of six paired networks and
   no more than 10% degradation of the other mode.

## Phase 6: figures and one-time held-out evaluation

1. Generate the two Fig.4-style figures from exactly the frozen isolated
   families.
2. Generate a field-neuron activity GIF only if sparse neuron activity was
   saved by the confirmation worker; do not synthesize neuron dots from 1-mm
   onset maps.
3. Open the patient held-out endpoint once and label it developmental because
   the historical held-out set has already been viewed.
4. Do not return to fitting after the held-out result.

## Phase 7: same-checkpoint field intervention

1. Freeze responsibility-patch construction before inspecting effects.
2. Branch sham, top-patch attenuation, mass-preserving relocation and matched
   random control from identical checkpoints.
3. Report mode-specific event rate, onset density and patient-training
   four-layer changes at the network level.

## Execution order and stop conditions

- Phase 0 and the historical inventory are mandatory before any new SNN run.
- A provenance, parity or artifact-schema failure stops new launches.
- Ordinary negative scientific results do not stop the bounded M3 canary.
- M3 alone cannot reject the free-field family. M4 shell-only search is allowed
  only around a usable M3 anchor.
- The primary M3 budget is 34 canary runs, 16 canary replications, up to 96 M3
  optimization runs, 12 selection runs and 12 confirmation runs: at most 170
  new 20-s runs before intervention. A registered M4-shell extension adds
  at most 88 runs, for a total ceiling of 258.
- A negative bounded result is `NOT_SUPPORTED_WITHIN_FROZEN_M3_BUDGET`, or
  `NOT_SUPPORTED_WITHIN_FROZEN_M3_M4_BUDGET` only when the M4 shell was validly
  tested. Neither rejects every continuous field or authorizes EE/E-to-I/Z/M
  rescue within rev14.
