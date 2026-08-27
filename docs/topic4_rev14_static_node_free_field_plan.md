# Topic 4 rev14 execution plan

## Phase 0: zero-simulation reference (complete)

1. Implemented a dedicated training-only producer for `exact_off` seeds
   2311--2313.
2. Freeze original event indices, causal-family masks, input hashes and contact
   order.
3. Produced per-network and equal-network soft scores, OOD/shaft support,
   natural KMeans diagnostics and a Fig.4 bundle.
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

1. Implement paired-phase whole-sheet Fourier evaluation and analytic
   roughness.
2. Reuse the frozen signed-depth and mass-projection mapping.
3. Add Fourier-to-spline storage parity with whole-sheet `h` correlation at
   least 0.995.
4. Test that basis values are independent of contact coordinates and contact
   count.
5. Test mass, bounds, deterministic hashes and exact-off engine parity.
6. Keep EE, E-to-I and Z/M off in the worker and manifest.

## Phase 3: bounded M3 canary

1. Freeze eight observation-free orthogonalized Sobol directions.
2. Generate both signs at RMS 0.8 and 1.4: 32 selectable fields.
3. Add uniform and Stage-AK `exact_off` as nonselectable benchmarks.
4. Prewarm one network cache, then launch 34 seed-2321 runs with
   `systemd-run --user` plus `nohup`.
5. Use one numerical thread per worker. Derive worker count from measured peak
   RSS, reserve at least 48 GiB and retain at least 40 GiB free disk.
6. Monitor at 600-s intervals; do not continuously poll.
7. Aggregate with the frozen training-only `J14` producer.

## Phase 4: replication and refinement

1. Freeze the best eight selectable M3 fields without looking at KMeans or
   patient held-out.
2. Run seeds 2322--2323 with common random numbers.
3. If a usable M3 anchor exists, run a bounded full-M3 CMA-ES: population 16,
   three generations and two CRN fit networks, with no restart under the
   frozen primary budget.
4. Release the M4 shell only around a usable frozen M3 anchor. Without such an
   anchor, a shell-only result is invalid and any M4 experiment must search M3
   and M4 jointly.
5. The optional M4-shell extension is capped at 16 canary runs, 8 replications
   and 64 optimization runs.
6. Do not add Gaussian cores, contact-centered basis functions or direct
   18 x 18 coefficient optimization.

## Phase 5: fresh-network selection

1. Freeze at most two candidates and the selection rule.
2. Run each candidate plus `exact_off` on six fresh network/noise seeds.
3. Score networks independently; use equal-network aggregation and paired
   differences.
4. Freeze one candidate only if improvement is not driven by one network or by
   overlap inflation.
5. Once frozen, apply the predeclared natural-KMeans acceptance contract. At
   least four of six networks must have both clusters, >=6 events per cluster,
   minority fraction >=0.20, seed AMI >=0.90, patient balanced alignment >=0.70
   and positive contact-split signed margin.
6. Require weakest-mode improvement in at least four of six paired networks and
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
  optimization runs, 18 selection runs and 6 final spatial controls: at most
  170 new 20-s runs before intervention. A registered M4-shell extension adds
  at most 88 runs, for a total ceiling of 258.
- A negative bounded result is `NOT_SUPPORTED_WITHIN_FROZEN_M3_BUDGET`, or
  `NOT_SUPPORTED_WITHIN_FROZEN_M3_M4_BUDGET` only when the M4 shell was validly
  tested. Neither rejects every continuous field or authorizes EE/E-to-I/Z/M
  rescue within rev14.
