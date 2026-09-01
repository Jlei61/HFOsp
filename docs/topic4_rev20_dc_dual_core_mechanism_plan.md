# Topic 4 rev20-DC implementation plan

## Task 0: freeze provenance

- Freeze the exact `dualcore_s39` record from commit `26bc4338` and verify the
  historical JSON blob hash.
- Hash the transition config, patient-training target, support classifier and
  frozen patient held-out endpoint.
- Generate one manifest containing the 31 unique one-factor candidates.

## Task 1: extend the corrected producer

- Add deterministic binary dual-core construction and I-neuron field queries.
- Extend `build_substrate` to accept manual dual-core Node overrides, Node gain,
  signed-depth shrinkage and learned EE/E-to-I doses.
- Add a fixed-topology, target-normalized EE ellipse mapper whose reference
  `(45 degrees, AR=2)` is bit-exact no-op.
- Keep Z/M off and preserve the current edge-supported causal-family producer.
- Store family boundaries, full contact onsets/ranks and edge structure audits.

## Task 2: endpoint producer

- Build the unconditional patient-training and held-out event-cloud contracts.
- Score every formal returned family; do not condition on KMeans or OOD.
- Compute natural KMeans on the readable subset only after scoring.
- Compute OOD on all returned isolated families, counting unreadable as OOD.
- Aggregate with equal network weight and paired seed bootstrap.

## Task 3: tests and three-seed canary

- Unit tests: binary budget/tie determinism, baseline candidate uniqueness,
  ellipse no-op, incoming-budget conservation, one-factor-only invariant,
  KMeans/OOD masks and unconditional distribution label invariance.
- Run the reference on seeds 2501-2503 for 20 s.
- Audit late runaway, event-family representation, distribution estimability
  and Fig.4 baseline visualization.

## Task 4: formal screen

- Run all 31 candidates on seeds 2511-2514 for 20 s with common random numbers.
- Use systemd-run plus nohup, one numerical thread per worker, measured RSS-based
  concurrency up to 12 workers while reserving at least 32 GiB RAM. The first
  real substrate rebuild reached 7.2 GiB before simulation; until the canary
  supplies a full-worker peak, scheduling assumes 18 GiB per worker.
- Monitor every 600 s; stop new launches on nonfinite output, provenance drift,
  OOM pressure or low disk. Do not continuously poll.

## Task 5: confirmation

- Within each family select at most one non-reference level by training
  complete-distribution distance only.
- Freeze selected IDs before reading validation metrics.
- Run reference plus selected levels on seeds 2521-2532.
- Open held-out distribution, KMeans/prototype and OOD endpoints once.

## Task 6: figures and report

- Main response atlas: aligned parameter curves for full distribution,
  two-template concordance and OOD; paired networks in light lines and
  equal-network mean with interval in color.
- Joint/Both inset: EE and E-to-I expression interaction relative to the two
  one-factor curves.
- Fig.4-style baseline audit and a representative causal-family GIF are
  generated from the same canary/confirmation artifacts.
- Add `figures/README.md`, PNG/PDF and metadata; inspect both renders.
- Report the safe claim, largest remaining gap and the next causal experiment.
