# Topic 4 rev11-NLC2: joint continuous Node and local connectivity fit

> Status: frozen design addendum after NLC1, 2026-08-14. The NLC1 design file
> remains immutable because it is a hashed input to the completed canary.

## 1. NLC1 evidence and boundary

NLC1 completed 39/39 paired workers (13 candidates by three network seeds)
without runaway. The measured sentinel peak RSS was 7.52 GiB; the dispatcher
used nine workers. Topology, delays and all GABA pathways remained unchanged.
The maximum incoming-budget errors were below 2e-11 for E->E and 1e-11 for
E->I.

Relative to Node, `joint_03` increased equal-network natural KMeans alignment
from 0.772 to 0.836 and contact-split patient rank margin from 0.077 to 0.200,
while reducing OOD from 0.501 to 0.105. `joint_04` reached natural alignment
0.958 and the best shaft-aware aggregate score (1.298 versus Node 4.456), but
its contact-split margin was only 0.108. These are three-network development
signals and not patient recovery.

The frozen direction classifier is not sign-flipped. On 30,049 patient
training events it agrees with the old A/B labels in 94.5%, and its rank-profile
matrix is approximately `[[+1.00,-0.63],[-0.64,+1.00]]`. Negative-diagonal
model matrices therefore represent residual mismatch rather than rank coding.

Current status:

```text
LOCAL_EE_ETOI_CONNECTIVITY_CHANGES_SAME_NETWORK_REPERTOIRE
/
PATIENT_RANK_GEOMETRY_PARTIAL_DEVELOPMENT_SIGNAL
/
COMPLETE_PATIENT_INTERICTAL_RECOVERY_NOT_SHOWN
```

## 2. NLC2 scientific question

Does joint variation of a continuous whole-sheet Node field and target-budget
conserving local E->E/E->I redistribution improve all three of:

1. same-network natural K=2 structure;
2. contact-split patient rank geometry;
3. shaft-aware recruitment and precedence?

Z/M, beta, topology, delays, I->E, I->I and all GABA weights remain frozen.
This is a static development fit, not an ictal experiment.

## 3. Candidate representation

The Node perturbation combines all 12 real whole-sheet Fourier modes through
harmonic two, then projects the combination into the existing 18 by 18 cubic
B-spline basis. Each combination is normalized by its sheet-wide RMS before
an amplitude in [0.15, 0.8] is applied. Candidate generation never reads
contact coordinates, shaft identity, patient events or labels. Total Node field
mass remains fixed by the existing level-set projection.

The E->E and E->I coefficient vectors use the NLC1 continuous local mapper.
Each pathway retains its own incoming target budget. NLC2 samples bounded local
perturbations around `joint_03` and `joint_04`, which bracket the NLC1 natural
KMeans and shaft-aware signals. The parents are search centres, not accepted
solutions.

The fit library contains:

- one Node/no-edge control;
- two frozen-Node NLC1 parent controls;
- 16 joint Node/edge candidates around each parent.

All 35 candidates use common network seeds 1521-1523 for 8000 ms.

## 4. Exploratory objective

For every candidate define equal-network quantities:

- `A_K`: natural KMeans balanced alignment to contact-split patient rank
  assignments;
- `M_P`: contact-split patient rank-profile signed margin;
- `D_SA`: worst-mode shaft-aware aggregate error;
- `D_R`: worst-mode recruitment error;
- `D_OOD`: OOD fraction;
- `B`: detector occupancy.

The scalar ordering is:

```text
J_NLC2 = LSE_0.15(
    1 - A_K,
    (1 - M_P) / 2,
    min(1, log(1 + D_SA) / log(9))
) + 0.2 D_R + 0.1 D_OOD + 0.05 B + 0.02 R
```

`R` is normalized Node/edge perturbation energy. The full Pareto coordinates
remain primary evidence; the scalar only provides a reproducible ordering.
Only non-finite simulation, late runaway and insufficient events for the stated
statistic are invalid. There are no sign gates or numerous blockers.

## 5. Decision boundary

- Improvement in KMeans only means generic repertoire separation.
- Improvement in cross-fit rank geometry only may be prototype assignment
  without a natural two-cluster repertoire.
- Improvement in shaft-aware score only may be recruitment support without
  the Fig.4 rank geometry.
- A fit candidate must improve the weakest of these on fresh selection networks
  before entering long confirmation.
- The canonical Fig.4 direct waveform and KMeans panels are produced only after
  fresh-network selection; fit-screen plots are diagnostic.
- No result in NLC2 is patient blind because the patient target is development
  data.
