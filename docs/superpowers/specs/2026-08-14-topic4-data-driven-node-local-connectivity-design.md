# Topic 4 rev11-NLC: data-driven node and local connectivity

> Status: design v1, 2026-08-14. This supersedes the interpretation that Z/M
> was introduced to fit the patient interictal modes. D6.3 remains an immutable
> historical Node-field replication experiment.

## 1. Scientific question

The primary question is whether patient event data can identify a continuous
nodal excitability field together with a local recurrent E-loop substrate that
produces a same-network event repertoire resembling the patient propagation
repertoire.

Z/M is not an optimization target for this question. It is the later common
slow-state interface used to test whether a frozen interictal substrate can be
embedded in an ictal lifecycle model. Static substrate discovery is performed
first; the frozen winner is then transferred, unchanged, to active Z/M.

Safe claim boundary:

```text
patient-development data constrain a continuous Node + local-connectivity
substrate; Fig.4-style direct readout and natural KMeans describe what that
substrate can reproduce. They do not establish a patient core, a patient-blind
generalization result, or an ictal lifecycle mechanism.
```

## 2. Current connectivity contract

The matrix convention is target first, source second. Therefore the existing
parameter names must be read as follows:

| Parameter | Biological direction | Current rule |
| --- | --- | --- |
| `C_EE`, `l_EE` | E -> E | 800 E sources per E target; rotated elliptical exponential kernel |
| `C_IE`, `l_IE` | E -> I | 800 E sources per I target; isotropic exponential kernel |
| `C_EI`, `l_EI` | I -> E | 200 I sources per E target; isotropic exponential kernel |
| `C_II`, `l_II` | I -> I | 200 I sources per I target; isotropic exponential kernel |

The E->E long axis is a kernel orientation around every E target. It does not
define a special chain or a separate population of "axis E neurons". Every E
neuron uses the same baseline rule. Source out-degree is emergent and varies
with position and the finite sheet boundary.

Existing Topic 4 edge mappers only redistribute weights on already sampled
E->E edges. They preserve each E target's total incoming recurrent-E weight,
topology and delay assignment, and leave E->I and all GABA channels unchanged.
They therefore do not test a learned local E-I loop.

## 3. Rev11-NLC representation

### 3.1 Continuous Node field

Use one continuous observation-invariant field over the full sheet:

\[
V_{\theta,i}=V_{\theta,0}-h(x_i)d_i.
\]

There is no Gaussian component count, peak count or contact-conditioned field
support. The field is represented by the existing uniform tensor B-spline.
Initial joint searches use smooth global perturbation modes around a frozen
continuous anchor; the basis is defined on the sheet before model events are
seen.

### 3.2 Continuous local E-source mapper

Evaluate the same field at every E and I neuron. Define field contrast

\[
q_i=2h(x_i)-1.
\]

For an existing edge from E source \(s\) to target \(t\), use pathway-specific
continuous features

\[
\phi(t,s)=\left[
q_s,\ q_tq_s,\ -q_s r/\ell,\ -q_tq_s r/\ell,
q_tq_s\Delta x/\ell,\ q_tq_s\Delta y/\ell
\right].
\]

The log multiplier is \(c^T\phi\). Separate coefficient vectors are used for
E->E and E->I. Target-wise normalization preserves the original incoming
weight budget separately for every E target and every I target.

This representation is:

- continuous over all neurons, including E neurons away from the current long
  axis or high-field support;
- local because it reweights an already local baseline graph and contains
  explicit distance terms;
- non-separable in source and target field values;
- observation-invariant because contact coordinates, shaft identity and
  KMeans labels are not inputs to the mapper;
- low-dimensional enough for a noisy derivative-free SNN search.

The first round does not alter topology, delays, I->E or I->I. If reweighting
cannot recover capacity, a later pre-registered round may resample local
topology with common random uniforms. That is a separate mechanism question.

## 4. Experimental arms

Use paired network and stochastic seeds:

1. `Node`: continuous Node field, baseline connectivity.
2. `Node+EE`: Node plus learned E->E local redistribution.
3. `Node+EtoI`: Node plus learned E->I local redistribution.
4. `Node+EE+EtoI`: Node plus both learned pathways.

`Null` and connectivity-only arms are retained as descriptive controls, not as
additional optimization targets. I->E is frozen in the primary experiment so
that altered inhibitory recruitment is not confounded with altered inhibitory
feedback. A secondary paired arm may release I->E only after E->I has a
replicable effect.

## 5. Data-driven objective

The objective is evaluated with equal network weight. It combines:

- natural within-network K=2 support and stability;
- cross-fitted patient A/B geometry on held-out contacts;
- mode-conditioned recruitment and precedence, with ICL and SCL kept distinct;
- a weak event-yield term that prevents silent candidates from appearing good;
- smooth-field and moderate-edge-redistribution penalties.

Use a weakest-mode continuous penalty, not a sign-only gate. Only non-finite
simulation, late runaway and insufficient events for the stated statistic are
hard invalid states. This is an exploratory capacity experiment, so descriptive
failures remain in the response surface instead of becoming many blockers.

The optimizer may not see patient held-out blocks. Fit, selection and final
network pools are disjoint. Fig.4-style direct waveforms and natural KMeans are
the mandatory final products for every selected candidate.

## 6. Search order

### NLC0: zero-simulation audit

Verify matrix direction, in-degree, distance, source out-degree, no-op parity,
incoming-weight conservation, topology/delay hashes and field coverage of both
E and I populations.

### NLC1: connectivity capacity canary

Freeze the current continuous Node anchor and scan a bounded low-discrepancy
set of E->E and E->I coefficient vectors on paired short networks. This asks
whether local connectivity changes the repertoire at all before paying for a
joint search.

### NLC2: joint Node-connectivity search

Jointly optimize smooth Node perturbation coordinates and the retained local
connectivity coefficients. Use common random numbers, multiple restarts and
network-level selection. Do not optimize Z/M parameters.

### NLC3: frozen-substrate confirmation

Run longer fresh-network slow-off confirmations and produce the two canonical
Fig.4 panels. The result is accepted as a data-driven interictal substrate only
if the same-network natural KMeans and patient-geometry effects replicate.

### NLC4: active-Z/M transfer

Transfer the frozen Node and connectivity parameters unchanged to active Z/M.
This stage tests compatibility with bounded entry, carrier, exit and recovery;
it is not another interictal fit. Failure here is a slow-state compatibility
failure, not evidence against the data-driven local substrate.

## 7. Decision interpretation

| Result | Interpretation |
| --- | --- |
| E->E improves natural KMeans but not patient geometry | generic repertoire separation, not patient recovery |
| E->I changes event yield but not mode geometry | inhibitory recruitment gain, not route learning |
| E->E + E->I improves both metrics beyond either arm | local E-loop interaction candidate |
| no reweighting candidate changes capacity | fixed topology may be limiting; test local topology resampling |
| slow-off passes but active Z/M runs away | static substrate found, seizure-unification interface unresolved |
| active Z/M passes lifecycle gates without refitting | shared interictal-to-ictal substrate candidate |
