# Template-specific propagation axes and spatial fields

> Current manuscript-facing method. The primary construction uses template-specific
> early-to-late gradient axes. D_AB, swap, decision-k and endpoint axes are retained
> only as historical or supplementary method checks and do not define the primary axis.

## Template-specific propagation axes

For each patient with two stable interictal group-event templates, TA and TB were
represented on the same set of contacts that had valid ranks in both templates and
valid three-dimensional SEEG coordinates. Let (r_T(i)) denote the propagation rank
of contact (i) in template (T\in\{A,B\}), with lower ranks indicating earlier
participation. We defined template earliness as

\[
e_T(i)=-\frac{r_T(i)-\overline{r_T}}{\operatorname{sd}(r_T)}.
\]

We fitted the spatial first-order trend of each template separately,

\[
e_T(i)=\alpha_T+\beta_T^\top(x_i-\bar{x})+\epsilon_i,
\]

using truncated least squares with a relative singular-value threshold of 0.05. Because
(\nabla e_T=\beta_T) points from later to earlier contacts, the propagation-positive
unit vector was defined as

\[
u_T=-\frac{\beta_T}{\|\beta_T\|},
\]

so that positive displacement always runs from early to late participation. The stored
formal axis uses this convention; the unflipped earliness gradient is retained under
explicit diagnostic fields and is never interpreted as the propagation-positive direction.

An axis was considered estimable when at least six common mapped contacts were available,
the template earliness field was non-constant, and the fitted gradient was non-zero.
Estimability was kept separate from spatial sampling quality. Two-dimensional geometry
was considered supported when at least two electrode shafts contributed and the truncated
coordinate matrix had effective rank at least two. Directional stability was quantified
by contact bootstrap and leave-one-shaft-out refitting. A strict-stability flag required
both TA and TB to have a median bootstrap cosine of at least 0.80 and a median
leave-one-shaft-out cosine of at least 0.50. Failure of this stability flag did not erase
an otherwise estimable direction.

## Collinearity and shared direction

The spatial relationship between TA and TB was evaluated from their fitted unit vectors,
not from a rank correlation between template values. Because collinearity is a property
of an unoriented line, the line angle was

\[
\phi=\cos^{-1}(|u_A^\top u_B|),\qquad 0^\circ\leq\phi\leq90^\circ.
\]

The broad collinearity definition was (\phi\leq60^\circ), equivalently
(|u_A^\top u_B|\geq0.5). Thresholds of 45° and 30° were retained as sensitivity
analyses. The sign of (u_A^\top u_B) separately distinguished propagation-positive
same-direction from reversed-direction pairs. For broadly collinear pairs, TB was
sign-aligned to TA and a shared angular bisector was defined as

\[
u_{\mathrm{shared}}=
\operatorname{normalize}\left(u_A+\operatorname{sign}(u_A^\top u_B)u_B\right).
\]

No shared direction was constructed for non-collinear template pairs. All axes,
quality metrics, collinearity labels and shared directions were estimated solely from
interictal data.

## Frozen two-dimensional interictal fields

Each patient's interictal construction was frozen before any seizure, onset, subtype or
ictal-energy value was read. TA and TB each retained their own propagation axis. Contacts
were projected onto an axis-aligned plane whose first coordinate was displacement along
the corresponding early-to-late vector. After removing this axial component from the
three-dimensional coordinates, the first singular vector of the residual coordinates
defined the transverse direction. Both coordinates were normalized by the robust
2.5th-to-97.5th percentile axial span.

For template (T), the earliness values were smoothed at the observed contact locations
with a Gaussian Nadaraya-Watson estimator. The smoothing support was the contact's
template-specific interictal event-participation fraction, and the bandwidth was the
median nearest-neighbour spacing in the normalized plane. The fixed artifact stores the
contact order, early-to-late axis, direction-validity metrics, plane coordinates,
bandwidth, support, contact-evaluated template field and kernel weights. Collinear
patients additionally retain TA and TB fields on the shared plane. A deterministic
SHA-256 fingerprint covers the frozen axes, contact order, planes, support, fields and
kernel weights; downstream loading fails if any frozen component has drifted.

The canonical producer is `scripts/build_topic5_interictal_template_fields.py`, and its
per-patient records are stored under
`results/interictal_propagation_masked/template_gradient_fields/per_subject/`. Future
seizure analyses join activation values to the frozen contact order by exact channel name.
They may not refit an axis, plane, bandwidth or template field from seizure data. Missing
activation contacts remain missing and do not change the interictal construction.

## Comparison with seizure activation

For each seizure, a newly defined activation vector is aligned by channel name to the
frozen interictal artifact and smoothed with the same template-specific kernel weights.
TA and TB are evaluated separately on their own planes; broadly collinear patients are
also evaluated on the shared plane. Identity and transverse-mirror candidates are
compared by absolute correlation, and maxAB is selected independently for every seizure.
The same selection is repeated within every permutation.

Channel shuffling provides a coarse spatial null, whereas within-shaft shuffling preserves
electrode-shaft geography and tests whether concordance exceeds the sampled anatomical
layout. Seizure-level values are first folded to a within-patient median and patients are
then treated as the cohort-level statistical units. Axis construction and quality strata
remain unchanged when onset definitions, seizure subtypes or energy features are revised.

## Supplementary method checks

Endpoint, swap, decision-k and D_AB constructions are not part of the primary method.
They may be reported in supplementary material as calibration analyses showing how the
continuous template-specific gradient relates to previous endpoint-based summaries.
They must not be used to redefine the frozen TA/TB axes or to select patients using ictal
outcomes.
