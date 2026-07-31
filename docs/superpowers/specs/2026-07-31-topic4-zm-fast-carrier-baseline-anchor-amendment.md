# Topic 4 Phase D — baseline conductance-anchor amendment

**Date:** 2026-07-31

**Status:** LOCKED PRE-RESULT AMENDMENT — revision 2

**Parent design:** `2026-07-31-topic4-zm-snn-fast-carrier-repair-design.md`

## 1. Why an amendment is required

The parent design defined a signed point anchor

\[
\kappa_I^{(0)}=\frac{1}{V_{\mathrm{ref}}-E_I}.
\]

The locked `pre_entry__natural` checkpoint contradicts the assumption needed
for a positive conductance. Among 31,998 free E cells:

- median voltage is 9.8161 mV;
- the locked inhibitory reversal is \(E_I=11\) mV;
- only 40.29% of free E cells are above \(E_I\);
- the 5th--95th percentile range is -84.787 to 14.399 mV.

Thus the original formula gives a negative \(\kappa_I\), which is not an
inhibitory conductance. The same current-based reference also crosses
\(E_K=0\), so a single signed point anchor is not valid for the sAHP term
either. This is a mathematical incompatibility between the broad voltage
distribution of the old additive-current model and a positive reversal-based
conductance; it is not a candidate-arm outcome.

This issue was found after the five-state Arm-A migration gate passed and
before any B/C/D carrier outcome was opened. The exact 500 ms migration result
is `armA_migration_equivalence.json`, manifest SHA
`b777ca15a22b1f05f04af9e667f75a34eceaec49392ac5e83804cc3311bb56c0`.

## 2. Why a scalar distribution median is still insufficient

Revision 1 used the median absolute reversal driving force. A 50 ms
pre-entry-only smoke test showed that this is positive but does not match the
actual Arm-A effective charge: Arm A has already divided recurrent excitation
by its locked \(S_G\), whereas the conductance arm intentionally removes that
legacy divisor. The central anchor therefore had an I/E charge ratio far above
the reference. No bounded-mid/late carrier outcome was inspected.

The final anchor must match the *effective charge terms* at the source
checkpoint, not only an unweighted voltage statistic.

## 3. Locked replacement anchor

Keep all reversals, arms, scale bounds and acceptance gates unchanged. On free
E cells in the locked pre-entry checkpoint define the Arm-A effective
excitatory drive

\[
I^{E,A}_i=I^E_i-
I^{E,\mathrm{rec}}_i\frac{\alpha_G S_G}{1+\alpha_G S_G},
\]

\[
\widetilde I^I_i=(1-\gamma)I^I_i+\gamma\langle I^I\rangle.
\]

The positive effective-charge anchor is

\[
\kappa_E^{(0)}=
\frac{\langle|I^{E,A}|\rangle}
{\langle|I^E(E_E-V)|\rangle},
\]

\[
\kappa_I^{(0)}=
\frac{\langle|zI^I|\rangle}
{\langle|z\widetilde I^I(E_I-V)|\rangle},
\]

\[
g_M^{(0)}=
\frac{\eta_m\langle|m|\rangle}
{\langle|m(E_K-V)|\rangle}.
\]

For the locked checkpoint and primary \(\gamma=1/6\), this gives approximately
\(\kappa_E^{(0)}=0.0291741\), \(\kappa_I^{(0)}=0.0168780\), and
\(g_M^{(0)}=1.76217\times10^{-5}\). At scale `(1,1,1)`, the three mean
absolute charge terms equal their Arm-A targets to numerical precision before
the continuation starts.

This anchor does **not** claim pointwise current-sign equivalence below a
reversal. Every calibration artifact must report the fractions of free/active
E samples above
\(E_I\) and \(E_K\), together with \(V_\infty\) and
\(\tau_{\mathrm{eff}}\).

## 4. What remains locked

- \(E_L=E_K=0\), \(E_I=11\), \(E_E=25\) mV are unchanged.
- The deterministic scale lattice remains
  \((s_E,s_I,s_M)\in\{0.8,1.0,1.2\}^3\).
- Calibration still uses only the pre-entry baseline before any bounded-state
  B/C/D outcome is inspected.
- Returning-event rate/count, event ordering, two-source geometry,
  \(V_\infty\), effective charge ratio, \(\tau_{\mathrm{eff}}\), prevention
  and plateau criteria remain hard gates.
- If no scale triplet preserves the baseline, the result is
  `NO_GO_baseline_calibration_failed`. Reversals or scale bounds must not be
  changed after that result.

The amendment therefore repairs an impossible initialization without adding a
candidate parameter or relaxing the scientific gate.
