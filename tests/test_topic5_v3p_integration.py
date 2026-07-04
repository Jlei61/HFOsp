"""Topic 5 V3p — feasibility pilot integration test (Task 1, DECISION GATE).

Runs the real `scripts/run_topic5_v3p_feasibility.py` CLI end-to-end against
the mounted artifacts (ictal field long cache + lagPat + rank-displacement
JSON) and checks the `feasibility.csv` contract columns exist. See
docs/superpowers/plans/2026-07-03-topic5-v3p-preictal-trajectory.md Task 1.
"""
import csv
import json

import pytest
import pandas as pd
import numpy as np


@pytest.mark.integration
def test_v3p_feasibility_writes_csv(tmp_path):
    from scripts.run_topic5_v3p_feasibility import main
    out = tmp_path / "feasibility.csv"
    main(["--cohort", "narrow", "--outdir", str(tmp_path)])
    df = pd.read_csv(out)
    for col in ["geometry_sufficient", "n_windows_P3", "n_seizures_ge_min_windows", "n_nonaxis",
                "roster_status", "axis_quality_gate_pass", "admitted", "n_unique_label_permutations_est"]:
        assert col in df.columns
    assert len(df) >= 1


# ---------------------------------------------------------------------------
# Task 7 trajectory runner — the two-tier integration contract (brief Step 1).
# Tier 1: header-only CSV must still carry the co-primary columns (skipped path).
# Tier 2: an eligible subject (253) produces finite co-primary surplus slopes +
# a well-formed label-null p in [0, 1].
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_v3p_trajectory_writes_csv_even_if_skipped(tmp_path):
    from scripts.run_topic5_v3p_trajectory import main
    main(["--cohort", "narrow", "--outdir", str(tmp_path), "--n-perm", "20", "--subjects", "__none__"])
    df = pd.read_csv(tmp_path / "v3p_trajectory_subject.csv")
    for c in ["net_offaxis_flux_surplus_slope", "mode_shift_density_surplus_slope",
              "p_label_slope_b", "p_label_slope_c", "status"]:
        assert c in df.columns


@pytest.mark.integration
def test_v3p_trajectory_runs_on_eligible_subject(tmp_path):
    from scripts.run_topic5_v3p_trajectory import main
    main(["--cohort", "narrow", "--outdir", str(tmp_path), "--n-perm", "50", "--subjects", "253"])
    df = pd.read_csv(tmp_path / "v3p_trajectory_subject.csv")
    row = df[df.subject.astype(str) == "epilepsiae_253"].iloc[0]
    if row["status"] == "ok":
        assert np.isfinite(row["net_offaxis_flux_surplus_slope"])
        assert np.isfinite(row["mode_shift_density_surplus_slope"])
        assert 0.0 <= row["p_label_slope_b"] <= 1.0


# ---------------------------------------------------------------------------
# Task 8 — supportive H3p-a (beta_axis) + secondary H3p-d (burden / self-sustain
# lag1-specific / gain_shift) columns, added onto the SAME trajectory runner +
# eligible subject as Task 7. module_support_flag_a is hard-coded False
# (supportive-only, never sole support).
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_v3p_trajectory_h3pa_h3pd_columns(tmp_path):
    from scripts.run_topic5_v3p_trajectory import main
    main(["--cohort", "narrow", "--outdir", str(tmp_path), "--n-perm", "50", "--subjects", "253"])
    df = pd.read_csv(tmp_path / "v3p_trajectory_subject.csv")
    for c in ["beta_axis_strength_slope", "beta_axis_reliable", "beta_axis_slope_z",
              "p_label_slope_a", "module_support_flag_a",
              "nonaxis_activation_burden_slope_raw", "nonaxis_activation_burden_slope_label_surplus",
              "nonaxis_activation_burden_slope_resid", "burden_slope_z", "p_label_burden",
              "N_self_sustain_lag1_slope", "N_self_sustain_lag0_slope",
              "N_self_sustain_lag1_specific_slope", "N_self_sustain_slope_z", "p_label_selfsustain",
              "gain_axis_slope", "gain_nonaxis_slope", "gain_shift_slope",
              "gain_nonaxis_surplus_slope", "gain_shift_slope_z"]:
        assert c in df.columns
    row = df[df.subject.astype(str) == "epilepsiae_253"].iloc[0]
    assert row["module_support_flag_a"] in (False, "False", 0)  # hard-coded, never sole support
    if row["status"] == "ok":
        assert np.isfinite(row["beta_axis_strength_slope"])
        assert np.isfinite(row["nonaxis_activation_burden_slope_resid"])
        assert np.isfinite(row["N_self_sustain_lag1_specific_slope"])
        assert np.isfinite(row["gain_nonaxis_surplus_slope"])
        assert 0.0 <= row["p_label_slope_a"] <= 1.0


# ---------------------------------------------------------------------------
# Task 9 -- summary + tier verdict (Holm co-primary). Unlike the tests above,
# this does NOT need real mounted data: given an already-computed trajectory
# CSV, the tier ladder / subject_support / Holm-correction / broad_core-vs-
# broad_expanded logic is pure arithmetic over its columns, so a small
# hand-verified synthetic fixture makes it exactly (not just "runs without
# crashing") checkable. Every p-value asserted below was cross-checked
# directly against ``scipy.stats.wilcoxon(..., alternative="greater")``: for
# n same-signed distinct values the exact one-sided p is 1/2**n, and Holm-2
# doubles whichever raw p is smaller (capped at 1).
# ---------------------------------------------------------------------------
_V3P_ROW_DEFAULTS = dict(
    status="ok", geometry_sufficient=True, in_broad_core=False,
    label_null_underpowered=False,
    module_support_flag_b=False, onset_jitter_pass_b=True,
    leave_one_contact_flux_pass=True, axis_only_flux_control_pass=True,
    near_onset_dependent_b=False,
    module_support_flag_c=False, onset_jitter_pass_c=True,
    single_contact_driven=False, leave_one_contact_mode_pass=True,
    axis_only_mode_control_pass=True, near_onset_dependent_c=False,
    beta_axis_strength_slope=-0.1, p_label_slope_a=0.2, beta_axis_reliable=True,
    net_offaxis_flux_surplus_slope=0.0, net_offaxis_flux_slope_z=0.0,
    mode_shift_density_surplus_slope=0.0, mode_shift_density_slope_z=0.0,
)
_V3P_FIELDNAMES = ["subject", "cohort"] + list(_V3P_ROW_DEFAULTS.keys())


def _v3p_row(subject, cohort, z_b=0.0, z_c=0.0, supported=False, **overrides):
    """One synthetic ``v3p_trajectory_subject.csv`` row: exactly the columns
    ``run_topic5_v3p_summary._subject_row`` reads (brief Task 9: "a few rows
    with controlled module_support_flag_b/c, slope_label_z, gate columns,
    in_broad_core, status"). ``supported=True`` flips the FULL H3p-b gate
    stack on so the row clears ``subject_support`` via the flux leg.
    """
    row = dict(_V3P_ROW_DEFAULTS)
    row["subject"], row["cohort"] = subject, cohort
    row["net_offaxis_flux_slope_z"] = z_b
    row["mode_shift_density_slope_z"] = z_c
    if supported:
        row["module_support_flag_b"] = True
    row.update(overrides)
    return row


def _write_v3p_trajectory_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_V3P_FIELDNAMES)
        w.writeheader()
        w.writerows(rows)


def test_v3p_summary_tier_ladder_holm_and_broad_core_divergence(tmp_path):
    """narrow: 6 eligible subjects with H3p-b slope_label_z=[6,5,4,3,2,1]
    (exact one-sided Wilcoxon p=1/64=0.015625; Holm-doubled=0.03125<0.05) +
    3/6 individually gate-stack-supported (>= config min_subject_support_narrow=2)
    -> narrow tier-3. A 7th row is status=skipped/geometry_sufficient=True
    with an otherwise full "looks supported" gate stack and an extreme
    z=-100 -- it MUST be excluded from every denominator/population despite
    geometry_sufficient=True (brief: "geometry_insufficient/status==skipped
    ... EXCLUDED").
    broad_core (6 subjects, same z_b pattern) independently ALSO clears the
    same H3p-b Holm bar -> would replicate narrow's leg alone
    (tier_broad_core=4). broad_expanded adds 3 admitted candidates with
    strongly opposite-signed z_b=[-10,-9,-8]; pooled over n=9 the same test
    is no longer significant (p_holm_b=1.0) -- rev2's "expansion adds power,
    never rescues a curated-subset null" (and, symmetrically, must not let a
    bad expansion silently borrow the core's clean pass either) -> tier
    stays 3. ``tier != tier_broad_core`` proves the two are genuinely
    computed from different row subsets, not aliased.
    """
    from scripts.run_topic5_v3p_summary import main

    narrow_rows = [
        _v3p_row("epilepsiae_9001", "narrow", z_b=6.0, z_c=-3.0, supported=True),
        _v3p_row("epilepsiae_9002", "narrow", z_b=5.0, z_c=-2.0, supported=True),
        _v3p_row("epilepsiae_9003", "narrow", z_b=4.0, z_c=-1.0, supported=True),
        _v3p_row("epilepsiae_9004", "narrow", z_b=3.0, z_c=-1.0, supported=False),
        _v3p_row("epilepsiae_9005", "narrow", z_b=2.0, z_c=-2.0, supported=False),
        _v3p_row("epilepsiae_9006", "narrow", z_b=1.0, z_c=-3.0, supported=False),
        _v3p_row("epilepsiae_9099", "narrow", z_b=-100.0, z_c=100.0, supported=True,
                 status="skipped", geometry_sufficient=True),
    ]
    broad_core_rows = [
        _v3p_row("epilepsiae_8001", "broad", z_b=6.0, z_c=-3.0, supported=True, in_broad_core=True),
        _v3p_row("epilepsiae_8002", "broad", z_b=5.0, z_c=-2.0, in_broad_core=True),
        _v3p_row("epilepsiae_8003", "broad", z_b=4.0, z_c=-1.0, in_broad_core=True),
        _v3p_row("epilepsiae_8004", "broad", z_b=3.0, z_c=-1.0, in_broad_core=True),
        _v3p_row("epilepsiae_8005", "broad", z_b=2.0, z_c=-2.0, in_broad_core=True),
        _v3p_row("epilepsiae_8006", "broad", z_b=1.0, z_c=-3.0, in_broad_core=True),
    ]
    broad_expanded_only_rows = [
        _v3p_row("epilepsiae_8101", "broad", z_b=-10.0, z_c=-1.0, in_broad_core=False),
        _v3p_row("epilepsiae_8102", "broad", z_b=-9.0, z_c=-1.0, in_broad_core=False),
        _v3p_row("epilepsiae_8103", "broad", z_b=-8.0, z_c=-1.0, in_broad_core=False),
    ]

    _write_v3p_trajectory_csv(tmp_path / "narrow" / "v3p_trajectory_subject.csv", narrow_rows)
    _write_v3p_trajectory_csv(
        tmp_path / "broad" / "v3p_trajectory_subject.csv", broad_core_rows + broad_expanded_only_rows
    )

    main(["--indir", str(tmp_path)])

    with (tmp_path / "narrow" / "v3p_summary_subject.csv").open(newline="") as fh:
        narrow_summary_rows = list(csv.DictReader(fh))
    assert len(narrow_summary_rows) == 7
    assert "tier" not in narrow_summary_rows[0]  # tier assigned ONLY in the cohort JSON
    excl = next(r for r in narrow_summary_rows if r["subject"] == "epilepsiae_9099")
    assert excl["excluded_from_denominator"] == "True"
    assert excl["subject_support"] == "False"  # status=skipped overrides a look-alike full gate stack
    by_subj = {r["subject"]: r for r in narrow_summary_rows}
    assert by_subj["epilepsiae_9001"]["subject_support"] == "True"
    assert by_subj["epilepsiae_9001"]["support_driver"] == "H3p-b"  # b-leg names itself as the driver

    narrow_payload = json.loads((tmp_path / "narrow" / "v3p_cohort_tier.json").read_text())
    broad_payload = json.loads((tmp_path / "broad" / "v3p_cohort_tier.json").read_text())
    assert narrow_payload == broad_payload  # same joint verdict written to both cohort dirs, never pooled into one
    assert "tier" not in narrow_payload["narrow"]  # nested per-cohort block carries no tier either

    payload = narrow_payload
    narrow_blk = payload["narrow"]
    assert (narrow_blk["n_total"], narrow_blk["n_eligible"], narrow_blk["n_excluded"]) == (7, 6, 1)
    assert narrow_blk["n_subject_support"] == 3
    assert narrow_blk["p_wilcoxon_b"] == pytest.approx(0.015625)
    assert narrow_blk["p_holm_b"] == pytest.approx(0.03125)
    assert narrow_blk["p_holm_c"] == pytest.approx(1.0)
    assert (narrow_blk["cohort_b_pass"], narrow_blk["cohort_c_pass"]) == (True, False)

    broad_blk, core_blk = payload["broad"], payload["broad_core"]
    assert (broad_blk["n_eligible"], core_blk["n_eligible"]) == (9, 6)
    assert broad_blk["p_holm_b"] == pytest.approx(1.0)     # dragged down by the 3 admitted candidates
    assert core_blk["p_holm_b"] == pytest.approx(0.03125)  # curated 9 alone still replicates cleanly
    assert (broad_blk["cohort_b_pass"], core_blk["cohort_b_pass"]) == (False, True)

    assert payload["broad_expanded_replicates"] is False
    assert payload["broad_core_replicates"] is True
    assert payload["tier"] == 3            # a bad expansion must not borrow the core's clean pass
    assert payload["tier_broad_core"] == 4  # ... even though the curated core alone would say 4
    assert payload["state_v3p_supported"] is True
    assert payload["pre_registered_negative"] is False


def test_v3p_summary_tier4_never_rescued_by_expansion_when_core_null(tmp_path):
    """rev2's PRIMARY stated concern, the mirror image of the divergence test
    above: if broad_core alone does NOT replicate narrow's H3p-b leg, admitting
    more candidate subjects into broad_expanded must NEVER manufacture a
    tier-4 replication the curated 9 do not themselves show ("expansion adds
    power, never rescues a curated-subset null"). Constructed so the raw
    broad_expanded arithmetic alone WOULD clear Holm (sanity-asserted below)
    while broad_core does not -- this is exactly the case the earlier
    divergence test's mutation check (dropping "and broad_core_replicates"
    from the tier formula) could NOT catch, because there broad_expanded
    itself already failed too.
    """
    from scripts.run_topic5_v3p_summary import main

    narrow_rows = [
        _v3p_row("epilepsiae_9001", "narrow", z_b=6.0, z_c=-3.0, supported=True),
        _v3p_row("epilepsiae_9002", "narrow", z_b=5.0, z_c=-2.0, supported=True),
        _v3p_row("epilepsiae_9003", "narrow", z_b=4.0, z_c=-1.0, supported=True),
        _v3p_row("epilepsiae_9004", "narrow", z_b=3.0, z_c=-1.0, supported=False),
        _v3p_row("epilepsiae_9005", "narrow", z_b=2.0, z_c=-2.0, supported=False),
        _v3p_row("epilepsiae_9006", "narrow", z_b=1.0, z_c=-3.0, supported=False),
    ]
    # broad_core (6): mixed sign, clearly not Wilcoxon-significant alone.
    broad_core_rows = [
        _v3p_row("epilepsiae_5001", "broad", z_b=-2.0, z_c=-1.0, in_broad_core=True),
        _v3p_row("epilepsiae_5002", "broad", z_b=-1.0, z_c=-1.0, in_broad_core=True),
        _v3p_row("epilepsiae_5003", "broad", z_b=-1.0, z_c=-1.0, in_broad_core=True),
        _v3p_row("epilepsiae_5004", "broad", z_b=1.0, z_c=-1.0, in_broad_core=True),
        _v3p_row("epilepsiae_5005", "broad", z_b=1.0, z_c=-1.0, in_broad_core=True),
        _v3p_row("epilepsiae_5006", "broad", z_b=1.2, z_c=-1.0, in_broad_core=True),
    ]
    # 6 admitted candidates, all strongly positive -- alone they tip the
    # POOLED n=12 expanded set to a raw p small enough to survive Holm, even
    # though the curated core (above) shows nothing.
    broad_expanded_only_rows = [
        _v3p_row(f"epilepsiae_51{i:02d}", "broad", z_b=50.0 + i, z_c=-1.0, in_broad_core=False)
        for i in range(6)
    ]

    _write_v3p_trajectory_csv(tmp_path / "narrow" / "v3p_trajectory_subject.csv", narrow_rows)
    _write_v3p_trajectory_csv(
        tmp_path / "broad" / "v3p_trajectory_subject.csv", broad_core_rows + broad_expanded_only_rows
    )

    main(["--indir", str(tmp_path)])
    payload = json.loads((tmp_path / "narrow" / "v3p_cohort_tier.json").read_text())

    # sanity: confirms the scenario is real -- expanded WOULD look significant
    # in isolation, while core does not (the interesting case, not a typo).
    assert payload["broad"]["p_holm_b"] == pytest.approx(0.02685546875)
    assert payload["broad_core"]["p_holm_b"] == pytest.approx(1.0)

    assert payload["broad_expanded_replicates"] is True
    assert payload["broad_core_replicates"] is False
    assert payload["tier"] == 3             # NOT rescued to 4 by the expansion
    assert payload["tier_broad_core"] == 3  # core alone also caps at narrow's own tier-3


def test_v3p_summary_honest_negative_tier(tmp_path):
    """Descriptive-direction-only path (tier rung 1): narrow's 3 subjects
    show a barely-positive H3p-b median (0.2) nowhere near Wilcoxon
    significance (n=3, mixed signs) and none individually clears the full
    gate stack -> no cohort Holm pass, 0 subject_support -> tier=1,
    ``pre_registered_negative=True`` -- V3p's honest-negative path (never
    rescued by a single subject) must be reachable, not just the positive one.
    """
    from scripts.run_topic5_v3p_summary import main

    narrow_rows = [
        _v3p_row("epilepsiae_7001", "narrow", z_b=0.5, z_c=-0.5),
        _v3p_row("epilepsiae_7002", "narrow", z_b=-0.3, z_c=-0.3),
        _v3p_row("epilepsiae_7003", "narrow", z_b=0.2, z_c=-0.2),
    ]
    broad_rows = [
        _v3p_row("epilepsiae_6001", "broad", z_b=0.1, z_c=-0.4, in_broad_core=True),
        _v3p_row("epilepsiae_6002", "broad", z_b=-0.2, z_c=-0.1, in_broad_core=True),
        _v3p_row("epilepsiae_6003", "broad", z_b=0.05, z_c=-0.2, in_broad_core=True),
    ]
    _write_v3p_trajectory_csv(tmp_path / "narrow" / "v3p_trajectory_subject.csv", narrow_rows)
    _write_v3p_trajectory_csv(tmp_path / "broad" / "v3p_trajectory_subject.csv", broad_rows)

    main(["--indir", str(tmp_path)])
    payload = json.loads((tmp_path / "narrow" / "v3p_cohort_tier.json").read_text())

    assert (payload["narrow"]["cohort_b_pass"], payload["narrow"]["cohort_c_pass"]) == (False, False)
    assert payload["narrow"]["n_subject_support"] == 0
    assert payload["tier"] == 1
    assert payload["tier_broad_core"] == 1
    assert payload["state_v3p_supported"] is False
    assert payload["pre_registered_negative"] is True


def test_v3p_summary_underpowered_subject_excluded_from_cohort_denominator(tmp_path):
    """Fix 1 (spec §7/§8/§11 "不计强阳性分母"): a ``label_null_underpowered``
    subject is a POPULATION-level exclusion (same class as
    geometry_insufficient/skipped), not merely a subject_support demotion --
    it must drop from BOTH the support-fraction denominator (``n_eligible``)
    AND the cohort Wilcoxon z population, exactly like a skipped subject.

    Constructed so the exclusion is DECISIVE, not cosmetic: 6 clean subjects
    with H3p-b z=[6,5,4,3,2,1] alone give p_holm_b=0.03125 (cohort_b_pass=True).
    A 7th subject is status=ok + geometry_sufficient=True (so NEITHER of the
    other two exclusion rules would catch it) but label_null_underpowered=True,
    carrying an extreme z_b=-100 that -- IF wrongly included -- drags the
    same Wilcoxon to p_holm_b=0.297 (cohort_b_pass=False). Every assertion
    below therefore flips between the pre-fix (included) and post-fix
    (excluded) code: a genuine RED->GREEN lock, not a no-op on this fixture.
    (On the ACTUAL cohort no subject is underpowered, so the production run is
    unaffected -- this test guards the general case + future data.)
    """
    from scripts.run_topic5_v3p_summary import main

    rows = [
        _v3p_row("epilepsiae_4001", "narrow", z_b=6.0, z_c=-3.0, supported=True),
        _v3p_row("epilepsiae_4002", "narrow", z_b=5.0, z_c=-2.0, supported=True),
        _v3p_row("epilepsiae_4003", "narrow", z_b=4.0, z_c=-1.0),
        _v3p_row("epilepsiae_4004", "narrow", z_b=3.0, z_c=-1.0),
        _v3p_row("epilepsiae_4005", "narrow", z_b=2.0, z_c=-2.0),
        _v3p_row("epilepsiae_4006", "narrow", z_b=1.0, z_c=-3.0),
        # status=ok + geometry_sufficient -> ONLY the label_null_underpowered
        # rule can exclude it; a module-supported-looking gate stack + extreme
        # z that WOULD swing the Wilcoxon if (wrongly) counted.
        _v3p_row("epilepsiae_4099", "narrow", z_b=-100.0, z_c=100.0, supported=True,
                 label_null_underpowered=True),
    ]
    _write_v3p_trajectory_csv(tmp_path / "narrow" / "v3p_trajectory_subject.csv", rows)
    main(["--cohort", "narrow", "--indir", str(tmp_path)])

    with (tmp_path / "narrow" / "v3p_summary_subject.csv").open(newline="") as fh:
        by_subj = {r["subject"]: r for r in csv.DictReader(fh)}
    # still APPEARS in the per-subject CSV (never silently dropped), but flagged
    # out of the denominator and granted no support
    assert len(by_subj) == 7
    assert by_subj["epilepsiae_4099"]["excluded_from_denominator"] == "True"
    assert by_subj["epilepsiae_4099"]["subject_support"] == "False"

    blk = json.loads((tmp_path / "narrow" / "v3p_cohort_tier.json").read_text())["narrow"]
    # (a) NOT counted in n_eligible -- 6 clean, not 7
    assert (blk["n_total"], blk["n_eligible"], blk["n_excluded"]) == (7, 6, 1)
    # (b) its z did NOT enter the Wilcoxon -- p is what the 6 clean give alone
    assert blk["p_wilcoxon_b"] == pytest.approx(0.015625)
    assert blk["p_holm_b"] == pytest.approx(0.03125)
    assert blk["cohort_b_pass"] is True


def test_v3p_summary_h3pc_leg_is_a_discriminating_branch(tmp_path):
    """Fix 2: prove the H3p-c gate stack genuinely drives subject_support and
    that EACH of its conjuncts individually can zero it. No other fixture sets
    ``module_support_flag_c=True``, so Python's ``and`` short-circuits on the
    always-default-False flag and the five downstream c-leg conjuncts
    (onset_jitter_pass_c / single_contact_driven / leave_one_contact_mode_pass
    / axis_only_mode_control_pass / near_onset_dependent_c) are never
    evaluated for effect -- a bug in any of them would pass every other test.

    Fix 3 (folded in): row c001 also exercises the ``axis_weakening_supportive``
    TRUE branch (elsewhere only its False branch is hit) via
    p_label_slope_a<alpha + beta_axis_reliable + slope<0, and asserts
    support_driver on a real supporting row.
    """
    from scripts.run_topic5_v3p_summary import main

    rows = [
        # c-leg fully supporting, b-leg off -> support DRIVEN BY H3p-c;
        # p_label_slope_a<alpha -> axis_weakening_supportive TRUE as well.
        _v3p_row("epilepsiae_c001", "narrow", z_c=3.0, module_support_flag_c=True,
                 p_label_slope_a=0.01),
        # each c-leg conjunct flipped to its failing value (c otherwise supporting)
        _v3p_row("epilepsiae_c002", "narrow", z_c=3.0, module_support_flag_c=True, onset_jitter_pass_c=False),
        _v3p_row("epilepsiae_c003", "narrow", z_c=3.0, module_support_flag_c=True, single_contact_driven=True),
        _v3p_row("epilepsiae_c004", "narrow", z_c=3.0, module_support_flag_c=True, leave_one_contact_mode_pass=False),
        _v3p_row("epilepsiae_c005", "narrow", z_c=3.0, module_support_flag_c=True, axis_only_mode_control_pass=False),
        _v3p_row("epilepsiae_c006", "narrow", z_c=3.0, module_support_flag_c=True, near_onset_dependent_c=True),
    ]
    _write_v3p_trajectory_csv(tmp_path / "narrow" / "v3p_trajectory_subject.csv", rows)
    main(["--cohort", "narrow", "--indir", str(tmp_path)])
    with (tmp_path / "narrow" / "v3p_summary_subject.csv").open(newline="") as fh:
        by_subj = {r["subject"]: r for r in csv.DictReader(fh)}

    # c-path alone drives support, names itself the driver, sets the c-leg
    # column True and the b-leg column False
    assert by_subj["epilepsiae_c001"]["subject_support"] == "True"
    assert by_subj["epilepsiae_c001"]["support_driver"] == "H3p-c"
    assert by_subj["epilepsiae_c001"]["mode_transition_supported"] == "True"
    assert by_subj["epilepsiae_c001"]["nonaxis_flux_amplification_supported"] == "False"
    assert by_subj["epilepsiae_c001"]["axis_weakening_supportive"] == "True"  # Fix 3 TRUE branch

    # each c-leg conjunct individually zeroes the c-path
    for sid in ("epilepsiae_c002", "epilepsiae_c003", "epilepsiae_c004",
                "epilepsiae_c005", "epilepsiae_c006"):
        assert by_subj[sid]["subject_support"] == "False", sid
        assert by_subj[sid]["support_driver"] == "none", sid


# ---------------------------------------------------------------------------
# Task 10 -- result figure (co-primary surplus slopes + preictal trajectory).
# A pure rendering smoke test: a tiny synthetic per-cohort fixture (no
# mounted data, no real pipeline run needed) -- just checks the figure
# (2 co-primary-endpoint rows + a pooled trajectory row + the optional
# null-relative-z row, since this fixture DOES carry slope-z values) renders
# without crashing and that the PNG + README land on disk, non-empty. Uses
# its own minimal fieldnames list (NOT ``_V3P_ROW_DEFAULTS``/
# ``_write_v3p_trajectory_csv`` above, which are Task-9-scoped and do not
# carry ``p_label_slope_{b,c}``) so this test stays fully independent of the
# Task-9 fixture machinery.
# ---------------------------------------------------------------------------
def _write_csv_rows(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def test_v3p_plot_summary_renders_png_and_readme(tmp_path):
    from scripts.plot_topic5_v3p_summary import main

    subj_fields = ["subject", "cohort", "in_broad_core",
                   "net_offaxis_flux_surplus_slope", "net_offaxis_flux_slope_z", "p_label_slope_b",
                   "mode_shift_density_surplus_slope", "mode_shift_density_slope_z", "p_label_slope_c"]
    narrow_subj = [
        dict(subject="epilepsiae_1096", cohort="narrow", in_broad_core=False,
             net_offaxis_flux_surplus_slope=0.0018, net_offaxis_flux_slope_z=2.4, p_label_slope_b=0.02,
             mode_shift_density_surplus_slope=0.0002, mode_shift_density_slope_z=-0.3, p_label_slope_c=0.4),
        dict(subject="epilepsiae_1125", cohort="narrow", in_broad_core=False,
             net_offaxis_flux_surplus_slope=0.0004, net_offaxis_flux_slope_z=0.6, p_label_slope_b=0.3,
             mode_shift_density_surplus_slope=0.00005, mode_shift_density_slope_z=0.2, p_label_slope_c=0.4),
        dict(subject="epilepsiae_253", cohort="narrow", in_broad_core=False,
             net_offaxis_flux_surplus_slope=-0.0006, net_offaxis_flux_slope_z=-0.4, p_label_slope_b=0.6,
             mode_shift_density_surplus_slope=-0.0001, mode_shift_density_slope_z=-1.1, p_label_slope_c=0.7),
    ]
    broad_subj = [
        dict(subject="epilepsiae_139", cohort="broad", in_broad_core=True,
             net_offaxis_flux_surplus_slope=0.0012, net_offaxis_flux_slope_z=1.8, p_label_slope_b=0.03,
             mode_shift_density_surplus_slope=0.0001, mode_shift_density_slope_z=0.4, p_label_slope_c=0.5),
        dict(subject="epilepsiae_620", cohort="broad", in_broad_core=True,
             net_offaxis_flux_surplus_slope=0.0003, net_offaxis_flux_slope_z=0.2, p_label_slope_b=0.5,
             mode_shift_density_surplus_slope=-0.00003, mode_shift_density_slope_z=-0.2, p_label_slope_c=0.6),
        dict(subject="epilepsiae_583", cohort="broad", in_broad_core=False,
             net_offaxis_flux_surplus_slope=0.0009, net_offaxis_flux_slope_z=0.5, p_label_slope_b=0.4,
             mode_shift_density_surplus_slope=0.00006, mode_shift_density_slope_z=0.1, p_label_slope_c=0.5),
    ]
    _write_csv_rows(tmp_path / "narrow" / "v3p_trajectory_subject.csv", subj_fields, narrow_subj)
    _write_csv_rows(tmp_path / "broad" / "v3p_trajectory_subject.csv", subj_fields, broad_subj)

    win_fields = ["subject", "cohort", "seizure_idx", "span", "phase", "t_center",
                  "net_offaxis_flux_lag1", "mode_shift_density"]
    rng = np.random.default_rng(0)
    win_bounds = [(-118, -92), (-88, -62), (-58, -32), (-28, -12)]
    for cohort, subs in (("narrow", ["epilepsiae_1096", "epilepsiae_1125"]), ("broad", ["epilepsiae_139"])):
        rows = []
        for sid in subs:
            for sz in range(2):
                for i, (lo, hi) in enumerate(win_bounds):
                    rows.append(dict(
                        subject=sid, cohort=cohort, seizure_idx=sz, span="full", phase=f"P{i}",
                        t_center=float(rng.uniform(lo, hi)),
                        net_offaxis_flux_lag1=0.1 + 0.02 * i + float(rng.normal(0, 0.01)),
                        mode_shift_density=0.01 + 0.002 * i + float(rng.normal(0, 0.001)),
                    ))
        _write_csv_rows(tmp_path / cohort / "v3p_window_detail.csv", win_fields, rows)

    # minimal tier payload -- exactly the keys _caption() reads
    tier_payload = {
        "tier": 2, "state_v3p_supported": False,
        "narrow": {"n_eligible": 3, "n_subject_support": 1, "p_holm_b": 0.08, "p_holm_c": 0.6},
        "broad": {"n_eligible": 3, "n_subject_support": 1, "p_holm_b": 0.1, "p_holm_c": 0.55},
    }
    for cohort in ("narrow", "broad"):
        (tmp_path / cohort / "v3p_cohort_tier.json").write_text(json.dumps(tier_payload))

    outdir = tmp_path / "figures"
    out_paths = main(["--indir", str(tmp_path), "--outdir", str(outdir)])

    assert len(out_paths) == 1
    png_path = out_paths[0]
    assert png_path.exists() and png_path.stat().st_size > 0
    readme_path = png_path.parent / "README.md"
    assert readme_path.exists() and readme_path.stat().st_size > 0


def test_v3p_plot_summary_degrades_gracefully_without_panel_c(tmp_path):
    """Brief: "the plot must degrade gracefully if the optional panel C data
    is absent". A subject CSV missing BOTH slope-z columns entirely (an
    older/incomplete upstream write) must still render a (2-row) figure,
    never crash."""
    import matplotlib.pyplot as plt
    from scripts.plot_topic5_v3p_summary import (
        main, _build_figure, _load_subject_rows, _load_window_rows,
        _load_tier_payload, _panel_c_available,
    )

    subj_fields = ["subject", "cohort", "in_broad_core", "net_offaxis_flux_surplus_slope",
                   "p_label_slope_b", "mode_shift_density_surplus_slope", "p_label_slope_c"]
    rows = [dict(subject="epilepsiae_1096", cohort="narrow", in_broad_core=False,
                  net_offaxis_flux_surplus_slope=0.001, p_label_slope_b=0.2,
                  mode_shift_density_surplus_slope=0.0001, p_label_slope_c=0.5)]
    _write_csv_rows(tmp_path / "narrow" / "v3p_trajectory_subject.csv", subj_fields, rows)
    _write_csv_rows(tmp_path / "broad" / "v3p_trajectory_subject.csv", subj_fields, [])

    win_fields = ["subject", "cohort", "seizure_idx", "span", "phase", "t_center",
                  "net_offaxis_flux_lag1", "mode_shift_density"]
    win_rows = [dict(subject="epilepsiae_1096", cohort="narrow", seizure_idx=0, span="full",
                      phase="P0", t_center=-100.0, net_offaxis_flux_lag1=0.1, mode_shift_density=0.01)]
    _write_csv_rows(tmp_path / "narrow" / "v3p_window_detail.csv", win_fields, win_rows)
    _write_csv_rows(tmp_path / "broad" / "v3p_window_detail.csv", win_fields, [])

    outdir = tmp_path / "figures"
    out_paths = main(["--indir", str(tmp_path), "--outdir", str(outdir)])  # no v3p_cohort_tier.json at all

    png_path = out_paths[0]
    assert png_path.exists() and png_path.stat().st_size > 0
    readme_path = png_path.parent / "README.md"
    assert readme_path.exists() and readme_path.stat().st_size > 0

    # Lock the *2-row* degradation directly (not just "did not crash"):
    # derive panel-C availability exactly as main() does -- it must be False --
    # and a rebuilt figure must then have exactly 2 rows x 2 cols (4 axes); a
    # 3-row panel-C figure would have 6.
    subject_rows = _load_subject_rows(tmp_path)
    panel_c_on = _panel_c_available(subject_rows)
    assert panel_c_on is False
    fig = _build_figure(subject_rows, _load_window_rows(tmp_path),
                        _load_tier_payload(tmp_path), panel_c_on)
    assert len(fig.axes) == 4
    plt.close(fig)
