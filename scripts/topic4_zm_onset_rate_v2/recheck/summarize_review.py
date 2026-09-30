"""Persist audit evidence separately and mark the prior figure's overclassification."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULT = ROOT / "results/topic4_sef_hfo/fig5_zm_rate_synchronized_20260917"
AUDIT = RESULT / "recheck_20260917"


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    equations = read(AUDIT / "independent_equation_audit.json")
    assert equations["status"] == "PASS_INTERNAL_EQUATIONS_ONLY"
    folds = []
    for name in ["LPC_small_max", "LPC_small_min"]:
        coarse = read(AUDIT / f"{name}_N256.json")
        fine = read(AUDIT / f"{name}_N512.json")
        folds.append({**fine, "D_time_mesh_difference": abs(fine["D"] - coarse["D"]),
                      "T_time_mesh_difference_ms": abs(fine["T_ms"] - coarse["T_ms"])})
    floquet = []
    for dt in ["0.05", "0.025"]:
        source = AUDIT / f"floquet/burstUp_0031_N512_dt{dt}.json"
        q = read(source)
        neutral = q["identified_neutral_index"]
        assert neutral is not None
        vals = [complex(*v) for v in q["multipliers"]]
        growing = max((i for i in range(len(vals)) if i != neutral), key=lambda i: abs(vals[i]))
        assert vals[growing].real > 1 and abs(vals[growing].imag) < 1e-10
        floquet.append(dict(source=str(source), dt_ms=q["dt_ms"],
                            multiplier=vals[growing].real, phase_multiplier=vals[neutral].real,
                            eigen_residual=q["residuals"][growing],
                            phase_tangent_defect=q["phase_tangent_relative_defect"],
                            period_ms=q["T_ms"]))
    difference = abs(floquet[1]["multiplier"] - floquet[0]["multiplier"])
    fine = floquet[1]
    review = dict(
        date="2026-09-18", review_execution="COMPLETE", bifurcation_analysis="INCOMPLETE",
        model_internal_consistency="PASS_CHECKED_ORBITS_AND_HOPF_MODE",
        native_SNN_equivalence="NOT_ACCEPTED_LOCAL_RESPONSE_COUNTEREVIDENCE",
        conditional_definition="Z fixed on prescribed spatial power path; M dynamic; J_EE_core=1",
        new_periodic_folds=folds,
        fold_pair_D_width=folds[0]["D"]-folds[1]["D"],
        middle_orbit_Floquet=floquet,
        middle_orbit_stability="UNSTABLE_SUPPORTED_BY_TWO_TIME_STEPS",
        multiplier_step_difference=difference,
        fine_distance_above_one_over_step_difference=(fine["multiplier"]-1)/difference,
        fine_efold_time_s=fine["period_ms"]/math.log(fine["multiplier"])/1000,
        whole_middle_branch_stability="NOT_EXHAUSTIVELY_CLASSIFIED",
        prior_main_LPC="LOCAL_EVIDENCE_RETAINED; NOT_FIRST_OR_ONLY_PERIODIC_FOLD",
        high_rate_Hopf="LOCAL_SUPERCRITICAL_EVIDENCE_RETAINED",
        prior_figure_stability_line="INVALID_OVERCLASSIFICATION_BEFORE_MAIN_LPC",
        SNN_onset_bifurcation="NOT_ESTABLISHED",
        separatrix="NOT_ESTABLISHED",
        spatial_grid_convergence="NOT_TESTED",
        human_visual_acceptance="PENDING",
        own_jobs_running=False,
        limits=["Representative internal equation checks, not all orbit/state combinations",
                "Qualitative numerical instability; not a rigorous full-spectrum enclosure",
                "No physical parameter or shared-model replacement in this review"],
    )
    write(AUDIT / "review_status.json", review)
    status_path = RESULT / "status.json"
    previous = read(status_path)
    backup = AUDIT / "status_before_recheck.json"
    if not backup.exists():
        write(backup, previous)
    previous.update(stage="SCIENTIFIC_RECHECK_CONFIRMED_GAPS", task_complete=False,
                    review_execution="COMPLETE", review_report=str(AUDIT / "scientific_review.md"),
                    latest_review_status=str(AUDIT / "review_status.json"), own_jobs_running=False,
                    own_jobs=[], bifurcation_completeness="INCOMPLETE_CONFIRMED_MISSED_PERIODIC_FOLDS",
                    Floquet_scope="Prior-stage sampled-orbit counts; additional weakly unstable audit orbit is recorded separately below",
                    additional_audit_Floquet={"burstUp_0031_N512": "UNSTABLE_SUPPORTED_BY_STEP_HALVING"},
                    new_periodic_folds_D=[q["D"] for q in folds],
                    prior_figure="REQUIRES_PERIODIC_STABILITY_LINE_CORRECTION",
                    onset_bifurcation_type="LOCAL_CONDITIONAL_LPC_RETAINED; NATIVE_GLOBAL_ONSET_NOT_ESTABLISHED")
    previous["remaining"] = [
        "Repair and independently validate depleted-workpoint rate-response closure against colored LIF and native SNN dynamics/propagation",
        "Correct omitted small periodic folds and classify weak instability without extrapolating from a single maximum-D split",
        "Connect the prescribed conditional Z path to the actual autonomous Z/M entry field and carried histories",
        "Check 1 mm to 0.5 mm spatial convergence of the relevant branches and modes",
        "Resolve relevant branch endpoints, remaining equilibrium stability and post-LPC attractor type",
    ]
    write(status_path, previous)
    print(json.dumps({"review": review["review_execution"], "new_folds": previous["new_periodic_folds_D"],
                      "Floquet": floquet, "efold_time_s": review["fine_efold_time_s"]}, indent=2))


if __name__ == "__main__":
    main()
