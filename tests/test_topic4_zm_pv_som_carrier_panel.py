from scripts.analyze_topic4_zm_conductance_homotopy import credible_carrier
from scripts.analyze_topic4_zm_pv_som_carrier import (
    BASE_ORDER, _gap_spatial_class, _label, _split_arm, adjudicate,
)


def _summary(*, rho=.5, g_max=None, e_exc=60., down=250., common=0.,
             som_tau_d=60., shunt=False, seed=1):
    """A locked PV/SOM arm; only the mode-H coordinates vary."""
    mode = {
        "rho_mode_H": rho,
        "tau_mode_H": 250.,
        "tau_mode_H_down": down,
        "mode_H_common_subtraction": common,
        "mode_H_persistent_e_exc": e_exc,
        "m_mode_half": 30.,
    }
    if g_max is not None:
        mode["mode_H_persistent_g_max"] = g_max
    subtype = {
        "som_source_fraction_realized": .25,
        "som_slow_integrated_budget_fraction": .35,
        "som_recruit_delay_scale": 3.,
        "tau_d_som_ms": som_tau_d,
        "seed": seed,
    }
    if shunt:
        subtype["slow_membrane_mode"] = "shunt"
    return {
        "state": "bounded_late__peak",
        "T_ms": 2500.,
        "mechanism": {
            "pv_som_inhibitory_subtypes": subtype,
            "state_selective_mode_H": mode,
        },
    }


def test_multiplicative_arms_survive_the_dose_series_extension():
    assert _label(_summary()) == "current_H250"
    assert _label(_summary(som_tau_d=30.)) == "current_SOM30"
    assert _label(_summary(down=1500.)) == "current_H1500"
    assert _label(_summary(down=1500., common=1.)) == "contrast_H1500"
    assert _label(_summary(shunt=True)) == "SOM_shunt"


def test_runs_predating_the_mechanism_default_to_the_parity_path():
    """No persistent key means g=0, which is the literal pre-change membrane."""
    assert "mode_H_persistent_g_max" not in (
        _summary()["mechanism"]["state_selective_mode_H"]
    )
    assert _label(_summary()) == "current_H250"


def test_dose_series_admits_its_own_matched_control():
    assert _label(_summary(rho=0., g_max=0.)) == "persistent_g0"
    assert _label(_summary(rho=0., g_max=.04)) == "persistent_g0.04"
    assert _label(_summary(rho=0., g_max=.32)) == "persistent_g0.32"


def test_dose_series_refuses_arms_that_are_not_like_for_like():
    # The reversal potential sets how much drive a given g delivers, so a
    # different value is a different mechanism, not another dose.
    assert _label(_summary(rho=0., g_max=.32, e_exc=40.)) is None
    # Substrate and H timescale must match the panel they are compared against.
    assert _label(_summary(rho=0., g_max=.32, som_tau_d=30.)) is None
    assert _label(_summary(rho=0., g_max=.32, down=1500.)) is None
    assert _label(_summary(rho=0., g_max=.32, common=1.)) is None
    # Persistent conductance requires the current-based membrane arm.
    assert _label(_summary(rho=0., g_max=.32, shunt=True)) is None
    # Partial multiplicative gain belongs to no declared family.
    assert _label(_summary(rho=.25, g_max=.32)) is None


def test_replicate_substrates_get_their_own_arm_identity():
    """A different SOM wiring is a different substrate, not a duplicate arm."""
    assert _label(_summary(rho=0., g_max=.32, seed=2)) == "persistent_g0.32__som2"
    assert _label(_summary(rho=0., g_max=0., seed=3)) == "persistent_g0__som3"
    assert _label(_summary(rho=0., g_max=.32)) == "persistent_g0.32"
    # The multiplicative comparison panel was only ever run on one substrate,
    # so a replicate must not be pooled into it under the same label.
    assert _label(_summary(seed=2)) is None


def test_dose_and_seed_are_recoverable_from_the_label():
    dose, seed = _split_arm(_label(_summary(rho=0., g_max=.32, seed=2)))
    assert (dose, seed) == (0.32, 2)
    assert _split_arm(_label(_summary(rho=0., g_max=.04))) == (0.04, 1)


def _row(*, gap=.7, pc1=.8, runaway=False):
    return {"post_onset_deep_gap_fraction": gap, "spatial_pc1": pc1,
            "runaway": runaway}


def _panel(arms):
    """Rows for {persistent arm label: passes gate}; the panel never passes."""
    rows = {}
    for label, ok in arms.items():
        row = _row(gap=0. if ok else .7, pc1=.82)
        row["credible_carrier"] = ok
        row["gap_spatial_class"] = _gap_spatial_class(row)
        rows[label] = row
    for label in BASE_ORDER:
        row = _row()
        row["credible_carrier"] = False
        row["gap_spatial_class"] = _gap_spatial_class(row)
        rows[label] = row
    return rows


def test_single_substrate_pass_is_not_yet_a_replicated_carrier():
    verdict = adjudicate(_panel({
        "persistent_g0": False, "persistent_g0.32": True,
    }))
    assert verdict["verdict"] == "PERSISTENT_SLOW_EXCITATION_CARRIER_CANDIDATE"
    assert verdict["passing_arms"] == ["persistent_g0.32"]


def test_a_wiring_with_no_passing_dose_at_all_is_substrate_dependent():
    verdict = adjudicate(_panel({
        "persistent_g0": False, "persistent_g0.32": True,
        "persistent_g0__som2": False, "persistent_g0.32__som2": False,
    }))
    assert verdict["verdict"] == (
        "PERSISTENT_SLOW_EXCITATION_CARRIER_IS_SUBSTRATE_DEPENDENT"
    )
    assert verdict["seed_replication"]["g0.32"]["seeds_tested"] == [1, 2]
    assert verdict["seed_replication"]["g0.32"]["seeds_passing_gate"] == [1]
    assert verdict["wirings_without_a_passing_dose"] == [2]


def test_a_wiring_that_passes_at_its_own_dose_is_not_substrate_dependent():
    """Every wiring carries; only the dose that clears the gate moves."""
    verdict = adjudicate(_panel({
        "persistent_g0.32": True, "persistent_g0.4": False,
        "persistent_g0.32__som2": True, "persistent_g0.4__som2": True,
        "persistent_g0.32__som3": False, "persistent_g0.4__som3": True,
    }))
    assert verdict["verdict"] == (
        "PERSISTENT_SLOW_EXCITATION_CARRIER_REPLICATES_AT_A_WIRING_SPECIFIC_DOSE"
    )
    assert verdict["wirings_without_a_passing_dose"] == []
    assert verdict["passing_dose_per_wiring"] == {"1": 0.32, "2": 0.32, "3": 0.4}


def test_one_dose_holding_on_every_tested_wiring_is_promoted():
    verdict = adjudicate(_panel({
        "persistent_g0": False, "persistent_g0.32": True,
        "persistent_g0.32__som2": True, "persistent_g0.32__som3": True,
    }))
    assert verdict["verdict"] == (
        "PERSISTENT_SLOW_EXCITATION_CARRIER_REPLICATES_ACROSS_SUBSTRATES"
    )
    assert verdict["seed_replication"]["g0.32"]["seeds_passing_gate"] == [1, 2, 3]
    assert verdict["passing_dose_per_wiring"] == {"1": 0.32, "2": 0.32, "3": 0.32}


def test_the_weakest_passing_dose_is_named_not_the_strongest():
    verdict = adjudicate(_panel({
        "persistent_g0": False, "persistent_g0.16": True,
        "persistent_g0.32": True, "persistent_g0.64": True,
    }))
    assert verdict["weakest_passing_dose"] == 0.16


def test_no_pass_anywhere_keeps_the_gap_verdict():
    verdict = adjudicate(_panel({
        "persistent_g0": False, "persistent_g0.04": False,
        "persistent_g0.08": False,
    }))
    assert verdict["verdict"] == "PV_SOM_SPATIAL_PATTERN_WITH_TEMPORAL_GAPS"
    assert verdict["weakest_passing_dose"] is None


def test_gap_spatial_class_separates_the_two_failure_axes():
    assert _gap_spatial_class(_row()) == "fragmented_spatially_distributed"
    assert _gap_spatial_class(_row(pc1=.97)) == "common_mode_fragmented"
    assert _gap_spatial_class(_row(gap=0.)) == "gaps_filled_spatially_distributed"
    assert _gap_spatial_class(_row(gap=0., pc1=.97)) == "common_mode_plateau"


def test_runaway_and_missing_episode_outrank_the_two_axes():
    assert _gap_spatial_class(_row(gap=0., pc1=.8, runaway=True)) == "runaway"
    assert _gap_spatial_class(_row(gap=None)) == "no_episode"
    assert _gap_spatial_class(_row(pc1=None)) == "no_episode"


def test_class_boundaries_track_the_locked_carrier_gate():
    """The class must not drift from the gate it is meant to explain."""
    assert _gap_spatial_class(_row(gap=.20, pc1=.95)) == (
        "gaps_filled_spatially_distributed"
    )
    assert _gap_spatial_class(_row(gap=.21, pc1=.95)) != (
        "gaps_filled_spatially_distributed"
    )
    assert _gap_spatial_class(_row(gap=.20, pc1=.951)) == "common_mode_plateau"

    # Anything the gate accepts must land in the one class that describes a
    # carrier; if the gate thresholds move, this pins the class to follow.
    passing = {
        "episode_status": "onset_persistent", "runaway": False,
        "median_vseeg_gain_db": 25.2, "energy_occupancy_6db": .55,
        "post_onset_deep_gap_fraction": 0., "spatial_pc1": .82,
        "core_mean_hz": 187., "core_rho80_active_fraction": 0.,
    }
    assert credible_carrier(passing)
    assert _gap_spatial_class(passing) == "gaps_filled_spatially_distributed"
