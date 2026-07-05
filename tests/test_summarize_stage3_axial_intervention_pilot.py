import json
import importlib.util
import os

_SPEC = importlib.util.spec_from_file_location(
    "_summ", os.path.join("scripts", "summarize_stage3_axial_intervention_pilot.py"))
_summ = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_summ)
summarize_file = _summ.summarize_file
summarize_dir = _summ.summarize_dir


def _toy(arm, seed, far_excl=0.2, far_raw=0.9):
    return dict(
        arm=arm, n_returned=12, n_neg=6, n_pos=5, n_collision=1, n_none=0, collision_rate=0.083,
        config=dict(seed=seed),
        pre_intervention_parity=(None if arm in ("baseline", "static_deadzone", "wall_only") else True),
        selected_baseline_event=(None if arm == "baseline"
                                 else dict(event_id=3, core_source_raw="neg", oracle_far_ratio=0.5,
                                           oracle_reach_mm=9.0)),
        selected_replay_event=(None if arm == "baseline"
                               else dict(event_id=3, oracle_far_ratio=0.1, oracle_reach_mm=3.0)),
        events=[dict(core_source_raw="neg", oracle_far_ratio=0.3, oracle_reach_mm=8.0,
                     instr_far_ratio=far_raw, instr_far_ratio_excl_target_contacts=far_excl)],
    )


def test_summary_groups_by_arm_and_seed(tmp_path):
    d = str(tmp_path)
    json.dump(_toy("baseline", 1), open(os.path.join(d, "baseline_s1.json"), "w"))
    json.dump(_toy("dynamic_on_axis", 1), open(os.path.join(d, "dynamic_on_axis_s1.json"), "w"))
    json.dump(_toy("dynamic_on_axis", 2), open(os.path.join(d, "dynamic_on_axis_s2.json"), "w"))
    rows = summarize_dir(d)
    assert len(rows) == 3
    keys = {(r["arm"], r["seed"]) for r in rows}
    assert keys == {("baseline", 1), ("dynamic_on_axis", 1), ("dynamic_on_axis", 2)}


def test_summary_preserves_fail_guard_fields():
    row = summarize_file(_toy("dynamic_on_axis", 1))
    for k in ("pre_intervention_parity", "collision_rate", "selected_event_id", "selected_source"):
        assert k in row
    assert row["pre_intervention_parity"] is True
    assert row["selected_event_id"] == 3 and row["selected_source"] == "neg"
    # paired baseline->replay far ratio is preserved for the directional read
    assert row["selected_baseline_far_ratio"] == 0.5 and row["selected_replay_far_ratio"] == 0.1


def test_summary_uses_excluded_target_contact_metric():
    # event has raw instr 0.9 but excluded-target instr 0.2; the summary must use 0.2, not 0.9
    row = summarize_file(_toy("dynamic_on_axis", 1, far_excl=0.2, far_raw=0.9))
    assert row["median_instr_far_ratio_excl_target_contacts"] == 0.2
    assert "median_instr_far_ratio" not in row   # raw metric must not be silently substituted
