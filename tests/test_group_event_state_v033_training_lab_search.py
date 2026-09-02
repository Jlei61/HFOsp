"""Task 6: search space, ASHA, seed policy, stop rules and failure classification (S1-S7)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.topic5_group_event_state.v033_training_lab.data import build_view
from src.topic5_group_event_state.v033_training_lab.models import GROUP_NAMES
from src.topic5_group_event_state.v033_training_lab.objective import ResidualCountTrainable
from src.topic5_group_event_state.v033_training_lab.search import (
    CATEGORICAL,
    LR_LOG_RANGE,
    SearchBudget,
    SearchSpace,
    asha_promote,
    classify_failure,
    run_search,
    run_search_batch,
)
from src.topic5_group_event_state.v033_training_lab.synthetic import plant_residual_signal
from tests.test_group_event_state_v032_model_toyutil import make_toy_bundle


def _keys(obj, found=None):
    found = [] if found is None else found
    if isinstance(obj, dict):
        for k, v in obj.items():
            found.append(str(k))
            _keys(v, found)
    elif isinstance(obj, list):
        for v in obj:
            _keys(v, found)
    return found


def _views(seed=12, beta=1.0):
    bundle, _ = make_toy_bundle(seed=seed, planted_beta=0.0)
    z = build_view(bundle, scaling="zscore")
    r = build_view(bundle, scaling="robust")
    pz, _ = plant_residual_signal(z, beta=beta, dispersion_r=8.0, generator_seed=1, noise_seed=2)
    pr, _ = plant_residual_signal(r, beta=beta, dispersion_r=8.0, generator_seed=1, noise_seed=2)
    return {"zscore": pz, "robust": pr}


def test_s1_sampler_is_seeded_log_uniform_in_lr_and_covers_every_categorical_value():
    space = SearchSpace.for_family("fixed_leaky")
    rng = np.random.default_rng(0)
    cfgs = [space.sample(rng) for _ in range(600)]
    rng_again = np.random.default_rng(0)
    again = [space.sample(rng_again) for _ in range(3)]
    assert [c.config_hash() for c in cfgs[:3]] == [c.config_hash() for c in again]
    for cfg in cfgs:
        for g in GROUP_NAMES:
            assert LR_LOG_RANGE[0] <= cfg.lr[g] <= LR_LOG_RANGE[1]
    logs = np.log([c.lr["encoder_weights"] for c in cfgs])
    mid = 0.5 * (math.log(LR_LOG_RANGE[0]) + math.log(LR_LOG_RANGE[1]))
    assert abs(float(np.median(logs)) - mid) < 0.4
    seen = {name: set() for name in CATEGORICAL}
    for cfg in cfgs:
        d = cfg.as_dict()
        seen["optimizer"].add(d["optimizer"]); seen["schedule"].add(d["schedule"])
        seen["warmup_fraction"].add(d["warmup_fraction"]); seen["dispersion"].add(d["dispersion"])
        seen["sampling"].add(d["sampling"]); seen["scaling"].add(d["scaling"])
        for key in ("init", "write_scale", "alpha_init", "hidden_norm", "depth", "width", "activation", "dropout",
                    "write_width"):
            seen[key].add(d["arch"][key])
        seen["time_bank"].add(tuple(d["arch"]["taus_seconds"]))
    for name, values in CATEGORICAL.items():
        expected = {tuple(v) if isinstance(v, (list, tuple)) else v for v in values}
        assert seen[name] == expected, name
    assert all(c.arch.state_family == "fixed_leaky" for c in cfgs)
    with pytest.raises(ValueError):
        SearchSpace.for_family("gated_exploratory")
    gated = SearchSpace.for_family("gated_exploratory", gated_approved=True)
    g = gated.sample(np.random.default_rng(1))
    assert g.arch.state_family == "gated_exploratory" and g.arch.tbptt_seconds in (1800.0, 3600.0, 7200.0)
    narrowed = SearchSpace.for_family("fixed_leaky", restrict={"write_width": (4,), "optimizer": ("adamw",)})
    assert all(c.arch.write_width == 4 and c.optimizer == "adamw"
               for c in (narrowed.sample(np.random.default_rng(i)) for i in range(20)))


def test_s2_asha_keeps_top_fraction_but_never_prunes_a_config_inside_its_grace_period():
    rows = [{"config_id": f"c{i}", "score": float(i), "grace_ok": True} for i in range(8)]
    rows[7]["grace_ok"] = False                     # worst score, but its groups activated too late
    decision = asha_promote(rows, eta=2)
    assert decision["n_keep"] == 4
    assert decision["promoted"][:4] == ["c0", "c1", "c2", "c3"]
    assert "c7" in decision["promoted"] and decision["grace_deferred"] == ["c7"]
    assert set(decision["pruned"]) == {"c4", "c5", "c6"}


def test_s3_search_batch_runs_rungs_with_the_seed_policy_and_resumes_between_rungs(tmp_path):
    views = _views()
    trainable = ResidualCountTrainable()
    space = SearchSpace.for_family("fixed_leaky", restrict={"schedule": ("constant",), "warmup_fraction": (0.0,),
                                                             "depth": (1,), "width": (32,), "dropout": (0.0,)})
    budget = SearchBudget(n_configs=4, max_steps=40, rung_steps=(20, 40), eta=2, seeds_low=1, seeds_mid=2,
                          seeds_final=2, n_final=2, validate_every=10)
    batch = run_search_batch(trainable, views, space, budget, base_seed=5, device="cpu", out_dir=tmp_path,
                             batch_index=0)
    rows = batch["rows"]
    rung0 = [r for r in rows if r["rung_index"] == 0]
    rung1 = [r for r in rows if r["rung_index"] == 1]
    assert len(rung0) == 4 and all(r["seed_index"] == 0 and r["steps_budget"] == 20 for r in rung0)
    assert len({r["config_id"] for r in rung1}) == 2 and len(rung1) == 4
    assert {r["seed_index"] for r in rung1} == {0, 1}
    seed0_final = [r for r in rung1 if r["seed_index"] == 0]
    assert all(r["resumed_from_step"] == 20 for r in seed0_final)
    inc = batch["incumbent"]
    assert inc["n_seeds"] == 2 and inc["rung_index"] == 1 and math.isfinite(inc["score"])
    assert inc["config_id"] in {r["config_id"] for r in rung1}
    trace = json.loads((tmp_path / "search_trace.json").read_text())
    assert len(trace["rows"]) == len(rows)
    for key in ("config_id", "config_hash", "seed", "seed_index", "rung_index", "steps_budget", "n_steps_run",
                "inner_val_nll", "gain_h_minus_model", "all_groups_active_step", "grace_ok", "status",
                "selected_in_warmup", "selected_at_budget_edge"):
        assert key in rows[0], key
    assert "dev_test" not in " ".join(_keys(batch)).lower()


def test_s4_search_stops_after_two_batches_without_improvement_or_on_a_stable_plateau(tmp_path):
    scores = iter([1.0, 1.0, 1.0, 1.0])

    def flat(*_a, batch_index, **_k):
        return {"batch_index": batch_index, "incumbent": {"config_id": f"c{batch_index}", "score": next(scores),
                                                          "plateau_reached": False, "config_hash": "h"}, "rows": []}

    out = run_search(None, {}, None, None, base_seed=0, device="cpu", out_dir=tmp_path / "a", max_batches=6,
                     batch_runner=flat)
    assert out["stop_reason"] == "no_improvement_two_batches" and out["n_batches"] == 3
    assert out["incumbent"]["config_id"] == "c0"

    def plateau(*_a, batch_index, **_k):
        return {"batch_index": batch_index, "incumbent": {"config_id": f"p{batch_index}", "score": 2.0,
                                                          "plateau_reached": True, "config_hash": "h"}, "rows": []}

    out = run_search(None, {}, None, None, base_seed=0, device="cpu", out_dir=tmp_path / "b", max_batches=6,
                     batch_runner=plateau)
    assert out["stop_reason"] == "stable_plateau" and out["n_batches"] == 2

    improving = iter([3.0, 2.0, 1.0, 0.5, 0.25, 0.1])

    def better(*_a, batch_index, **_k):
        return {"batch_index": batch_index, "incumbent": {"config_id": f"i{batch_index}", "score": next(improving),
                                                          "plateau_reached": False, "config_hash": "h"}, "rows": []}

    out = run_search(None, {}, None, None, base_seed=0, device="cpu", out_dir=tmp_path / "c", max_batches=3,
                     batch_runner=better)
    assert out["stop_reason"] == "max_batches" and out["incumbent"]["config_id"] == "i2"


def test_s5_failure_classification_covers_every_row_with_a_fixed_priority():
    base = dict(tiny_overfit_pass=True, all_groups_active=True, train_learned=True, inner_val_gain_ci_low=0.05,
                synthetic_recovery_pass=True, random_reservoir_equivalent=False, selected_at_budget_edge=False,
                search_no_improvement_batches=0, effective_windows=30, min_effective_windows=8)
    assert classify_failure(base)["category"] == "none"
    assert classify_failure({**base, "effective_windows": 4})["category"] == "support_insufficient"
    assert classify_failure({**base, "tiny_overfit_pass": False})["category"] == "gradient_path"
    assert classify_failure({**base, "all_groups_active": False})["category"] == "gradient_path"
    assert classify_failure({**base, "selected_at_budget_edge": True})["category"] == "budget_edge"
    assert classify_failure({**base, "train_learned": False, "inner_val_gain_ci_low": -0.1})["category"] == "underfit"
    assert classify_failure({**base, "inner_val_gain_ci_low": -0.1, "random_reservoir_equivalent": True})["category"] == "objective_or_support"
    assert classify_failure({**base, "inner_val_gain_ci_low": -0.1})["category"] == "overfit_or_objective"
    assert classify_failure({**base, "search_no_improvement_batches": 2})["category"] == "search_exhausted"
    both = classify_failure({**base, "effective_windows": 4, "tiny_overfit_pass": False})
    assert both["category"] == "support_insufficient" and set(both["all_matches"]) >= {"support_insufficient", "gradient_path"}
    assert "next_action" in both and "rule" in both


def test_s6_budget_from_request_validates_rungs():
    budget = SearchBudget.from_request({"n_configs": 4, "max_steps": 60, "rung_steps": [20, 60], "eta": 2,
                                        "seeds_low": 1, "seeds_mid": 3, "seeds_final": 5, "n_final": 2})
    assert budget.rung_steps == (20, 60) and budget.seeds_final == 5
    with pytest.raises(ValueError):
        SearchBudget(n_configs=4, max_steps=60, rung_steps=(60, 20), eta=2)
    with pytest.raises(ValueError):
        SearchBudget(n_configs=4, max_steps=60, rung_steps=(20, 50), eta=2)
