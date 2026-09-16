"""Control-arm parity contracts for v0.3.7.

The v0.3.7 first round left every load-bearing comparison uninterpretable in the
same way: the arm that carries the scientific claim was optimised while the arm
it must beat was not.  Three separate layers reproduced the identical signature
(a control or an interface whose weights stayed exactly at their zero
initialisation), so the tests below are written against the *mechanism*, not
against any particular number.

Contracts asserted here:

1.  Every added arm retains its fitted parent as an explicit zero-increment
    candidate.  A non-zero optimiser start may help gradients, but can never
    replace the parent in a nested scientific comparison.
2.  Every optimiser recipe searched for a model family also searches the
    baseline, and the trainability gate covers the baseline family.
3.  The frozen-decoder modulation interface starts in exact parity, updates its
    output maps on the first step, and passes gradient to the state path from
    the second step onward; every v037 call site sets this explicitly rather
    than inheriting the v035 default.
4.  Held-out sample size for a physical horizon is reported as non-overlapping
    windows, not as overlapping five-minute anchors.
5.  Cards record the code version that produced them.
"""

from __future__ import annotations

import ast
from dataclasses import fields
import inspect
from pathlib import Path
import re

import numpy as np
import torch

from src.topic5_group_event_state.v035.stepwise_decoder import (
    DynamicStepAdapter,
    StepwiseAdapterConfig,
)
from src.topic5_group_event_state.v037 import h2a as h2a_module
from src.topic5_group_event_state.v037.h1_train import (
    H1TrainConfig,
    NestedH1Readout,
    independent_window_count,
)
from src.topic5_group_event_state.v037.h2a import H2ATrainConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
SEARCH_SCRIPT = REPO_ROOT / "scripts/run_group_event_state_v037_optimizer_search.py"
FINALIZE_SCRIPT = REPO_ROOT / "scripts/finalize_group_event_state_v037_optimizer_search.py"
OPTIMIZED_H1_SCRIPT = REPO_ROOT / "scripts/run_group_event_state_v037_optimized_h1.py"

_WIDTHS = {"count": 1, "burden": 2, "community": 3, "coupling": 4, "mixture": 3, "embedding": 2, "mark": 5}


# ---------------------------------------------------------------- 1. baseline parity


def test_baseline_readout_can_be_given_the_same_non_zero_start_as_the_state() -> None:
    model = NestedH1Readout(2, 3, 4, 2, 3, _WIDTHS, 2, state_readout_init_std=1e-2, bmark_readout_init_std=1e-2)
    baseline = torch.cat([module.weight.reshape(-1) for module in model.bmark.values()])
    state = torch.cat([module.weight.reshape(-1) for module in model.state.values()])
    assert float(baseline.abs().max()) > 0.0, "baseline readout must not start at exactly zero"
    assert float(state.abs().max()) > 0.0


def test_baseline_readout_default_stays_at_zero_for_backward_compatibility() -> None:
    model = NestedH1Readout(2, 3, 4, 2, 3, _WIDTHS, 2)
    baseline = torch.cat([module.weight.reshape(-1) for module in model.bmark.values()])
    assert float(baseline.abs().max()) == 0.0


def test_rate_control_has_a_deterministic_zero_start() -> None:
    first = NestedH1Readout(2, 3, 4, 2, 3, _WIDTHS, 2)
    second = NestedH1Readout(2, 3, 4, 2, 3, _WIDTHS, 2)
    for name in first.ENDPOINTS:
        assert torch.equal(first.q[name].weight, torch.zeros_like(first.q[name].weight))
        assert torch.equal(first.q[name].bias, torch.zeros_like(first.q[name].bias))
        assert torch.equal(first.q[name].weight, second.q[name].weight)


def test_train_config_exposes_baseline_initialisation_and_warmup() -> None:
    names = {field.name for field in fields(H1TrainConfig)}
    assert "bmark_readout_init_std" in names, "the control arm needs its own initialisation knob"
    assert "warmup_steps_bmark" in names, "the control arm needs its own warm-up knob"


def test_baseline_stage_applies_its_own_warmup() -> None:
    source = inspect.getsource(__import__(
        "src.topic5_group_event_state.v037.h1_train", fromlist=["_train_stage"]
    )._train_stage)
    assert "warmup_steps_bmark" in source, "_train_stage must honour the baseline warm-up"


def test_nested_stage_keeps_parent_parity_candidate() -> None:
    source = inspect.getsource(__import__(
        "src.topic5_group_event_state.v037.h1_train", fromlist=["_train_stage"]
    )._train_stage)
    assert "parent_inner_loss" in source
    assert "optimiser_initial_inner_loss" in source
    assert "selected_parent_parity" in source
    assert "peak_parameter_delta_from_stage_start" in source


def test_control_gate_distinguishes_pathway_exploration_from_scientific_gain() -> None:
    source = (REPO_ROOT / "scripts/finalize_group_event_state_v037_baseline_search.py").read_text(
        encoding="utf-8"
    )
    assert "mark_pathway_explored_fraction" in source
    assert "mark_peak_parameter_delta" in source
    assert "rate_pathway_explored_fraction" in source
    assert "rate_peak_parameter_delta" in source
    assert 'mark["gain_over_parent"]' in source
    assert "and statistics.median(row[\"mark_gain\"]" not in source


# ---------------------------------------------------------------- 2. equal search


def _literal(script: Path, name: str) -> dict:
    tree = ast.parse(script.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return _eval_recipe_block(node.value)
    raise AssertionError(f"{name} not found in {script}")


def _eval_recipe_block(node: ast.AST) -> dict:
    out: dict = {}
    assert isinstance(node, ast.Dict)
    for key, value in zip(node.keys, node.values):
        assert isinstance(key, ast.Constant)
        assert isinstance(value, ast.Call)
        out[key.value] = {kw.arg for kw in value.keywords}
    return out


def test_the_control_arm_gets_its_own_searched_recipe_grid() -> None:
    baseline = _literal(SEARCH_SCRIPT, "BASELINE_RECIPES")
    assert len(baseline) >= 3, "one untested control setting is what made round one uninterpretable"
    tuned = {
        "lr_bmark", "max_steps_bmark", "warmup_steps_bmark", "bmark_readout_init_std",
        # The rate stage is the innermost control; a gain nested above an
        # unconverged rate stage cannot be attributed to the mark bank.
        "lr_q", "max_steps_q",
    }
    varied: set[str] = set()
    for keys in baseline.values():
        varied |= keys & tuned
    assert tuned <= varied, f"control recipes never vary {sorted(tuned - varied)}"
    # The legacy setting must stay in the grid so the first round remains a
    # comparable point rather than being quietly redefined away.
    assert any("legacy" in name for name in baseline)


def test_trainability_gate_covers_the_baseline_stage() -> None:
    source = FINALIZE_SCRIPT.read_text(encoding="utf-8")
    assert "bmark" in source, "the trainability gate must read the baseline stage"
    assert "baseline_trainability_eligible" in source, (
        "the gate must expose a separate eligibility flag for the control arm"
    )


def test_integrated_summary_preserves_the_patient_first_control_gate() -> None:
    source = FINALIZE_SCRIPT.read_text(encoding="utf-8")
    assert 'frozen_recipe.get(' in source
    assert '"eligible"' in source
    assert '"per_subject_stage_eligible"' in source


def test_model_trainability_is_not_defined_by_a_human_effect() -> None:
    source = FINALIZE_SCRIPT.read_text(encoding="utf-8")
    assert "pathway_explored_fraction" in source
    assert "inner_increment_supported" in source
    trainability = re.search(
        r'"trainability_eligible": \((.*?)\n\s*\),', source, flags=re.S
    )
    assert trainability is not None
    assert "median_gain" not in trainability.group(1)
    assert "non_init_fraction" not in trainability.group(1)


def test_trainability_gate_also_requires_a_converged_rate_stage() -> None:
    source = FINALIZE_SCRIPT.read_text(encoding="utf-8")
    assert "rate_stage_eligible" in source, (
        "the mark bank is nested above the rate stage; a rate stage stuck at its "
        "budget edge makes the nested gain unattributable"
    )
    assert "rate_training_budget_exhausted" in source
    assert "rate_stage_median_inner_gain" in source, (
        "the innermost rate control must improve as well as merely leave step zero"
    )


def test_budget_gate_uses_training_termination_not_checkpoint_location() -> None:
    source = FINALIZE_SCRIPT.read_text(encoding="utf-8")
    assert 'row["training_budget_exhausted"]' in source


def test_optimizer_supervisor_actually_schedules_the_control_grid() -> None:
    source = (REPO_ROOT / "scripts/supervise_group_event_state_v037_optimizer_search.py").read_text(
        encoding="utf-8"
    )
    assert "BASELINE_RECIPES" in source
    assert '("bmark", recipe, subject, seed)' in source, (
        "defining control recipes in the unit runner is insufficient if the queue never launches them"
    )
    assert "MODEL_SEARCH_ON_FROZEN_CONTROL" in source
    assert "--baseline-recipe" in source


def test_model_search_runner_refuses_an_unfrozen_control() -> None:
    source = SEARCH_SCRIPT.read_text(encoding="utf-8")
    assert "learned-model search requires the already frozen --baseline-recipe" in source


def test_optimised_h1_refuses_to_run_when_the_baseline_never_trained() -> None:
    source = OPTIMIZED_H1_SCRIPT.read_text(encoding="utf-8")
    assert "baseline_selection_status" in source, (
        "the formal runner must fail closed when the control arm has no trainable recipe"
    )


# ---------------------------------------------------------------- 3. modulation interface


def test_modulation_interface_passes_gradient_to_the_state_on_the_first_step() -> None:
    adapter = DynamicStepAdapter(
        StepwiseAdapterConfig(context_dim=6, rank=4, output_init_std=1e-2),
        hidden_width=8,
        n_contacts=5,
    )
    context = torch.randn(3, 6, requires_grad=True)
    outputs = adapter(context, torch.rand(3))
    tensors = [v for v in (outputs if isinstance(outputs, (tuple, list)) else [outputs]) if torch.is_tensor(v)]
    loss = sum((v * torch.randn_like(v)).sum() for v in tensors)
    loss.backward()
    assert float(context.grad.abs().max()) > 0.0, (
        "instrument check 6: the state input must receive gradient on the first optimiser step"
    )
    assert float(adapter.down.weight.grad.abs().max()) > 0.0


def test_modulation_interface_default_reproduces_the_frozen_decoder_exactly() -> None:
    adapter = DynamicStepAdapter(StepwiseAdapterConfig(context_dim=6, rank=4), hidden_width=8, n_contacts=5)
    outputs = adapter(torch.randn(3, 6), torch.rand(3))
    tensors = [v for v in (outputs if isinstance(outputs, (tuple, list)) else [outputs]) if torch.is_tensor(v)]
    assert all(float(v.abs().max()) == 0.0 for v in tensors), "zero-modulation parity must stay available"


def test_every_v037_adapter_call_site_sets_the_initialisation_explicitly() -> None:
    source = Path(inspect.getfile(h2a_module)).read_text(encoding="utf-8")
    calls = re.findall(r"StepwiseAdapterConfig\((?:[^()]|\([^()]*\))*\)", source)
    assert calls, "no adapter construction found in the v037 H2a module"
    for call in calls:
        assert "output_init_std" in call, (
            f"call site inherits the v035 zero-initialisation default: {call}"
        )


def test_h2a_config_gives_the_state_arm_its_own_patience_and_warmup() -> None:
    names = {field.name for field in fields(H2ATrainConfig)}
    assert {"state_modulation_init_std", "patience_epochs_state", "warmup_epochs_state"} <= names
    config = H2ATrainConfig()
    assert config.patience_epochs_state > config.patience_epochs, (
        "the arm that must climb out of the origin needs more patience than the arms that do not; "
        "12 epochs stopped it at zero in three of four pilot patients"
    )
    assert config.warmup_epochs_state > 0


def test_h2a_state_arm_starts_in_exact_parity_with_the_frozen_decoder() -> None:
    # A non-zero start would perturb the frozen decoder, so the state arm would
    # have to repay that perturbation before it could show any gain -- the
    # mirror image of the bug being fixed.  Parity keeps the nested contrast
    # one-sided: the state arm can win or tie, never start behind.
    assert H2ATrainConfig().state_modulation_init_std == 0.0


def test_modulation_path_is_live_after_one_optimiser_step() -> None:
    adapter = DynamicStepAdapter(
        StepwiseAdapterConfig(context_dim=6, rank=4, output_init_std=0.0),
        hidden_width=8,
        n_contacts=5,
    )
    optimiser = torch.optim.SGD(adapter.parameters(), lr=0.1)

    def step() -> float:
        context = torch.randn(3, 6, requires_grad=True)
        outputs = adapter(context, torch.rand(3))
        tensors = [v for v in outputs if torch.is_tensor(v)]
        optimiser.zero_grad(set_to_none=True)
        loss = sum((v * torch.randn_like(v)).sum() for v in tensors)
        loss.backward()
        optimiser.step()
        return float(context.grad.abs().max())

    assert step() == 0.0, "zero output maps mean the first step cannot reach the state; that is expected"
    assert step() > 0.0, (
        "after one step the output maps are non-zero, so the state input must receive gradient; "
        "a permanently blocked path would make any early stop look like 'state has no effect'"
    )


# ---------------------------------------------------------------- 4. independent windows


def test_independent_window_count_ignores_overlapping_anchors() -> None:
    # 24 h of five-minute anchors; an 8 h horizon admits three disjoint windows.
    anchors = np.arange(0.0, 24 * 3600.0, 300.0)
    assert independent_window_count(anchors, 8 * 3600.0) == 3
    assert independent_window_count(anchors, 2 * 3600.0) == 12
    assert independent_window_count(anchors, 300.0) == anchors.size


def test_independent_window_count_handles_empty_and_unsorted_input() -> None:
    assert independent_window_count(np.asarray([]), 3600.0) == 0
    assert independent_window_count(np.asarray([7200.0, 0.0, 3600.0]), 3600.0) == 3


def test_independent_window_count_respects_recording_segments() -> None:
    # Two sessions two days apart must not be bridged into one long window.
    anchors = np.concatenate([np.arange(0.0, 3600.0, 300.0), np.arange(172800.0, 176400.0, 300.0)])
    segment = np.concatenate([np.zeros(12, dtype=int), np.ones(12, dtype=int)])
    assert independent_window_count(anchors, 3600.0, segment=segment) == 2


# ---------------------------------------------------------------- 5. code provenance


def test_h1_card_records_the_code_version_that_produced_it() -> None:
    source = inspect.getsource(__import__(
        "src.topic5_group_event_state.v037.h1_train", fromlist=["train_h1_subject"]
    ).train_h1_subject)
    assert "code_provenance" in source, (
        "a card with no code version let a stale result be summarised by newer code"
    )
    assert "independent_windows_by_horizon" in source
    assert "B_mark_gain_over_B_rate" in source


def test_primary_human_memory_bank_is_shorter_than_the_holdout_span() -> None:
    from src.topic5_group_event_state.v037.h1_train import H1TrainConfig

    assert max(H1TrainConfig().taus_seconds) == 16 * 3600.0


def test_dual_model_honours_the_selected_transparent_baseline_recipe() -> None:
    from src.topic5_group_event_state.v037.h1_dual_train import NestedDualReadout

    readout = NestedDualReadout(
        q_dim=3, bmark_burden_dim=2, bmark_grammar_dim=2,
        background_current_dim=2, background_state_dim=2,
        event_burden_dim=2, event_grammar_dim=2,
        widths={name: 1 for name in NestedDualReadout.ENDPOINTS}, n_horizon=1,
        bmark_readout_init_std=1e-2,
    )
    assert max(float(layer.weight.abs().max()) for layer in readout.bmark.values()) > 0.0


def test_final_report_separates_path_exploration_from_origin_selection() -> None:
    from scripts.finalize_group_event_state_v037 import _training_adequacy

    explored_null = {
        "stages": {
            "B_mark": {
                "selected_at_init": True,
                "adapter_moved_during_training": True,
                "training_budget_exhausted": False,
            }
        }
    }
    dead_path = {
        "stages": {
            "B_mark": {
                "selected_at_init": True,
                "adapter_moved_during_training": False,
                "training_budget_exhausted": False,
            }
        }
    }
    assert _training_adequacy([explored_null])["interpretable"] is True
    assert _training_adequacy([dead_path])["interpretable"] is False


def test_final_report_recognises_joint_h2a_update_provenance() -> None:
    from scripts.finalize_group_event_state_v037 import _h2a_joint

    card = {
        "subject": "synthetic",
        "producer_updated_by": "interictal H2a grammar likelihood only",
        "h2a_primary_contrasts": {},
        "h1_mandatory_reevaluation": {"primary_contrasts": {}},
    }
    row = _h2a_joint([card])["rows"][0]
    assert row["producer_updated_only_by_interictal_h2a"] is True


# ---------------------------------------------------------------- 6. null arms must be floored


def test_fitted_wrong_time_null_is_floored_at_the_no_edge_model() -> None:
    # A same-capacity placebo picks its own ridge on INNER and can generalise
    # worse than having no edge at all.  When it does, "real beats placebo"
    # measures the placebo's overfitting, not the real edge's skill, so the
    # reportable contrast floors the null at the no-edge model.
    from src.topic5_group_event_state.v037.h3_persistent import _floored_gain

    harmful = {"status": "ESTIMATED", "placebo": 3.7325, "no_edge": 3.0694,
               "placebo_over_no_edge": -0.6631, "placebo_worse_than_no_edge": True,
               "floored": 3.0694}
    real = 2.9800
    assert abs(_floored_gain(harmful, real) - (3.0694 - 2.9800)) < 1e-9
    raw = harmful["placebo"] - real
    assert raw > 0.7 and _floored_gain(harmful, real) < 0.1, (
        "the raw contrast must not be reportable when the null is worse than no edge"
    )

    benign = {"status": "ESTIMATED", "placebo": 1.0021, "no_edge": 1.0281,
              "placebo_over_no_edge": 0.0260, "placebo_worse_than_no_edge": False,
              "floored": 1.0021}
    assert abs(_floored_gain(benign, 1.0003) - (1.0021 - 1.0003)) < 1e-9

    assert _floored_gain({"status": "NOT_ESTIMABLE"}, 1.0) is None


def test_h3_persistent_card_exposes_the_no_edge_floor() -> None:
    source = (REPO_ROOT / "src/topic5_group_event_state/v037/h3_persistent.py").read_text(
        encoding="utf-8"
    )
    assert "no_edge_floor" in source
    assert "placebo_worse_than_no_edge" in source
    assert "persistent_count_real_over_floored_wrong_time_background" in source
    assert "NOT_COMPARABLE_DIFFERENT_SUPPORT" in source, (
        "flooring is only valid when the null and the no-edge model share the evaluation support"
    )


def test_finaliser_reports_wrong_time_control_quality() -> None:
    source = (REPO_ROOT / "scripts/finalize_group_event_state_v037.py").read_text(encoding="utf-8")
    assert "wrong_time_control_quality" in source
    assert "shifted_arm_worse_than_constant" in source, (
        "a shifted arm worse than the constant arm is misleading, not null, and inflates the gain"
    )
    assert "median_donor_offset_hours" in source, (
        "the half-roll donor offset is an accident of coverage fragmentation, so it must be "
        "reported before any cross-patient correct-time statement"
    )
    assert "clock_matched_control_available" in source
    assert "h3_persistent_floored_null" in source


def test_donor_offset_is_measured_inside_each_coverage_segment() -> None:
    source = (REPO_ROOT / "scripts/finalize_group_event_state_v037.py").read_text(encoding="utf-8")
    block = source.split("def _donor_offset_hours", 1)[1].split("\ndef ", 1)[0]
    assert "segment" in block and "np.unique" in block, (
        "rolling the whole held-out block reports an offset the donor rule never used"
    )
    assert "SEGMENT_NOT_STORED" in block, "the offset must be marked unavailable, never guessed"
