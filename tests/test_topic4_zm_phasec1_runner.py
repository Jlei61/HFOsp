"""Runner-only tests for Phase-C1 immutable routing and resource guards."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

import scripts.run_topic4_zm_phasec_cell as CELL
import scripts.run_topic4_zm_phasec1_parallel as COORD
import scripts.analyze_topic4_zm_phasec1_gain as GAIN


def test_c1_worker_swap_skips_valid_published_terminal_artifact(
    tmp_path, monkeypatch
):
    output = tmp_path / "terminal.json"
    output.write_text("{}")

    class Process:
        pid = 31

    captured = {}

    def validator(path, task, *, producer_locks):
        assert producer_locks == {"runner.py": "a" * 64}
        return True, "valid", {}

    def fake_snapshot(processes, *, published_terminal_pids=()):
        captured["pids"] = [process.pid for process in processes]
        captured["published"] = set(published_terminal_pids)
        return {"worker_swap_total_kb": 0}

    monkeypatch.setattr(
        COORD.PRES, "worker_process_swap_snapshot", fake_snapshot
    )
    result = COORD.worker_swap_snapshot(
        [{"output": str(output), "process": Process()}],
        producer_locks={"runner.py": "a" * 64},
        validator=validator,
    )
    assert result == {"worker_swap_total_kb": 0}
    assert captured == {"pids": [31], "published": {31}}


def _self_hashed(payload):
    out = dict(payload)
    out["manifest_sha256"] = CELL._object_sha(payload)
    return out


def _cell(cell_id="primary__rising__bounded_mid", status="valid"):
    return {
        "cell_id": cell_id,
        "tier": "primary_convex",
        "array_row": 0,
        "status": status,
        "reasons": [] if status == "valid" else ["z_hard_bounds"],
        "trajectory_id": "rising",
        "path_index": 2,
        "path_direction": "forward",
        "state_sha256": "s" * 64,
    }


def test_restore_coordinate_changes_only_e_slow_fields():
    base = {
        "slow.z": np.r_[np.ones(4), np.ones(2)],
        "slow.m": np.r_[np.zeros(4), np.zeros(2)],
        "slow.S_G": np.asarray(0.1),
        "V": np.arange(6.0),
        "rng_state": {"state": {"state": "1", "inc": "3"}},
    }
    coordinate = {
        "z": np.asarray([0.2, 0.3, 0.4, 0.5]),
        "m": np.asarray([1.0, 2.0, 3.0, 4.0]),
        "S_G": 0.7,
    }
    ctx = {"S": {"NE": 4}}
    out = CELL._restore_coordinate(base, coordinate, ctx)
    np.testing.assert_allclose(out["slow.z"], [0.2, 0.3, 0.4, 0.5, 1, 1])
    np.testing.assert_allclose(out["slow.m"], [1, 2, 3, 4, 0, 0])
    assert float(out["slow.S_G"]) == 0.7
    np.testing.assert_array_equal(out["V"], base["V"])
    assert not np.shares_memory(out["V"], base["V"])

    bad = dict(base)
    bad["slow.z"] = np.r_[np.ones(4), [0.9, 1.0]]
    with pytest.raises(RuntimeError, match="I-cell Z/M constants"):
        CELL._restore_coordinate(bad, coordinate, ctx)


def test_coordinate_contract_is_authorized_only_by_final_forward_lock(
    tmp_path, monkeypatch
):
    producer = tmp_path / "builder.py"
    producer.write_text("# fixed\n")
    semantic_body = {
        "schema": "coordinate",
        "parent_phasec_input_manifest_sha256": "bootstrap",
        "parent_phasec_input_manifest_path": "phasec_input.json",
        "parent_phasec_input_manifest_file_sha256": "i" * 64,
        "producer_file_sha256": {
            "builder.py": CELL._sha(producer),
        },
        "seeds": {},
    }
    coordinate = _self_hashed({
        **semantic_body,
        "semantic_sha256": CELL._object_sha(semantic_body),
    })
    path = tmp_path / "coordinate.json"
    path.write_text(json.dumps(coordinate))
    final = {
        "provenance": {
            "phasec_input_manifest_path": "phasec_input.json",
            "phasec_input_manifest_file_sha256": "i" * 64,
            "phasec_input_manifest_manifest_sha256": "bootstrap",
        },
        "c1": {
            "coordinate_manifests": {
                "dt": {
                    "path": "coordinate.json",
                    "file_sha256": CELL._sha(path),
                    "manifest_sha256": coordinate["manifest_sha256"],
                    "semantic_sha256": coordinate["semantic_sha256"],
                },
            }
        }
    }
    monkeypatch.setattr(CELL, "ROOT", str(tmp_path))
    assert CELL._coordinate_contract(
        final, str(path)
    )["manifest_sha256"] == coordinate["manifest_sha256"]
    final["c1"]["coordinate_manifests"]["dt"]["file_sha256"] = "x" * 64
    with pytest.raises(RuntimeError, match="file SHA"):
        CELL._coordinate_contract(final, str(path))


def test_coordinate_contract_rejects_semantic_mutation_even_when_rehashed(
    tmp_path, monkeypatch
):
    producer = tmp_path / "builder.py"
    producer.write_text("# fixed\n")
    body = {
        "schema": "coordinate",
        "parent_phasec_input_manifest_sha256": "bootstrap",
        "parent_phasec_input_manifest_path": "phasec_input.json",
        "parent_phasec_input_manifest_file_sha256": "i" * 64,
        "producer_file_sha256": {
            "builder.py": CELL._sha(producer),
        },
        "seeds": {},
    }
    semantic_sha = CELL._object_sha(body)
    coordinate = _self_hashed({
        **body,
        "semantic_sha256": semantic_sha,
    })
    coordinate["seeds"] = {"1": {"mutated": True}}
    coordinate["manifest_sha256"] = CELL._object_sha({
        key: value for key, value in coordinate.items()
        if key != "manifest_sha256"
    })
    path = tmp_path / "coordinate.json"
    path.write_text(json.dumps(coordinate))
    final = {
        "provenance": {
            "phasec_input_manifest_path": "phasec_input.json",
            "phasec_input_manifest_file_sha256": "i" * 64,
            "phasec_input_manifest_manifest_sha256": "bootstrap",
        },
        "c1": {
            "coordinate_manifests": {
                "dt": {
                    "path": "coordinate.json",
                    "file_sha256": CELL._sha(path),
                    "manifest_sha256": coordinate["manifest_sha256"],
                    "semantic_sha256": semantic_sha,
                },
            },
        },
    }
    monkeypatch.setattr(CELL, "ROOT", str(tmp_path))
    with pytest.raises(RuntimeError, match="semantic SHA"):
        CELL._coordinate_contract(final, str(path))


def test_base_task_matrix_skips_invalid_physical_explicitly(monkeypatch):
    monkeypatch.setattr(COORD, "MANIFEST_PATH", Path("/tmp/phasec.json"))
    monkeypatch.setattr(
        COORD, "COORDINATE_MANIFEST_PATH", Path("/tmp/coordinate.json")
    )
    monkeypatch.setattr(COORD, "_sha", lambda path: "f" * 64)
    valid = _cell()
    invalid = _cell("primary__rising__bounded_late", "invalid_physical")
    coordinate = {
        "manifest_sha256": "c" * 64,
        "semantic_sha256": "s" * 64,
        "producer_file_sha256": {"builder.py": "b" * 64},
        "seeds": {
            "1": {
                "config_sha": "k" * 64,
                "npz_file_sha256": "n" * 64,
                "npz_semantic_sha256": "q" * 64,
                "cells": [valid, invalid],
            }
        },
    }
    rows, invalid_rows = COORD.base_tasks(
        {"manifest_sha256": "m" * 64}, coordinate
    )
    assert len(rows) == 2 * 3
    assert len(invalid_rows) == 1
    assert all(valid["cell_id"] in row["key"] for row in rows)
    assert all(invalid["cell_id"] not in row["key"] for row in rows)


def test_gain_tasks_come_only_from_write_once_trigger(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(COORD, "ROOT", tmp_path)
    monkeypatch.setattr(CELL, "ROOT", str(tmp_path))
    phasec_path = tmp_path / "phasec.json"
    coordinate_path = tmp_path / "coordinate.json"
    trigger_path = tmp_path / "trigger.json"
    for path in (phasec_path, coordinate_path, trigger_path):
        path.write_text("{}")
    monkeypatch.setattr(COORD, "MANIFEST_PATH", phasec_path)
    monkeypatch.setattr(COORD, "COORDINATE_MANIFEST_PATH", coordinate_path)
    monkeypatch.setattr(COORD, "TRIGGER_MANIFEST_PATH", trigger_path)
    cell = _cell()
    cell["state_sha256"] = "a" * 64
    coordinate = {
        "manifest_sha256": "c" * 64,
        "semantic_sha256": "s" * 64,
        "producer_file_sha256": {"builder.py": "b" * 64},
        "seeds": {
            "1": {
                "config_sha": "k" * 64,
                "npz_file_sha256": "n" * 64,
                "npz_semantic_sha256": "q" * 64,
                "cells": [cell],
            }
        },
    }
    base_refs = []
    for phase in COORD.PHASES:
        for noise in COORD.NOISES:
            path = tmp_path / f"base/{phase}/{noise}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
            base_refs.append({
                "part_path": str(path.relative_to(tmp_path)),
                "part_sha256": COORD._sha(path),
            })
    arms = []
    for phase in COORD.PHASES:
        for noise in COORD.NOISES:
            for delta in COORD.DELTAS:
                arms.append({
                    "path": CELL._c1_gain_relative_path(
                        "dt", 1, "primary_convex", cell["cell_id"],
                        phase, noise, delta,
                    )
                })
    trigger = {
        "manifest_sha256": "t" * 64,
        "producer_file_sha256": {"trigger.py": "u" * 64},
        "triggered_cells": [{
            "seed": 1,
            "tier": "primary_convex",
            "cell_id": cell["cell_id"],
            "slow_state_sha256": cell["state_sha256"],
            "triggering_base_parts": base_refs,
            "expected_carrier_gain_arms": arms,
        }],
    }
    rows = COORD.gain_tasks(
        {"manifest_sha256": "m" * 64}, coordinate, trigger
    )
    assert len(rows) == 30
    assert len({row["output"] for row in rows}) == 30
    trigger["triggered_cells"] = []
    assert COORD.gain_tasks(
        {"manifest_sha256": "m" * 64}, coordinate, trigger
    ) == []


def test_resource_cap_keeps_reserves_and_never_exceeds_twelve(monkeypatch):
    args = type("Args", (), {
        "reserve_cpus": 8,
        "reserve_gb": 96.0,
        "worker_rss_gb": 4.0,
        "max_workers": 99,
    })()
    monkeypatch.setattr(COORD, "mem_available_gb", lambda: 240.0)
    monkeypatch.setattr(COORD.os, "cpu_count", lambda: 64)
    assert COORD._resource_cap(args) == 12
    monkeypatch.setattr(COORD, "mem_available_gb", lambda: 100.0)
    assert COORD._resource_cap(args) == 0


def test_swap_guard_has_bounded_jitter_allowance(monkeypatch):
    baseline = 900_000
    monkeypatch.setattr(COORD, "swap_used_kb", lambda: baseline + 1024)
    assert COORD.swap_growth_exceeded(baseline, 64.0) is False
    monkeypatch.setattr(
        COORD, "swap_used_kb", lambda: baseline + 64 * 1024 + 1
    )
    assert COORD.swap_growth_exceeded(baseline, 64.0) is True


def test_semantic_slow_hash_is_dtype_sensitive():
    z64 = np.asarray([0.1, 0.2], np.float64)
    m64 = np.asarray([1.0, 2.0], np.float64)
    assert CELL._slow_state_sha(z64, m64, 0.5) != CELL._slow_state_sha(
        z64.astype(np.float32), m64.astype(np.float32), 0.5
    )


def test_runner_source_contains_no_duplicate_literal_dict_keys():
    for module in (CELL, COORD, GAIN):
        tree = ast.parse(Path(module.__file__).read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            literal = [
                key.value for key in node.keys
                if isinstance(key, ast.Constant)
                and isinstance(key.value, (str, int, float))
            ]
            assert len(literal) == len(set(literal)), (
                f"duplicate literal dict key in {module.__file__}: {literal}"
            )


def _gain_rows(gain_hz_per_mV):
    rows = {}
    baseline = 100.0
    for delta in GAIN.DELTAS:
        # Positive threshold offset suppresses rate.
        value = baseline - float(gain_hz_per_mV) * delta
        rows[delta] = {
            "status": "complete",
            "gain_plateau_gate_pass": True,
            "noise_bank_sha": "b" * 64,
            "core_rate_500ms_hz": [value, value],
        }
    return rows


def test_conditional_gain_curve_and_hierarchical_ratio_are_signed():
    curve = GAIN.gain_curve(_gain_rows(20.0))
    assert curve["status"] == "ok"
    np.testing.assert_allclose(curve["gain_hz_per_mV_blocks"], [20, 20])
    rows = []
    for phase in ("rising", "peak"):
        for index in range(3):
            rows.append({
                "phase": phase,
                "noise": f"noise{index}",
                "ratio_blocks": np.asarray([0.8, 0.9]),
                "ratio_point": 0.85,
            })
    interval = GAIN.gain_ratio_interval(rows, seed=1, n_boot=200)
    assert interval["lo"] >= 0.8
    assert interval["hi"] <= 0.9

    wrong = _gain_rows(20.0)
    wrong[0.10]["noise_bank_sha"] = "c" * 64
    with pytest.raises(RuntimeError, match="paired future-noise"):
        GAIN.gain_curve(wrong)


def test_conditional_gain_scientific_failure_is_indeterminate():
    rows = _gain_rows(20.0)
    rows[-0.10] = {
        **rows[-0.10],
        "status": "scientific_failure",
        "scientific_end_reason": "runaway",
    }
    out = GAIN.gain_curve(rows)
    assert out["status"] == "scientific_indeterminate"
    assert out["linearity_pass"] is False


def test_raw_conditional_gain_requires_full_locked_provenance():
    phasec_producers = {"runner.py": "1" * 64}
    coordinate_producers = {"builder.py": "2" * 64}
    trigger_producers = {"trigger.py": "3" * 64}
    coordinate_seed = {
        "coordinate_npz_file_sha256": "4" * 64,
        "coordinate_npz_semantic_sha256": "5" * 64,
    }
    trigger = {
        "manifest_sha256": "6" * 64,
        "phasec_manifest_sha256": "7" * 64,
        "phasec_manifest_file_sha256": "8" * 64,
        "phasec_producer_file_sha256": phasec_producers,
        "coordinate_manifest_sha256": "9" * 64,
        "coordinate_manifest_semantic_sha256": "a" * 64,
        "coordinate_manifest_file_sha256": "b" * 64,
        "coordinate_producer_file_sha256": coordinate_producers,
        "producer_file_sha256": trigger_producers,
    }
    arm = {
        "resolution": "dt",
        "seed": 1,
        "tier": "primary_convex",
        "cell_id": "cell",
        "trajectory_id": "rising",
        "path_index": 2,
        "path_direction": "forward",
        "phase": "rising",
        "noise": "noise_replay",
        "delta_mV": 0.05,
        "threshold_offset_mV": 0.05,
        "burn_in_ms": 500.0,
        "measure_ms": 1000.0,
        "slow_state_sha256": "c" * 64,
        "config_sha": "d" * 64,
        "fast_base_state_hash": "e" * 64,
        "state_file_sha256": "f" * 64,
        "noise_bank_sha": "0" * 64,
        "phasec_manifest_sha256": trigger["phasec_manifest_sha256"],
        "phasec_manifest_file_sha256": trigger[
            "phasec_manifest_file_sha256"
        ],
        "coordinate_manifest_sha256": trigger[
            "coordinate_manifest_sha256"
        ],
        "coordinate_manifest_semantic_sha256": trigger[
            "coordinate_manifest_semantic_sha256"
        ],
        "coordinate_manifest_file_sha256": trigger[
            "coordinate_manifest_file_sha256"
        ],
        **coordinate_seed,
    }
    part = {
        "schema": CELL.C1_GAIN_PART_SCHEMA,
        "trigger_manifest_sha256": trigger["manifest_sha256"],
        "trigger_manifest_file_sha256": "1" * 64,
        **arm,
        "runtime_provenance": {
            "manifest_sha256": trigger["phasec_manifest_sha256"],
            "manifest_file_sha256": trigger[
                "phasec_manifest_file_sha256"
            ],
            "producer_sha256": phasec_producers,
            "state_file_sha256": arm["state_file_sha256"],
            "noise_bank_sha": arm["noise_bank_sha"],
            "coordinate_manifest_sha256": trigger[
                "coordinate_manifest_sha256"
            ],
            "coordinate_manifest_semantic_sha256": trigger[
                "coordinate_manifest_semantic_sha256"
            ],
            "coordinate_manifest_file_sha256": trigger[
                "coordinate_manifest_file_sha256"
            ],
            **coordinate_seed,
            "coordinate_producer_sha256": coordinate_producers,
            "trigger_manifest_sha256": trigger["manifest_sha256"],
            "trigger_manifest_file_sha256": "1" * 64,
            "trigger_producer_sha256": trigger_producers,
        },
    }
    GAIN._validate_carrier_part(
        part,
        arm,
        trigger,
        trigger_file_sha256="1" * 64,
        coordinate_seed_provenance=coordinate_seed,
    )
    part["fast_base_state_hash"] = "x" * 64
    with pytest.raises(RuntimeError, match="raw provenance mismatch"):
        GAIN._validate_carrier_part(
            part,
            arm,
            trigger,
            trigger_file_sha256="1" * 64,
            coordinate_seed_provenance=coordinate_seed,
        )
    part["fast_base_state_hash"] = arm["fast_base_state_hash"]
    part["runtime_provenance"]["coordinate_producer_sha256"] = {
        "builder.py": "x" * 64
    }
    with pytest.raises(RuntimeError, match="runtime provenance mismatch"):
        GAIN._validate_carrier_part(
            part,
            arm,
            trigger,
            trigger_file_sha256="1" * 64,
            coordinate_seed_provenance=coordinate_seed,
        )
