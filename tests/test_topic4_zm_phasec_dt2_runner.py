import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.lock_topic4_zm_phasec1_dt2_confirmation as L
import scripts.run_topic4_zm_phasec0_parallel as C0
import scripts.run_topic4_zm_phasec_dt2_parallel as D
import scripts.run_topic4_zm_phasec_cell as CELL


def _panel():
    return {
        "panel_sha256": "p" * 64,
        "analysis_panel_E_ids": [0],
        "pairwise_panel_E_ids": [0],
    }


def _family(seed, phase, config):
    step = 1000 + seed + (0 if phase == "rising" else 10)
    state = {
        "state_hash": f"h{seed}{phase}",
        "file_sha256": f"f{seed}{phase}",
        "path": f"state/{seed}/{phase}.npz",
        "t_step": step,
    }
    return {
        "state": state,
        "noise_banks": [
            {
                "replicate": noise,
                "bank_sha": f"b{seed}{phase}{noise}",
                "start_step": step,
            }
            for noise in C0.NOISES
        ],
    }


def _manifest():
    rows = {}
    for seed in (1, 3, 4):
        panel = _panel()
        native = {
            "canonical_config_sha": f"native{seed}",
            "fixed_panels": panel,
            "c0_carrier_states": {
                phase: _family(seed, phase, f"native{seed}")
                for phase in ("rising", "peak")
            },
            "c0_pre_entry_gain_control": _family(
                seed, "pre", f"native{seed}"
            ),
            "resolution_confirmations": {},
        }
        if seed in (1, 3):
            native["resolution_confirmations"]["dt2"] = {
                "resolution": "dt2",
                "dt_ms": 0.05,
                "config_sha": f"dt2{seed}",
                "parent_config_sha": f"native{seed}",
                "anchor_path": f"anchor/{seed}.json",
                "anchor_file_sha256": f"a{seed}",
                "config_path": f"config/{seed}.json",
                "config_file_sha256": f"c{seed}",
                "panel_selection_config_sha": f"native{seed}",
                "panel_selection_resolution": "parent_native_dt",
                "fixed_panels": panel,
                "c0_carrier_states": {
                    phase: _family(seed, phase, f"dt2{seed}")
                    for phase in ("rising", "peak")
                },
                "c0_pre_entry_gain_control": _family(
                    seed, "pre", f"dt2{seed}"
                ),
            }
        rows[str(seed)] = native
    return {"manifest_sha256": "m" * 64, "per_seed": rows}


def test_c0_dt2_matrix_is_complete_and_resolution_local():
    manifest = _manifest()
    rows = C0.tasks(
        ("identity", "gain"), manifest, resolution="dt2", seeds=(1, 3)
    )
    assert len(rows) == 102  # 12 identity + 2*3 states*3 noises*5 gain arms
    assert {row["expected"]["resolution"] for row in rows} == {"dt2"}
    assert all("/dt2/" in row["output"] for row in rows)
    assert all(
        row["cmd"][row["cmd"].index("--resolution") + 1] == "dt2"
        for row in rows
    )
    assert {
        row["expected"]["config_sha"] for row in rows
    } == {"dt21", "dt23"}


def test_dt2_reuses_native_panel_and_rejects_wrong_parent():
    manifest = _manifest()
    native, dt2 = CELL._resolution_seed_row(manifest, 1, "dt2")
    assert dt2["fixed_panels"] == native["fixed_panels"]
    manifest["per_seed"]["1"]["resolution_confirmations"]["dt2"][
        "parent_config_sha"
    ] = "wrong"
    with pytest.raises(RuntimeError, match="lineage"):
        CELL._resolution_seed_row(manifest, 1, "dt2")
    with pytest.raises(RuntimeError, match="lacks independent"):
        CELL._resolution_seed_row(_manifest(), 4, "dt2")


def _native_summary():
    windows = [
        {
            "phenotype": "periodic_non_tonic_carrier",
            "direction": "rising",
            "cells": ["cell_a", "cell_b"],
        }
    ]
    seed_results = {
        str(seed): {"windows": windows if seed in (1, 3) else []}
        for seed in (1, 3, 4)
    }
    adjudication = {
        "status": "local_maturation_window",
        "candidates": [{
            "phenotype": "periodic_non_tonic_carrier",
            "direction": "rising",
        }],
        "seed_results": seed_results,
    }
    return {
        "schema": "zm_phasec1_summary_v1_2026-07-28",
        "resolution": "dt",
        "phasec_manifest_sha256": "m" * 64,
        "phasec_manifest_file_sha256": None,
        "coordinate_manifest_sha256": "2" * 64,
        "coordinate_manifest_file_sha256": "1" * 64,
        "primary_adjudication": adjudication,
        "secondary_shell_adjudication": {
            "status": "no_window", "candidates": [], "seed_results": {}
        },
    }


def _coordinate():
    seeds = {}
    for seed in (1, 3):
        cells = []
        for index, cell_id in enumerate(("cell_a", "cell_b")):
            cells.append({
                "tier": "primary_convex",
                "cell_id": cell_id,
                "status": "valid",
                "trajectory_id": "rising",
                "path_index": index,
                "path_direction": "rising",
                "state_sha256": f"slow{seed}{cell_id}",
            })
        seeds[str(seed)] = {
            "npz_file_sha256": f"npzf{seed}",
            "npz_semantic_sha256": f"npzs{seed}",
            "cells": cells,
        }
    return {
        "manifest_sha256": "d" * 64,
        "semantic_sha256": "s" * 64,
        "producer_file_sha256": {"builder.py": "b" * 64},
        "resolution": "dt2",
        "seeds": seeds,
    }


def test_dt2_confirmation_lock_selects_only_homologous_native_window(
    tmp_path, monkeypatch
):
    manifest = _manifest()
    manifest.update({
        "production_authorized": True,
        "c1": {
            "coordinate_manifests": {
                "dt": {
                    "path": "dt.json", "file_sha256": "1" * 64,
                    "manifest_sha256": "2" * 64,
                    "semantic_sha256": "3" * 64,
                },
                "dt2": {
                    "path": "dt2.json", "file_sha256": "4" * 64,
                    "manifest_sha256": "d" * 64,
                    "semantic_sha256": "s" * 64,
                },
            }
        },
    })
    phasec = tmp_path / "phasec.json"
    summary_path = tmp_path / "native.json"
    phasec.write_text(json.dumps(manifest))
    summary = _native_summary()
    summary["gain_trigger_manifest_sha256"] = "g" * 64
    summary["phasec_manifest_file_sha256"] = L._sha(phasec)
    summary_path.write_text(json.dumps(summary))
    trigger_path = tmp_path / "gain-trigger.json"
    trigger_path.write_text("{}")
    monkeypatch.setattr(L.PCC, "validate_manifest", lambda _value: None)
    monkeypatch.setattr(
        L,
        "_validate_gain_trigger",
        lambda *_args: {"manifest_sha256": "g" * 64},
    )
    monkeypatch.setattr(L, "_dt2_coordinate", lambda _value: (
        _coordinate(), manifest["c1"]["coordinate_manifests"]["dt2"]
    ))
    monkeypatch.setattr(L, "_rel", lambda value: str(value))
    payload = L.build_payload(
        phasec_path=phasec,
        native_summary_path=summary_path,
        gain_trigger_path=trigger_path,
    )
    assert payload["selection_is_closed"] is True
    assert len(payload["selected_cells"]) == 4
    assert len(payload["expected_base_arms"]) == 24
    assert {row["seed"] for row in payload["expected_base_arms"]} == {1, 3}
    assert all(row["resolution"] == "dt2" for row in payload["expected_base_arms"])
    assert all(
        "coordinate_npz_file_sha256" in row
        and "coordinate_npz_semantic_sha256" in row
        and row["burn_in_ms"] == 500.0
        and row["measure_ms"] == 8000.0
        for row in payload["expected_base_arms"]
    )
    assert payload["gain_trigger_manifest"]["manifest_sha256"] == "g" * 64


def _write_closed_gain_trigger(tmp_path, manifest, phasec_path):
    coordinate_body = {
        "resolution": "dt",
        "semantic_sha256": "3" * 64,
        "producer_file_sha256": {"coordinate.py": "c" * 64},
    }
    coordinate = {
        **coordinate_body,
        "manifest_sha256": L._object_sha(coordinate_body),
    }
    coordinate_path = tmp_path / "coordinate-dt.json"
    coordinate_path.write_text(json.dumps(coordinate))
    manifest["c1"]["coordinate_manifests"]["dt"] = {
        "path": str(coordinate_path),
        "file_sha256": L._sha(coordinate_path),
        "manifest_sha256": coordinate["manifest_sha256"],
        "semantic_sha256": coordinate["semantic_sha256"],
    }
    manifest["provenance"] = {
        "producer_file_sha256": {"phasec.py": "p" * 64}
    }
    phasec_path.write_text(json.dumps(manifest))
    trigger_body = {
        "schema": L.GAIN_TRIGGER_SCHEMA,
        "selection_is_closed": True,
        "resolution": "dt",
        "phasec_manifest_sha256": manifest["manifest_sha256"],
        "phasec_manifest_file_sha256": L._sha(phasec_path),
        "coordinate_manifest_sha256": coordinate["manifest_sha256"],
        "coordinate_manifest_semantic_sha256": coordinate["semantic_sha256"],
        "coordinate_manifest_file_sha256": L._sha(coordinate_path),
        "phasec_producer_file_sha256": manifest["provenance"][
            "producer_file_sha256"
        ],
        "coordinate_producer_file_sha256": coordinate[
            "producer_file_sha256"
        ],
        "producer_file_sha256": manifest["provenance"][
            "producer_file_sha256"
        ],
    }
    trigger = {
        **trigger_body,
        "manifest_sha256": L._object_sha(trigger_body),
    }
    trigger_path = tmp_path / "gain-trigger.json"
    trigger_path.write_text(json.dumps(trigger))
    return trigger_path, trigger


def test_dt2_confirmation_requires_closed_parent_matched_gain_trigger(
    tmp_path,
):
    manifest = _manifest()
    manifest.update({
        "production_authorized": True,
        "c1": {"coordinate_manifests": {"dt": {}, "dt2": {}}},
    })
    phasec = tmp_path / "phasec.json"
    trigger_path, trigger = _write_closed_gain_trigger(
        tmp_path, manifest, phasec
    )
    assert L._validate_gain_trigger(
        trigger_path, manifest, phasec
    )["manifest_sha256"] == trigger["manifest_sha256"]

    unclosed = dict(trigger)
    unclosed["selection_is_closed"] = False
    unclosed["manifest_sha256"] = L._object_sha({
        key: value for key, value in unclosed.items()
        if key != "manifest_sha256"
    })
    trigger_path.write_text(json.dumps(unclosed))
    with pytest.raises(RuntimeError, match="selection_is_closed"):
        L._validate_gain_trigger(trigger_path, manifest, phasec)

    trigger_path.unlink()
    with pytest.raises(RuntimeError, match="closed canonical"):
        L._validate_gain_trigger(trigger_path, manifest, phasec)


def test_dt2_confirmation_rejects_native_summary_from_other_gain_trigger(
    tmp_path, monkeypatch
):
    manifest = _manifest()
    manifest.update({
        "production_authorized": True,
        "c1": {
            "coordinate_manifests": {
                "dt": {
                    "path": "dt.json", "file_sha256": "1" * 64,
                    "manifest_sha256": "2" * 64,
                    "semantic_sha256": "3" * 64,
                },
                "dt2": {
                    "path": "dt2.json", "file_sha256": "4" * 64,
                    "manifest_sha256": "d" * 64,
                    "semantic_sha256": "s" * 64,
                },
            }
        },
    })
    phasec = tmp_path / "phasec.json"
    phasec.write_text(json.dumps(manifest))
    summary = _native_summary()
    summary["phasec_manifest_file_sha256"] = L._sha(phasec)
    summary["gain_trigger_manifest_sha256"] = "wrong"
    summary_path = tmp_path / "native.json"
    summary_path.write_text(json.dumps(summary))
    monkeypatch.setattr(L.PCC, "validate_manifest", lambda _value: None)
    monkeypatch.setattr(
        L,
        "_validate_gain_trigger",
        lambda *_args: {"manifest_sha256": "g" * 64},
    )
    with pytest.raises(RuntimeError, match="does not match"):
        L.build_payload(
            phasec_path=phasec,
            native_summary_path=summary_path,
            gain_trigger_path=tmp_path / "trigger.json",
        )


def test_dt2_confirmation_lock_fails_without_two_independent_supporting_seeds():
    summary = _native_summary()
    summary["primary_adjudication"]["seed_results"]["3"]["windows"] = []
    with pytest.raises(RuntimeError, match="seeds 1 and 3"):
        L._native_positive_windows(summary)


def test_dt2_c1_tasks_are_exact_closed_selection(monkeypatch):
    manifest = {
        "c1": {
            "coordinate_manifests": {
                "dt2": {"path": "dt2_coordinate.json"}
            }
        }
    }
    arm = {
        "schema": CELL.C1_BASE_PART_SCHEMA,
        "phasec_manifest_sha256": "m",
        "phasec_manifest_file_sha256": "mf",
        "coordinate_manifest_sha256": "c",
        "coordinate_manifest_semantic_sha256": "cs",
        "coordinate_manifest_file_sha256": "cf",
        "coordinate_npz_file_sha256": "nf",
        "coordinate_npz_semantic_sha256": "ns",
        "seed": 1,
        "tier": "primary_convex",
        "cell_id": "cell_a",
        "trajectory_id": "rising",
        "path_index": 0,
        "path_direction": "rising",
        "phase": "rising",
        "noise": "noise_replay",
        "resolution": "dt2",
        "slow_state_sha256": "slow",
        "config_sha": "cfg",
        "fast_base_state_hash": "fast",
        "state_file_sha256": "state",
        "noise_bank_sha": "noise",
        "burn_in_ms": 500.0,
        "measure_ms": 8000.0,
        "path": CELL._c1_base_relative_path(
            "dt2", 1, "primary_convex", "cell_a",
            "rising", "noise_replay",
        ),
    }
    selection = {
        "manifest_sha256": "sel",
        "coordinate_producer_file_sha256": {"builder.py": "b" * 64},
        "expected_base_arms": [arm],
    }
    monkeypatch.setattr(D.C1, "_sha", lambda _path: "selfile")
    rows = D.c1_tasks(manifest, selection)
    assert len(rows) == 1
    assert rows[0]["expected"]["dt2_confirmation_manifest_sha256"] == "sel"
    assert "--resolution" in rows[0]["cmd"]
    assert rows[0]["cmd"][rows[0]["cmd"].index("--resolution") + 1] == "dt2"
    selection["expected_base_arms"].append(dict(arm))
    with pytest.raises(RuntimeError, match="duplicate"):
        D.c1_tasks(manifest, selection)


def test_c1_observables_lock_readout_kernel_width():
    source = Path(CELL.__file__).read_text()
    assert '"readout_kernel_width_mm"' in source
    assert 'float(ctx["S"]["p"].Rr)' in source
