#!/usr/bin/env python3
"""Close FCXR-LC6A from frozen artifacts without changing scientific labels."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "src/snn_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import aggregate_topic4_fcxr_lc6a_phenotypes as AGG  # noqa: E402
import run_topic4_fcxr_lc6a_natural_trajectory as NAT  # noqa: E402
from src.topic4_fcxr_lc6_phenotype import (  # noqa: E402
    front_readout_degeneracy, q_matched_control_set,
)


OUT = NAT.OUT
MANIFEST = ROOT / "config/topic4_fcxr_lc6a_patient_axis_surround.json"
PHENOTYPE = OUT / "phenotype_map.json"
GRAPH = OUT / "graph_audit.json"
TWO_HOP = OUT / "two_hop_kernel_audit.json"
FUNCTIONAL = OUT / "impulse_response_audit.json"
GAINS = OUT / "gain_forks.json"
CONFIRMATION = OUT / "confirmation_summary.json"
LC5_AUTHORIZATION = OUT / "lc5_to_lc6a_authorization.json"
RUN_MANIFEST = OUT / "run_manifest.json"
STATUS = OUT / "STATUS.md"
RESOURCE_LOG = OUT / "resource_log.jsonl"
DONE = OUT / "DONE_LC6A_COMPLETE.json"
ARCHIVE = ROOT / "docs/archive/topic4/fcxr_lc6a_patient_axis_surround_no_carrier_2026-08-15.md"
FIGURES = OUT / "figures"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"required LC6A artifact missing: {path}")
    return json.loads(path.read_text())


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(value)
    os.replace(tmp, path)


def _write_json(path: Path, payload: dict) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _git(*args: str) -> str:
    return subprocess.check_output(
        ("git", *args), cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
    ).strip()


def _source_hashes() -> dict[str, str]:
    paths = [
        MANIFEST,
        ROOT / "config/topic4_fcxr_lc6a_functional_probe_prelock.json",
        ROOT / "config/topic4_fcxr_lc6a_gain_fork_prelock.json",
        ROOT / "config/topic4_fcxr_lc6a_confirmation_prelock.json",
    ]
    paths.extend(sorted((ROOT / "src").glob("topic4_fcxr_lc6*.py")))
    paths.extend(sorted((ROOT / "scripts").glob("*topic4_fcxr_lc6a*.py")))
    return {
        str(path.relative_to(ROOT)): _sha(path)
        for path in paths if path.is_file()
    }


def _verify_engine_hashes(manifest: dict) -> dict[str, dict]:
    result = {}
    for relative, expected in manifest["blessed_engine_sha256"].items():
        actual = _sha(ROOT / relative)
        result[relative] = {
            "expected": expected, "actual": actual, "match": actual == expected,
        }
    if not all(row["match"] for row in result.values()):
        raise RuntimeError("blessed-engine hash drift at LC6A closeout")
    return result


def _verify_complete() -> tuple[dict, dict, dict, dict, dict, dict, dict]:
    manifest = _json(MANIFEST)
    graph = _json(GRAPH)
    two_hop = _json(TWO_HOP)
    functional = _json(FUNCTIONAL)
    phenotype = _json(PHENOTYPE)
    gains = _json(GAINS)
    confirmation = _json(CONFIRMATION)
    if graph.get("status") != "COMPLETE" or not graph.get("all_graphs_legal"):
        raise RuntimeError("graph family is not complete and legal")
    if two_hop.get("status") != "COMPLETE":
        raise RuntimeError("two-hop audit incomplete")
    if functional.get("status") != "COMPLETE":
        raise RuntimeError("functional characterization incomplete")
    if phenotype.get("status") != "COMPLETE" or len(phenotype.get("rows", [])) != 5:
        raise RuntimeError("fixed five-arm phenotype map incomplete")
    if gains.get("status") != "COMPLETE":
        raise RuntimeError("gain forks incomplete")
    if confirmation.get("status") != "COMPLETE_NOT_TRIGGERED":
        raise RuntimeError("negative LC6A must finish with confirmation not triggered")
    if phenotype.get("carrier_candidates"):
        raise RuntimeError("this closeout is only valid for the no-carrier outcome")
    if phenotype.get("headline_counts") != {"SATURATED_HIGH_STATE": 5}:
        raise RuntimeError("unexpected LC6A headline vector")
    active_running = sorted(
        str(path.relative_to(OUT)) for path in OUT.rglob("RUNNING_*.json")
        if "superseded" not in path.parts
    )
    active_failed = sorted(
        str(path.relative_to(OUT)) for path in OUT.rglob("FAILED_*.json")
        if "superseded" not in path.parts
    )
    if active_running or active_failed:
        raise RuntimeError(
            f"active LC6A sentinel remains: running={active_running}, failed={active_failed}"
        )
    return manifest, graph, two_hop, functional, phenotype, gains, confirmation


def _graph_rows(graph: dict, two_hop: dict) -> list[dict]:
    two_by = two_hop["audits"]
    rows = []
    for condition in ("C0", "C1", "Q1", "Q2", "Q3"):
        row = graph["audits"][condition]
        hop = two_by[condition]
        rows.append({
            "condition": condition,
            "construction_q": float(row["construction_q"]),
            "two_hop_q": float(hop["operator"]["q_parallel_two_hop"]),
            "surround_center_ratio": float(hop["operator"]["surround_center_ratio"]),
            "two_hop_latency_q95_ms": float(hop["latency"]["q95_ms"]),
            "graph_sha256": row["graph_sha256"],
        })
    return rows


SHEET_AREA_MM2 = 400.0
AXIAL_BIN_MM = 20.0 / 32.0
MICROSTATE_Q_TOLERANCE = 0.05


def _trajectory_rows(
    phenotype: dict, natural_summaries: dict[str, dict] | None = None,
) -> list[dict]:
    if natural_summaries is None:
        natural_summaries = {
            condition: _json(OUT / f"trajectories/{condition}/summary.json")
            for condition in ("C0", "C1", "Q1", "Q2", "Q3")
        }
    control = q_matched_control_set(
        {row["condition"]: row["construction_q"] for row in phenotype["rows"]},
        reference="C0", tolerance=MICROSTATE_Q_TOLERANCE,
    )
    rows = []
    for row in phenotype["rows"]:
        natural = natural_summaries[row["condition"]]
        degeneracy = front_readout_degeneracy(
            row["spatial_slow_flow"],
            sheet_area_mm2=SHEET_AREA_MM2, axial_bin_mm=AXIAL_BIN_MM,
        )
        rows.append({
            "condition": row["condition"],
            "construction_q": float(row["construction_q"]),
            "in_q_matched_control_band": row["condition"] in control["members"],
            "onset_s": float(row["effective_onset_ms"]) / 1000.0,
            # baseline_metrics is deliberately capped at the C0 baseline horizon for
            # across-arm equivalence.  The entry ledger belongs to the arm's own onset.
            "n_returning_pre_onset": int(natural["n_returning_pre_onset"]),
            "peak_global_rate_100ms_hz": float(row["global_rate_100ms_peak_hz"]),
            "peak_local_q99_hz": float(row["local_rate_q99_peak_hz"]),
            "active_area_mm2": float(row["spatial_slow_flow"]["max_active_area_mm2"]),
            "active_area_sheet_fraction": float(
                row["spatial_slow_flow"]["max_active_area_mm2"] / SHEET_AREA_MM2
            ),
            "front_readout_degeneracy": degeneracy,
            "boundedness_margin_mixed_units": float(row["boundedness"]["boundedness_margin"]),
            "baseline_tradeoff": bool(row["baseline_tradeoff"]["tradeoff"]),
            "headline": row["headline"],
        })
    return rows


def _escalation_after_entry(
    control_members: list[str], natural_summaries: dict[str, dict],
) -> dict:
    """Compare the post-entry ramp against the spread of the same-q realizations."""

    aligned = {}
    for condition, summary in natural_summaries.items():
        rate = list(summary["per_second_mean_rate_hz"])
        aligned[condition] = rate[int(float(summary["onset_ms"]) // 1000):]
    depth = min(len(values) for values in aligned.values())
    steps = []
    for step in range(depth):
        every = [aligned[condition][step] for condition in aligned]
        matched = [aligned[condition][step] for condition in control_members]
        steps.append({
            "seconds_after_entry": step,
            "all_arm_min_hz": min(every), "all_arm_max_hz": max(every),
            "q_matched_min_hz": min(matched), "q_matched_max_hz": max(matched),
            "all_arm_spread_ratio": max(every) / min(every),
            "q_matched_spread_ratio": max(matched) / min(matched),
        })
    return {
        "seconds_from_entry_to_arm_end": {
            condition: len(values) - 1 for condition, values in aligned.items()
        },
        "compared_depth_s": depth,
        "per_second": steps,
        "max_all_arm_spread_ratio": max(step["all_arm_spread_ratio"] for step in steps),
        "max_q_matched_spread_ratio": max(step["q_matched_spread_ratio"] for step in steps),
    }


def _fmt_table(rows: list[dict]) -> str:
    lines = [
        "| 条件 | q(two-hop) | 同 q 对照带内 | onset | onset 前 IED | "
        "全局峰值(100 ms) | baseline 代价 | 结局 |",
        "|---|---:|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['condition']} | {row['two_hop_q']:.3f} | "
            f"{'是' if row['in_q_matched_control_band'] else '否'} | "
            f"{row['onset_s']:.1f} s | {row['n_returning_pre_onset']} | "
            f"{row['peak_global_rate_100ms_hz']:.1f} Hz | "
            f"{'有' if row['baseline_tradeoff'] else '无'} | saturation |"
        )
    return "\n".join(lines)


def _write_readme() -> None:
    sections = [
        ("lc6a_graph_and_twohop", "连接几何审计：构图坐标与实际两跳 E→I→E 算子的宽度、周边/中心质量比、两跳延迟，以及垂轴宽度这个 confound 读数。图本身不包含自然发作，也不能证明 carrier。", "看两跳 q 与 surround/center 确实随条件增强，同时确认垂轴宽度几乎不动——改造是轴向的。"),
        ("lc6a_functional_response", "配对弱 E patch 的四个独立读数：直接足迹的空间范围、中心偏转的兴奋/抑制成分、晚窗是否有东西传出去、以及哪些探针停在“没有任何脉冲改变”的严格区间。", "晚窗偏转只出现在三个位置且与 reach 不成序，不能读成 Mexican-hat 周边信号。"),
        ("lc6a_trajectory_phenotypes", "五条固定自然轨迹的四个独立问题：进入时刻相对同 q 微状态对照带、进入后的升级曲线、三条注册上限里哪一条被突破、以及间期基线付出的代价。轴向 D-halo / front-speed 读数因退化已移除。", "只有 Q3 的进入落在同 q 对照带外；进入之后五臂的升级几乎重合，说明加宽周边只改了什么时候开始，没改升级本身。"),
        ("lc6a_gain_forks", "预注册选出的 C0/Q2 exact-state 弱输入 fork，报告 susceptibility、回落时间、rate 和面积偏离，并标出每个 fork 前 1 s 的实际全局放电率。响应性与 boundedness 分开判读。", "onset+2 s 的两个 fork 采到的是 34/52 Hz 的升级斜坡而非高态；只有 C0 的 onset+6 s 是高 rate 状态。非零响应不等于 carrier。"),
    ]
    text = []
    for stem, description, focus in sections:
        for suffix, qualifier in (("png", "位图版本"), ("pdf", "矢量版本")):
            path = FIGURES / f"{stem}.{suffix}"
            if not path.is_file():
                continue
            body = description if suffix == "png" else f"与 PNG 内容相同的{qualifier}。"
            text.append(
                f"### {path.name}\n\n{body}\n\n**关注点**：{focus}\n"
            )
    _atomic_text(FIGURES / "README.md", "\n".join(text).rstrip() + "\n")


def run() -> dict:
    manifest, graph, two_hop, functional, phenotype, gains, confirmation = _verify_complete()
    lc5 = _json(LC5_AUTHORIZATION)
    if not (
        lc5.get("status") == "COMPLETE"
        and lc5.get("authorize_lc6a_40k_dynamics") is True
        and lc5.get("lc5_outcome") == "ESCALATING_SATURATION"
        and lc5.get("checks", {}).get("classifier_replay_complete") is True
    ):
        raise RuntimeError("LC5 right-censor closeout does not authorize LC6A")
    lc5_summary = Path(lc5["lc5_summary"])
    if _sha(lc5_summary) != lc5["lc5_summary_sha256"]:
        raise RuntimeError("LC5 right-censor summary drift")
    engine = _verify_engine_hashes(manifest)
    graph_rows = _graph_rows(graph, two_hop)
    natural_summaries = {
        condition: _json(OUT / f"trajectories/{condition}/summary.json")
        for condition in ("C0", "C1", "Q1", "Q2", "Q3")
    }
    trajectory = _trajectory_rows(phenotype, natural_summaries)
    by_condition = {row["condition"]: row for row in graph_rows}
    for row in trajectory:
        row.update({
            "two_hop_q": by_condition[row["condition"]]["two_hop_q"],
            "surround_center_ratio": by_condition[row["condition"]]["surround_center_ratio"],
        })

    control = q_matched_control_set(
        {row["condition"]: row["construction_q"] for row in phenotype["rows"]},
        reference="C0", tolerance=MICROSTATE_Q_TOLERANCE,
    )
    band_onsets = sorted(
        row["onset_s"] for row in trajectory if row["in_q_matched_control_band"]
    )
    q_ladder = {
        "preregistered_targets": {
            entry["id"]: entry["q_marginal"] for entry in manifest["graph_family"]
        },
        "spec_expected_c0_q_marginal": 0.66,
        "measured_c0_q_marginal": trajectory[0]["construction_q"],
        "microstate_control_tolerance": MICROSTATE_Q_TOLERANCE,
        "q_matched_realizations": control["members"],
        "distinct_reach_rungs": control["reach_rungs"],
        "q_matched_onset_band_s": [band_onsets[0], band_onsets[-1]],
        "onset_outside_q_matched_band": [
            row["condition"] for row in trajectory
            if not row["in_q_matched_control_band"]
            and not band_onsets[0] <= row["onset_s"] <= band_onsets[-1]
        ],
        "note": (
            "C0 measured 0.934 rather than the 0.66 the spec expected, so the "
            "pre-registered Q1=1.00 rung landed inside C1's own same-q microstate "
            "tolerance.  The realized design is three q-matched realizations plus two "
            "reach rungs, not a five-rung ladder."
        ),
    }
    escalation = _escalation_after_entry(control["members"], natural_summaries)
    fork_context = [
        {
            "condition": row["condition"],
            "checkpoint": checkpoint["checkpoint"],
            "preceding_1s_global_rate_hz": float(
                checkpoint["preceding_1s"]["global_rate_hz"]
            ),
            "arm_peak_1s_global_rate_hz": max(
                natural_summaries[row["condition"]]["per_second_mean_rate_hz"]
            ),
            "susceptibility_hz_s_per_l2_current_s": checkpoint["paired"][
                "susceptibility_hz_s_per_l2_current_s"
            ],
            "duplicate_determinism_checked": checkpoint.get("duplicate_determinism") is not None,
        }
        for row in gains["rows"] for checkpoint in row["checkpoints"]
    ]
    for entry in fork_context:
        entry["fraction_of_arm_peak"] = (
            entry["preceding_1s_global_rate_hz"] / entry["arm_peak_1s_global_rate_hz"]
        )
    corrections = [
        {
            "id": "AXIAL_FRONT_READOUTS_DEGENERATE",
            "superseded": (
                "the D-halo lead / halo width / recruitment-front speed columns read as "
                "a slow spatial mechanism that changes with E-to-I reach"
            ),
            "evidence": (
                "D_halo_lead_mm is finite only from the first 1 s bin that crosses the "
                "local rate threshold; in every arm every later bin is below "
                f"{AXIAL_BIN_MM:.3f} mm, i.e. one axial grid bin.  Its single finite value "
                "tracks how much sheet was already recruited inside that bin "
                "(C1 10% -> 1.24 mm, Q2 7% -> 1.21 mm, Q1 17% -> 1.19 mm, Q3 70% -> 0.57 mm, "
                "C0 39% -> 0.01 mm) and is unordered in q.  D_halo_width_q05_q95_mm is "
                "18.09-18.16 mm in every arm at every second including pre-onset, i.e. the "
                "sheet width.  rate_front_q95_axis_mm is pinned at 15.5 mm, so the fitted "
                "recruitment speed is 0.0007-0.18 mm/s for a 20 mm sheet recruited inside 1 s."
            ),
            "corrected": (
                "the axial front / D-halo family has no post-onset dynamic range on this "
                "substrate and supports no mechanism claim; the sheet recruits globally "
                "within one 1 s bin."
            ),
        },
        {
            "id": "ENTRY_TIMING_NEEDS_THE_SAME_Q_CONTROL",
            "superseded": "Q1/Q2 delayed onset to 13/12 s while Q3 advanced it to 6 s",
            "evidence": (
                f"{', '.join(control['members'])} all sit inside the registered "
                f"+/-{MICROSTATE_Q_TOLERANCE:g} same-q microstate tolerance of C0 and their "
                f"onsets already span {band_onsets[0]:.0f}-{band_onsets[-1]:.0f} s.  Q2 "
                "(12 s) falls inside that band; only Q3 (6 s) falls outside it."
            ),
            "corrected": (
                "only Q3 moved entry beyond the graph-microstate reference spread; the "
                "Q1/Q2 differences are inside it."
            ),
        },
        {
            "id": "GAIN_FORKS_ARE_NOT_HIGH_STATE_SNAPSHOTS",
            "superseded": "the paired weak-patch forks measure high-state susceptibility",
            "evidence": (
                "the 1 s preceding each fork ran at "
                + ", ".join(
                    f"{entry['condition']} {entry['checkpoint']} "
                    f"{entry['preceding_1s_global_rate_hz']:.1f} Hz "
                    f"({entry['fraction_of_arm_peak'] * 100:.0f}% of that arm's peak 1 s rate)"
                    for entry in fork_context
                )
                + "; high_state_dwell_s is defined as total_ms - onset_ms, not a measured episode."
            ),
            "corrected": (
                "onset+2 s forks probed the early escalation ramp, not the high state; only "
                "the C0 onset+6 s fork sampled a genuinely high-rate state."
            ),
        },
        {
            "id": "LATE_FUNCTIONAL_DEFLECTION_IS_NOT_A_REACH_READOUT",
            "superseded": (
                "the late-window centre/surround deflection tracks the widening E-to-I surround"
            ),
            "evidence": (
                "no probe changed the per-window firing rate in any spatial bin, so the "
                "far field is exactly 0 unless a spike time moved inside a window.  It is "
                "nonzero at exactly "
                + ", ".join(functional["spike_timing_divergent_locations"])
                + " -- unordered in q (C0 and Q2 are zero, Q3 is zero at its core-adjacent "
                "patch and nonzero at its neutral patch).  excess_spikes is a net count and "
                "cannot see a within-window shift."
            ),
            "corrected": (
                "the late deflection is a binary spike-timing divergence, so the paired probe "
                "supports no statement about reach-dependent functional surround; "
                "any_registered_zero_crossing is driven entirely by those three locations."
            ),
        },
        {
            "id": "BOUNDEDNESS_MARGIN_MIXES_UNITS",
            "superseded": (
                "boundedness_margin is a single normalized margin comparable across criteria"
            ),
            "evidence": (
                "classify_high_state takes min() over (250-rate)/250 and (0.05-f_ref)/0.05, "
                "which are fractions of their thresholds, together with (0.05 - drift), which "
                "is a raw s^-1 difference.  Re-normalizing the drift terms by 0.05 leaves the "
                "ranking C0 > Q2 > Q1 > C1 > Q3 unchanged, so the pre-registered fork "
                "selection (largest margin, then most different phenotype) is unaffected."
            ),
            "corrected": (
                "the scalar is a mixed-unit worst-case indicator, not a normalized margin; the "
                "figure now shows the three registered limits separately."
            ),
        },
        {
            "id": "FUNCTIONAL_PROBE_BASELINE_IS_NOT_H_EQUALS_ZERO",
            "superseded": "the paired probes were run at the spec's Z=1, H=0 baseline",
            "evidence": (
                "the frozen probe state is the C0 exact state at 2100.05 ms: slow z = 1.0, "
                "m = 0, x_relay = 1.0 hold exactly, but H has already risen to mean 0.453 "
                "from the returning IEDs that precede it."
            ),
            "corrected": (
                "Z/U/M/X hold as specified; H does not, because the amplitude lock needed a "
                "realistic interictal state rather than the t=0 state."
            ),
        },
    ]

    # Plot-only repair: preserve the frozen phenotype JSON and its registered hash.
    AGG._plot(phenotype["rows"])
    _write_readme()

    key_artifacts = [
        LC5_AUTHORIZATION, lc5_summary, GRAPH, TWO_HOP, FUNCTIONAL,
        PHENOTYPE, GAINS, CONFIRMATION,
        FIGURES / "lc6a_graph_and_twohop.png",
        FIGURES / "lc6a_functional_response.png",
        FIGURES / "lc6a_trajectory_phenotypes.png",
        FIGURES / "lc6a_gain_forks.png",
    ]
    artifact_hashes = {
        str(path.relative_to(ROOT)): _sha(path) for path in key_artifacts if path.is_file()
    }
    now = datetime.now(timezone.utc).astimezone().isoformat()
    payload = {
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
        "status": "COMPLETE",
        "decision": "CANONICAL_SEED_AXIAL_REACH_FAMILY_NO_CARRIER",
        "safe_claim": (
            "Under the locked legacy Z/H substrate and canonical graph/noise realization, "
            "widening patient-axis E-to-I reach from the legacy family through q≈1.5 did not "
            "open a bounded high-state carrier: all five arms entered from returning IEDs and "
            "escalated to registered saturation, and the post-entry escalation was "
            "indistinguishable across arms.  Only the strongest rung (Q3) moved entry beyond "
            "the spread of the three same-q graph realizations, and Q2/Q3 shifted the "
            "interictal baseline event statistics."
        ),
        "output_root_executed": str(OUT.relative_to(ROOT)),
        "output_root_spec_example_mismatch": {
            "present": True,
            "spec_example": "results/topic4_sef_hfo/mz_full_conductance_spatial_relay/lc6a_patient_axis_surround",
            "executed_manifest_family": str(OUT.relative_to(ROOT)),
            "scientific_or_numeric_effect": False,
        },
        "protocol": manifest["protocol"],
        "execution_manifest": str(MANIFEST.relative_to(ROOT)),
        "execution_manifest_sha256": _sha(MANIFEST),
        "git_head": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "completed_at": now,
        "graph_rows": graph_rows,
        "trajectory_rows": trajectory,
        "lc5_right_censor_closeout": {
            "decision": lc5["decision"],
            "onset_ms": lc5["lc5_onset_ms"],
            "terminal_ms": lc5["lc5_terminal_ms"],
            "end_rate_hz": lc5["lc5_end_rate_hz"],
            "D_end": lc5["lc5_D_end"],
            "H_end": lc5["lc5_H_end"],
            "summary": str(lc5_summary),
            "summary_sha256": lc5["lc5_summary_sha256"],
            "first_continuation_chunk_input_hash_unavailable": True,
            "classifier_snapshot_replay_complete": True,
        },
        "q_ladder_realized": q_ladder,
        "escalation_after_entry": escalation,
        "gain_forks": gains,
        "gain_fork_state_context": fork_context,
        "functional_probe_perturbation": {
            "spike_timing_divergent_locations": functional["spike_timing_divergent_locations"],
            "late_deflection_orders_with_reach": False,
        },
        "post_hoc_corrections": corrections,
        "confirmation": confirmation,
        "engineering": {
            "graph_family_legal": True,
            "functional_characterization_complete": True,
            "five_natural_trajectories_complete": True,
            "gain_forks_complete": True,
            "confirmation_not_triggered": True,
            "blessed_engine_hashes": engine,
            "source_sha256": _source_hashes(),
            "artifact_sha256": artifact_hashes,
        },
        "scientific_result_vector": {
            "two_hop_surround_changed": True,
            "natural_entry_all_arms": True,
            "entry_moved_beyond_q_matched_band": q_ladder["onset_outside_q_matched_band"],
            "baseline_tradeoff_conditions": [
                row["condition"] for row in trajectory if row["baseline_tradeoff"]
            ],
            "bounded_high_branch_opened": False,
            "global_saturation_all_arms": True,
            "local_refractory_ceiling_fraction_low": True,
            "axial_front_readouts_degenerate": [
                row["condition"] for row in trajectory
                if row["front_readout_degeneracy"]["degenerate"]
            ],
            "carrier_confirmation_triggered": False,
            "termination_tested": False,
            "lifecycle_tested": False,
        },
        "claim_boundary": {
            "allowed": (
                "Canonical-seed axial E-to-I reach family did not create a bounded carrier "
                "under the legacy substrate; the strongest rung moved entry earlier and "
                "Q2/Q3 shifted the interictal baseline."
            ),
            "forbidden": [
                "Mexican-hat connectivity is universally ineffective",
                "termination failed in LC6A",
                "complete lifecycle was tested",
                "the U mechanism is invalid",
                "wider E-to-I reach changed the D depletion halo geometry",
                "Q1 or Q2 delayed natural entry",
                "the gain forks measured high-state susceptibility at onset+2 s",
                "the late paired functional deflection is a reach-dependent surround signal",
            ],
        },
    }
    _write_json(RUN_MANIFEST, payload)

    graph_resource_rows = [
        row["resource_end"] for row in graph["audits"].values()
        if "resource_end" in row
    ]
    resource = {
        "timestamp": now,
        "scope": "closeout summary; not a continuous sampler",
        "natural_arm_measured_peak_rss_gib": 6.891483,
        "graph_swap_used_mib_min": min(float(row["swap_used_mib"]) for row in graph_resource_rows),
        "graph_swap_used_mib_max": max(float(row["swap_used_mib"]) for row in graph_resource_rows),
        "resource_integrity_gate_triggered": False,
    }
    _atomic_text(RESOURCE_LOG, json.dumps(resource, sort_keys=True) + "\n")

    table = _fmt_table(trajectory)
    status = f"""# FCXR-LC6A status

状态：**COMPLETE — CANONICAL_SEED_AXIAL_REACH_FAMILY_NO_CARRIER**

五个固定图和五条自然轨迹均完成。实际 two-hop 抑制宽度从 C0 的 {graph_rows[0]['two_hop_q']:.3f} 增至 Q3 的 {graph_rows[-1]['two_hop_q']:.3f}；五臂都保留自然进入，但随后均进入注册 saturation，没有打开 bounded high-state carrier。

LC5v2.1 唯一右删失格已续跑裁决：23 s onset 后在 27 s 达到 405.9 Hz，D=0.573、H=25.763，结局 `ESCALATING_SATURATION`。25--26 s 的 reducer 故障使该段输入 hash/诊断 trace 不可用，但 exact checkpoint 已恢复，28 个 classifier bundles 重放完成，注册 saturation 出现在后续完整 26--27 s 段。

进入时刻必须对着"同一 reach、只换连线微状态"的对照带读：{'/'.join(q_ladder['q_matched_realizations'])} 三张图的 q 都落在注册的 ±{MICROSTATE_Q_TOLERANCE:g} 同 q 容差内，它们的 onset 已经自己散在 {band_onsets[0]:.0f}–{band_onsets[-1]:.0f} s。Q2 的 {next(row['onset_s'] for row in trajectory if row['condition'] == 'Q2'):.0f} s 落在这个带里；只有 Q3 的 {next(row['onset_s'] for row in trajectory if row['condition'] == 'Q3'):.0f} s 在带外。Q2/Q3 同时改变了基线事件统计。

进入之后的升级过程与 reach 无关：五臂都在进入后 {min(escalation['seconds_from_entry_to_arm_end'].values())}–{max(escalation['seconds_from_entry_to_arm_end'].values())} s 抵达停机点（越过注册饱和线后再跑 1 s），逐秒 rate 的跨臂离散度最大 {escalation['max_all_arm_spread_ratio']:.2f}×，而三张同 q 图之间已经有 {escalation['max_q_matched_spread_ratio']:.2f}×。

轴向 D-halo / front-speed 读数在本轮不可用：它们在进入后没有动态范围（见 `run_manifest.json` 的 `post_hoc_corrections`），不支持任何"更宽 reach 加速 D 耗竭"的机制说法。

gain fork 只回答某个精确状态对弱输入还有没有非零响应；onset+2 s 的两次 fork 分别落在 {fork_context[0]['preceding_1s_global_rate_hz']:.0f} Hz 和 {fork_context[-1]['preceding_1s_global_rate_hz']:.0f} Hz，即升级斜坡上而非高态。没有 carrier，因此 graph-realization confirmation 按合同未触发。

termination：**NOT_TESTED**  
lifecycle：**NOT_TESTED**

执行根：`{OUT.relative_to(ROOT)}`。spec 中旧的嵌套 output 示例与实际根不一致；该偏差只影响路径，不影响方程、图、轨迹或判决，已写入 `run_manifest.json`。
"""
    _atomic_text(STATUS, status)

    archive = f"""# FCXR-LC6A 患者轴 E→I surround：canonical-seed bounded-negative

日期：2026-08-15

## 1. 一句话结论

我们确实把患者轴方向的有效两跳抑制周边做宽了，但在当前 legacy Z/H substrate 上，它没有创造一个可停留的中间高态：五条自然轨迹都从 returning IED 进入，随后继续升级到注册 saturation。

正式标签：`CANONICAL_SEED_AXIAL_REACH_FAMILY_NO_CARRIER`。

## 2. LC5v2.1 右删失格收口

唯一右删失格在 23 s onset；从 25 s exact state 继续后，于 27 s 达到 405.9 Hz，D=0.573、H=25.763，判 `ESCALATING_SATURATION`，没有 offset。25--26 s 曾在 reducer 路径失败，因此该段输入 hash 和诊断 trace 不可用；但 exact checkpoint 被恢复，28 个 classifier snapshot bundles 全部重放，注册 saturation 位于完整记录的 26--27 s 段。这个缺口限制第一续跑秒的细粒度诊断，不改变终局 saturation 裁决。

## 3. 这轮真正改了什么

只改变代码 `IE`，即生物学 E→I 的患者轴 reach；EE、I→E、I→I、权重、Z/H、两个 core、噪声与所有慢机制保持冻结。graph-only two-hop 审计显示 q 从 {graph_rows[0]['two_hop_q']:.3f} 增至 {graph_rows[-1]['two_hop_q']:.3f}，surround/center 质量比从 {graph_rows[0]['surround_center_ratio']:.2f} 增至 {graph_rows[-1]['surround_center_ratio']:.2f}，所以这不是“图没改到位”的假阴性。

改造是轴向的：垂轴 marginal 宽度从 {graph['audits']['C0']['marginal_e_to_i']['sigma_perpendicular_mm']:.4f} mm 只走到 {graph['audits']['Q3']['marginal_e_to_i']['sigma_perpendicular_mm']:.4f} mm，两跳垂轴宽度同样保持在 1.414 mm 附近；source out-degree 的 CV/q95/q99/interior-edge 比值相对 C0 的偏差都在 ±10% 容差内（最大 −5.8%），对轴向与垂轴位置的 Spearman 偏移都 ≤0.005。spec 要求的垂轴 confound 与 out-degree 容差都通过。

## 4. 实际跑出来的不是五级阶梯

spec 预期 C0 的 `q_parallel^marginal` 约 0.66，实测是 {q_ladder['measured_c0_q_marginal']:.3f}。因此预注册的 Q1=1.00 这一档落在了 C1 自己的 ±{MICROSTATE_Q_TOLERANCE:g} 同 q 容差里。实际结构是：{'/'.join(q_ladder['q_matched_realizations'])} 三张同 q 但连线微状态不同的图（互换边比例 20–21%），加上 Q2、Q3 两个真正的 reach 档。这三张同 q 图就是这一轮唯一的内建对照。

## 5. 五条自然轨迹

{table}

所有条件的 active area 都到 400 mm²，也就是 20×20 mm 全片 100%；近 refractory ceiling 的细胞比例很低（最大 {max(row['max_near_refractory_fraction'] for row in phenotype['rows']) * 100:.2f}%，注册线 5%），但全局 1 s 均值仍跨过 250 Hz 的注册 saturation。因此这轮不是“有限面积内的健康 carrier”，而是全片 escalating high state。

**进入时刻**：三张同 q 图的 onset 已经自己散在 {band_onsets[0]:.0f}–{band_onsets[-1]:.0f} s。Q2 的 {next(row['onset_s'] for row in trajectory if row['condition'] == 'Q2'):.0f} s 在这个带内，Q3 的 {next(row['onset_s'] for row in trajectory if row['condition'] == 'Q3'):.0f} s 在带外。可以说的只有“最强那一档把进入提前到对照带之外”，不能说“Q1/Q2 推迟了进入”。

**进入之后**：五臂都在进入后 {min(escalation['seconds_from_entry_to_arm_end'].values())}–{max(escalation['seconds_from_entry_to_arm_end'].values())} s 抵达各自的停机点（越过注册饱和线后再跑 1 s 停），对齐到进入时刻后逐秒 rate 的跨臂离散度最大 {escalation['max_all_arm_spread_ratio']:.2f}×，而仅三张同 q 图之间就已经有 {escalation['max_q_matched_spread_ratio']:.2f}×。换句话说，把两跳抑制周边加宽 60% 只改变了什么时候开始升级，没有改变升级本身。

**轴向 D-halo / front-speed 不可用**：这三个读数在本 substrate 上没有进入后的动态范围。`D_halo_lead_mm` 只在“第一个越过局部 rate 阈值的 1 s 窗”里有限，其数值由那一窗里已经点亮了多少面积决定（C1 10%→1.24 mm、Q2 7%→1.21 mm、Q1 17%→1.19 mm、Q3 70%→0.57 mm、C0 39%→0.01 mm），与 q 完全不成序；之后每一窗都 <0.03 mm，即不到一个 {AXIAL_BIN_MM:.3f} mm 的空间格。`D_halo_width` 在所有臂所有秒（含进入前）都是 18.1 mm，就是整片的宽度。因此不能写“更宽 E→I 加速了 D 耗竭 halo”。详见 `run_manifest.json` 的 `post_hoc_corrections`。

## 6. 短功能探针

八个探针位置全部满足预锁的亚阈值合同：分窗放电率差在每个空间格点上精确为零。其中五个位置的远场（|x|>1.5 mm）也精确为零——没有任何脉冲改变过。另外三个（{', '.join(functional['spike_timing_divergent_locations'])}）远场非零，说明有脉冲在窗内挪了位置；它们正是唯一出现晚窗“零交叉”的三个位置。这三个位置与 reach 不成序（C0 与 Q2 为零，Q3 换到 core-adjacent 位点也为零），所以这一轮的配对探针**不支持任何关于 reach 依赖功能周边的说法**，只支持“探针确实是亚阈值的、直接足迹只落在被刺激的小块上、中心偏转由抑制驱动力而非被招募的兴奋构成”。

另外，探针基线是 C0 在 2100 ms 的精确状态：Z=1、U=M=0、X=1 精确成立，但 H 已经涨到均值 0.453，不是 spec 字面写的 H=0。

## 7. gain fork 的意义与边界

按预注册规则（boundedness margin 最大 + 表型最不同）选择 C0 与 Q2。必须注明这些 fork 采样的是什么状态：onset+2 s 时 C0 前 1 s 为 {fork_context[0]['preceding_1s_global_rate_hz']:.1f} Hz、Q2 为 {fork_context[-1]['preceding_1s_global_rate_hz']:.1f} Hz，分别只有各自 1 s 峰值的 {fork_context[0]['fraction_of_arm_peak'] * 100:.0f}% 和 {fork_context[-1]['fraction_of_arm_peak'] * 100:.0f}%——这是升级斜坡，不是高态。只有 C0 的 onset+6 s（前 1 s {fork_context[1]['preceding_1s_global_rate_hz']:.1f} Hz）真正采到了高 rate 状态，它给出非零响应。因此“C0 早期响应为零”只能读作“斜坡早期对这个弱输入没有可测响应”，不能读作“高态是惰性的”。

`high_state_dwell_s` 也需要注意：它按 `总时长 − onset` 计算，不是实测的高态时长，所以 spec §10.1 第 2 条的“高态至少持续 5 s”在实现里是观察时长而非高态时长。本轮五臂都因其他条件失败，这个宽松定义没有改变结论。

## 8. 可以说与不能说

可以说：在 canonical graph/noise 与锁定 legacy substrate 下，单独把患者轴 E→I reach 扫到 q≈1.5，没有打开 bounded carrier；进入之后的升级过程与 reach 无关；最强那一档把进入提前到同 q 对照带之外；Q2/Q3 带来 baseline tradeoff。

不能说：Mexican-hat 普遍无效；U 被否定；LC6A 测过 termination 或完整 lifecycle；更宽 reach 改变了 D 耗竭 halo 的几何；Q1/Q2 推迟了进入；gain fork 测到了高态的响应性；晚窗配对偏转是 reach 依赖的周边信号。LC6A 从设计上只测 carrier capability。

## 9. 下一机制分支

固定宽核把 800 条 E→I 输入从近处重新分配到远处，也削弱了局部 center。若继续，优先做 spec 已预留但未授权的 center-preserving two-component E→I kernel（70–75% legacy local + 25–30% wide axial），而不是继续扩单一 q 网格。若仍是全局 saturation，应转向 H source/transfer；不能再把问题包装成“刹车剂量不足”。

## 10. 工程与边界

graph legality、two-hop、functional、自然轨迹、两个 gain phenotype、四组主图和未触发 confirmation 均完成；六个 blessed engine hash 一致。C0 自然轨迹对参考路径 bitwise parity 通过（spike sha256 一致、rate 最大差 0 Hz）。无 carrier，所以 confirmation 不运行是合同结果，不是缺失实验。

本 archive 的 §4–§7 由 2026-08-15 的复审重写；被取代的旧表述与逐条证据记录在 `run_manifest.json` 的 `post_hoc_corrections`（六条：轴向 front 读数退化、进入时刻需要同 q 对照、gain fork 不是高态、晚窗功能偏转是脉冲时刻发散、boundedness margin 单位混用、探针基线 H≠0）。原始 per-arm JSON、图与哈希链未改动。

结果根：`{OUT.relative_to(ROOT)}`。spec 的旧嵌套路径示例未被 runner 使用；这一纯路径偏差已在 `run_manifest.json` 留痕。

termination：`NOT_TESTED`。lifecycle：`NOT_TESTED`。
"""
    _atomic_text(ARCHIVE, archive)
    _write_json(DONE, {
        "status": "DONE",
        "decision": payload["decision"],
        "run_manifest": str(RUN_MANIFEST),
        "run_manifest_sha256": _sha(RUN_MANIFEST),
        "status_artifact": str(STATUS),
        "status_sha256": _sha(STATUS),
        "archive": str(ARCHIVE.relative_to(ROOT)),
        "archive_sha256": _sha(ARCHIVE),
    })
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    if not args.confirm_run:
        raise SystemExit("LC6A closeout requires --confirm-run")
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
