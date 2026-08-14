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


def _trajectory_rows(
    phenotype: dict, natural_summaries: dict[str, dict] | None = None,
) -> list[dict]:
    if natural_summaries is None:
        natural_summaries = {
            condition: _json(OUT / f"trajectories/{condition}/summary.json")
            for condition in ("C0", "C1", "Q1", "Q2", "Q3")
        }
    rows = []
    for row in phenotype["rows"]:
        natural = natural_summaries[row["condition"]]
        rows.append({
            "condition": row["condition"],
            "onset_s": float(row["effective_onset_ms"]) / 1000.0,
            # baseline_metrics is deliberately capped at the C0 baseline horizon for
            # across-arm equivalence.  The entry ledger belongs to the arm's own onset.
            "n_returning_pre_onset": int(natural["n_returning_pre_onset"]),
            "peak_global_rate_hz": float(row["global_rate_100ms_peak_hz"]),
            "peak_local_q99_hz": float(row["local_rate_q99_peak_hz"]),
            "D_halo_lead_mm": float(row["spatial_slow_flow"]["max_D_halo_lead_mm"]),
            "active_area_mm2": float(row["spatial_slow_flow"]["max_active_area_mm2"]),
            "boundedness_margin": float(row["boundedness"]["boundedness_margin"]),
            "baseline_tradeoff": bool(row["baseline_tradeoff"]["tradeoff"]),
            "headline": row["headline"],
        })
    return rows


def _fmt_table(rows: list[dict]) -> str:
    lines = [
        "| 条件 | q(two-hop) | onset | onset前IED | 全局峰值 | D halo | baseline代价 | 结局 |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['condition']} | {row['two_hop_q']:.3f} | {row['onset_s']:.1f} s | "
            f"{row['n_returning_pre_onset']} | {row['peak_global_rate_hz']:.1f} Hz | "
            f"{row['D_halo_lead_mm']:.2f} mm | "
            f"{'有' if row['baseline_tradeoff'] else '无'} | saturation |"
        )
    return "\n".join(lines)


def _write_readme() -> None:
    sections = [
        ("lc6a_graph_and_twohop", "连接与两跳抑制几何审计。它证明患者轴 E→I reach 和实际 E→I→E surround 随条件增强；图本身不包含自然发作，也不能证明 carrier。", "重点看 two-hop q 和 surround/center，而不是只看 sampler 输入宽度。"),
        ("lc6a_functional_response", "同一弱 E patch 在三个时间窗内的配对突触膜贡献。背景事件混杂已显式标记，零交叉只作描述，不决定自然轨迹权限。", "看晚窗响应是否随 reach 改变；不要把无零交叉解释成自然动力学 no-go。"),
        ("lc6a_trajectory_phenotypes", "五条固定自然轨迹的 onset、全局/局部活动、D halo、活动面积和 boundedness margin。D halo 与面积使用不同纵轴，避免毫米量被 400 mm² 尺度压扁。", "五臂均自然进入后升级到注册 saturation，且没有任何 bounded carrier。"),
        ("lc6a_gain_forks", "预注册选出的 C0/Q2 exact-state 弱输入 fork，报告 susceptibility、回落时间、rate 和面积偏离。响应性与 boundedness 分开判读。", "非零响应不等于 carrier；termination 和 lifecycle 均未测试。"),
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
    trajectory = _trajectory_rows(phenotype)
    by_condition = {row["condition"]: row for row in graph_rows}
    for row in trajectory:
        row.update({
            "two_hop_q": by_condition[row["condition"]]["two_hop_q"],
            "surround_center_ratio": by_condition[row["condition"]]["surround_center_ratio"],
        })

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
            "changing patient-axis E-to-I reach from the legacy family through q≈1.5 changed "
            "entry timing and slow spatial readouts but did not open a bounded high-state carrier."
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
        "gain_forks": gains,
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
            "baseline_tradeoff_conditions": [
                row["condition"] for row in trajectory if row["baseline_tradeoff"]
            ],
            "bounded_high_branch_opened": False,
            "global_saturation_all_arms": True,
            "local_refractory_ceiling_fraction_low": True,
            "carrier_confirmation_triggered": False,
            "termination_tested": False,
            "lifecycle_tested": False,
        },
        "claim_boundary": {
            "allowed": (
                "Canonical-seed axial E-to-I reach family changed entry and D-halo geometry "
                "but did not create a bounded carrier under the legacy substrate."
            ),
            "forbidden": [
                "Mexican-hat connectivity is universally ineffective",
                "termination failed in LC6A",
                "complete lifecycle was tested",
                "the U mechanism is invalid",
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

Q2/Q3 改变了基线事件统计；Q3 将 onset 从 C0 的 11 s 提前到 6 s，说明更宽 E→I reach 也可能通过更早/更广的抑制使用推动 D 耗竭，而不只是稳定网络。

gain fork 只回答高态是否还对弱输入有非零响应；它不覆盖 boundedness。没有 carrier，因此 graph-realization confirmation 按合同未触发。

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

只改变代码 `IE`，即生物学 E→I 的患者轴 reach；EE、I→E、I→I、权重、Z/H、两个 core、噪声与所有慢机制保持冻结。graph-only two-hop 审计显示 q 从 {graph_rows[0]['two_hop_q']:.3f} 增至 {graph_rows[-1]['two_hop_q']:.3f}，所以这不是“图没改到位”的假阴性。

## 4. 五条自然轨迹

{table}

所有条件的 active area 都到 400 mm²；近 refractory ceiling 的细胞比例很低，但全局和局部 rate 仍跨过注册 saturation。因此这轮不是“有限面积内的健康 carrier”，而是全片 escalating high state。

Q1/Q2 把 onset 推迟到 13/12 s，Q3 却提前到 6 s；更宽 E→I reach 不是单向稳定旋钮。它可能先招募更远的 I 使用，继而在 wavefront 前方加速 D=1-Z 的耗竭。

## 5. gain fork 的意义

按预注册规则选择 C0 与 Q2。fork 只测 exact high-state snapshot 对弱局部输入是否仍有有限非零响应，不参与 boundedness 标签。即使存在很小的非零响应，也不能把已经升级到 saturation 的状态改称 carrier。

## 6. 可以说与不能说

可以说：在 canonical graph/noise 与锁定 legacy substrate 下，单独把患者轴 E→I reach 扫到 q≈1.5，没有打开 bounded carrier；Q2/Q3 还带来 baseline tradeoff。

不能说：Mexican-hat 普遍无效；U 被否定；LC6A 测过 termination 或完整 lifecycle。LC6A 从设计上只测 carrier capability。

## 7. 下一机制分支

固定宽核把 800 条 E→I 输入从近处重新分配到远处，也削弱了局部 center。若继续，优先做 spec 已预留但未授权的 center-preserving two-component E→I kernel（70–75% legacy local + 25–30% wide axial），而不是继续扩单一 q 网格。若仍是全局 saturation，应转向 H source/transfer；不能再把问题包装成“刹车剂量不足”。

## 8. 工程与边界

graph legality、two-hop、functional、自然轨迹、两个 gain phenotype、四组主图和未触发 confirmation 均完成；六个 blessed engine hash 一致。无 carrier，所以 confirmation 不运行是合同结果，不是缺失实验。

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
