#!/usr/bin/env python3
"""Row-level audit of the v0.3.2 E1146 discrepancy (+0.1277 model side vs -0.3291 eval side).

Read-only on ``/data/hfosp_group_event_state_v0_3_2``.  Writes
``e1146_discrepancy_audit.json`` into the Agent A result directory and mirrors it
under ``/data/hfosp_group_event_state_v0_3_3/agent_a/``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v033_evaluator import e1146_audit as A  # noqa: E402
from src.topic5_group_event_state.v033_evaluator import canonical as C  # noqa: E402


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _summary(report: dict) -> dict:
    means = report["published_means"]
    canon = report["canonical_rescore_means"]
    decomp = report["model_side_decomposition_means"]
    steps = report["per_seed"][0]["steps"]
    by = {s["step"]: s for s in steps}
    return {
        "engineering_consistency": [
            "同一 checkpoint、同一 113 个 dev_test anchor、同一 count 目标，两条路径逐行一致；"
            "分叉从 prediction_H 开始：model 侧用 registry 冻结的 H_strong（%d 特征，commit %s），"
            "eval 侧在 H1 运行里重新拟合 H_strong（%d 特征，含 days_since_recording_start，commit %s）。"
            % (by["prediction_H"]["n_features_model"], by["checkpoint"]["registry_commit"][:10],
               by["prediction_H"]["n_features_eval"], by["checkpoint"]["eval_commit"][:10]),
            "该 H 差异本身很小（max |Δlog μ_H| = %.4f，dev_test 平均 NLL 7.1457 vs 7.1462），不足以翻转符号。"
            % by["prediction_H"]["max_abs_delta_log_mu"],
            "符号翻转发生在 prediction_H_plus_state：model 侧的 H+S_correct 是 checkpoint 自己的读出 "
            "log μ_H + α wᵀS̃（调制 RMS ≈ 0.04–0.07 nats），eval 侧的 H+S_correct 是一个重新拟合的 NB ridge GLM"
            "（H_strong ⊕ 12 列原始 anchor state，系数/截距/dispersion 全部重估），二者不是同一个预测器。",
            "两个公开数字都被 canonical evaluator 从各自的逐 anchor 行逐行复现（all_published_reproduced=%s）；"
            "评分公式一致，分歧完全在被评分的预测和 dispersion 规则。" % report["all_published_reproduced"],
        ],
        "what_the_numbers_mean": [
            "model 侧 +%.4f 几乎全部来自 dispersion 重估：H 用 registry log r=0.597，adapter 训练出更小的 log r，"
            "H − mean(S_train) 臂（调制恒为 0）= %+.4f，而 mean − correct（真正的动态贡献）= %+.4f。"
            % (means["model_h_minus_correct"], decomp["dispersion_component_h_minus_mean"],
               decomp["dynamic_component_mean_minus_correct"]),
            "把 checkpoint 自己的预测放在共享 H dispersion 下用 canonical evaluator 评分，动态状态的增量为 %+.4f nats/anchor"
            "（三 seed 均值）；这才是 v0.3.2 对 E1146 '正确时刻状态是否有增量' 的可比读数。"
            % canon["model_predictions_shared_H_dispersion"],
            "eval 侧 −%.4f（shared alpha）/ −%.4f（per-arm）度量的是：给 GLM 加 12 列原始 state 后在 80–100%% 段外推变差，"
            "这是一个 probe 结果，不是 checkpoint 读出的结果。"
            % (-means["eval_shared_gain"], -means["eval_per_arm_gain"]),
        ],
        "allowed_conclusions": [
            "v0.3.2 的 +0.1277 与 −0.3291 不是同一比较的两个答案；它们对同一 checkpoint/anchor 使用了不同的 prediction_H、"
            "不同的 prediction_H_plus_state 构造和不同的 dispersion 规则。",
            "v0.3.3 canonical evaluator 规则：训练/评价/画图必须评分 checkpoint 自己的预测；dispersion 规则显式声明；"
            "任何 GLM probe 必须作为单独命名的臂报告。",
        ],
        "forbidden_conclusions": [
            "不能说 E1146 上存在或不存在 30 分钟 count 增量的状态——本审计只解释两条路径为何不同。",
            "不能把 +0.1277 当作状态证据（它是 dispersion 重估），也不能把 −0.3291 当作 checkpoint 阴性（它评分的是另一个模型）。",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v032-root", type=Path, default=Path("/data/hfosp_group_event_state_v0_3_2"))
    parser.add_argument("--subject", default="epilepsiae_1146")
    parser.add_argument("--seeds", nargs="+", type=int, default=(20260902, 20260903, 20260904))
    parser.add_argument("--out", type=Path,
                        default=ROOT / "results/group_event_state/v0_3_3/evaluator_assay/e1146_discrepancy_audit.json")
    parser.add_argument("--mirror", type=Path,
                        default=Path("/data/hfosp_group_event_state_v0_3_3/agent_a/e1146_discrepancy_audit.json"))
    args = parser.parse_args()

    arts = [A.load_seed_artifacts(args.v032_root, args.subject, seed) for seed in args.seeds]
    report = A.aggregate_seeds(arts, subject=args.subject, model_row_tolerance=1e-5, eval_row_tolerance=1e-8)
    inputs = {}
    for seed in args.seeds:
        run_dir = args.v032_root / "model/runs/leaky_bank" / args.subject / f"seed_{seed}"
        h1_dir = args.v032_root / "evaluation/h1" / args.subject
        inputs[str(seed)] = {
            "model_evaluation_json": _sha(run_dir / "evaluation.json"),
            "model_result_json": _sha(run_dir / "result.json"),
            "eval_h1_result_json": _sha(h1_dir / f"h1_result_seed_{seed}.json"),
            "eval_h1_arrays_npz": _sha(h1_dir / f"h1_arrays_seed_{seed}.npz"),
        }
    inputs["history_baseline_registry_json"] = _sha(args.v032_root / "shared/history_baseline_registry.json")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    payload = {
        "format": "group_event_state_v0_3_3_e1146_discrepancy_audit",
        "generated": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_commit": commit,
        "canonical_schema_version": C.SCHEMA_VERSION,
        "canonical_tolerance_nats": C.TOLERANCE_NATS,
        "read_only_inputs_sha256": inputs,
        "step_order": list(A.STEP_ORDER),
        "audit": report,
        "summary": _summary(report),
        "evidence_label": "DIAGNOSTIC (engineering audit of v0.3.2 artefacts; no human result is asserted)",
        "sealed_partition_opened": False,
    }
    _atomic_json(args.out, payload)
    _atomic_json(args.mirror, payload)
    print(json.dumps({
        "out": str(args.out), "first_divergence": report["first_divergence"],
        "sign_flip_origin_by_seed": report["sign_flip_origin_by_seed"],
        "published_means": report["published_means"],
        "canonical_rescore_means": report["canonical_rescore_means"],
        "model_side_decomposition_means": report["model_side_decomposition_means"],
        "all_published_reproduced": report["all_published_reproduced"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
