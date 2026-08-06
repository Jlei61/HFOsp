#!/usr/bin/env python3
"""Build the M2-backed shared early-recruitment readout artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic4_early_recruitment_readout import (  # noqa: E402
    build_m2_early_recruitment_readout,
    load_readout_config,
)


DEFAULT_OUT = ROOT / "results/topic4_sef_hfo/early_recruitment_readout"


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _primary(summary, target, level="contact_space", support="all"):
    ref = summary["reference_primary"]
    return summary["targets"][target]["by_reference"][ref][level][support]["10_50ms"]


def _write_status(out, summary):
    lines = [
        "# Topic 4 early-recruitment readout — M2 adapter",
        "",
        "**Framing:** `model_side_readout_infrastructure_not_seizure_validation`. "
        "本产物只检查稳定间期响应的空间次序能否预测临界附近固定早期窗内的兴奋率能量，"
        "不把 runaway 命名为真实发作，也不改写 M2 的 ignition/spread verdict。",
        "",
        "## 合同",
        "- 间期场：positive kick-minus-control E-rate 的逐位置 half-peak arrival time。",
        "- 早期场：10–50 ms 内 positive excess E-rate 的平方均值；5–30、10–75 ms 为敏感性。",
        "- 预期方向：`arrival_energy_spearman < 0`，等价于 `earliness_energy_spearman > 0`。",
        "- 饱和红线：早期窗结束前 escape 的轨迹不可判，不使用截断窗口。",
        "",
        "## 当前运行摘要",
    ]
    reference = summary["references"][summary["reference_primary"]]
    max_time = summary["provenance"]["perturbation"]["max_time_ms"]
    display_end = summary["provenance"].get("display_window_ms")
    lines.append(
        f"- stable interictal 动态检查到 `{max_time:g} ms`："
        f"contact recruitment=`{reference['n_contact_participating']}/15`，"
        "未招募触点保持无 finite arrival，不做补值。"
    )
    if "pre_runaway" in summary["targets"]:
        probe = summary["targets"]["pre_runaway"]
        lines.append(
            f"- `pre_runaway` visualization probe：contact recruitment="
            f"`{probe['n_contact_participating']}/15`，escape_at=`{probe['escape_at_ms']} ms`，"
            f"主图仅显示 `0–{display_end:g} ms`。这是独立状态探针，不命名为完整发作。"
        )
    for target in summary["targets"]:
        row = _primary(summary, target)
        cmp = row["comparison"]
        null = row["within_group_null"] or {}
        ref = summary["reference_primary"]
        src_x = summary["targets"][target]["by_reference"][ref]["source_space"]["core_excluded"]["10_50ms"]
        con_x = summary["targets"][target]["by_reference"][ref]["contact_space"]["core_excluded"]["10_50ms"]
        con_x_cmp = con_x["comparison"]
        con_x_null = con_x["within_group_null"] or {}
        lines.append(
            f"- `{target}` contact/all 10–50 ms: energy_status=`{row['energy_status']}`, "
            f"n=`{cmp['n']}`, earliness_rho=`{cmp['earliness_energy_spearman']}`, "
            f"within_shaft_p=`{null.get('p_one_sided')}` (`{null.get('method')}`). "
            f"去直接核后 source=`{src_x['comparison']['status']}` (n={src_x['comparison']['n']}), "
            f"contact=`{con_x_cmp['status']}` (n={con_x_cmp['n']}, "
            f"earliness_rho={con_x_cmp.get('earliness_energy_spearman')}, "
            f"within_shaft_p={con_x_null.get('p_one_sided')})."
        )
    lines += [
        "",
        "## 有界解释",
        "共享 readout 已经接入 accepted E1146 montage。source-space 含核关系方向较强，但去掉直接 perturbation "
        "core 后 source arrival field 退化；E1146 contact-space 含核仅弱相关，去 direct-core loading 后符号反转，"
        "within-shaft null 也不支持升级。"
        "因此这次运行只能说明基础设施和同平面显示闭合，不能写成间期→发作早期 bridge 已建立。",
        "本 STATUS 只描述 M2 数值 association adapter；当前 onset-locked 主图另由 "
        "`run_topic4_m3_runaway_readout.py` 复用 accepted M3 continuous q_I→runaway 轨迹生成。"
        "两层结果不得混写。",
        "",
        "逐状态、逐窗口、source/contact、含核/去核统计见 `early_recruitment_readout.json`；"
        "逐时间动态场和 mask 见 `early_recruitment_readout_arrays.npz`。",
    ]
    (out / "STATUS.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument(
        "--geometry-npz", default=None,
        help="accepted subject-SNN figdata override (useful from a results-light worktree)",
    )
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary, arrays = build_m2_early_recruitment_readout(
        args.config, geometry_npz=args.geometry_npz)
    (out / "early_recruitment_readout.json").write_text(
        json.dumps(_jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(out / "early_recruitment_readout_arrays.npz", **arrays)
    _write_status(out, summary)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
