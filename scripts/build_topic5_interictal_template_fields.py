#!/usr/bin/env python3
"""Build reusable interictal-only TA/TB gradient-axis and 2D field artifacts.

This producer never reads seizure, onset, subtype, or ictal-energy data.  Its
per-subject output freezes the early-to-late axes, direction-validity tiers,
own-template planes/fields, and (when broadly collinear) the shared plane/fields.
Future ictal analyses must name-join their activation values to these artifacts.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import zlib
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.interictal_propagation import load_subject_propagation_events  # noqa: E402
from src.lagpat_rank_audit import mask_phantom_ranks  # noqa: E402
from src.propagation_skeleton_geometry import assign_events_to_templates, parse_shaft  # noqa: E402
from src.seeg_coord_loader import load_subject_coords  # noqa: E402
from src.topic5_template_axis_field import (  # noqa: E402
    INTERICTAL_FIELD_CONTRACT,
    TEMPLATE_AXIS_DEFINITION,
    TEMPLATE_AXIS_DIRECTION,
    build_interictal_template_field_record,
)

RANKDISP = ROOT / "results/interictal_propagation_masked/rank_displacement/per_subject"
TEMPLATE_RECORDS = (ROOT / "results/spatial_modulation/propagation_geometry/"
                    "observation_readout/real_subjects")
DEFAULT_OUT = ROOT / "results/interictal_propagation_masked/template_gradient_fields"
YUQUAN_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
EPILEPSIAE_ROOT = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")


def _subject_dir(dataset: str, subject: str) -> Path:
    return YUQUAN_ROOT / subject if dataset == "yuquan" else EPILEPSIAE_ROOT / subject / "all_recs"


def _seed(token: str, base: int = 0) -> int:
    return int((zlib.crc32(token.encode("utf-8")) + int(base)) % (2**32 - 1))


def _jsonable(x):
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(float(x)) else float(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    return x


def _failed_record(sid: str, stable_k, status: str, **extra) -> Dict[str, object]:
    dataset, subject = sid.split("_", 1)
    return {
        "contract": INTERICTAL_FIELD_CONTRACT,
        "subject_id": sid,
        "dataset": dataset,
        "subject": subject,
        "stable_k": stable_k,
        "template_labels": {"a": "TA", "b": "TB"},
        "axis_definition": TEMPLATE_AXIS_DEFINITION,
        "axis_direction_convention": TEMPLATE_AXIS_DIRECTION,
        "status": status,
        "direction_validity": {
            "ta": {"estimable": False, "reason_codes": [status]},
            "tb": {"estimable": False, "reason_codes": [status]},
            "pair": {"axis_pair_estimable": False, "geometry_2d_supported": False,
                     "strict_stability_pass": False},
        },
        "interictal_field": {"status": "axis_not_available"},
        **extra,
    }


def _load_rank_pair(sid: str) -> Dict[str, object]:
    path = RANKDISP / f"{sid}.json"
    if not path.exists():
        return _failed_record(sid, None, "missing_rank_displacement_source")
    data = json.loads(path.read_text())
    stable_k = data.get("stable_k")
    if stable_k != 2:
        return _failed_record(sid, stable_k, "stable_k_not_2",
                              source={"rank_displacement": str(path.relative_to(ROOT))})
    pairs = data.get("pairs") or []
    if not pairs:
        return _failed_record(sid, stable_k, "missing_template_pair",
                              source={"rank_displacement": str(path.relative_to(ROOT))})
    pair = pairs[0]
    names = list(pair.get("channel_names") or [])
    joint = np.asarray(pair.get("joint_valid"), bool)
    rank_a = np.asarray(pair.get("rank_a_dense_full"), float)
    rank_b = np.asarray(pair.get("rank_b_dense_full"), float)
    if not (len(names) == len(joint) == len(rank_a) == len(rank_b)):
        return _failed_record(sid, stable_k, "rank_displacement_shape_mismatch",
                              source={"rank_displacement": str(path.relative_to(ROOT))})
    dataset, subject = sid.split("_", 1)
    names_joint = [names[i] for i in np.where(joint)[0]]
    try:
        coord_record = load_subject_coords(dataset, subject, names_joint)
    except Exception as exc:
        return _failed_record(
            sid, stable_k, "coordinate_load_failed", error=str(exc)[:200],
            source={"rank_displacement": str(path.relative_to(ROOT))},
        )
    coords = np.asarray(coord_record.coords_array_in_requested_order, float)
    mapped = np.asarray(coord_record.mapped_mask_in_requested_order, bool)
    ra, rb = rank_a[joint], rank_b[joint]
    valid = mapped & np.isfinite(coords).all(1) & np.isfinite(ra) & np.isfinite(rb)
    if int(valid.sum()) < 6:
        return _failed_record(
            sid, stable_k, "insufficient_joint_mapped", n_joint_mapped=int(valid.sum()),
            source={"rank_displacement": str(path.relative_to(ROOT))},
        )
    names_used = [names_joint[i] for i in np.where(valid)[0]]
    return {
        "status": "source_ready",
        "subject_id": sid,
        "dataset": dataset,
        "subject": subject,
        "stable_k": stable_k,
        "names": names_used,
        "coords": coords[valid],
        "rank_a": ra[valid],
        "rank_b": rb[valid],
        "shafts": [parse_shaft(name)[0] for name in names_used],
        "source": {
            "rank_displacement": str(path.relative_to(ROOT)),
            "pair_index": 0,
            "template_pair_source": "masked stable-k=2 TA/TB joint-valid rank fields",
        },
    }


def _load_support(source: Mapping[str, object]) -> Dict[str, object]:
    dataset, subject = str(source["dataset"]), str(source["subject"])
    names = [str(x) for x in source["names"]]
    rank_a_by_name = dict(zip(names, np.asarray(source["rank_a"], float)))
    rank_b_by_name = dict(zip(names, np.asarray(source["rank_b"], float)))
    pa = TEMPLATE_RECORDS / f"{dataset}_{subject}_t_a.json"
    pb = TEMPLATE_RECORDS / f"{dataset}_{subject}_t_b.json"
    if pa.exists() and pb.exists():
        da, db = json.loads(pa.read_text()), json.loads(pb.read_text())
        cha, chb = da.get("channels") or [], db.get("channels") or []
        if cha and chb:
            sa = {str(c["name"]): float(c["support"]) for c in cha}
            sb = {str(c["name"]): float(c["support"]) for c in chb}
            return {
                "a": np.asarray([sa.get(name, 0.0) for name in names], float),
                "b": np.asarray([sb.get(name, 0.0) for name in names], float),
                "source": "canonical_template_participation_records",
                "n_events": {"a": None, "b": None, "unassigned": None},
            }

    events = load_subject_propagation_events(_subject_dir(dataset, subject))
    bools = np.asarray(events["bools"], bool)
    if bools.ndim != 2 or bools.shape[1] == 0:
        raise ValueError("empty interictal event participation matrix")
    event_names = [str(x) for x in events["channel_names"]]
    ranks = np.asarray(events["ranks"], float)
    masked = mask_phantom_ranks(ranks, bools, normalize=True)
    ta = np.asarray([rank_a_by_name.get(name, np.nan) for name in event_names], float)
    tb = np.asarray([rank_b_by_name.get(name, np.nan) for name in event_names], float)
    labels = assign_events_to_templates(masked, ta, tb)

    def support_for(label: int) -> Dict[str, float]:
        selected = labels == label
        values = bools[:, selected].mean(axis=1) if np.any(selected) else np.zeros(len(event_names))
        return {name: float(value) for name, value in zip(event_names, values)}

    sa, sb = support_for(0), support_for(1)
    return {
        "a": np.asarray([sa.get(name, 0.0) for name in names], float),
        "b": np.asarray([sb.get(name, 0.0) for name in names], float),
        "source": "recomputed_from_masked_events",
        "n_events": {
            "a": int(np.sum(labels == 0)),
            "b": int(np.sum(labels == 1)),
            "unassigned": int(np.sum(labels < 0)),
        },
    }


def build_subject(sid: str) -> Dict[str, object]:
    source = _load_rank_pair(sid)
    if source.get("status") != "source_ready":
        return source
    try:
        support = _load_support(source)
    except Exception as exc:
        support = {
            "a": np.zeros(len(source["names"]), float),
            "b": np.zeros(len(source["names"]), float),
            "source": f"support_load_failed:{str(exc)[:160]}",
            "n_events": {},
        }
    record = build_interictal_template_field_record(
        subject_id=sid,
        dataset=str(source["dataset"]),
        subject=str(source["subject"]),
        stable_k=int(source["stable_k"]),
        names=source["names"],
        coords=np.asarray(source["coords"], float),
        rank_ta=np.asarray(source["rank_a"], float),
        rank_tb=np.asarray(source["rank_b"], float),
        shafts=source["shafts"],
        support_ta=np.asarray(support["a"], float),
        support_tb=np.asarray(support["b"], float),
        support_source=str(support["source"]),
        template_event_counts=support.get("n_events"),
        n_axis_boot=200,
        n_pair_boot=500,
        line_threshold=0.50,
        seed=_seed(sid, 17),
    )
    record["source"] = source["source"]
    return record


def _cohort_row(record: Mapping[str, object]) -> Dict[str, object]:
    row = {key: record.get(key) for key in ("subject_id", "dataset", "subject", "stable_k", "status")}
    field = record.get("interictal_field") or {}
    row.update({
        "interictal_field_status": field.get("status"),
        "n_field_contacts": field.get("n_contacts"),
        "support_source": record.get("support_source"),
    })
    pair = record.get("axis_pair") or {}
    if pair.get("status") != "ok":
        return row
    ta, tb = pair["axis_a"], pair["axis_b"]
    relation, boot = pair["relation"], pair["pair_bootstrap"]
    row.update({
        "axis_pair_estimable": pair.get("axis_pair_estimable"),
        "geometry_2d_supported": pair.get("geometry_2d_supported"),
        "strict_stability_pass": pair.get("strict_stability_pass"),
        "ta_n": ta.get("n"), "tb_n": tb.get("n"),
        "ta_n_shafts": ta.get("n_shafts"), "tb_n_shafts": tb.get("n_shafts"),
        "ta_effective_rank": ta.get("effective_rank"), "tb_effective_rank": tb.get("effective_rank"),
        "ta_R2": ta.get("R2"), "tb_R2": tb.get("R2"),
        "ta_bootstrap_cosine": ta.get("bootstrap_cosine"),
        "tb_bootstrap_cosine": tb.get("bootstrap_cosine"),
        "ta_loso_cosine": ta.get("loso_cosine"), "tb_loso_cosine": tb.get("loso_cosine"),
        "cos_ta_tb": relation.get("cosine"), "abs_cos_ta_tb": relation.get("abs_cosine"),
        "line_angle_deg": relation.get("line_angle_deg"),
        "collinear_60deg": relation.get("collinear"), "relation": relation.get("relation"),
        "pair_boot_p_collinear": boot.get("p_collinear"),
        "pair_boot_p_sign_stable": boot.get("p_sign_stable"),
        "robust_collinear": boot.get("robust_collinear"),
        "shared_field_available": "shared_a" in (field.get("field_models") or {}),
    })
    return row


def _summary(records: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    axes = [r for r in records if (r.get("axis_pair") or {}).get("status") == "ok"]
    geometry = [r for r in axes if r["axis_pair"].get("geometry_2d_supported")]
    strict = [r for r in axes if r["axis_pair"].get("strict_stability_pass")]
    fields = [r for r in axes if (r.get("interictal_field") or {}).get("status") == "ok"]
    shared = [r for r in fields if "shared_a" in r["interictal_field"].get("field_models", {})]

    def counts(items):
        return {name: sum(r["axis_pair"]["relation"]["relation"] == name for r in items)
                for name in ("same", "reversed", "different")}

    return {
        "contract": INTERICTAL_FIELD_CONTRACT,
        "axis_definition": TEMPLATE_AXIS_DEFINITION,
        "axis_direction_convention": TEMPLATE_AXIS_DIRECTION,
        "ictal_independence": "no seizure/onset/subtype/energy input is read by this producer",
        "denominators": {
            "template_pair_inputs": len(records),
            "axis_pair_estimable": len(axes),
            "geometry_2d_supported": len(geometry),
            "strict_stability_pass": len(strict),
            "own_ta_tb_fields_ready": len(fields),
            "shared_fields_ready": len(shared),
        },
        "axis_relation_counts_all_estimable": counts(axes),
        "axis_relation_counts_2d_geometry": counts(geometry),
        "axis_relation_counts_strict_stability": counts(strict),
        "reuse_contract": {
            "canonical_subject_artifact": "per_subject/<dataset>_<subject>.json",
            "activation_join": "exact channel-name join to interictal_field.contact_order",
            "frozen_before_ictal": ["TA/TB axes", "axis validity", "plane", "sigma", "support",
                                    "template field", "kernel weights"],
        },
    }


README = """# Template gradient fields

这里保存纯间期、患者特异的 TA/TB 传播轴和二维 field。producer 不读取任何发作、onset、
subtype 或能量数据，因此后续发作定义变化时必须复用这里的冻结 artifact，而不是重新建轴。

- `per_subject/<dataset>_<subject>.json`：canonical artifact。`axis_pair.axis_a/axis_b.u`
  的正方向固定为 early→late；`direction_validity` 分开记录可估性、二维几何和 strict stability。
- `interictal_field.field_models.own_a/own_b`：TA/TB 各自平面的固定 contact field、support、
  bandwidth 与 kernel weights。共线患者另有 `shared_a/shared_b`。
- `interictal_field.fingerprint_sha256`：冻结轴、平面、support、field 与 kernel weights 的确定性指纹；
  下游加载时自动校验，artifact 漂移会 fail closed。
- `axis_cohort.csv`：全 cohort 的方向质量和 TA/TB 线夹角。
- `cohort_summary.json`：分母和关系分布。

下游必须把新的 activation vector 按 `interictal_field.contact_order` 做 exact channel-name join；
不得使用发作能量重新拟合轴、平面、bandwidth 或 template field。单杆患者可以有 early→late
方向，但 `geometry_2d_supported=false`，不能解释为有效二维 field 几何。

最小复用 API：

```python
record = json.loads(subject_artifact.read_text())
scorers = scorers_from_interictal_record(record)  # 自动校验 fingerprint
aligned = align_activation_to_interictal_field(record, activation_names, activation_values)
scores = score_scorer_bundle(scorers, aligned["values"])
```

`own_a/own_b` 分别对应 TA/TB；仅共线患者存在 `shared_a/shared_b`。下游应先检查
`aligned["n_finite"] >= 6`，并在每次 null permutation 中重新计算 maxAB。
"""


def run(subjects: Sequence[str], out_dir: Path) -> Dict[str, object]:
    per_subject = out_dir / "per_subject"
    per_subject.mkdir(parents=True, exist_ok=True)
    records = []
    rows = []
    for index, sid in enumerate(subjects, 1):
        record = build_subject(sid)
        records.append(record)
        rows.append(_cohort_row(record))
        (per_subject / f"{sid}.json").write_text(
            json.dumps(_jsonable(record), ensure_ascii=False, indent=2)
        )
        pair = record.get("axis_pair") or {}
        relation = (pair.get("relation") or {}).get("relation", "-")
        print(
            f"[{index:02d}/{len(subjects)}] {sid}: axis={record.get('status')} "
            f"strict={pair.get('strict_stability_pass')} relation={relation} "
            f"field={(record.get('interictal_field') or {}).get('status')}",
            flush=True,
        )
    columns = sorted({key for row in rows for key in row})
    with (out_dir / "axis_cohort.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(_jsonable(rows))
    summary = _summary(records)
    (out_dir / "cohort_summary.json").write_text(
        json.dumps(_jsonable(summary), ensure_ascii=False, indent=2)
    )
    (out_dir / "README.md").write_text(README)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="*", default=None, help="dataset_subject tokens")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    subjects = args.subjects or sorted(path.stem for path in RANKDISP.glob("*.json"))
    summary = run(subjects, args.out_dir)
    print(json.dumps(summary["denominators"], ensure_ascii=False, indent=2))
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
