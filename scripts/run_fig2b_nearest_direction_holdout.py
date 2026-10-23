#!/usr/bin/env python3
"""Re-score frozen holdouts by the nearest 3D template direction, without rank gating."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zlib

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_fig2b_two_target_holdout import OUT as SOURCE, METRICS, SPLITS
from scripts.run_fig2b_rank_reconstruction_holdout import FROZEN, write_json, paired_summary, stars, jsonable
from src.interictal_rank_holdout import METHODS
from src.interictal_nearest_direction_holdout import nearest_direction_errors, shuffled_template_axes, shuffled_direction_errors
from src.interictal_two_target_holdout import holm_adjust
from src.topic5_interictal_direction_rose import fit_event_directions_3d
from src.seeg_coord_loader import load_subject_coords

OUT = ROOT/'results/topic1/nearest_direction_holdout_20261023'
SEED = 20261023
NULL_DESCRIPTION = 'training-template rank-contact permutation, fixed patient geometry, refitted axes and nearest-angle reassignment'


def run_patient(sid, out, n_draws):
    with threadpool_limits(limits=1):
        return _run_patient(sid, Path(out), n_draws)


def _run_patient(sid, out, n_draws):
    source_path = SOURCE/'per_subject'/f'{sid}.json'
    source = json.loads(source_path.read_text())
    frozen = json.loads((FROZEN/'per_subject'/f'{sid}.json').read_text())
    dataset, subject = sid.split('_', 1)
    coord_result = load_subject_coords(dataset, subject, frozen['channel_names'])
    coords = np.asarray(coord_result.coords_array_in_requested_order)
    record = copy.deepcopy(source)
    record.pop('diagnostics', None)
    record.update(direction_assignment='nearest_3d_angle', n_direction_null_draws=n_draws,
                  source_record_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                  direction_null_description=NULL_DESCRIPTION,
                  coordinate_space=coord_result.coord_space, coordinate_units=coord_result.coord_units,
                  previous_rank_assigned_direction={m:source['combined'][m]['angular_mae_deg'] for m in METHODS})
    arrays = dict(coords=coords)
    with np.load(SOURCE/'per_subject'/f'{sid}_heldout.npz') as cache:
        for f, row in enumerate(record['folds']):
            assert not set(row['train_blocks']) & set(row['test_blocks'])
            # The source cache covered every test sequence. The new direction
            # readout imposes no rank-template selection or rank-support gate.
            assert cache[f'f{f}_rank_eligible'].all(), 'source cache excludes test sequences'
            finite = cache[f'f{f}_direction_valid']
            directions = cache[f'f{f}_test_directions']
            test_indices = cache[f'f{f}_test_indices'][finite]
            assert len(directions) == finite.sum() == row['n_direction']
            axes = np.array([row['models'][m]['axes'] for m in METHODS])
            templates = np.array([row['models'][m]['templates'] for m in METHODS])
            refitted = fit_event_directions_3d(templates.reshape(4, -1).T, coords)['directions'].reshape(2, 2, 3)
            np.testing.assert_allclose(refitted, axes, atol=1e-12)
            evaluated = nearest_direction_errors(directions, axes)
            row.pop('diagnostics', None)
            row['previous_rank_assigned_direction'] = {m:row['metrics'][m]['angular_mae_deg'] for m in METHODS}
            row['assignment_changed_fraction'] = {}
            row['nearest_direction_counts'] = {}
            for i, m in enumerate(METHODS):
                old_errors = cache[f'f{f}_{m}_angular_mae_deg']
                assert np.all(evaluated['angular_mae_deg'][i] <= old_errors+1e-8)
                for metric in ('angular_mae_deg', 'cosine_loss'):
                    row['metrics'][m][metric] = float(evaluated[metric][i].mean())
                    arrays[f'f{f}_{m}_{metric}'] = evaluated[metric][i]
                arrays[f'f{f}_{m}_assignments'] = evaluated['assignments'][i]
                row['assignment_changed_fraction'][m] = float(np.mean(evaluated['assignments'][i] != cache[f'f{f}_{m}_assignments'][finite]))
                row['nearest_direction_counts'][m] = np.bincount(evaluated['assignments'][i], minlength=2)
            arrays[f'f{f}_test_indices'] = test_indices
            arrays[f'f{f}_test_directions'] = directions
            if f < 2:
                null_axes, info = shuffled_template_axes(templates, coords, n_draws, SEED+f*10000+zlib.crc32(sid.encode()))
                null = shuffled_direction_errors(directions, null_axes)
                row['direction_null_draws'] = {m:null[i] for i, m in enumerate(METHODS)}
                row['direction_null_info'] = info
                arrays[f'f{f}_null_axes'] = null_axes
    for m in METHODS:
        for metric in ('angular_mae_deg', 'cosine_loss'):
            record['combined'][m][metric] = float(np.mean([row['metrics'][m][metric] for row in record['folds'][:2]]))
        record['null'][m]['angular_mae_deg'] = float(np.mean([row['direction_null_draws'][m] for row in record['folds'][:2]]))
    out.joinpath('per_subject').mkdir(exist_ok=True)
    np.savez_compressed(out/'per_subject'/f'{sid}_heldout.npz', **arrays)
    write_json(out/'per_subject'/f'{sid}.json', record)
    print(sid+' nearest 3D: '+' / '.join(f'{record["combined"][m]["angular_mae_deg"]:.3f}' for m in METHODS)+
          '; null: '+' / '.join(f'{record["null"][m]["angular_mae_deg"]:.3f}' for m in METHODS), flush=True)
    return record


def summarize(records, subjects, out):
    records.sort(key=lambda r:subjects.index(r['subject_id']))
    audit = []
    for f, split in [(-1, 'alternating_combined')]+list(enumerate(SPLITS)):
        for metric in METRICS:
            values = {m:[r['combined'][m][metric] if f < 0 else r['folds'][f]['metrics'][m][metric] for r in records] for m in METHODS}
            audit.append(dict(split=split, metric=metric, **paired_summary(values[METHODS[0]], values[METHODS[1]])))
    for row, p in zip(audit, holm_adjust([r['wilcoxon_two_sided_p'] for r in audit])):
        row.update(holm_35_p=float(p), stars=stars(p))
    primary = {}
    for target in ('rank_mae', 'angular_mae_deg'):
        comparisons = {'sequence_vs_trajectory':paired_summary([r['combined'][METHODS[0]][target] for r in records],
                                                               [r['combined'][METHODS[1]][target] for r in records])}
        for m in METHODS:
            comparisons[f'{m}_null_vs_real'] = paired_summary([r['null'][m][target] for r in records], [r['combined'][m][target] for r in records])
        primary[target] = dict(comparisons=comparisons)
    family = [row for item in primary.values() for row in item['comparisons'].values()]
    for row, p in zip(family, holm_adjust([r['wilcoxon_two_sided_p'] for r in family])):
        row.update(holm_p=float(p), stars=stars(p))
    summary = dict(status='complete', n_requested=len(subjects), n_complete=len(records), failures=[],
                   n_rank=sum(r['n_rank'] for r in records), n_direction=sum(r['n_direction'] for r in records),
                   n_events=sum(r['n_events'] for r in records), primary=primary, audit=audit,
                   direction_assignment='nearest_3d_angle', direction_null_description=NULL_DESCRIPTION,
                   main_family_size=6, audit_family_size=35, statistical_unit='patient',
                   new_clustering_fits=0, frozen_model_folds=len(records)*6,
                   rejected_null_draws=sum(f['direction_null_info']['rejected_degenerate_draws'] for r in records for f in r['folds'][:2]),
                   followup_status='EXPLORATORY_AUTHOR_CORRECTED_DIRECTION_TARGET',
                   human_visual_acceptance='PENDING_AUTHOR_VISUAL_REVIEW')
    write_json(out/'summary.json', summary)
    for filename, rows in [('metric_holdout_comparisons.csv', audit), ('patients.csv', [dict(
        subject_id=r['subject_id'], n_direction=r['n_direction'],
        **{f'{m}_{k}':v for m, d in r['combined'].items() for k, v in d.items()},
        **{f'{m}_null_direction':r['null'][m]['angular_mae_deg'] for m in METHODS},
        **{f'{m}_old_rank_selected_direction':r['previous_rank_assigned_direction'][m] for m in METHODS}) for r in records])]:
        with (out/filename).open('w', newline='') as handle:
            writer=csv.DictWriter(handle, fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    result = primary['angular_mae_deg']['comparisons']['sequence_vs_trajectory']
    (out/'README.md').write_text(
        '# Figure 2B：留出三维方向最近模板误差\n\n'
        '2026-10-23按作者确认，方向任务的分配与评分均改为三维方向：每个测试事件取到TA/TB两条冻结训练方向的最小有向夹角，单位为度。不再使用rank MAE或旧Matching Index选择模板。每折对事件取均值，再对互补两折等权；统计单位为患者。\n\n'
        '沿用两种方法原有训练结果：仅rank聚类与rank+空间方向聚类，每种各两个模板；本次未重训、未依据测试方向转动模板。两折共26人，'+str(summary['n_direction'])+'个方向可估计事件，完整复用六个训练/测试划分，全部测试序列在旧缓存中均有覆盖。参考事件方向仍由完整rank与触点坐标拟合，目标是方向覆盖/重建，不是早期预测或独立测量的传播真值。\n\n'
        '## 新null\n\n'
        '固定训练模板及患者电极坐标，在每个模板的有限rank与已定位触点支持内置换rank值，重拟合3D梯度轴；相同随机键跨四个模板共享，相同支持时使用完全相同的通道置换。保持模板rank值集合、参与支持、真实测试方向和电极几何；破坏rank与空间位置的对应。每次置换后重新按最小三维夹角选择模板，200次置换只用于两个主分析折。该null是模板空间对应对照，不是重新聚类的空间信息消融，也不保持两个模板的夹角。原事件—模板配对置换对于最小夹角任务不成立，未沿用。\n\n'
        '## 结果\n\n'
        f'患者中位误差：Rank SEQ {result["a_median"]:.4f}°，Spatial TRAJ. {result["b_median"]:.4f}°；{result["n_b_lower"]}/{result["n"]}人空间模板误差较低。双侧患者配对Wilcoxon原始P={result["wilcoxon_two_sided_p"]:.8g}，两张图六比较Holm P={result["holm_p"]:.8g}。\n\n'
        'Rank版的MAE和rank-channel shuffle结果原样复用；主图六项比较重新一起Holm校正。五种误差×七种读出完整保存在metric_holdout_comparisons.csv，35项统一Holm；不同读出不作为独立重复。旧按rank选模板后算方向的结果保留在`../rank_direction_holdout_20261022/`。\n\n'
        '## 图与复现\n\n'
        '候选图：`../../paper-ready-figure/fig2/candidates/nearest_direction_holdout_20261023/figures/`。主图右侧灰柱仍为第一折真实方向的二维投影，标为Fold-1 projection；红蓝线为Spatial TRAJ.两个训练模板，不能将二维rose角距离当作三维误差。另附E10/E14的两折真实3D误差分布，其等权混合均值精确复现左侧对应患者的点。\n\n'
        '```bash\npython scripts/run_fig2b_nearest_direction_holdout.py --workers 2 --n-null 200\n'
        'PYTHONPATH=/tmp/hfosp_fig5_pdf_tools python scripts/paper_figures/plot_fig2b_nearest_direction_holdout.py --dpi 600\n'
        'PYTHONPATH=/tmp/hfosp_fig5_pdf_tools python scripts/paper_figures/validate_fig2b_nearest_direction_holdout.py\n```\n')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--n-null', type=int, default=200)
    parser.add_argument('--subjects', nargs='+')
    args=parser.parse_args();args.output.mkdir(parents=True, exist_ok=True)
    subjects=args.subjects or json.loads((SOURCE/'run_manifest.json').read_text())['subjects']
    snap=args.output/'source_snapshot';snap.mkdir(exist_ok=True)
    sources=[Path(__file__), ROOT/'src/interictal_nearest_direction_holdout.py', ROOT/'docs/fig2b_nearest_direction_holdout.md']
    for path in sources:shutil.copy2(path, snap/path.name)
    write_json(args.output/'run_manifest.json', dict(subjects=subjects, seed=SEED, n_null_draws=args.n_null,
        analysis_parent=str(SOURCE), frozen_training_root=str(FROZEN), main_direction_assignment='nearest_3d_angle',
        direction_null_description=NULL_DESCRIPTION, rose_event_label='Fold-1 projection',
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}))
    records=[];failures=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(run_patient, sid, str(args.output), args.n_null):sid for sid in subjects}
        for future in as_completed(jobs):
            try:records.append(future.result())
            except Exception as exc:
                failures.append(dict(subject=jobs[future], error=repr(exc)));print('FAILED', jobs[future], repr(exc), flush=True)
    if failures:
        write_json(args.output/'summary.json', dict(status='partial', n_complete=len(records), failures=failures))
        raise RuntimeError(f'{len(failures)} patients failed')
    result=summarize(records, subjects, args.output)
    print(json.dumps(jsonable(result['primary']['angular_mae_deg']), indent=2), flush=True)


if __name__=='__main__':main()
