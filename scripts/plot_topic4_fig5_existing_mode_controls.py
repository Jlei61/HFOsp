#!/usr/bin/env python3
"""Render existing native M trajectories as explicitly historical Fig5 controls."""
from pathlib import Path
import json
import numpy as np
import plot_topic4_m_parameter_modes as fig5
from run_topic4_m_parameter_modes import tracker_step, fresh_tracker

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/topic4_sef_hfo/fig5_manual_core_release_v1/m_runaway_return_v1/runs'
OUT = ROOT / 'results/topic4_sef_hfo/fig5_m_overnight_exploration_20260913/existing_mode_figures'


def convert(name):
    path = SOURCE / (name + '.npz')
    source_meta = json.loads(path.with_suffix('.json').read_text())
    with np.load(path) as f, np.load(fig5.OUT / 'geometry.npz') as geometry:
        assert source_meta['frozen_identity'] == fig5.read(fig5.OUT / 'protocol.json')['identity']
        a = dict(time_ms=np.arange(len(f['field_e_count_1ms'])) + .5,
            spikes_1ms=np.rint(np.c_[f['rate_e_hz'].reshape(-1,10).mean(1)*32,
                                     f['rate_i_hz'].reshape(-1,10).mean(1)*8]).astype(np.uint16),
            regions_1ms=f['region_spikes_1ms'], field_1ms=f['field_e_count_1ms'],
            raster=f['sample_spikes'][:, geometry['sample_source_indices']],
            slow_time_ms=f['z_time_ms'], Z=f['z_stats'][:, :9], M=f['m_stats'][:, [0,5,6,7]],
            currents=f['currents_5ms'][:, :3], lfp_time_ms=f['lfp_time_ms'], lfp_raw=f['lfp_raw'])
        for key in ['centers_mm', 'contact_xy', 'contact_names', 'positions_e', 'cell_e_counts', 'region_counts']:
            assert np.array_equal(f[key], geometry[key]), key
        assert np.array_equal(f['sample_ids'][geometry['sample_source_indices']], geometry['sample_ids'])
        a.update({k: geometry[k] for k in geometry.files})
    rate = a['spikes_1ms'].reshape(-1, 10, 2).sum(1)[:, 0] / 32000 / .01
    tr = fresh_tracker()
    for i, value in enumerate(rate): tracker_step(tr, value, (i+1)*.01, rescue=False)
    assert tr['restore_s'] is None
    r = dict(job=dict(name='existing_' + name, eta_m=source_meta['eta_m'],
        tau_M_s=source_meta['tau_adp_ms']/1000, seed=source_meta['job']['seed']), tracker=tr)
    return a, r, path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    g = fig5.grid_summary(); records = []
    for name in ['weak_fast', 'weak_20s']:
        a, r, source = convert(name)
        m = fig5.analyze(a, r)
        dest = OUT / name
        fig5.render(a, r, m, dest / 'figures', g)
        lineage = dict(source=str(source), source_metadata=str(source.with_suffix('.json')),
            reused_existing_trajectory=True, new_batch_result=False, duration_s=m['duration_s'],
            native_only=True, external_Z_refill=False, source_identity_and_geometry_checked=True,
            tracker_recomputed_from_saved_native_spikes=True, mode=m['mode'],
            full_1_to_5_observed=m['full_1_to_5_observed'], independent_new_replicate=False,
            F_source='New forty-job M grid only; historical 90-s censoring not counted as 180-s censoring',
            E2_patient_reference='Frozen Fig3C E1146/SZ3', agent_visual_review='PENDING', human_review='PENDING')
        fig5.write(dest / 'source.json', lineage)
        records.append(dict(name=name, **lineage, figure=str(dest / 'figures/fig5.png')))
        del a
    fig5.write(OUT / 'index.json', records)
    lines = ['# 完整Fig5：已有原生M条件作为对照', '',
        '这些图使用此前已完成的原生90秒轨迹，不是今晚40条新搜索的结果，不充当新重复样本。两条都未做外部Z补充。', '',
        '| 条件 | 参数 | 实際观察模式 | 完整图 |', '|---|---|---|---|']
    for row in records:
        tau = 2 if row['name'] == 'weak_fast' else 20
        lines.append(f'| {row["name"]} | ηM=0.02，τM={tau}s | {row["mode"]} | [Fig5]({row["figure"]}) |')
    lines += ['', 'F仅引用新40条扫描当前已完整结果，不将旧90秒未进入当成180秒未进入。正式患者Fig3C保持原样；原生场或触点功率为负则明确画为下降，不把高率等同于功率增强。', '',
        '所有未实际出现的②–⑤保留空缺。不同模式的完整布局不代表完整①–⑤过程已经实现；当前仍需用户目视及科学验收。']
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps([{k:r[k] for k in ['name','mode','duration_s','figure']} for r in records]))


if __name__ == '__main__': main()
