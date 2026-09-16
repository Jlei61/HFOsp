#!/usr/bin/env python3
"""One matched E-only clamp + native OU trajectory, then automatic readout."""
from run_topic4_snn_raster_transition import OUT, ROOT, read, write
from plot_topic4_snn_raster_transition import analyze, single
from pathlib import Path
import hashlib
import subprocess
import sys
import time
import numpy as np


def main():
    arm = 'z_current_e_ou'
    status = OUT / 'e_only_ou_followup_status.json'
    started = time.time()
    write(status, {'status': 'RUNNING', 'arm': arm, 'started_unix': started})
    try:
        subprocess.run([sys.executable, str(ROOT / 'scripts/run_topic4_snn_raster_transition.py'),
                        '--arm', arm], check=True)
        path = OUT / 'runs' / f'{arm}_seed9108401.npz'
        meta = read(path.with_suffix('.json'))
        data = np.load(path)
        baseline = np.load(OUT / 'runs/z_current_e_seed9108401.npz')
        ou_reference = np.load(OUT / 'runs/jump_ou_seed9108401.npz')
        reference_meta = read(OUT / 'runs/z_current_e_seed9108401.json')
        assert meta['frozen_identity'] == reference_meta['frozen_identity']
        assert np.array_equal(data['q_1ms'], baseline['q_1ms'])
        assert np.array_equal(data['sample_ids'], baseline['sample_ids'])
        prefix = round(1000 / float(data['dt_ms']))
        prefix_checks = {key: bool(np.array_equal(data[key][:prefix], ou_reference[key][:prefix]))
                         for key in ['sample_spikes', 'rate_e_hz', 'rate_i_hz']}
        assert all(prefix_checks.values()), prefix_checks
        assert all(np.isfinite(data[key]).all() for key in data.files)
        counts = np.rint(data['rate_e_hz'] * 32000 * .0001).astype(np.int64).reshape(-1, 10).sum(1)
        assert np.array_equal(counts, data['field_e_count_1ms'].sum(1))
        assert np.array_equal(counts, data['region_spikes_1ms'].sum(1))
        summaries = {'OU_off': analyze(baseline, reference_meta), 'OU_on': analyze(data, meta)}
        write(OUT / 'e_only_ou_comparison.json', {
            'status': 'COMPLETE_PENDING_SCIENTIFIC_AND_VISUAL_REVIEW',
            'same_OU_input_prefix_as_jump_ou': prefix_checks,
            'summaries': summaries,
            'scope': 'Same fixed carrier, same uniform E-only q(t); native global and spatial OU off/on. No endogenous Z/M.',
            'limitations': 'One seed; do not infer autonomous recovery, all working points, or bifurcation class.'})
        single(arm)
        readme = OUT / 'figures/README.md'
        entry = '\n### raster_z_current_e_ou.png\n仅 E 群的 GABA 电流乘统一的外部 q(t)，恢复原全局及空间 OU；使用与无 OU 的 C 分支相同底物、参数周期及抽样神经元。真实 spike 与全群率由完整连续仿真记录，PDF 为同名版本。\n**关注点**：直接补足 C 的 OU 背景对照；Z/M 自主演化仍关闭，结果待科学与目视审阅。\n'
        if '### raster_z_current_e_ou.png' not in readme.read_text():
            with readme.open('a') as stream:
                stream.write(entry)
        write(status, {'status': 'COMPLETE_PENDING_SCIENTIFIC_AND_VISUAL_REVIEW',
                       'arm': arm, 'seconds': time.time() - started,
                       'analysis': str(OUT / 'e_only_ou_comparison.json'),
                       'figure': str(OUT / 'figures/raster_z_current_e_ou.png'),
                       'next_round': False})
    except Exception as exc:
        write(status, {'status': 'FAILED', 'error': repr(exc), 'next_round': False})
        raise


if __name__ == '__main__':
    main()
