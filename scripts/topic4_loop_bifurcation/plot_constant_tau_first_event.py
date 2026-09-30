#!/usr/bin/env python3
"""An unchanged-observer spatial look at the first recovered constant-tau event."""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
import shutil
import numpy as np
from campaign import ROOT, read, write, sha
from analyze_natural_exit_mediators import DEST, OUT as PROBES, CONTROLS, stream
from analyze_feedback_tail import load, SOURCE
import zoom_topic4_return_core_propagation as zoom

OUT = ROOT/'constant_tau_first_return_spatial'


def main():
    OUT.mkdir(exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
    result = read(DEST/'result.json');row = result['rows'][2]
    events = row['complete_brief_after_Z_reference'];assert len(events) == 1
    event = events[0];folder = PROBES/'runs'/CONTROLS[1]
    geo = dict(np.load(PROBES/'geometry.npz'));counts = np.r_[32000, geo['region_counts'][:3]]
    data = stream(folder, 'chunks', 'time_ms', ['spikes_1ms', 'regions_1ms'])
    field = stream(folder, 'chunks', 'field_time_ms', ['field_5ms'])
    raw = np.c_[data['spikes_1ms'][:, 0], data['regions_1ms'][:, :3]]
    rates = raw.reshape(-1, 5, 4).sum(1)/counts/.005
    assert np.array_equal(field['field_5ms'].sum(1), raw[:, 0].reshape(-1, 5).sum(1))
    new = dict(rates=rates, field_rates=field['field_5ms']/geo['cell_e_counts']/.005)
    metrics = zoom.event_metrics(new, event)
    # Use the unchanged original reference example, not a best-looking matched event.
    with np.load(SOURCE/'references/native_s9108405.npz') as reference:
        raw0 = np.c_[reference['spikes_1ms'][:, 0], reference['regions_1ms'][:, :3]]
        old = dict(rates=raw0.reshape(-1, 5, 4).sum(1)/counts/.005,
                   field_rates=reference['field_5ms']/geo['cell_e_counts']/.005)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(16, 6));grid = fig.add_gridspec(2, 8, width_ratios=[1.5, 1.5, 1, 1, 1, 1, 1, 1],
        left=.065, right=.925, bottom=.10, top=.88, wspace=.35, hspace=.7)
    examples = [('Original interictal reference', old, .57, .57),
                ('First brief after core Z recovery', new, event['start_s'], event['start_s']+16.8)]
    arrays = {}
    for i, (label, d, start, absolute) in enumerate(examples):
        a = fig.add_subplot(grid[i, :2]);zoom.trace(a, d, start, start+.15, relative=True)
        a.set_title(label+f'\nt = {absolute:.3f} s', fontsize=11)
        if i == 0:a.legend(frameon=False, fontsize=8, ncol=2, loc='upper right')
        else:a.set_xlabel('Time from window start (ms)')
        frames = []
        for j, offset in enumerate([0, 25, 50, 75, 100, 125]):
            a = fig.add_subplot(grid[i, j+2]);v = d['field_rates'][round(start*200)+offset//5]
            im = zoom.field(a, v, geo);frames.append(v)
            a.set_title(f'+{offset}–{offset+5} ms', fontsize=9)
            if j:a.set_yticklabels([])
            else:a.set_ylabel('y (mm)')
            if i == 1:a.set_xlabel('x (mm)')
        arrays[f'row{i}_field_frames'] = np.array(frames)
    fig.colorbar(im, cax=fig.add_axes([.94, .23, .013, .51]), label='Native E rate (Hz; 5 ms)')
    fig.suptitle('Constant 0.5 s K decay: one recovered brief event, same native spatial readout', fontsize=13)
    for ext in ['png', 'svg']:fig.savefig(ROOT/f'figures/constant_tau_first_return_spatial.{ext}', dpi=180)
    plt.close(fig)
    np.savez_compressed(OUT/'display_data.npz', **arrays)
    write(OUT/'result.json', dict(status='COMPLETE_ONE_EVENT_SPATIAL_CANDIDATE', original_reference_start_s=.57,
        event_absolute_start_s=event['start_s']+16.8, event_duration_s=event['duration_s'],
        both_core_Z_reference_s=row['first_both_core_Z_reference_s'], metrics=metrics,
        scope='One complete brief after recovery; population-threshold order is descriptive, not causal source localization. Sustained return and propagation fidelity require the registered continuation and visual review.',
        producer_sha256=sha(__file__), agent_visual_review='PENDING', human_visual_review='PENDING',
        counts_as_independent_seed=False, counts_as_autonomous_loop=False))
    shutil.copy2(__file__, OUT/'producer.py')
    title = '### constant_tau_first_return_spatial.png / constant_tau_first_return_spatial.svg'
    path = ROOT/'figures/README.md'
    if title not in path.read_text():
        with path.open('a') as f:f.write('\n\n'+title+'\n采用原有空间放大模式，对比原间期固定示例与恒定0.5秒K衰减候选在两核Z达到参考后的第一个完整短事件。全部为空间1mm、时间5ms原生计数，保留共同0–500Hz色标及两核位置。\n**关注点**：观察起始位置、核参与及外传；单个事件不能替代持续间期返回，人工待审。\n')
    print(metrics, flush=True)


if __name__ == '__main__':main()
