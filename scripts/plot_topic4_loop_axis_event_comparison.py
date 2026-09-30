#!/usr/bin/env python3
"""Three matched graphs: chronological first tail event and native5ms fields.

Diagnostic companion to the conditional response map, not a new event detector
or a replacement for the accepted autonomous Figure5 layout.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import review_topic4_loop_native_states as view

ROOT = Path('/data/hfosp/topic4_sef_hfo/fig5_autonomous_loop_zk_20260924')
LABELS = {'current': 'Current axis', 'rotated': 'Rotated structure',
          'isotropic': 'Isotropic structure'}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def feedback_tail(folder, lo, hi):
    result = dict(window_s=[lo, hi])
    for stream, keys in [('feedback_chunks', ['G_raw', 'G_applied_mean', 'K_mean']),
                         ('mechanism_chunks', ['global_E_rate_Hz']),
                         ('global_response_chunks', ['q'])]:
        data = view.load(folder, ['time_ms'] + keys, subdir=stream)
        keep = (data['time_ms'] >= lo * 1000) & (data['time_ms'] < hi * 1000)
        for key in keys:
            values = data[key][keep]
            result[key] = dict(n=len(values), minimum=float(values.min()),
                               maximum=float(values.max()), mean=float(values.mean()))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--history', choices=['high', 'interictal'], required=True)
    args = parser.parse_args()
    name = f'z0.95_k0.02_{args.history}'
    roots = [('current', ROOT)] + [(c, ROOT / 'axis_controls/conditional_runs' / c)
                                   for c in ['rotated', 'isotropic']]
    rows = []; inputs = None; reference_geometry = None
    for condition, root in roots:
        folder = root / 'runs' / name
        result = read(folder / 'result.json')
        assert result['status'] == 'COMPLETE'
        job = result['job']; first = float(job['branch_start_s'])
        assert job['conditional_clamp'] and job['target_Z'] == .95 and job['target_K'] == .02
        review_path = root / 'spatial_review' / name / 'review.json'
        review = read(review_path)
        assert review['result_sha256'] == sha(folder / 'result.json')
        assert review['field_spike_integrity'] == 'PASS'
        group = review['groups'][0]
        assert group['label'] == 'conditional_tail' and group['relative_window_s'] == [20., 30.]
        example = group['example']; source_image = Path(example['file'])
        source_npz = source_image.with_name(source_image.stem + '_inputs.npz')
        with np.load(source_npz) as f:
            fields = f['field_counts']; frame_times = f['field_start_s']
            cell_counts = f['cell_e_counts']
        assert fields.shape == (6, 400)
        reference = example['frame_reference_s']
        assert np.allclose(frame_times - reference, np.arange(6) * .025, rtol=0, atol=1e-12)
        with np.load(root / 'geometry.npz') as f:
            geometry = {key: f[key] for key in ['centers_mm', 'core_radius_mm', 'region_counts', 'cell_e_counts']}
        assert np.array_equal(cell_counts, geometry['cell_e_counts']) and np.all(cell_counts > 0)
        if reference_geometry is not None:
            for key in geometry:
                assert np.array_equal(geometry[key], reference_geometry[key]), key
        reference_geometry = geometry
        data = view.load(folder, ['spikes_1ms', 'regions_1ms', 'inputs'], first_step=round(first * 10000))
        assert len(data['spikes_1ms']) == 30000
        assert np.array_equal(data['regions_1ms'][:, :3].sum(1), data['spikes_1ms'][:, 0])
        if inputs is not None:
            assert np.array_equal(data['inputs'], inputs)
        inputs = data['inputs']
        raw = np.c_[data['spikes_1ms'][:, 0], data['regions_1ms'][:, :3]]
        rates = raw.reshape(-1, 5, 4).sum(1) / np.r_[32000, geometry['region_counts'][:3]] / .005
        t = (first + (np.arange(len(rates)) + .5) * .005 - reference) * 1000
        keep = (t >= -40 - 1e-9) & (t < 260 - 1e-9)
        rows.append(dict(condition=condition, geometry=geometry, fields=fields / cell_counts / .005,
            time_ms=t[keep], rates=rates[keep], reference_s=reference, source_npz=str(source_npz),
            source_npz_sha256=sha(source_npz), review=str(review_path), review_sha256=sha(review_path),
            selection=example['selection'], tail_brief_count=group['events']['brief_count'],
            core_recruitment=group['core_recruitment'], frame_absolute_times_s=frame_times.tolist(),
            feedback_tail=feedback_tail(folder, first + 20., first + 30.)))
    plt.rcParams.update({'font.size':9, 'axes.labelsize':9, 'axes.titlesize':10,
                         'xtick.labelsize':8, 'ytick.labelsize':8})
    fig = plt.figure(figsize=(15, 7.5))
    grid = fig.add_gridspec(3, 7, width_ratios=[2.1, 1, 1, 1, 1, 1, 1],
                           left=.055, right=.925, bottom=.12, top=.85, hspace=.58, wspace=.28)
    for i, row in enumerate(rows):
        ax = fig.add_subplot(grid[i, 0])
        for j, label in enumerate(['All E', 'Core A', 'Core B', 'Other E']):
            ax.plot(row['time_ms'], row['rates'][:, j], c=view.COLORS[j], lw=1, label=label)
        ax.axvline(0, c='#bbbbbb', lw=.6, ls=':')
        is_event = row['selection'] == 'first_complete_brief'
        selected = 'First brief' if is_event else 'Fixed window reference'
        ax.set(xlim=(-40, 260), ylim=(0, 510), xticks=[0, 100, 200], yticks=[0, 250, 500],
               ylabel='E rate (Hz)', title=f'{LABELS[row["condition"]]}\n{selected}: {row["reference_s"]:.2f} s')
        ax.spines[['top', 'right']].set_visible(False)
        if i == 2:
            ax.set_xlabel('Time from row reference (ms)')
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
        for j in range(6):
            ax = fig.add_subplot(grid[i, j + 1])
            im = ax.imshow(row['fields'][j].reshape(20, 20), origin='lower', extent=(0, 20, 0, 20),
                           interpolation='nearest', cmap='magma', vmin=0, vmax=500)
            for core, center in zip('AB', row['geometry']['centers_mm']):
                ax.add_patch(Circle(center, float(row['geometry']['core_radius_mm']),
                                    fill=False, ec='#36dbd3', lw=.8))
                ax.text(center[0], center[1] + 2., core, ha='center', color='#36dbd3', fontsize=7)
            ax.set(xticks=[0, 10, 20], yticks=[0, 10, 20], title=f'+{j*25}–{j*25+5} ms')
            if j == 0:
                ax.set_ylabel('y (mm)')
            else:
                ax.set_yticklabels([])
            if i == 2:
                ax.set_xlabel('x (mm)')
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.51, .935), ncol=4, frameon=False)
    fig.colorbar(im, cax=fig.add_axes([.943, .17, .010, .59]), label='E rate (Hz)')
    history = 'High-state history' if args.history == 'high' else 'Interictal history'
    fig.suptitle(f'{history} • prescribed mean Z = 0.95, K / gL = 0.02', y=.982, fontsize=12)
    fig.text(.5, .052, 'First chronological brief in each final 10 s window, or fixed final 300 ms when absent; native 5 ms fields.', ha='center', fontsize=9)
    fig.text(.5, .023, 'One paired-noise trajectory per graph. Event alignment is illustrative; it does not establish equivalent dynamics or pure orientation effects.', ha='center', fontsize=8)
    dest = ROOT / 'axis_controls/conditional_runs/event_comparison' / args.history
    figures = dest / 'figures'; figures.mkdir(parents=True, exist_ok=True)
    outputs = []
    for ext in ['png', 'pdf', 'svg']:
        path = figures / f'axis_highZ_lowK_propagation.{ext}'
        fig.savefig(path, dpi=200, facecolor='white')
        outputs.append(dict(path=str(path), sha256=sha(path)))
    plt.close(fig)
    metadata = dict(status='COMPLETE_CANDIDATE', history=args.history, coordinate=dict(Z=.95, K=.02),
        rows=[{k:v for k,v in row.items() if k not in ['geometry', 'fields', 'time_ms', 'rates']} for row in rows],
        paired_future_input_records_exact=True, geometry_exact=True,
        role='Visual diagnostic companion, not the main Figure5 layout or a new state criterion.',
        limits='One selected illustration per completed trajectory; all final10s events remain in the native reviews. '
               'Each row uses its own first complete brief, or the existing fixed final300ms fallback when absent. '
               'Rows may have different absolute event times; no temporal warping or event identity is inferred. '
               'Graph controls also change outdegree and low-threshold source strength.',
        producer_sha256=sha(Path(__file__)), files=outputs,
        agent_visual_review='PENDING', human_review='PENDING')
    (dest / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (dest / 'feedback_tail.json').write_text(json.dumps(dict(
        status='COMPLETE_THREE_MATCHED_GRAPHS',
        rows=[dict(condition=row['condition'], **row['feedback_tail']) for row in rows],
        sampling='G/K every20ms; causal R and q every1ms; fixed final10s, one carried history. No new thresholds or simulations.',
        producer_sha256=sha(Path(__file__))), indent=2) + '\n')
    (figures / 'README.md').write_text(
        '### axis_highZ_lowK_propagation.png\n\n'
        '同一高Z、低K条件和同一完整历史下，三种连接结构各自在末10秒内按时间顺序取首个完整短事件，'
        '沿用已有原生审阅的5ms放电率与六帧空间计数；若没有短事件，明确标注原有固定末300ms切片。'
        '三行共用空间坐标、物理core圈和色标，横向时间相对各行参考；这是一例形态比较，不代表全部事件统计。\n\n'
        '**关注点**：共同Z/K与未来输入下的core募集和传播覆盖是否改变；结构重配仍有输出度和来源强度差异，'
        '不能称纯方向效应、吸引子等价或自主回返。这是条件比较诊断图，正式自主Figure5布局未替换，图待人工目视。\n')
    print(json.dumps(dict(history=args.history, figure=str(outputs[0]['path']),
                          brief_counts={row['condition']:row['tail_brief_count'] for row in rows})))


if __name__ == '__main__':
    main()
