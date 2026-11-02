#!/usr/bin/env python3
"""Normalize entry propagation rows, restore early state 5 and group D by source."""
import csv
import json
from pathlib import Path
import shutil

import build_fig6_selected_r08_r09_enlarged_c as previous
import build_fig6_selected_r08_r09_panela as panel_a

base, matched, layout = previous.base, previous.matched, previous.layout
np, plt = previous.np, previous.plt
SOURCE = previous.OUT
OUT = SOURCE.with_name('selected_r08_r09_normalized_onsets_20261102')
NATIVE_EXPORT = previous.previous.NATIVE_EXPORT
NORMALIZATION = {}
PATIENT_BLUE = '#214e86'


def refresh_a(meta):
    # Reuse the original complete-signal filter and A renderer, with just state 5 moved.
    panel_a.OUT = OUT/'onset_stage_revision'
    panel_a.STATE_TIMES = [s['time_s'] for s in meta['states']]
    panel_a.NUMBER_LABEL_TIMES = [s['label_anchor_time_s'] for s in meta['states']]
    panel_a.STATE_TIMES[4] = panel_a.NUMBER_LABEL_TIMES[4] = 59.30
    panel_a.main()
    wave = json.loads((panel_a.OUT/'figure_metadata.json').read_text())
    for ext in ['pdf','png','svg']:
        shutil.copy2(panel_a.OUT/'figures'/f'fig6-panela.{ext}',OUT/'figures'/f'fig6-panela.{ext}')
    for name in ['R08_display_data.npz','R09_display_data.npz']:
        with np.load(SOURCE/name) as old, np.load(panel_a.OUT/name) as new:
            for key in old.files:
                np.testing.assert_array_equal(old[key],new[key])
    shutil.copy2(panel_a.OUT/'stage_readouts.npz',OUT/'stage_readouts.npz')
    with np.load(SOURCE/'stage_readouts.npz') as old, np.load(OUT/'stage_readouts.npz') as new:
        keep = [0,1,2,3,5,6,7]
        np.testing.assert_array_equal(old['field_rate20ms_Hz'][keep],new['field_rate20ms_Hz'][keep])
        maps, cell_counts = new['field_rate20ms_Hz'].copy(),new['cell_e_counts'].copy()
    states = wave['states']
    frames = []
    for state, field in zip(states,maps):
        frames.append(dict(**state,E_rate_Hz=field,
            native_spike_count=int(np.rint(np.sum(field*cell_counts*.02))),field_matches_A_stage_readout=True))
    meta['waveform'] = wave
    meta['states'] = states
    meta['spatial_maps']['frames'] = frames
    meta['stage_revisions'] = {'5':[59.45,59.30]}
    with (OUT/'A_C_stage_correspondence.csv').open('w') as handle:
        writer = csv.writer(handle)
        writer.writerow(['number','episode','phase','A_time_s','C_time_s','window_start_s','window_end_s','native_spike_count'])
        for state,frame in zip(states,frames):
            writer.writerow([state['number'],state['episode'],state['label'],state['time_s'],frame['time_s'],
                             *state['sample_window_s'],frame['native_spike_count']])
    assert states[4]['time_s'] == wave['segments'][1]['displayed_onset_s'] == 59.30
    return maps,cell_counts


def normalization(meta,maps,cell_counts):
    # Match Fig5's per-event, across-time maximum convention while preserving
    # Fig6's original 20-ms rate estimator and explicitly defining its entry windows.
    windows = [[meta['states'][4*r]['time_s'],meta['states'][4*r+3]['time_s']] for r in range(2)]
    chunks = []
    for path in sorted((panel_a.previous.RAW/f'runs/seed{panel_a.SEED}/chunks').glob('*.npz')):
        lo,hi = [int(v)/10000 for v in path.stem.split('_')]
        if any(hi > start-.01 and lo < end+.01 for start,end in windows):
            chunks.append(path)
    times,counts = [],[]
    for path in chunks:
        with np.load(path) as data:
            times.append(data['field_time_ms']/1000)
            counts.append(data['field_5ms'])
    t,raw = np.concatenate(times),np.concatenate(counts)
    records,denominators,arrays = [],[],{}
    for row,(start,end) in enumerate(windows):
        centers = np.arange(round(start*1000),round(end*1000)+1,5,dtype=np.int64)/1000
        fields = np.array([raw[(t>=sec-.01)&(t<sec+.01)].sum(0)/cell_counts/.02 for sec in centers])
        denominator = float(fields.max())
        assert denominator > 0
        for j in range(4):
            state = meta['states'][4*row+j]
            selected = np.flatnonzero(centers == state['time_s'])
            assert len(selected) == 1
            np.testing.assert_array_equal(fields[selected[0]],maps[4*row+j])
        normalized = maps[row*4:row*4+4]/denominator
        assert 0 <= normalized.min() <= normalized.max() <= 1
        np.testing.assert_allclose(normalized*denominator,maps[row*4:row*4+4],rtol=1e-15,atol=1e-12)
        records.append(dict(row=row+1,center_time_window_s=[start,end],native_support_window_s=[start-.01,end+.01],
            sample_width_ms=20,center_step_ms=5,n_continuous_frames=len(centers),
            denominator_Hz=denominator,selected_frame_peak_normalized=normalized.max(axis=1),
            peak_center_time_s=float(centers[np.unravel_index(fields.argmax(),fields.shape)[0]])))
        denominators.append(denominator)
        arrays[f'row{row+1}_center_time_s'] = centers
        arrays[f'row{row+1}_continuous_field_Hz'] = fields
    NORMALIZATION.update(method='One maximum across spatial cells and all 20-ms frames in each continuous Local-to-Ictal entry window; same denominator for all four displayed frames in a row.',
        fig5_convention_source=str(base.MAIN/'scripts/paper_figures/build_fig5_aligned_layout.py'),
        normalization_domain='Continuous illustrated entry window, not the entire ictal episode.',
        denominators_Hz=denominators,rows=records,per_frame_normalization=False,
        between_row_absolute_amplitude_comparable=False,color_limits=[0,1],color_mapping='linear',
        colorbar_label='Norm. Firing rate',native_rates_retained=True)
    np.savez_compressed(OUT/'normalized_propagation_fields.npz',
        raw_selected_fields_Hz=maps,normalized_selected_fields=maps/np.repeat(denominators,4)[:,None],
        denominators_Hz=denominators,**arrays)
    base.write(OUT/'propagation_normalization.json',NORMALIZATION)


def normalized_c_export(fig,stem):
    for j,ax in enumerate(fig.axes[:8]):
        im = ax.images[0]
        im.set_data(np.asarray(im.get_array())/NORMALIZATION['denominators_Hz'][j//4])
        im.set_norm(base.Normalize(0,1))
    for row,ax in enumerate(fig.axes[8:]):
        cb = ax._colorbar
        cb.update_normal(fig.axes[4*row].images[0])
        cb.set_ticks([0,.5,1])
        cb.ax.set_yticklabels(['0','0.5','1'])
        cb.set_label('Norm. Firing rate',fontsize=layout.LABEL,labelpad=4)
    previous.enlarged_c_export(fig,stem)


def draw_c(meta):
    previous.OUT, previous.SOURCE = OUT,SOURCE
    matched.OUT, matched.SOURCE = OUT,OUT
    matched.export = normalized_c_export
    try:
        result = matched.draw_c(meta)
    finally:
        matched.export = NATIVE_EXPORT
    result.update(previous.C_DISPLAY)
    result.update(normalization=NORMALIZATION,individual_row_colorbars=previous.C_DISPLAY['colorbar_bounds'],
                  eight_native_frames_unchanged=False,seven_other_native_frames_unchanged=True)
    return result


def draw_d():
    positions = []
    def grouped_export(fig,stem):
        assert stem == 'fig6-paneld' and len(fig.axes)==5
        fig.set_size_inches(245/25.4,71/25.4)
        order = [0,2,1,3]
        for col,(index,x) in enumerate(zip(order,[15.,66.,119.,170.])):
            ax = fig.axes[index]
            ax.set_position([x/245,15/71,46/245,46/71])
            ax.title.set_color('black' if col<2 else PATIENT_BLUE)
            ax.title.set_fontweight('bold')
            positions.append(ax.get_position().bounds)
        fig.axes[-1].set_position([220/245,15/71,1.7/245,46/71])
        fig.texts[0].set_position((115.5/245,8.5/71))
        matched.OUT = OUT
        NATIVE_EXPORT(fig,stem)
    layout.OUT = OUT
    layout.export = grouped_export
    matches = json.loads((SOURCE/'model_selected_patient_matches.json').read_text())
    _,alias = layout.load_stat()
    result = layout.draw_d(matches,alias)
    result.update(final_size_mm=[245,71],titles=['Model Onset 1','Model Onset 2','E10 SZ13','E10 SZ14'],
        field_plot_height_mm=46.,colorbar_height_mm=46.,field_axes_bounds=positions,
        colorbar_bounds=[220/245,15/71,1.7/245,46/71],model_title_color='black',
        patient_title_colors=[PATIENT_BLUE,PATIENT_BLUE],title_fontweight='bold',
        display_order_source_indices=[0,2,1,3])
    result.pop('paired_title_colors',None)
    return result


def documents(meta):
    (OUT/'caption_zh.md').write_text('''Figure 6：连续发作、间期轨迹与患者能量场。

A：同一原生连续仿真的两个显示窗为 8.10–13.80 秒和 58.50–60.40 秒，每秒等宽。1–4 与 5–8 保留 Local、Spread、Entry、Ictal 语义；5 移至第二段浅红底色起点 59.30 秒，其余采样时刻不变。波形、滤波、增益与底色范围保持；实际首次终止在省略区内的 17.20 秒，Termination 文字为阶段示意。

B：保留 0–118 秒完整原生轨迹及 Interictal 小窗，1–3 在小窗、4 在主轨迹；同一时间色条与三维坐标框等高。小窗包含间期事件及连续进入段，不把轨迹视为预测证据或解析分岔证明。

C：两行四帧分别与 A 的 1–4、5–8 严格对应，20 毫秒窗口内每个网格的 E 放电率保持原估计。沿用当前 Figure 5 的每行统一分母原则，对每行完整 Local-to-Ictal 进入窗口（第一行中心时刻 8.85–9.41 秒，第二行 59.30–60.24 秒；每 5 毫秒取一个 20 毫秒窗口）所有网格、所有时刻的最大放电率归一化，线性显示 0–1。这不是逐帧拉满，也不是按整段癫痫发作取峰值；行内保留强弱变化，两行颜色不代表可直接比较的绝对 Hz。分母及原始 Hz 均保留于 propagation_normalization.json 和 normalized_propagation_fields.npz。

D：从左至右为 Model Onset 1、Model Onset 2、E10 SZ13、E10 SZ14；模型标题黑色加粗、患者标题统一深蓝加粗。患者、场值与匹配算法保持：每个固定模型 onset 窗口从该患者全部 25 次合格发作中取最高有符号场相关者；描述性匹配不等同于独立验证。模型能量窗仍为 8.85–9.85 秒与 59.30–60.30 秒，各场以 min–max 显示，评分使用原始 robust-z 场。

E：原历史 13 人、140 次合格发作的 full_energy_plus200ms__db 配对统计保持，p=0.00708，图面显示 **（p<0.01）。数据、统计、宽高及布局未重算或改变。本版待作者目视检查。
''')
    desc={
        'fig6-complete-layout':'完整 A–E 沿用此前布局。更新 A/C 的第 5 帧、C 的归一化色标与 D 的顺序和标题。',
        'fig6-panela':'原生波形和底色不变，第 5 个标记提前到 59.30 秒。它与 C 第二行首帧使用同一个 20 毫秒采样窗。',
        'fig6-panelb':'全范围轨迹及 Interictal 小窗保留。与上一版导出字节一致。',
        'fig6-panelb-interictal':'独立间期及进入段保留。与上一版导出字节一致。',
        'fig6-panelb-full-trajectory':'独立全范围轨迹保留。与上一版导出字节一致。',
        'fig6-panelc':'八帧空间传播与 A 的阶段对应。每行按连续进入窗口的时空峰值归一化，线性 0–1 色标；不逐帧归一化。',
        'fig6-paneld':'先展示两个模型，再展示 E10 SZ13 和 E10 SZ14。模型标题黑色加粗，患者标题深蓝加粗，原始场和患者选择保持。',
        'fig6-panele':'历史患者统计保持。与上一版导出字节一致。'}
    (OUT/'figures/README.md').write_text('\n\n'.join(f'### {stem}.png / .pdf / .svg\n\n{text}\n**关注点**：本版待作者目视检查；数值定义见同级图注与元数据。' for stem,text in desc.items())+'\n')
    (OUT/'README.md').write_text('# Figure 6：模型优先排列与归一化传播\n\n'
        '[完整图](figures/fig6-complete-layout.png) · [PDF](figures/fig6-complete-layout.pdf)\n\n'
        f'复现：`python {Path(__file__).resolve()}`。沿用冻结原生仿真，不重跑模型或匹配统计。\n')


def main():
    pointer = base.MAIN/'results/paper-ready-figure/fig6/current_version.json'
    frozen = pointer.read_bytes()
    shutil.copytree(SOURCE,OUT,dirs_exist_ok=True,ignore=shutil.ignore_patterns('validation.json','visual_review.json'))
    plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans',
                         'pdf.fonttype':42,'svg.fonttype':'none','path.simplify':False})
    meta = json.loads((SOURCE/'figure_metadata.json').read_text())
    print('Updating state 5 with the native A renderer.',flush=True)
    maps,cell_counts = refresh_a(meta)
    normalization(meta,maps,cell_counts)
    meta['C_display'] = draw_c(meta)
    meta['D'] = draw_d()
    meta['spatial_maps'].update(normalization=NORMALIZATION,
        separate_row_colorbars=[dict(row=row+1,bounds=pos,limits=[0,1],denominator_Hz=NORMALIZATION['denominators_Hz'][row])
                               for row,pos in enumerate(meta['C_display']['colorbar_bounds'])],
        both_colorbars_same_normalization=False,both_colorbars_same_display_limits=True)
    meta.update(producer=str(Path(__file__).resolve()),previous_candidate=str(SOURCE),
        revision_scope='Move state 5 to 59.30 s, normalize C per entry and group D models before patients.',
        status='PENDING_AUTHOR_VISUAL_REVIEW')
    base.write(OUT/'figure_metadata.json',meta)
    previous.compose()
    manifest = json.loads((OUT/'layout_manifest.json').read_text())
    manifest['layout_unchanged_from'] = str(SOURCE)
    base.write(OUT/'layout_manifest.json',manifest)
    unchanged = []
    for stem in ['fig6-panelb','fig6-panelb-interictal','fig6-panelb-full-trajectory','fig6-panele']:
        for ext in ['png','pdf','svg']:
            name=f'{stem}.{ext}'
            assert (OUT/'figures'/name).read_bytes()==(SOURCE/'figures'/name).read_bytes()
            unchanged.append(name)
    for name in ['trajectory_data.npz','field_comparisons.json','model_selected_patient_matches.json',
                 'model_selected_patient_arrays.npz','all_patient_seizure_matches.csv','cohort_statistics_selected.json']:
        assert (OUT/name).read_bytes()==(SOURCE/name).read_bytes()
    checks={}
    for stem in ['fig6-panela','fig6-panelc','fig6-paneld','fig6-complete-layout']:
        with base.fitz.open(OUT/'figures'/f'{stem}.pdf') as doc:
            checks[stem]=all(page.rect.contains(base.fitz.Rect(s['bbox'])) for page in doc
                for b in page.get_text('dict')['blocks'] if b['type']==0 for line in b['lines'] for s in line['spans'])
    assert all(checks.values()),checks
    assert pointer.read_bytes()==frozen
    base.write(OUT/'validation.json',dict(unchanged_panel_files=unchanged,
        waveform_arrays_gains_filters_unchanged=True,seven_other_stage_fields_exact=True,
        state5_matches_A_shading_start_s=59.30,A_C_all_times_and_fields_consistent=True,
        C_normalization_roundtrip=True,C_per_frame_normalization=False,
        D_fields_and_selected_patients_unchanged=True,cohort_statistics_unchanged=True,
        PDF_text_inside_canvas=checks,formal_pointer_unchanged=True,human_visual_acceptance='PENDING'))
    documents(meta)
    shutil.copy2(__file__,OUT/'source'/Path(__file__).name)
    print(json.dumps(dict(output=str(OUT),denominators_Hz=NORMALIZATION['denominators_Hz'],state5=meta['states'][4],checks=checks),indent=2),flush=True)


if __name__=='__main__':
    main()
