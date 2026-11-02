#!/usr/bin/env python3
"""Full recovery loops, interictal inset, and two fixed patient reference fields."""
import hashlib
import json
from pathlib import Path
import shutil

import build_fig6_selected_r08_r09_complete as parent

base = parent.base
np, plt = base.np, base.plt
OUT = parent.OUT.with_name('selected_r08_r09_recovery_references_20261031')
PREVIOUS = parent.OUT
HORIZON = 118.


def history():
    path = base.MAIN / 'results/topic4_sef_hfo/adaptation_history_20261031/long120s/results.json'
    return next(r for r in json.loads(path.read_text()) if r['seed'] == parent.SEED)


def add_curve(ax, t, xyz, bounds):
    inside = np.all((xyz >= bounds[:, 0]) & (xyz <= bounds[:, 1]), axis=1)
    keep = inside[:-1] & inside[1:]
    segments = np.stack([xyz[:-1], xyz[1:]], axis=1)[keep]
    ax.add_collection3d(base.Line3DCollection(segments, cmap='viridis', norm=base.Normalize(0, HORIZON),
        array=((t[:-1]+t[1:])/2)[keep], linewidths=.8, zorder=6, clip_on=False))
    ax.set(xlim=bounds[0], ylim=bounds[1], zlim=bounds[2])
    ax.view_init(elev=26, azim=-125)
    ax.set_proj_type('ortho')
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor('#f2f5fa')
    return segments, keep


def check_projected(fig, ax, segments):
    fig.canvas.draw()
    vertices = np.unique(segments.reshape(-1, 3), axis=0)
    x, y, _ = base.proj3d.proj_transform(*vertices.T, ax.get_proj())
    xy = ax.transData.transform(np.c_[x, y])
    assert np.all((xy[:, 0] >= 0) & (xy[:, 0] <= fig.bbox.width)
        & (xy[:, 1] >= 0) & (xy[:, 1] <= fig.bbox.height))


def draw_b(a, states):
    rate = a['spikes_1ms'][:, 0].reshape(-1, 5).sum(1)/32000/.005
    all_t = (np.arange(len(rate))+.5)*.005
    st = a['slow_time_ms']/1000
    all_xyz = np.c_[np.interp(all_t, st, a['Z'][:, 0]),
        .0005*np.interp(all_t, st, a['M'][:, 0]), base.gaussian_filter1d(rate, 1)]
    return draw_b_trajectory(all_t, all_xyz, states)


def draw_b_trajectory(all_t, all_xyz, states, *, inset_bounds=None,
        inset_title='Interictal zoom', export_full=True):
    """Render existing trajectory coordinates without rerunning the observer."""
    np.savez_compressed(OUT/'trajectory_data.npz', time_s=all_t, Z_etaM_rate=all_xyz)
    take = all_t < HORIZON
    t, xyz = all_t[take], all_xyz[take]
    bounds = np.array([[0., 1.], [0., 1000.], [0., 500.]])
    assert np.all((xyz >= bounds[:, 0]) & (xyz <= bounds[:, 1]))
    zoom_bounds = np.array(inset_bounds if inset_bounds is not None else [[.70, 1.], [0., .10], [0., 250.]])
    hr = history()
    returns = [min(event['start_s'] for event in hr['briefs']
        if event.get('previous_episode') == interval['previous_episode']
        and interval['exit_s'] <= event['start_s'] < interval['end_s'])
        for interval in hr['intervals'][:2]]
    assert all(interval['interictal_return_10events_5s'] for interval in hr['intervals'][:2])
    records = {}
    outputs = [(True, 'fig6-panelb')]
    if export_full:
        outputs.append((False, 'fig6-panelb-full-trajectory'))
    for inset, name in outputs:
        fig = plt.figure(figsize=(7.5, 6.7) if inset else (10., 8.5))
        box = [.065, .06, .75, .81] if inset else [.075, .08, .75, .86]
        ax = fig.add_axes(box, projection='3d', computed_zorder=False)
        seg, keep = add_curve(ax, t, xyz, bounds)
        assert keep.all(), 'Full view must retain every adjacent native trajectory sample.'
        ax.set_box_aspect((1.6, 1., 1.25), zoom=1.03 if inset else 1.07)
        ax.set(xlabel='$Z$', ylabel=r'$\eta_M M$ (mV)', xticks=[0, .5, 1.],
            yticks=[0, 500, 1000], zticks=[0, 250, 500])
        ax.set_yticklabels(['', '500', '1000'])
        fs = 13 if inset else 16
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.label.set_fontsize(fs)
            axis.labelpad = 5
        ax.tick_params(labelsize=fs-2, pad=1)
        fig.text(.012, .48, r'$r_E$ (Hz)', rotation=90, va='center', fontsize=fs)
        # Return markers identify actual brief events after each recovery, not
        # merely recovery of Z or a low firing rate.
        for sec in returns:
            pt = [np.interp(sec, t, xyz[:, k]) for k in range(3)]
            ax.scatter(*pt, marker='D', s=35 if inset else 60,
                color=plt.colormaps['viridis'](sec/HORIZON), edgecolor='black', linewidth=.7,
                depthshade=False, zorder=12)
        labels = parent.reference.prior.label_states(ax, states[3:4], t, xyz, fontsize=11 if inset else 14)
        fig.text(.07, .965, 'Full trajectory', ha='left', va='top', fontsize=13 if inset else 17)
        inset_record = None
        if inset:
            zoom = fig.add_axes([.57, .65, .34, .30], projection='3d', computed_zorder=False,
                facecolor='white')
            zs, zk = add_curve(zoom, t, xyz, zoom_bounds)
            zoom.view_init(elev=22, azim=-115)
            zoom.set_box_aspect((1.55, 1., 1.25), zoom=1.05)
            zoom.set(xticks=zoom_bounds[0], yticks=zoom_bounds[1], zticks=zoom_bounds[2], xlabel='$Z$',
                ylabel=r'$\eta_M M$', zlabel=r'$r_E$')
            zoom.set_yticklabels(['', f'{zoom_bounds[1,1]:g}'])
            zoom.tick_params(labelsize=8, pad=-1)
            for axis in (zoom.xaxis, zoom.yaxis, zoom.zaxis):
                axis.label.set_fontsize(9)
                axis.labelpad = -2
            fig.text(.74, .965, inset_title, ha='center', va='top', fontsize=11)
            labels += parent.reference.prior.label_states(zoom, states[:3], t, xyz, fontsize=9)
            check_projected(fig, zoom, zs)
            inset_record = dict(limits=zoom_bounds, segment_count=int(zk.sum()), visible_state_numbers=[1, 2, 3],
                no_bridging_across_hidden_segments=True, units_match_main_axes=True)
        cb = fig.colorbar(base.ScalarMappable(norm=base.Normalize(0, HORIZON), cmap='viridis'),
            cax=fig.add_axes([.89, .19, .023, .40]), ticks=[0, 60, HORIZON])
        cb.set_label('Time (s)', fontsize=fs-1, labelpad=5)
        cb.ax.tick_params(labelsize=fs-2, length=3, pad=3)
        check_projected(fig, ax, seg)
        parent.export(fig, name)
        records[name] = dict(limits=bounds, samples=len(t), segment_count=int(keep.sum()),
            all_native_segments_visible=True, inset=inset_record, states=labels)
    return dict(seed=parent.SEED, time_s=[0, HORIZON], full_saved_time_s=[0, 120], limits=bounds,
        coordinates=['mean Z', 'eta_M mean M (mV)', 'all-E rate (Hz)'],
        rate_bin_ms=5, display_smoothing_sigma_ms=5, Z_truncation=False,
        all_native_segments_within_display_time_visible=True,
        visible_state_numbers=[1, 2, 3, 4], main_state_numbers=[4], inset_state_numbers=[1, 2, 3],
        first_returned_brief_s=returns, return_marker='diamond',
        return_definition='First brief in each post-termination interval meeting at least 10 brief events across 5 s',
        third_entry_process_start_s=hr['episodes'][2]['process_start_s'],
        view_end_reason='Show both complete recovery cycles and stop before the third entry.',
        colorbar_orientation='vertical', colorbar_units='s', legend=False, outputs=records)


def patient_references():
    energy = base.native.energy
    clinical = energy.clinical
    record, _ = clinical._load_record('epilepsiae_1146')
    fz = clinical.load_frozen('epilepsiae_1146')
    rows = clinical._checkpoint_rows('epilepsiae_1146', len(fz['names']))
    # A is the already locked reference. B is fixed by patient-only evidence,
    # before measuring either model-to-patient similarity.
    ref_a = next(r for r in rows if r['seizure_idx'] == 25)
    assert ref_a['shared_best_template'] == 'A'
    eligible_b = []
    for row in rows:
        if row['shared_best_template'] != 'B':
            continue
        path = energy.PATIENT_CACHE/f"seizure_{row['seizure_idx']:03d}.json"
        values = np.array(json.loads(path.read_text())['activation_clinical_order'])
        if np.isfinite(values).all() and values.mean() > 0:
            eligible_b.append(row)
    ref_b = sorted(eligible_b, key=lambda r: (-r['shared_b_abs'], r['seizure_idx']))[0]
    refs = []
    for label, row in [('A', ref_a), ('B', ref_b)]:
        idx = row['seizure_idx']
        path = energy.PATIENT_CACHE/f'seizure_{idx:03d}.json'
        case = json.loads(path.read_text())
        assert case['clinical_contact_names'] == list(fz['names'])
        values = np.asarray(case['activation_clinical_order'])
        meta = case['metadata']
        assert meta['reference'] == 'car' and meta['band_hz'] == [1., 150.]
        assert meta['clinical_window_sec'] == [0., 10.] and meta['baseline_eeg_sec'] == [-120., -90.]
        checkpoint, checkpoint_path = clinical._checkpoint_event('epilepsiae_1146', idx)
        audit = clinical._score_audit(record, values, checkpoint)
        error = max(r.get('abs_error', 0) for r in audit['checkpoint_comparison'].values())
        assert error <= 1e-12
        refs.append(dict(label=label, patient='E1146', public_seizure=f'SZ{idx+1}', seizure_idx=idx,
            energy_type=f'T{label}', values=values, source=str(path),
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), extraction=meta,
            checkpoint=str(checkpoint_path), checkpoint_max_error=error, original_type_row=row))
    return record, fz, refs, dict(A='Existing fixed E1146 SZ26 retained.',
        B='Among existing TB checkpoints with complete 15-contact CAR 1-150 Hz and positive mean robust z, highest original abs(r_TB); lower seizure index breaks ties.',
        model_similarity_used_for_reference_selection=False,
        B_eligible_seizure_indices=[r['seizure_idx'] for r in eligible_b],
        type_definition='Existing frozen shared_best_template based on absolute template correlation; not a new clinical label.')


def draw_d(comparisons):
    record, fz, refs, selection = patient_references()
    energy = base.native.energy
    weights = energy.scorers_from_interictal_record(record)['shared_a']['weight_id']
    patient_fields = [energy._smooth_from_weights(r['values'], weights) for r in refs]
    model_fields = [energy._smooth_from_weights(np.asarray(c['model_robust_z']), weights) for c in comparisons]
    similarity = np.array([[np.corrcoef(m, p)[0, 1] for p in patient_fields] for m in model_fields])
    np.testing.assert_allclose(similarity[:, 0], [c['selected']['field_r'] for c in comparisons], atol=1e-12)
    np.testing.assert_array_equal(refs[0]['values'], comparisons[0]['patient_robust_z'])
    fig = plt.figure(figsize=(10.4, 6.5))
    gs = fig.add_gridspec(2, 2, left=.08, right=.86, bottom=.15, top=.91, hspace=.72, wspace=.28)
    axes = []
    for j in range(4):
        row, col = divmod(j, 2)
        ax = fig.add_subplot(gs[row, col]); axes.append(ax)
        if row == 0:
            values = refs[col]['values']
            title = f"Reference {'AB'[col]} (T{'AB'[col]})\nE1146 {refs[col]['public_seizure']}"
        else:
            values = np.asarray(comparisons[col]['model_robust_z'])
            lo, hi = comparisons[col]['model_early_s']
            title = f"Model onset {'AB'[col]}\n{lo:.2f}–{hi:.2f} s"
        energy.clinical._draw_field(ax, fz, energy.clinical._normalize_minmax(values),
            np.asarray(fz['support_a']), cmap='Blues', colorbar_values=np.array([0., 1.]),
            title=title, title_color='black', show_y=col==0)
        ax.set_xticks([-20, 0, 20]); ax.set_yticks([-10, 0, 10, 20])
        ax.tick_params(labelsize=11, labelbottom=row==1, length=2)
        ax.set_xlabel('shared TA axis (mm)' if row else '', fontsize=11, labelpad=3)
        ax.yaxis.label.set_fontsize(11)
        ax.title.set_fontsize(13); ax.title.set_fontweight('normal')
        if row:
            ax.text(.5, -.36, f"$r_A$ = {similarity[col, 0]:+.3f}    $r_B$ = {similarity[col, 1]:+.3f}",
                transform=ax.transAxes, ha='center', va='top', fontsize=12)
    fig.canvas.draw()
    bars = []
    for row in range(2):
        pos = axes[row*2+1].get_position()
        cb = fig.colorbar(base.ScalarMappable(norm=base.Normalize(0, 1), cmap='Blues'),
            cax=fig.add_axes([.89, pos.y0, .014, pos.height]), ticks=[0, .5, 1])
        cb.set_label('Normalized z field', fontsize=11, labelpad=4)
        cb.ax.tick_params(labelsize=10, length=2, pad=3)
        np.testing.assert_allclose(cb.ax.get_position().extents[[1, 3]], pos.extents[[1, 3]])
        bars.append(cb.ax.get_position().bounds)
    parent.export(fig, 'fig6-paneld')
    result = dict(layout='Two fixed patient references above two model onsets',
        patient_field_displayed=True, analysis_contact_count=15,
        correlations=similarity[:, 0].tolist(), similarity_matrix=similarity,
        similarity_rows=['Model onset A: first seizure', 'Model onset B: second seizure'],
        similarity_columns=['Reference A: TA-type E1146 SZ26', f"Reference B: TB-type E1146 {refs[1]['public_seizure']}"],
        model_windows_s=[c['model_early_s'] for c in comparisons], patient_references=refs,
        reference_selection=selection, field_scoring='Signed Pearson of contact-evaluated fields, frozen shared-A identity weights; no sign flip, mirror, or contact exclusion.',
        display_normalization='Each field min-max mapped for display; similarity uses the original robust-z field values.',
        colorbar_aligned_to_field_height=True, colorbar_bounds=bars,
        model_A_B_labels='Chronological model examples, not inferred matching reference classes.')
    base.write(OUT/'patient_reference_comparisons.json', result)
    np.savez_compressed(OUT/'patient_reference_arrays.npz', contact_names=fz['names'],
        reference_robust_z=[r['values'] for r in refs], model_robust_z=[c['model_robust_z'] for c in comparisons],
        frozen_weight_id=weights, signed_r=similarity)
    return result


def compose():
    mm = 72/25.4
    doc = base.fitz.open(); page = doc.new_page(width=320*mm, height=382*mm)
    boxes = {'A': (0, 1, 320, 108), 'B': (3, 115, 142, 239),
        'C': (144, 115, 317, 239), 'D': (3, 250, 218, 379), 'E': (225, 250, 317, 379)}
    letters = {'B': (3, 112), 'C': (144, 112), 'D': (3, 247), 'E': (224, 247)}
    records = {}
    for label, box in boxes.items():
        path = OUT/'figures'/f'fig6-panel{label.lower()}.pdf'
        with base.fitz.open(path) as src:
            # Top-align content instead of centering short C inside a tall row.
            x0, y0, x1, y1 = box
            scale = min((x1-x0)*mm/src[0].rect.width, (y1-y0)*mm/src[0].rect.height)
            width, height = src[0].rect.width*scale, src[0].rect.height*scale
            rect = base.fitz.Rect(x0*mm, y0*mm, x0*mm+width, y0*mm+height)
            page.show_pdf_page(rect, src, 0, keep_proportion=True)
        if label in letters:
            page.insert_text(tuple(v*mm for v in letters[label]), label, fontsize=16, fontname='hebo')
        records[label] = dict(source=str(path), allotted_box_mm=box, actual_box_mm=[v/mm for v in rect], vertical_alignment='top')
    path = OUT/'figures/fig6-complete-layout.pdf'
    doc.save(path, garbage=4, deflate=True); doc.close()
    with base.fitz.open(path) as doc:
        doc[0].get_pixmap(matrix=base.fitz.Matrix(220/72, 220/72), alpha=False).save(path.with_suffix('.png'))
        path.with_suffix('.svg').write_text(doc[0].get_svg_image(text_as_path=False))
    base.write(OUT/'layout_manifest.json', dict(canvas_mm=[320, 382], panels=records,
        B='Untruncated 0-118s full state trajectory with a low-adaptation interictal inset',
        C='Top-aligned two rows of four frames, one colorbar per row',
        D='Patient reference TA/TB fields and both model-to-both-reference signed correlations', E='Unchanged historical cohort'))


def documents(meta):
    returns = meta['trajectory']['first_returned_brief_s']
    matrix = meta['D']['similarity_matrix']
    ref_b = meta['D']['patient_references'][1]['public_seizure']
    (OUT/'caption_zh.md').write_text(f'''Figure 6：两次发作及返回间期的连续状态轨迹。

A：保留种子 {parent.SEED} 的原生信号、两个显示窗、固定增益和 8.85/59.30 s onset 底纹。按作者要求，5 从 59.30 移至原 6 的 59.45 s，6 顺延到 59.55 s；7/8 仍为 59.67/60.24 s。第一轮 1–4 为 8.85/8.90/8.96/9.41 s。编号是说明性采样帧，不重定义最初起点；原始统计和实际 17.20 s 终止时间均不变，Termination 仍为省略区事件的阶段说明。

B：全体 E 平均 Z、eta_M M、放电率，展示 0–118 s 的每一条相邻原生轨迹线段，Z=0–1，没有截去低 Z 部分。两个完整发作先后终止于 17.20 和 75.90 s；随后首次返回的间期短事件在 {returns[0]:.2f} 和 {returns[1]:.2f} s，以菱形标记，两段都满足既定至少 10 次短事件、跨度 5 s 的间期返回观察条件。第三次进入从 118.756 s 开始，因此主图止于 118 s。色条为真实时间。小窗线性放大 Z=0.70–1、eta_M M=0–0.10 mV、r_E=0–250 Hz，完整主图不裁切；1–3 放在小窗，4 在主图。放电率为 5 ms 分箱、5 ms sigma 平滑，完整 120 s 数据另存。额外大图 fig6-panelb-full-trajectory 保留同一完整范围和两个返回标记。

C：与 A 一致的八个中心时刻，各用原生 20 ms 空间率窗口，两个同尺度色条分别匹配两排图块。新 5 的 core A/B 率约 144/274 Hz；新 6 约 279/0 Hz，展示后续 A 核持续招募与 B 核暂时减弱。不能仅据移动后的编号把更早的核外或 B 核短暂活动改写为 A 最早起始。Local、Spread、Entry、Ictal 仍为展示过程的四阶段，非自动分期算法。

D：Reference A 保留既有 E1146 SZ26（冻结 TA 偏好标签）；Reference B 为 E1146 {ref_b}（冻结 TB 偏好标签），在完整 15 触点、1–150 Hz CAR、平均 robust z 为正的既有 TB 发作中按原始 |r_TB| 最大选取，不使用新模型相似度来选参考。两例患者均使用各自 EEG 前 [-120,-90] s 基线和临床 onset 后 [0,10] s 能量窗。Model onset A/B 分别是第一、二次模型发作的 8.85–9.85 和 59.30–60.30 s 窗口；字母是示例编号，不预先宣称匹配哪种类型。固定 shared-A identity 核在 15 触点位置平滑，计算有符号 Pearson r，不做镜像或符号翻转。模型 A 对 Reference A/B 分别为 {matrix[0,0]:+.6f}/{matrix[0,1]:+.6f}，模型 B 分别为 {matrix[1,0]:+.6f}/{matrix[1,1]:+.6f}。四幅图各自 min–max 显示只用于看空间形态，r 使用原始 robust-z 场。当前两个参考仅为描述性实例，不是独立泛化检验。

E：历史 13 人队列图和统计字节一致，未用此次示例重算。本修订为候选，正式入口保留，待作者目视检查。
''')
    descriptions = {
        'fig6-complete-layout': '完整 A–E 修订，B/C 主体上移靠近标签，B 展示两次完整恢复并含间期小窗。D 新增患者 TA/TB 发作能量参考场，每个模型都对两者计算相似度。',
        'fig6-panela': '原生波形及显示范围不变，新 5/6 为 59.45/59.55 s，7/8 保持不变。编号与 C 完全同源。',
        'fig6-panelb': '0–118 s 全范围三维轨迹，Z=0–1，未裁去任何主图线段。间期低适应区域另作小窗，两个菱形对应真正恢复的间期事件。',
        'fig6-panelb-full-trajectory': '与 B 相同的全范围轨迹独立大图，不含小窗。时间范围包括两次发作及其恢复，终止在第三次进入之前。',
        'fig6-panelc': '两排四帧，各用 A 同时刻的 20 ms 原生空间率。新 5/6 展示初步双核招募和随后 A 核持续活动。',
        'fig6-paneld': '上排患者 Reference A/B，下排 Model onset A/B，模型下方分别列出 r_A 与 r_B。所有场复用固定患者几何和原能量算法，参考选择不依赖本轮模型相似度。',
        'fig6-panele': '历史队列统计原样保留。本次未重新估计队列效应。'}
    (OUT/'figures/README.md').write_text('\n\n'.join(f'### {name}.png / .pdf / .svg\n\n{desc}\n'
        '**关注点**：' + ('主图保留完整 Z 范围，小窗仅放大间期细节。' if 'panelb' in name
            else 'r 为有符号场相关，模型编号 A/B 不代表先验分型。' if name=='fig6-paneld'
            else '逐一检查原生采样时刻和图上显示，当前仍待作者目视确认。') for name, desc in descriptions.items())+'\n')
    (OUT/'README.md').write_text(f'# Figure 6：完整恢复与双患者参考\n\n'
        '[完整图](figures/fig6-complete-layout.png) · [PDF](figures/fig6-complete-layout.pdf) · '
        '[全范围 B 大图](figures/fig6-panelb-full-trajectory.png) · [图注](caption_zh.md)\n\n'
        'A/C 同时刻表见 A_C_stage_correspondence.csv；D 两参考的选择与四个相关系数见 patient_reference_comparisons.json。\n\n'
        f'复现：`python {Path(__file__).resolve()}`。\n')


def main():
    parent.OUT = OUT
    parent.HORIZON = HORIZON
    parent.panel_a.STATE_TIMES = [8.85, 8.90, 8.96, 9.41, 59.45, 59.55, 59.67, 60.24]
    parent.panel_a.NUMBER_LABEL_TIMES = [8.81, 8.90, 8.99, 9.41, 59.45, 59.55, 59.67, 60.24]
    parent.draw_b = draw_b
    parent.reference.draw_d = draw_d
    parent.compose = compose
    parent.documents = documents
    parent.main()
    shutil.copy2(__file__, OUT/'source'/Path(__file__).name)
    metadata = json.loads((OUT/'figure_metadata.json').read_text())
    metadata.update(producer=str(Path(__file__).resolve()), previous_candidate=str(PREVIOUS),
        stage_revisions={'5': [59.30, 59.45], '6': [59.45, 59.55]},
        model_energy_windows_unchanged=True, patient_reference_confirmation='Author explicitly selected two patient seizure energy fields, TA and TB.')
    base.write(OUT/'figure_metadata.json', metadata)
    with np.load(OUT/'stage_readouts.npz') as new, np.load(PREVIOUS/'stage_readouts.npz') as old:
        np.testing.assert_array_equal(new['field_rate20ms_Hz'][[0,1,2,3,6,7]], old['field_rate20ms_Hz'][[0,1,2,3,6,7]])
        np.testing.assert_array_equal(new['field_rate20ms_Hz'][4], old['field_rate20ms_Hz'][5])
    old_comparisons = json.loads((PREVIOUS/'field_comparisons.json').read_text())
    new_comparisons = json.loads((OUT/'field_comparisons.json').read_text())
    for old, new in zip(old_comparisons, new_comparisons):
        np.testing.assert_array_equal(new['model_robust_z'], old['model_robust_z'])
    validation = json.loads((OUT/'validation.json').read_text())
    validation.update(full_B_Z_range_and_every_native_segment_visible=True, B_contains_two_qualified_interictal_returns=True,
        unchanged_six_C_frames_exact=True, new5_equals_previous6_exact=True, model_energy_fields_unchanged=True,
        all_four_model_to_patient_correlations_computed=True, patient_references_pass_original_checkpoint=True)
    base.write(OUT/'validation.json', validation)


if __name__ == '__main__':
    main()
