#!/usr/bin/env python3
"""Unify final-size type and display model-first best-matched patient fields."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

import build_fig6_selected_r08_r09_interictal_detail as last

previous, parent, base = last.previous, last.parent, last.base
np, plt = last.np, last.plt
SOURCE = last.OUT
OUT = SOURCE.with_name('selected_r08_r09_matched_row_20261031')
MM = 72/25.4
A_SCALE = 320/(17*25.4)
TICK, LABEL, TITLE, LETTER = [v*A_SCALE for v in [13, 16, 14, 20]]
SIZES_MM = {'B': (139, 124), 'C': (173, 116), 'D': (215, 80)}


def export(fig, stem):
    path = OUT/'figures'/f'{stem}.pdf'
    fig.savefig(path, dpi=220, facecolor='white')
    fig.savefig(path.with_suffix('.svg'), dpi=220, facecolor='white')
    plt.close(fig)
    with base.fitz.open(path) as doc:
        doc[0].get_pixmap(matrix=base.fitz.Matrix(220/72, 220/72), alpha=False).save(path.with_suffix('.png'))


def draw_b(meta):
    with np.load(SOURCE/'trajectory_data.npz') as data:
        take = data['time_s'] < previous.HORIZON
        t, xyz = data['time_s'][take], data['Z_etaM_rate'][take]
    fig = plt.figure(figsize=np.array(SIZES_MM['B'])/25.4)
    ax = fig.add_axes([.065, .065, .75, .80], projection='3d', computed_zorder=False)
    bounds = np.array([[0., 1.], [0., 1000.], [0., 500.]])
    seg, keep = previous.add_curve(ax, t, xyz, bounds)
    assert keep.all()
    ax.set_box_aspect((1.6, 1., 1.25), zoom=1.03)
    ax.set(xlabel='$Z$', ylabel=r'$\eta_M M$ (mV)', xticks=[0, .5, 1.], yticks=[0, 500, 1000], zticks=[0, 250, 500])
    ax.set_yticklabels(['', '500', '1000'])
    ax.tick_params(labelsize=TICK, pad=1)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.label.set_fontsize(LABEL); axis.labelpad=4
    fig.text(.012, .48, r'$r_E$ (Hz)', rotation=90, va='center', fontsize=LABEL)
    for sec in meta['trajectory']['first_returned_brief_s']:
        pt = [np.interp(sec, t, xyz[:, k]) for k in range(3)]
        ax.scatter(*pt, marker='D', s=28, color=plt.colormaps['viridis'](sec/previous.HORIZON),
            edgecolor='black', linewidth=.6, depthshade=False, zorder=12)
    main_labels = parent.reference.prior.label_states(ax, meta['states'][3:4], t, xyz, fontsize=TITLE)
    fig.text(.07, .97, 'Full trajectory', fontsize=LABEL, va='top')
    zoom = fig.add_axes([.565, .64, .37, .31], projection='3d', computed_zorder=False, facecolor='white')
    zs, zk = previous.add_curve(zoom, t, xyz, last.BOUNDS)
    zoom.view_init(elev=22, azim=-115)
    zoom.set_box_aspect((1.55, 1., 1.25), zoom=1.04)
    zoom.set(xticks=[.75, 1.], yticks=[0, 1.5], zticks=[0, 200], xlabel='$Z$',
        ylabel=r'$\eta_M M$', zlabel=r'$r_E$')
    zoom.set_yticklabels(['', '1.5'])
    zoom.tick_params(labelsize=TICK, pad=-1)
    for axis in [zoom.xaxis, zoom.yaxis, zoom.zaxis]:
        axis.label.set_fontsize(TITLE); axis.labelpad=-2
    fig.text(.75, .98, 'Interictal', fontsize=LABEL, ha='center', va='top')
    inset_labels = parent.reference.prior.label_states(zoom, meta['states'][:3], t, xyz, fontsize=TITLE)
    cb = fig.colorbar(base.ScalarMappable(norm=base.Normalize(0, previous.HORIZON), cmap='viridis'),
        cax=fig.add_axes([.865, .19, .023, .40]), ticks=[0, 60, previous.HORIZON])
    cb.set_label('Time (s)', fontsize=LABEL, labelpad=4)
    cb.ax.tick_params(labelsize=TICK, length=2, pad=3)
    previous.check_projected(fig, ax, seg); previous.check_projected(fig, zoom, zs)
    export(fig, 'fig6-panelb')
    return dict(final_size_mm=SIZES_MM['B'], font_label_pt=LABEL, font_tick_pt=TICK,
        inset_font_label_pt=TITLE, inset_font_tick_pt=TICK, limits=bounds, inset_limits=last.BOUNDS,
        segment_count=int(keep.sum()), inset_segment_count=int(zk.sum()),
        states=main_labels+inset_labels, unchanged_trajectory=True)


def draw_c(meta):
    with np.load(SOURCE/'stage_readouts.npz') as data:
        times, maps = data['time_s'].copy(), data['field_rate20ms_Hz'].copy()
    with np.load(parent.panel_a.previous.RAW/'geometry.npz') as data:
        centers, radius = data['centers_mm'].copy(), float(data['core_radius_mm'])
    np.testing.assert_array_equal(times, [s['time_s'] for s in meta['states']])
    for j, frame in enumerate(meta['spatial_maps']['frames']):
        np.testing.assert_array_equal(maps[j], frame['E_rate_Hz'])
    fig = plt.figure(figsize=np.array(SIZES_MM['C'])/25.4)
    gs = fig.add_gridspec(2, 4, left=.11, right=.875, bottom=.12, top=.89, wspace=.15, hspace=.67)
    row_axes = [[], []]
    for j, state in enumerate(meta['states']):
        row, col = divmod(j, 4); ax=fig.add_subplot(gs[row, col]); row_axes[row].append(ax)
        im = ax.imshow(maps[j].reshape(20,20), origin='lower', extent=[0,20,0,20],
            cmap='magma', norm=base.PowerNorm(.6,0,500), interpolation='nearest')
        for k, xy in enumerate(centers):
            ax.add_patch(base.Circle(xy, radius, ec='#2dd4cd', fc='none', lw=.9))
            ax.text(xy[0], xy[1]+2.1, 'AB'[k], color='#10656a', fontsize=TICK, ha='center', weight='bold',
                bbox=dict(fc='white', ec='none', pad=.12, alpha=.85))
        ax.set(xticks=[0,10,20], yticks=[0,10,20])
        ax.tick_params(labelsize=TICK, length=2, pad=2, labelbottom=row==1, labelleft=col==0)
        if row==1: ax.set_xlabel('x (mm)', fontsize=LABEL, labelpad=3)
        if col==0: ax.set_ylabel('y (mm)', fontsize=LABEL, labelpad=3)
        ax.set_title(f"{state['number']}  {state['label']}\n{state['time_s']:.2f} s", color=state['color'], fontsize=TITLE, pad=5)
    fig.canvas.draw()
    bars=[]
    for row, axes in enumerate(row_axes):
        pos=axes[-1].get_position()
        cb=fig.colorbar(im, cax=fig.add_axes([.902,pos.y0,.012,pos.height]), ticks=[0,250,500])
        cb.set_label('E rate (Hz)', fontsize=LABEL, labelpad=4)
        cb.ax.tick_params(labelsize=TICK,length=2,pad=2)
        fig.text(.021,pos.y0+pos.height/2,f'Model Onset {row+1}',rotation=90,
            fontsize=LABEL,ha='center',va='center')
        np.testing.assert_allclose(cb.ax.get_position().extents[[1,3]],pos.extents[[1,3]])
        bars.append(cb.ax.get_position().bounds)
    export(fig,'fig6-panelc')
    return dict(final_size_mm=SIZES_MM['C'], row_labels=['Model Onset 1','Model Onset 2'],
        font_label_pt=LABEL,font_tick_pt=TICK,font_title_pt=TITLE,
        eight_native_frames_unchanged=True,A_C_times_identical=True,individual_row_colorbars=bars)


def find_matches():
    energy=base.native.energy; clinical=energy.clinical
    record,_=clinical._load_record('epilepsiae_1146'); fz=clinical.load_frozen('epilepsiae_1146')
    rows=clinical._checkpoint_rows('epilepsiae_1146',len(fz['names']))
    assert len(rows)==25
    weights=energy.scorers_from_interictal_record(record)['shared_a']['weight_id']
    comparisons=json.loads((SOURCE/'field_comparisons.json').read_text())
    cases=[]
    for row in rows:
        idx=row['seizure_idx']; path=energy.PATIENT_CACHE/f'seizure_{idx:03d}.json'
        case=json.loads(path.read_text()); values=np.asarray(case['activation_clinical_order'])
        assert case['clinical_contact_names']==list(fz['names'])==comparisons[0]['contact_names']
        extraction=case['metadata']
        assert extraction['reference']=='car' and extraction['band_hz']==[1.,150.]
        assert extraction['clinical_window_sec']==[0.,10.] and extraction['baseline_eeg_sec']==[-120.,-90.]
        assert extraction['n_baseline_frames']>=50 and np.isfinite(values).all()
        checkpoint,cp_path=clinical._checkpoint_event('epilepsiae_1146',idx)
        audit=clinical._score_audit(record,values,checkpoint)
        error=max(x.get('abs_error',0.) for x in audit['checkpoint_comparison'].values())
        assert error<=1e-12
        cases.append(dict(seizure_idx=idx,public_seizure=f'SZ{idx+1}',
            existing_type='T'+row['shared_best_template'],values=values,
            mean_robust_z=float(values.mean()),field=energy._smooth_from_weights(values,weights),
            source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            checkpoint=str(cp_path),checkpoint_max_error=error,extraction=extraction))
    matches=[]; rankings=[]; model_fields=[]
    for j,comparison in enumerate(comparisons):
        model=np.asarray(comparison['model_robust_z']); field=energy._smooth_from_weights(model,weights)
        model_fields.append(field)
        ranked=[dict(model='AB'[j],seizure_idx=c['seizure_idx'],public_seizure=c['public_seizure'],
            existing_type=c['existing_type'],signed_field_r=float(np.corrcoef(field,c['field'])[0,1]),
            signed_contact_r=float(np.corrcoef(model,c['values'])[0,1]),
            mean_robust_z=c['mean_robust_z'],positive_mean_enhancement=c['mean_robust_z']>0) for c in cases]
        ranked.sort(key=lambda r:(-r['signed_field_r'],r['seizure_idx']))
        for rank,row in enumerate(ranked,1): row['rank']=rank
        rankings.extend(ranked)
        winner=ranked[0]
        # In this run the all-candidate winner also passes the prior E/F
        # positive-mean enhancement gate; no change of gate is needed.
        assert winner['positive_mean_enhancement']
        chosen=next(c for c in cases if c['seizure_idx']==winner['seizure_idx'])
        matches.append(dict(model='AB'[j],reference='AB'[j],winner=winner,patient=chosen,
            model_robust_z=model,model_window_s=comparison['model_early_s']))
    matrix=np.array([[np.corrcoef(m,p['patient']['field'])[0,1] for p in matches] for m in model_fields])
    for j,match in enumerate(matches):
        assert matrix[j,j]==match['winner']['signed_field_r']
    with (OUT/'all_patient_seizure_matches.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rankings[0]));writer.writeheader();writer.writerows(rankings)
    result=dict(patient='E1146',n_candidate_seizures=len(cases),n_model_examples=2,
        selection='For each fixed model window, maximize signed Pearson correlation across all 25 complete exact-band patient seizure fields; lower seizure index breaks ties.',
        candidate_support='Complete 15-contact clinical-onset 0-10 s, CAR, 1-150 Hz, EEG -120 to -90 s robust-z baseline.',
        positive_mean_gate_changes_selection=False,template_type_restriction=False,
        field_scoring='Frozen shared-A identity-kernel contact-evaluated fields; no mirroring or sign flipping.',
        matches=matches,all_candidates=rankings,signed_similarity_matrix=matrix,
        matrix_rows=['Model Onset A','Model Onset B'],matrix_columns=['Reference seizure A','Reference seizure B'],
        interpretation='Model-selected descriptive examples; correlations are selection maxima, not held-out validation or a new cohort result.',
        reference_A_B_meaning='Paired example identifiers; patient frozen TA/TB labels are reported separately.',
        model_fields_and_windows_unchanged=True)
    base.write(OUT/'model_selected_patient_matches.json',result)
    np.savez_compressed(OUT/'model_selected_patient_arrays.npz',contact_names=fz['names'],frozen_weight_id=weights,
        model_robust_z=[m['model_robust_z'] for m in matches],
        reference_robust_z=[m['patient']['values'] for m in matches],signed_r=matrix)
    return fz,result


def draw_d(fz,result):
    clinical=base.native.energy.clinical
    width,height=SIZES_MM['D']; fig=plt.figure(figsize=(width/25.4,height/25.4))
    fields=[]; axes=[]
    for j,match in enumerate(result['matches']):
        fields.extend([(f"Model Onset {'AB'[j]}",np.asarray(match['model_robust_z']),True,j),
            (f"Reference seizure {'AB'[j]}\nE1146 {match['patient']['public_seizure']} · {match['patient']['existing_type']}",
             np.asarray(match['patient']['values']),False,j)])
    x_positions=[15.,60.,107.,152.]
    for col,(title,values,is_model,j) in enumerate(fields):
        ax=fig.add_axes([x_positions[col]/width,25/height,40/width,40/height]);axes.append(ax)
        clinical._draw_field(ax,fz,clinical._normalize_minmax(values),np.asarray(fz['support_a']),
            cmap='Blues',colorbar_values=np.array([0.,1.]),title=title,title_color='black',show_y=col==0)
        ax.set_xticks([-20,0,20]);ax.set_yticks([-10,0,10,20]);ax.set_xlabel('')
        ax.tick_params(labelsize=TICK,length=2,pad=2)
        ax.yaxis.label.set_fontsize(LABEL);ax.yaxis.labelpad=3
        ax.title.set_fontsize(TITLE);ax.title.set_fontweight('normal');ax.title.set_position((.5,1.))
        if is_model:
            correlations=result['signed_similarity_matrix'][j]
            fig.text((x_positions[col]+20)/width,10.5/height,
                f"$r_A$ = {correlations[0]:+.3f}\n$r_B$ = {correlations[1]:+.3f}",
                ha='center',va='top',fontsize=TITLE,linespacing=1.10)
    fig.text(103.5/width,18.5/height,'Shared TA axis (mm)',ha='center',va='top',fontsize=LABEL)
    fig.canvas.draw();pos=axes[-1].get_position()
    cb=fig.colorbar(base.ScalarMappable(norm=base.Normalize(0,1),cmap='Blues'),
        cax=fig.add_axes([196/width,pos.y0,1.7/width,pos.height]),ticks=[0,.5,1])
    cb.set_label('Normalized z field',fontsize=LABEL,labelpad=3)
    cb.ax.tick_params(labelsize=TICK,length=2,pad=2)
    np.testing.assert_allclose(cb.ax.get_position().extents[[1,3]],pos.extents[[1,3]])
    fig.canvas.draw()
    title_boxes=[ax.title.get_window_extent(fig.canvas.get_renderer()) for ax in axes]
    assert all(a.x1<b.x0 for a,b in zip(title_boxes,title_boxes[1:])), 'D titles overlap'
    export(fig,'fig6-paneld')
    return dict(final_size_mm=SIZES_MM['D'],layout_order=['Model Onset A','Reference seizure A','Model Onset B','Reference seizure B'],
        model_seconds_displayed=False,paired_matches=[m['winner'] for m in result['matches']],
        signed_similarity_matrix=result['signed_similarity_matrix'],colorbar_orientation='vertical',
        field_height_mm=40,neighbor_gap_mm=[5,7,5],colorbar_gap_mm=4,
        field_axes_bounds=[ax.get_position().bounds for ax in axes],colorbar_bounds=cb.ax.get_position().bounds,
        font_label_pt=LABEL,font_tick_pt=TICK,font_title_pt=TITLE,
        label_A_B_is_pair_identifier=True,selection_record='model_selected_patient_matches.json')


def compose():
    from matplotlib.font_manager import findfont,FontProperties
    font=findfont(FontProperties(family='DejaVu Sans',weight='bold'))
    doc=base.fitz.open();page=doc.new_page(width=320*MM,height=334*MM)
    boxes={'A':(0,1,320,108),'B':(3,115,142,239),'C':(144,115,317,231),
        'D':(3,250,218,330),'E':(225,250,317,330)}
    letters={'B':(3,112),'C':(144,112),'D':(3,247),'E':(224,247)}
    page.insert_font(fontname='FigureBold',fontfile=font)
    records={}
    for label,box in boxes.items():
        path=OUT/'figures'/f'fig6-panel{label.lower()}.pdf'
        with base.fitz.open(path) as src:
            x0,y0,x1,y1=box
            scale=min((x1-x0)*MM/src[0].rect.width,(y1-y0)*MM/src[0].rect.height)
            rect=base.fitz.Rect(x0*MM,y0*MM,x0*MM+src[0].rect.width*scale,y0*MM+src[0].rect.height*scale)
            page.show_pdf_page(rect,src,0,keep_proportion=True)
        if label in letters:
            page.insert_text(tuple(v*MM for v in letters[label]),label,fontsize=LETTER,fontname='FigureBold')
        if label in SIZES_MM: np.testing.assert_allclose(scale,1,rtol=0,atol=1e-6)
        records[label]=dict(source=str(path),box_mm=[v/MM for v in rect],scale=scale)
    path=OUT/'figures/fig6-complete-layout.pdf';doc.save(path,garbage=4,deflate=True);doc.close()
    with base.fitz.open(path) as doc:
        doc[0].get_pixmap(matrix=base.fitz.Matrix(220/72,220/72),alpha=False).save(path.with_suffix('.png'))
        path.with_suffix('.svg').write_text(doc[0].get_svg_image(text_as_path=False))
    base.write(OUT/'layout_manifest.json',dict(canvas_mm=[320,334],panels=records,
        bottom_row=['Model Onset A','Reference seizure A','Model Onset B','Reference seizure B','E'],
        typography=dict(family='DejaVu Sans',label_pt=LABEL,tick_pt=TICK,title_pt=TITLE,panel_letter_pt=LETTER,
            basis='Final rendered font sizes of existing panel A; B/C/D authored at final physical dimensions.')))


def documents(meta,match):
    a,b=match['matches']; matrix=match['signed_similarity_matrix']
    (OUT/'caption_zh.md').write_text(f'''Figure 6：双次发作进入、恢复以及模型驱动的患者参考配对。

A 的原生波形、1–8 时刻与前版保持一致。B 保留 0–118 s 完整轨迹及 Interictal 小窗，Z=0.75–1、eta_M M=0–1.5 mV、全体 E 率 0–200 Hz；本版仅放大并统一标签字号。C 两行改为 Model Onset 1 与 Model Onset 2；八个空间率窗口及两排色条均不变，仍逐一对应 A。

D 从左到右为 Model Onset A、Reference seizure A、Model Onset B、Reference seizure B。模型 A/B 是同一连续种子的第一/第二次示例，能量窗口仍为 8.85–9.85 s 与 59.30–60.30 s，图内按作者要求省略秒数。以每个模型为查询，在 E1146 全部 25 次具有完整 15 触点、临床 onset 后 0–10 s、CAR、1–150 Hz 及 EEG 前 [-120,-90] s robust-z 基线的发作中，按冻结 shared-A identity 核平滑后的有符号 Pearson 场相关最大选择；两例最高者同时满足旧协议的平均增强为正条件，不加 TA/TB 类型限制、不翻转符号、不镜像。

模型 A 选中 {a['patient']['public_seizure']}（既有 {a['patient']['existing_type']} 标签），r={matrix[0,0]:+.6f}；模型 B 选中 {b['patient']['public_seizure']}（既有 {b['patient']['existing_type']} 标签），r={matrix[1,1]:+.6f}。r_A/r_B 分别指与本版 Reference seizure A/B 的相关；交叉相关为模型 A 对参考 B {matrix[0,1]:+.6f}、模型 B 对参考 A {matrix[1,0]:+.6f}。参考 A/B 是配对编号，不是 TA/TB 类型名。各场单独 min–max 显示，评分使用原始 robust-z 场。25 次完整排名保留在 all_patient_seizure_matches.csv。

本次 D 是作者指定的“寻找最像的患者发作”展示：相关系数是筛选后的最大值，不能作为独立预测或泛化检验，也不替换模型自身的模板分类。E 原样保留历史 13 人队列结果，没有使用本轮选例重新计算。D/E 位于同一行，竖色条紧邻第四个场图。字号在合成后按 A 的字号统一，字体为 DejaVu Sans。当前为候选，待作者目视检查；正式入口不变。
''')
    descriptions={
        'fig6-complete-layout':'完整 A–E：B/C 标签字号放大，C 改为 Model Onset 1/2。底部四幅模型/患者场与 E 排成同一行。',
        'fig6-panela':'连续两次发作的原生波形及八个状态编号。与前版字节一致。',
        'fig6-panelb':'完整状态轨迹与 Interictal 小窗保持原范围和原生曲线。轴、刻度与小窗文字按整图实际尺寸放大。',
        'fig6-panelc':'两行分别为 Model Onset 1 和 Model Onset 2，各保留四个原生空间状态。时刻及每排色条与前版一致。',
        'fig6-paneld':'模型 A、其最相似患者参考、模型 B、其最相似患者参考横排。各模型的两个有符号相关都标出，竖色条贴近场图。',
        'fig6-panele':'历史队列统计原样保留。没有用本轮最大相关选例重算队列。',
        'fig6-panelb-full-trajectory':'全范围 B 的独立大图保留。与前版字节一致。',
        'fig6-panelb-interictal':'Interictal 区域的独立大图保留。与前版字节一致。'}
    (OUT/'figures/README.md').write_text('\n\n'.join(f'### {name}.png / .pdf / .svg\n\n{desc}\n**关注点**：'
        +('患者参考通过本模型的最高有符号场相关选择，是描述性展示。' if name in ['fig6-paneld','fig6-complete-layout']
          else '数据及采样语义保持，当前候选待作者目视检查。') for name,desc in descriptions.items())+'\n')
    (OUT/'README.md').write_text(f'# Figure 6：统一字号与模型驱动配对\n\n'
        '[完整图](figures/fig6-complete-layout.png) · [PDF](figures/fig6-complete-layout.pdf) · '
        '[D 配对图](figures/fig6-paneld.png) · [全量患者排名](all_patient_seizure_matches.csv)\n\n'
        f'复现：`python {Path(__file__).resolve()}`。\n')


def main():
    pointer=base.MAIN/'results/paper-ready-figure/fig6/current_version.json';pointer_bytes=pointer.read_bytes()
    shutil.copytree(SOURCE,OUT,dirs_exist_ok=True,ignore=shutil.ignore_patterns('visual_review.json','validation.json',
        'patient_reference_comparisons.json','patient_reference_arrays.npz'))
    meta=json.loads((SOURCE/'figure_metadata.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans','pdf.fonttype':42,
        'svg.fonttype':'none','path.simplify':False})
    print('Rendering B/C at final figure font sizes.',flush=True)
    bmeta=draw_b(meta);cmeta=draw_c(meta)
    print('Matching each model to all 25 complete E1146 seizure fields.',flush=True)
    fz,match=find_matches();dmeta=draw_d(fz,match);compose()
    meta.update(producer=str(Path(__file__).resolve()),previous_candidate=str(SOURCE),status='PENDING_AUTHOR_VISUAL_REVIEW',
        B_display=bmeta,C_display=cmeta,D=dmeta,
        revision_scope='Final-size typography, C row names, model-first patient selection and single-row D/E layout.',
        patient_reference_confirmation='Author requested the most similar patient seizure for each model onset; A/B name the paired examples and frozen TA/TB labels are reported separately.',
        patient_reference_selection='Maximum signed model-to-seizure field correlation across 25 complete same-patient seizures.')
    meta['spatial_maps']['row_labels']=cmeta['row_labels']
    meta['trajectory']['display_revision']=bmeta
    base.write(OUT/'figure_metadata.json',meta)
    files=[]
    for stem in ['fig6-panela','fig6-panele','fig6-panelb-full-trajectory','fig6-panelb-interictal']:
        for ext in ['png','pdf','svg']:
            name=f'{stem}.{ext}'
            assert (OUT/'figures'/name).read_bytes()==(SOURCE/'figures'/name).read_bytes();files.append(name)
    for name in ['trajectory_data.npz','stage_readouts.npz','field_comparisons.json','A_C_stage_correspondence.csv']:
        assert (OUT/name).read_bytes()==(SOURCE/name).read_bytes()
    checks={}
    for stem in ['fig6-panelb','fig6-panelc','fig6-paneld','fig6-complete-layout']:
        with base.fitz.open(OUT/'figures'/f'{stem}.pdf') as doc:
            checks[stem]=all(page.rect.contains(base.fitz.Rect(span['bbox'])) for page in doc
                for block in page.get_text('dict')['blocks'] if block['type']==0
                for line in block['lines'] for span in line['spans'])
            if stem=='fig6-paneld':
                text=''.join(page.get_text() for page in doc)
                assert '8.85' not in text and '59.30' not in text
    assert all(checks.values()),checks
    assert pointer.read_bytes()==pointer_bytes
    base.write(OUT/'validation.json',dict(unchanged_panel_files=files,A_C_native_times_and_fields_unchanged=True,
        trajectory_and_model_energy_arrays_unchanged=True,all_25_patient_cases_checkpoint_parity=True,
        model_selected_references_are_signed_maxima=True,all_four_signed_correlations_saved=True,
        B_C_D_at_final_physical_dimensions=True,all_new_PDF_text_inside_canvas=checks,
        formal_pointer_unchanged=True,human_visual_acceptance='PENDING'))
    documents(meta,match)
    shutil.copy2(__file__,OUT/'source'/Path(__file__).name)
    print(json.dumps(dict(output=str(OUT),matched=[m['winner'] for m in match['matches']],checks=checks),indent=2))


if __name__=='__main__': main()
