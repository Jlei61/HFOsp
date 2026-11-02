#!/usr/bin/env python3
"""Extend the native interictal inset continuously into the first entry."""
import json
from pathlib import Path
import shutil

import build_fig6_selected_r08_r09_integrated_detail as previous

base, recovery = previous.base, previous.recovery
np, plt = previous.np, previous.plt
SOURCE = previous.OUT
OUT = SOURCE.with_name('selected_r08_r09_entry_runup_20261031')
WINDOW = (5.90, 9.24)
BOUNDS = np.array([[.735,.830],[.015,.055],[0.,250.]])
ENTRY = 8.85  # Unchanged A/C state 1, not a new detector threshold.


def export(fig, stem):
    # Reuse the exact B layout; remove both old time-range annotations.
    for item in list(fig.texts):
        if item.get_text() == '5.90–8.27 s':
            item.remove()
        elif item.get_text().startswith('Interictal'):
            item.set_text('Interictal → entry')
    previous.previous.OUT = OUT
    previous.previous.export(fig,stem)


def add_detail(ax,t,xyz,*,inset):
    from scipy.signal import find_peaks
    use=(t>=WINDOW[0])&(t<WINDOW[1]);st,sx=t[use],xyz[use]
    segments,keep=recovery.add_curve(ax,st,sx,BOUNDS)
    assert keep.all(), 'The complete continuous interval must fit inside the inset.'
    mid=(st[:-1]+st[1:])/2
    ax.collections[-1].set_linewidth(np.where(mid>=ENTRY,1.35 if inset else 1.8,.85 if inset else 1.2))
    ax.view_init(elev=22,azim=-115)
    ax.set_box_aspect((1.55,1.,1.25),zoom=1.04 if inset else 1.06)
    ax.set(xticks=[.74,.83],yticks=[.015,.05],zticks=[0,250],xlabel='$Z$',
        ylabel=r'$\eta_M M$' if inset else r'$\eta_M M$ (mV)',zlabel=r'$r_E$' if inset else '')
    ax.set_yticklabels(['','0.05'])
    ax.tick_params(labelsize=previous.TICK,pad=-1 if inset else 1)
    for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
        axis.label.set_fontsize(previous.TITLE if inset else previous.LABEL)
        axis.labelpad=-2 if inset else 4
    ax.figure.canvas.draw()
    # Preserve the same three illustrative ascent arrows as the previous inset.
    original=st<8.27
    peaks,_=find_peaks(sx[original,2],height=50.,prominence=40.,distance=30)
    assert len(peaks)==9
    arrows=[]
    for i in peaks[1::3]:
        candidates=np.arange(max(1,i-10),i)
        middle=candidates[np.argmin(abs(sx[candidates,2]-.5*sx[i,2]))]
        arrows.append((middle-1,middle+1,'brief ascent'))
    # The last arrow is on the recorded trajectory, not an extrapolation.
    arrows.append((len(st)-9,len(st)-1,'first entry continuation'))
    records=[]
    for lo,hi,kind in arrows:
        q=np.array([base.proj3d.proj_transform(*p,ax.get_proj())[:2] for p in sx[[lo,hi]]])
        ax.annotate('',xy=q[1],xytext=q[0],arrowprops=dict(arrowstyle='-|>',
            color=plt.colormaps['viridis'](((st[lo]+st[hi])/2)/recovery.HORIZON),
            lw=1.,mutation_scale=(11 if inset else 15) if kind=='first entry continuation' else (8 if inset else 12),
            shrinkA=0,shrinkB=0),zorder=15)
        records.append(dict(from_time_s=float(st[lo]),to_time_s=float(st[hi]),kind=kind))
    recovery.check_projected(ax.figure,ax,segments)
    history=recovery.history()
    events=[{k:event[k] for k in ['event_detection_index','start_s','end_s','phase','participation']}
        for event in history['briefs'] if WINDOW[0]<=event['start_s']<history['episodes'][0]['process_start_s']]
    assert all(e['end_s']<WINDOW[1] for e in events)
    assert WINDOW[1]>8.96 and WINDOW[1]<history['episodes'][0]['global_high_s']
    return dict(time_s=WINDOW,limits=BOUNDS,samples=len(st),segment_count=int(keep.sum()),
        selection='Continuous original interval retaining previous nine-event slice and extending into first entry; no reordering, extrapolation or rescaling of time colors.',
        detected_preentry_brief_events=events,both_core_event_count=sum(e['participation']=='both_cores' for e in events),
        previous_nine_event_slice_s=[5.90,8.27],previous_slice_preserved=True,
        entry_highlight_start_s=ENTRY,entry_highlight_definition='Existing A/C Local state 1',
        first_global_high_s=history['episodes'][0]['global_high_s'],
        end_is_before_global_high=True,all_native_segments_visible=True,
        direction_arrows=records,time_color_scale_s=[0,recovery.HORIZON],
        minimum_Z_etaM_rate=sx.min(0),maximum_Z_etaM_rate=sx.max(0),
        annotation_state_numbers=[],time_range_text_displayed=False,title='Interictal → entry',
        rate_bin_ms=5,display_smoothing_sigma_ms=5)


def documents(meta):
    caption=(SOURCE/'caption_zh.md').read_text()
    old=caption.split('\n\n')[1]
    new='A、C 与上一版完全一致，八个时刻保持对应。B 主图保持完整 0–118 s 轨迹及 1–4 标记；小窗取消具体时间标注，标题为 Interictal → entry。小窗原有连续九次主要间期活动完整保留，并将窗口从 5.90–8.27 s 延长至 5.90–9.24 s，包含随后局部再激活、扩散及进入发作的持续抬升段。显示范围为 Z=0.735–0.830、eta_M M=0.015–0.055 mV、全体 E 率 0–250 Hz；所有原生样本及相邻线段均保留。自 A/C 既有 1 时刻 8.85 s 起稍加粗，末端箭头沿真实记录的轨迹指向，不作外推。小窗终点早于全局高活动确认的 9.39 s，后续完整发作由 B 主图展示。该段是已实现的进入轨迹，不把画面视为预测证据或解析分岔证明。小窗和主图仍共享同一 0–118 s 时间色条。'
    (OUT/'caption_zh.md').write_text(caption.replace(old,new))
    readme=(SOURCE/'figures/README.md').read_text()
    blocks=readme.split('\n\n### ')
    for j,block in enumerate(blocks):
        head=block.split('\n',1)[0].removeprefix('### ')
        if head.startswith('fig6-panelb.png'):
            blocks[j]=('### ' if j==0 else '')+head+'\n\nB 主图保持原版完整轨迹，Interictal → entry 小窗取消具体时间标注。小窗保留原有九次主要间期活动，并连续延长至第一次进入段，稍加粗进入后的真实轨迹。\n**关注点**：末端箭头是实测样本之间的方向，后续完整发作仍在主图。'
        elif head.startswith('fig6-panelb-interictal.png'):
            blocks[j]=('### ' if j==0 else '')+head+'\n\n独立放大小窗的同一段间期至进入轨迹。九次主要事件及后续持续抬升连续显示，不拼接或外推。\n**关注点**：该小窗只显示进入过程的早期部分。'
    (OUT/'figures/README.md').write_text('\n\n### '.join(blocks))
    (OUT/'README.md').write_text('# Figure 6：间期活动连续延长至第一次进入\n\n'
        '[完整图](figures/fig6-complete-layout.png) · [PDF](figures/fig6-complete-layout.pdf) · '
        '[间期至进入细节](figures/fig6-panelb-interictal.png)\n\n'
        f'复现：`python {Path(__file__).resolve()}`。本次仅修改 B 小窗，A/C/D/E 均保持。\n')


def main():
    pointer=base.MAIN/'results/paper-ready-figure/fig6/current_version.json';frozen=pointer.read_bytes()
    shutil.copytree(SOURCE,OUT,dirs_exist_ok=True,ignore=shutil.ignore_patterns('validation.json','visual_review.json'))
    plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans','pdf.fonttype':42,
        'svg.fonttype':'none','path.simplify':False})
    previous.SOURCE=SOURCE;previous.OUT=OUT
    previous.add_interictal=add_detail;previous.export=export
    meta=json.loads((SOURCE/'figure_metadata.json').read_text())
    bmeta=previous.draw_b(meta)
    previous.compose()
    meta.update(producer=str(Path(__file__).resolve()),previous_candidate=str(SOURCE),B_display=bmeta,
        revision_scope='B inset: remove explicit times and extend continuously from interictal activity into first entry.',
        status='PENDING_AUTHOR_VISUAL_REVIEW')
    meta['trajectory'].update(display_revision=bmeta,inset_time_s=WINDOW,
        inset_title='Interictal → entry',
        inset_bounds_rationale='Retain all original interictal excursions and early first-entry continuation in a linear viewport; later high activity remains in main view.')
    meta['trajectory']['outputs']['fig6-panelb']=bmeta
    meta['trajectory']['outputs']['fig6-panelb-interictal']=bmeta['standalone_interictal']
    base.write(OUT/'figure_metadata.json',meta)
    base.write(OUT/'interictal_display_selection.json',bmeta['inset'])
    copied=[]
    for stem in ['fig6-panela','fig6-panelc','fig6-paneld','fig6-panele','fig6-panelb-full-trajectory']:
        for ext in ['png','pdf','svg']:
            name=f'{stem}.{ext}';assert (OUT/'figures'/name).read_bytes()==(SOURCE/'figures'/name).read_bytes();copied.append(name)
    for name in ['trajectory_data.npz','stage_readouts.npz','field_comparisons.json','A_C_stage_correspondence.csv',
        'model_selected_patient_matches.json','model_selected_patient_arrays.npz','all_patient_seizure_matches.csv','cohort_statistics_selected.json']:
        assert (OUT/name).read_bytes()==(SOURCE/name).read_bytes()
    checks={}
    for stem in ['fig6-panelb','fig6-panelb-interictal','fig6-complete-layout']:
        with base.fitz.open(OUT/'figures'/f'{stem}.pdf') as doc:
            checks[stem]=all(page.rect.contains(base.fitz.Rect(s['bbox'])) for page in doc
                for b in page.get_text('dict')['blocks'] if b['type']==0 for line in b['lines'] for s in line['spans'])
            if stem!='fig6-complete-layout':
                text=doc[0].get_text();assert '5.90' not in text and '8.27' not in text and '9.24' not in text
                assert 'Interictal' in text and 'entry' in text
    assert all(checks.values()),checks
    assert pointer.read_bytes()==frozen
    base.write(OUT/'validation.json',dict(unchanged_panel_files=copied,
        all_underlying_trajectory_fields_stage_times_matches_and_statistics_unchanged=True,
        original_nine_event_slice_retained=True,inset_contiguous_through_first_entry=True,
        inset_all_native_samples_and_segments_visible=True,B_shared_normalization_and_colorbar=True,
        no_inset_time_text=True,PDF_text_inside_canvas=checks,formal_pointer_unchanged=True,
        human_visual_acceptance='PENDING'))
    documents(meta)
    shutil.copy2(__file__,OUT/'source'/Path(__file__).name)
    print(json.dumps(dict(output=str(OUT),checks=checks),indent=2))


if __name__=='__main__':main()
