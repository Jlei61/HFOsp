#!/usr/bin/env python3
"""Wide A; D/C middle; E/F bottom; slow traces retained as supplementary."""
import copy,hashlib,json,shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.patches import Circle
import build_fig6_unified_adaptation as base
import build_fig6_unified_adaptation_focus14 as focus

PARENT=base.OUT.with_name('unified_current_adaptation_readout14_20261030')
OUT=PARENT.with_name('unified_current_adaptation_rearranged_20261030')
AUDIT=base.MAIN/'results/topic4_sef_hfo/unified_adaptation_entry_switch_20261030'

def export_pdf(path):
    with base.fitz.open(path) as d:
        d[0].get_pixmap(matrix=base.fitz.Matrix(2.6,2.6),alpha=False).save(path.with_suffix('.png'))
        path.with_suffix('.svg').write_text(d[0].get_svg_image())

def main():
    base.OUT=OUT;(OUT/'figures').mkdir(parents=True,exist_ok=True)
    a=base.load('unified_ref48_s9108405')
    meta=json.loads((PARENT/'figure_metadata.json').read_text())
    sel=json.loads((PARENT/'display_selection.json').read_text());states=sel['states']
    audit=json.loads((AUDIT/'current_entry_audit.json').read_text())
    onset=audit['native']['episodes'][0]['onset_all_A_B_surround_s'][2]
    assert abs(onset-8.99)<1e-10
    focus.INTER_EVENT=sel['interictal_event'];focus.TERMINATION=sel['first_low_onset_s']
    base.native.WINDOW=focus.WINDOW;base.native.RECRUIT=9.56
    base.native.MAIN_BAND_HZ=(80.,1000.);base.native.GAIN_PERCENTILE=99.5
    base.native.DISPLAY_GAIN=.65;base.native.marks=focus.marks
    base.native.HIDDEN_STATE_NUMBERS={1,2,3,4}
    plt.rcParams.update({'font.size':15,'axes.labelsize':17,'axes.titlesize':16,
                        'xtick.labelsize':13,'ytick.labelsize':13,'pdf.fonttype':42,
                        'svg.fonttype':'none','path.simplify':False})
    fig=plt.figure(figsize=(15.7,6.3));ax=fig.add_axes([.075,.11,.905,.74])
    waveform=base.native.draw_a(ax,a,states,onset)
    for s,x in zip(states,[9.10,9.34,9.60,13.505]):
        ax.annotate(str(s['number']),xy=(s['time_s'],1),xycoords=ax.get_xaxis_transform(),
            xytext=(x,1.035),textcoords=ax.get_xaxis_transform(),ha='center',va='bottom',
            color=s['color'],weight='bold',fontsize=15,annotation_clip=False,
            arrowprops=dict(arrowstyle='-',lw=.6,color=s['color']))
    ax.annotate('Interictal',xy=(focus.INTER_EVENT['peak_s'],1),xycoords=ax.get_xaxis_transform(),
        xytext=(8.66,1.13),textcoords=ax.get_xaxis_transform(),ha='left',va='bottom',
        color=focus.INTER_COLOR,fontsize=14,annotation_clip=False,
        arrowprops=dict(arrowstyle='-',lw=.6,color=focus.INTER_COLOR))
    ax.set_xlabel('Time (s)');ax.tick_params(axis='x',labelbottom=True)
    assert np.array_equal(waveform['gain_per_contact'],meta['waveform']['gain_per_contact'])
    base.save(fig,'fig6-panela')
    # Same C samples and values, rearranged into a readable two-by-two block.
    fig=plt.figure(figsize=(8.3,7.4));gs=fig.add_gridspec(2,2,left=.10,right=.86,bottom=.10,top=.91,wspace=.18,hspace=.42)
    maps=[]
    for j,s in enumerate(states):
        ax=fig.add_subplot(gs[j//2,j%2]);idx=round(s['time_s']/.005)
        count=a['field_5ms'][idx-2:idx+2].sum(0);hz=count/a['cell_e_counts']/.02
        assert np.array_equal(hz,np.asarray(meta['spatial_maps'][j]['E_rate_Hz']))
        im=ax.imshow(hz.reshape(20,20),origin='lower',extent=[0,20,0,20],cmap='magma',norm=PowerNorm(.6,0,500),interpolation='nearest')
        for k,xy in enumerate(a['centers_mm']):
            ax.add_patch(Circle(xy,float(a['core_radius_mm']),ec='#2dd4cd',fc='none',lw=1.7))
            ax.text(xy[0],xy[1]+2.1,'AB'[k],color='#10656a',fontsize=12,ha='center',weight='bold',bbox=dict(fc='white',ec='none',pad=.2,alpha=.8))
        ax.set(xticks=[0,10,20],yticks=[0,10,20],xlabel='x (mm)')
        if j<2:ax.set_xlabel('')
        if j%2==0:ax.set_ylabel('y (mm)')
        else:ax.tick_params(labelleft=False)
        ax.set_title(f"{s['number']}  {s['label']}\n{s['time_s']:.3f} s",color=s['color'],fontsize=15,pad=9)
    cb=fig.colorbar(im,cax=fig.add_axes([.90,.16,.022,.64]),ticks=[0,250,500]);cb.set_label('E rate (Hz)')
    base.save(fig,'fig6-panelc')
    for ext in ['png','pdf','svg']:shutil.copy2(PARENT/'figures'/f'fig6-paneld.{ext}',OUT/'figures'/f'fig6-paneld.{ext}')
    # E is now anchored to native local recruitment, not the later global gate.
    c=json.loads((PARENT/'field_comparison.json').read_text())
    row=next(r for r in audit['fields'] if abs(r['window_s'][0]-onset)<1e-10)
    c.update(model_robust_z=row['robust_z'],model_early_s=row['window_s'],model_entry_s=onset,
        exact_entry_s=onset,model_early_log_power=row['log_power'],model_mean_z=float(np.mean(row['robust_z'])),
        model_positive_contacts=int((np.asarray(row['robust_z'])>0).sum()),
        high_threshold_onset_s=9.56,local_recruitment_onset_s=onset,
        source_onset_comparison=str(PARENT/'field_comparison.json'),
        producer=str(Path(__file__).resolve()),producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    c['contract']['model_onset']='First local core recruitment: core B>100Hz for50ms after global quiet, at8.99s; global200Hz gate remains9.56s.'
    c['selected']['field_r']=row['patient_field_r']
    c['selected']['contact_r']=float(np.corrcoef(c['model_robust_z'],c['patient_robust_z'])[0,1])
    from scipy.stats import spearmanr
    c['selected']['contact_rho']=float(spearmanr(c['model_robust_z'],c['patient_robust_z']).statistic)
    energy=base.native.energy;rec,_=energy.clinical._load_record('epilepsiae_1146');fz=energy.clinical.load_frozen('epilepsiae_1146')
    case=json.loads(Path(c['patient_source']).read_text())
    assert fz['fingerprint']==c['frozen_fingerprint']
    assert hashlib.sha256(Path(c['patient_source']).read_bytes()).hexdigest()==c['patient_source_sha256']
    energy.patient_cases=lambda:(rec,fz,[case])
    fig=plt.figure(figsize=(9.0,4.7));spec=fig.add_gridspec(1,1,left=.05,right=.94,top=.92,bottom=.18)[0]
    _,bars,_=base.native.draw_onset_fields(fig,spec,a,c,'minmax_z')
    bars[0][1].set_title('Model entry\n8.99–9.99 s',fontsize=15,weight='bold')
    base.save(fig,'fig6-panele');base.write(OUT/'field_comparison.json',c)
    # Remove the requested small n footer from F, keeping all data and p value.
    with base.fitz.open(PARENT/'figures/fig6-panelf.pdf') as src:
        clip=base.fitz.Rect(8,20,410,358);doc=base.fitz.open();page=doc.new_page(width=clip.width,height=clip.height)
        page.show_pdf_page(page.rect,src,0,clip=clip);doc.save(OUT/'figures/fig6-panelf.pdf',garbage=4,deflate=True)
    export_pdf(OUT/'figures/fig6-panelf.pdf')
    supp=OUT/'supplementary_candidates/figures';supp.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf','svg']:shutil.copy2(PARENT/'figures'/f'fig6-panelb.{ext}',supp/f'slow-state-traces.{ext}')
    doc=base.fitz.open();mm=72/25.4;page=doc.new_page(width=320*mm,height=330*mm)
    boxes={'A':(4,7,316,132),'D':(5,141,154,232),'C':(164,140,316,233),'E':(5,244,203,325),'F':(216,244,316,325)}
    letters={'A':(4,7),'D':(4,139),'C':(163,139),'E':(4,242),'F':(214,242)}
    manifest={}
    for label,box in boxes.items():
        srcpath=OUT/'figures'/f'fig6-panel{label.lower()}.pdf'
        with base.fitz.open(srcpath) as src:page.show_pdf_page(base.fitz.Rect(*(v*mm for v in box)),src,0,keep_proportion=True)
        page.insert_text(tuple(v*mm for v in letters[label]),label,fontsize=16,fontname='hebo')
        manifest[label]=dict(box_mm=box,source=str(srcpath))
    doc.save(OUT/'figures/fig6-complete-layout.pdf',garbage=4,deflate=True);export_pdf(OUT/'figures/fig6-complete-layout.pdf')
    base.write(OUT/'layout_manifest.json',dict(canvas_mm=[320,330],panels=manifest,
        arrangement='Wide A across former A+D; D left/C right middle; E left/F right bottom.',
        old_B='supplementary_candidates/figures/slow-state-traces.pdf',F_footer_removed=True))
    meta.update(waveform=waveform,local_recruitment_onset_s=onset,high_threshold_onset_s=9.56,
        energy_window_s=row['window_s'],field_r=row['patient_field_r'],rearranged_from=str(PARENT),
        shaded_intervals=[dict(state='recruitment_and_high',start_s=onset,end_s=focus.TERMINATION),
                          dict(state='post_high',start_s=focus.TERMINATION,end_s=focus.WINDOW[1])],
        F_scope='Unchanged previous-model cohort; n=13 remains in caption, not new-model validation.')
    base.write(OUT/'figure_metadata.json',meta)
    for name in ['scientific_result.json','trajectory_data.npz']:shutil.copy2(PARENT/name,OUT/name)
    sel.update(entry_process_start_s=onset,global_high_threshold_s=9.56,E_window_s=row['window_s'])
    base.write(OUT/'display_selection.json',sel)
    srcdir=OUT/'source';srcdir.mkdir(exist_ok=True);shutil.copy2(Path(__file__),srcdir/Path(__file__).name)
    valid={}
    for p in (OUT/'figures').glob('*.pdf'):
        with base.fitz.open(p) as d:
            valid[p.name]=all(pg.rect.contains(base.fitz.Rect(s['bbox'])) for pg in d for b in pg.get_text('dict')['blocks'] if b['type']==0 for l in b['lines'] for s in l['spans'])
    assert all(valid.values()),valid
    base.write(OUT/'validation.json',dict(status='PASS',C_state_values_unchanged=True,D_unchanged=True,
        A_gain_unchanged=True,E_patient_fixed=True,E_window_updated=[onset,onset+1],
        B_preserved_in_supplement=True,F_data_and_p_unchanged=True,PDF_text_inside=valid,human_visual_acceptance='PENDING'))
    (OUT/'figures/README.md').write_text('''### fig6-complete-layout.png / .pdf
A扩大为顶部通栏，中排左D右C，底排左E右F；原B移入补图候选。1–4代表时刻与空间率不变，E改为局部招募起点8.99–9.99 s。
**关注点**：局部招募起点与全局高活动阈值分开；候选待作者目视检查。

### fig6-panela.png / .pdf
同一8.65–14.00 s连续读出，沿用滤波、逐触点增益与0.65显示系数。高活动过程底纹从原生局部招募8.99 s开始。
**关注点**：不增加5或termination标记。

### fig6-panelc.png / .pdf
同一Local、Spread、Entry、Ictal四幅空间图重排为2×2，仍用原20 ms窗口与统一率色条。
**关注点**：编号3保留原全局阈值含义，不把它当作整个招募过程的起点。

### fig6-paneld.png / .pdf
完整60 s状态轨迹保持原值，移至原B所在的左侧区域。
**关注点**：第一次终止后的再进入仍可核查。

### fig6-panele.png / .pdf
局部B核启动后1 s的模型场与原SZ26患者场，沿用1–150 Hz robust-z和共享几何。
**关注点**：能量场相关与核招募顺序是两种读出，图中各场独立归一化。

### fig6-panelf.png / .pdf
保留历史13人队列的全部数据及p值，仅去掉小脚注和图内历史说明。
**关注点**：其历史模型身份仍在图注中明确，不作为新模型队列验证。
''')
    (supp/'README.md').write_text('''### slow-state-traces.png / .pdf
原B的Z与单变量适应电流逐字节保留，作为supplementary候选。仍对应原8.65–14.00 s轨迹。
**关注点**：仅改变放置位置，没有删除机制变量证据。
''')
    (OUT/'caption_zh.md').write_text('''A显示同一模型的8.65–14.00 s连续读出。局部招募从8.99 s开始：Core B率先持续激活，Core A约9.14 s被招募；9.56 s是全局E率达到200 Hz并持续200 ms的阈值时刻。1–4仍分别为9.100、9.300、9.560和13.505 s的Local、Spread、Entry、Ictal代表图，不增加termination编号。

D为同一模型完整60 s的Z、适应电流及E率轨迹。C为与原版完全相同的四幅20 ms空间率图，重排为2×2。

E以局部招募起点为锚，显示8.99–9.99 s的1–150 Hz CAR robust-z能量场，与原固定E1146 SZ26对照。F仍为旧模型的13名患者队列参考，p=0.0071，未重新拟合或计算，不能用作当前统一适应模型的队列验证；图内的小脚注移到本图注。原B另存supplementary候选。全部结果待作者目视检查，未替换正式入口。
''')
    (OUT/'README.md').write_text(f'''# Fig6 招募起点与大A重排候选

[完整图](figures/fig6-complete-layout.png) / [PDF](figures/fig6-complete-layout.pdf) / [图注](caption_zh.md)。顶部大A，中排左D右C，底排左E右F；原B保存在[supplementary候选](supplementary_candidates/figures/slow-state-traces.pdf)。

局部Core B招募从8.99 s开始，1的9.10 s已属于entry过程；9.56 s保留为全局高活动阈值。E使用局部起点后的完整1 s（8.99–9.99 s），患者仍为原SZ26。1–4时刻及C/D数据不变。F去掉小脚注，其原模型、13人样本和既有统计身份完整保留于图注。

本候选仍展示原seed9108405；新种子搜索另存于{AUDIT}，不把尚未完成的搜索当成阳性结果。正式入口未覆盖，待作者目视检查。

复现：`python {Path(__file__).resolve()}`。
''')
    print(OUT,flush=True)

if __name__=='__main__':main()
