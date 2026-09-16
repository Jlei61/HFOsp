"""Exhaustive fixed-example review, independent of candidate score or completion order."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


def render(units, cases, seeds, patients, out, label, event_population='原primary合格事件', support_only=False):
    """Use already loaded full trajectories; never alter fitting or select by patient distance."""
    root=Path(out)/('participant_conditioned_time_review' if support_only else 'all_condition_time_review')
    figures=root/'figures';figures.mkdir(parents=True,exist_ok=True)
    display=[f'SCL{i}' for i in range(9,5,-1)]+[f'ICL{i}' for i in range(11,0,-1)]
    assert len(seeds)==2
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10,'pdf.fonttype':42})
    selections=[];files=[];descriptions=[]
    with PdfPages(root/'all_conditions.pdf') as pdf:
        for case in cases:
            data={}
            for mode,mode_name in [(1,'TA'),(0,'TB')]:
                pat=patients[mode_name];order=[list(pat['names']).index(n) for n in display]
                data[(mode,0)]=dict(mass=pat['mass'][order],mask=pat['mask'][order],event=int(pat['event_id']),n=None)
                for column,seed in enumerate(seeds,1):
                    unit=units.get((case['id'],seed));entry=None
                    if unit is not None:
                        result,arrays,ids=unit
                        selected=ids[arrays['event_mode'][ids]==mode]
                        if len(selected):
                            phi=arrays['event_phi'][selected]
                            index=int(selected[np.argmin(np.sum((phi-phi.mean(0))**2,axis=1))])
                            lo,hi=result['events'][index]['window_ms']
                            order=[list(arrays['contact_names']).index(n) for n in display]
                            dt=float(arrays['contact_envelope_dt_ms'])
                            mass=arrays['contact_envelope'][round(lo/dt):round(hi/dt)][:,order].T
                            assert mass.shape[0]==15 and abs(mass.shape[1]*dt-250)<1e-6
                            entry=dict(mass=mass,mask=np.isfinite(arrays['centroid_ms'][index])[order],event=index,n=len(selected))
                            selections.append(dict(candidate=case['id'],seed=seed,mode=mode_name,event_index=index,
                                window_ms=[lo,hi],mode_n=len(selected),analysis_n=len(ids),arrays_sha256=result['arrays_sha256']))
                    data[(mode,column)]=entry
            for scale in ['event','contact']:
                fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
                for row,(mode,mode_name) in enumerate([(1,'TA'),(0,'TB')]):
                    for column in range(3):
                        ax=axes[row,column];entry=data[(mode,column)]
                        prefix='Fig2C 源事件包络' if column==0 else f'模型噪声 {seeds[column-1]}'
                        if entry is None:
                            state='该运行尚无完整输出' if column and units.get((case['id'],seeds[column-1])) is None else '完整运行的本事件集合中无可读样例'
                            ax.set_facecolor('black');ax.text(.5,.5,state,color='white',ha='center',transform=ax.transAxes)
                            ax.set(title=f'{prefix} · {mode_name}',xticks=[],yticks=[])
                            continue
                        mass=entry['mass'].astype(float,copy=True)
                        if support_only:mass[~entry['mask']]=np.nan
                        maximum=np.max(np.where(np.isfinite(mass),mass,0),axis=1,keepdims=True)
                        denominator=max(float(maximum.max()),1e-20) if scale=='event' else np.maximum(maximum,1e-20)
                        cmap=plt.get_cmap('magma').copy();cmap.set_bad('#808080')
                        im=ax.imshow(mass/denominator,aspect='auto',extent=[0,250,14.5,-.5],cmap=cmap,vmin=0,vmax=1,interpolation='nearest')
                        ax.axhline(3.5,color='#66bbbb',lw=.7)
                        suffix='' if column==0 else f'，该类 n={entry["n"]}'
                        ax.set(title=f'{prefix} · {mode_name} · 事件 {entry["event"]}{suffix}',
                            yticks=range(15),yticklabels=[n+(' *' if not ok else '') for n,ok in zip(display,entry['mask'])],xlabel='原始窗口时间 (ms)')
                        ax.tick_params(axis='y',labelsize=8)
                fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.65,label='包络 / '+('整事件峰值' if scale=='event' else '各触点峰值'))
                layer='仅显示参与触点；灰行为未参与' if support_only else '全触点窗口 QC；* 为未参与但仍保留信号'
                fig.suptitle(label(case)+'：患者与两条配对噪声\n患者：HFO 包络；模型：发放密度包络；'+event_population+'\n各模型例取自身模式均值附近，不按患者相似度挑选；完整 250 ms、不拉伸；'+layer)
                path=figures/f'{case["id"]}_{scale}_scale.png'
                fig.savefig(path,dpi=150,bbox_inches='tight');pdf.savefig(fig,bbox_inches='tight');plt.close(fig)
                with Image.open(path) as image:image.load()
                files.append(dict(file=path.name,candidate=case['id'],scale=scale,pdf_page=len(files)+1))
                descriptions.append('### '+path.name+'\n\n'+label(case)+'的两条配对噪声，患者 Fig2C 源事件 TA/TB 的 HFO 包络固定在左，模型分别在中、右列；这里不是 Fig2C 左侧的原始 STFT 频谱。每例最接近模型自身模式特征均值，按'+('整事件' if scale=='event' else '各触点')+'峰值归一化，'+layer+'，保留原始 250 ms 时标；模型发放密度与患者 HFO 幅度不可物理等同。**关注点**：SCL/ICL 参与、具体时序与宽度；示例不是该条件全部分布，图中的 n 仅说明模式观测支持。')
    (figures/'README.md').write_text('\n\n'.join(descriptions)+'\n')
    (root/'manifest.json').write_text(json.dumps(dict(status='RENDERED_PENDING_SCIENTIFIC_REVIEW',cases=[c['id'] for c in cases],seeds=seeds,
        pending_units=[dict(candidate=c['id'],seed=s) for c in cases for s in seeds if (c['id'],s) not in units],
        event_population=event_population,display_population='participants only; other rows gray' if support_only else 'all contacts including nonparticipants; QC only',rule='All supplied conditions in supplied order. One event nearest its own run-mode feature mean; supplied condition selection must be reported by the caller.',
        selections=selections,figures=files,pdf_pages=len(files),scientific_acceptance=False),indent=2))
    (root/'README.md').write_text('# 传入条件的时序审阅\n\n按调用者提供的条件及顺序呈现，条件池选择由上游 provenance 说明；每条件两页，分别为整事件与逐触点峰值归一化。患者固定为 Fig2C 的 TA 6344、TB 937，展示 HFO 包络而非原图的 STFT 频谱；患者原示例经过方向可读性筛选，不能视作全模式分布的无条件典型。'+layer+'。同一张网络的两条模型噪声分列展示，代表事件最接近本运行自身类均值。样例用于定位形态差异，需结合全部事件分布、全触点 QC 和原生活动，不能据此接受机制恢复。所有页面在 all_conditions.pdf；索引与选例在 manifest.json。\n')
    return dict(output=str(root),conditions=len(cases),pages=len(files))
