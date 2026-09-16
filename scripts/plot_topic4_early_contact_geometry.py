"""Where do the three earliest participating centroids fall?

This is an observation diagnostic in the shared 2D projection, not a neural
source estimator or a new training constraint. All participating-contact masks
remain frozen. Patient FIT is development evidence, not independent validation.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scripts import analyze_topic4_propagation_recovery_night as s


def summary(x,names,xy):
    ok=np.isfinite(x).sum(1)>=3;x=np.asarray(x)[ok]
    if not len(x):return dict(n=0,early_probability=None,mixed_rods_fraction=None)
    order=np.argsort(np.where(np.isfinite(x),x,np.inf),axis=1,kind='stable')
    ix=order[:,:3];pos=xy[ix];width=np.ptp(pos,axis=1)
    mixed=np.any(np.char.startswith(names[ix],'SCL'),1)&np.any(np.char.startswith(names[ix],'ICL'),1)
    probs=np.bincount(ix.ravel(),minlength=len(names))/len(x)
    assert abs(probs.sum()-3)<1e-12
    values=np.take_along_axis(np.where(np.isfinite(x),x,np.inf),order,axis=1)
    tie=np.isfinite(values[:,3])&np.isclose(values[:,2],values[:,3],rtol=0,atol=1e-9)
    return dict(n=len(x),early_probability=probs.tolist(),mixed_rods_fraction=float(mixed.mean()),
        x_span_mm=np.quantile(width[:,0],[.05,.5,.95]).tolist(),y_span_mm=np.quantile(width[:,1],[.05,.5,.95]).tolist(),
        earliest_three_centroid_span_ms=np.quantile(values[:,2]-values[:,0],[.05,.5,.95]).tolist(),
        ties_at_third_fourth=int(tie.sum()),participation=np.isfinite(x).mean(0).tolist())


def main():
    parent=s.rt.read(s.an.run.PARENT);ev=s.rt.load_evaluator(parent);names=np.asarray(s.rt.load_observation_contract(parent)['contact_names'])
    p=s.an.run.result_path('recovery_long_20260911','refine_mid_EE075',2511,847101)
    with np.load(p.with_suffix('.npz')) as z:
        assert np.array_equal(z['contact_names'],names);xy=z['contact_xy_mm']
    patient={mode:summary(np.asarray(ev.fit)[np.asarray(ev.fit_labels)==label],names,xy) for label,mode in [(1,'TA'),(0,'TB')]}
    candidates=['refine_mid_EE075','refine_mid_EE085_mean095','refine_midpoint_EE085_A115_mean095']
    bp=s.night.OUT/'final_B_selection.json'
    if bp.exists():candidates.extend(c['id'] for c in s.rt.read(bp)['candidates'])
    model={};metadata={};sources=[]
    for cid in candidates:
        c=s.rt.read(s.an.run.OUT/'candidates'/f'{cid}.json');metadata[cid]=c
        for seed in [847101,847102]:
            p=s.an.run.result_path(c['stage'],cid,2511,seed)
            if not s.an.run.complete(p):continue
            r=s.rt.read(p)
            with np.load(p.with_suffix('.npz')) as z:
                assert np.array_equal(z['contact_names'],names) and np.array_equal(z['contact_xy_mm'],xy)
                ids=np.intersect1d(z['primary_event_indices'],s.an.all_detected_ids(r,1500.))
                model[(cid,seed)]={mode:summary(z['centroid_ms'][ids[z['event_mode'][ids]==label]],names,xy) for label,mode in [(1,'TA'),(0,'TB')]}
            sources.append(dict(candidate=cid,seed=seed,path=str(p),arrays_sha256=r['arrays_sha256']))
    dest=s.night.OUT/'early_contact_geometry';F=dest/'figures';F.mkdir(parents=True,exist_ok=True)
    rows=[dict(source='patient_FIT',mode=mode,**v) for mode,v in patient.items()]
    rows.extend(dict(source=cid,seed=seed,mode=mode,**v) for (cid,seed),mm in model.items() for mode,v in mm.items())
    s.rt.write(dest/'data.json',dict(patient=patient,rows=rows,contact_names=names.tolist(),xy_mm=xy.tolist(),sources=sources,
        question='Do the earliest three participating contact centroids occupy both shafts?',
        observable='Per-contact fraction among first3, and 2D x/y range of these contacts. First3 is based on centroids, not onset or causal neural origin.',
        ties='Stable contact-order tie-break; third/fourth ties counted explicitly.',
        source_unit='Patient FIT events or one fixed model topology/noise60s replay; no pooling noise runs.',
        not_training=True,not_independent_validation=True,producer=__file__,producer_sha256=s.rt.sha(__file__)))
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':9,'pdf.fonttype':42})
    panels=[('current',candidates[:3],['上移3mm；EE0.75','上移3mm；EE0.85\n输入均值0.95','中点；EE0.85\n均值0.95、左核降幅1.15'])]
    for recipe,title in [('mid_EE075','EE0.75'),('mid_EE085_mean095','EE0.85、均值0.95')]:
        ids=['refine_'+recipe,'coreOU_'+recipe+'_rho050','coreOU_'+recipe+'_rho000']
        if all((cid,seed) in model for cid in ids for seed in [847101,847102]):panels.append((recipe,ids,[title+'\nρ=1',title+'\nρ=0.5',title+'\nρ=0']))
    files=[]
    for slug,ids,titles in panels:
      for seed in [847101,847102]:
        fig,axes=plt.subplots(2,4,figsize=(12,6.8),layout='constrained')
        for col,(cid,title) in enumerate([(None,'患者 FIT')]+list(zip(ids,titles))):
          for row,mode in enumerate(['TA','TB']):
            ax=axes[row,col];v=patient[mode] if cid is None else model[(cid,seed)][mode]
            for shaft,color in [('ICL','#dc8722'),('SCL','#42a9b5')]:
                ix=np.char.startswith(names,shaft);ax.plot(xy[ix,0],xy[ix,1],c=color,lw=1,zorder=1)
            if v['n']:
                ax.scatter(*xy.T,c=v['early_probability'],cmap='Blues',vmin=0,vmax=1,s=55,edgecolor='.4',linewidth=.4,zorder=3)
            else:ax.scatter(*xy.T,facecolors='none',edgecolors='.6',s=55)
            if cid:
                c=metadata[cid]
                for center,radius in zip(c['centers_mm'],c['radii_mm']):ax.add_patch(Circle(center,radius,fill=False,edgecolor='#c54b46',lw=1))
            for name in ['SCL9','SCL6','ICL11','ICL1']:
                k=list(names).index(name);ax.annotate(name,xy[k],xytext=(-3,5) if name=='ICL1' else (2,5),ha='right' if name=='ICL1' else 'left',textcoords='offset points',fontsize=7)
            ax.set(xlim=(0,20),ylim=(0,20),aspect='equal',xticks=[0,10,20],yticks=[0,10,20],xlabel='x (mm)')
            if col==0:ax.set_ylabel(mode+'\ny (mm)')
            ax.set_title(title if row==0 else '',fontsize=10)
            text=f"n={v['n']}；跨两杆 {v['mixed_rods_fraction']:.0%}" if v['n'] else '本运行无该类合格事件'
            ax.text(.02,.97,text,transform=ax.transAxes,va='top',fontsize=8)
        fig.colorbar(ScalarMappable(Normalize(0,1),cmap='Blues'),ax=axes.ravel().tolist(),location='bottom',shrink=.6,aspect=45,pad=.06,label='该触点进入本事件最早3个参与质心的概率')
        fig.suptitle(f'早位触点的空间覆盖｜固定图2511，噪声{seed}\n原合格事件；蓝色越深表示越常进入最早3位；红圈为实际模型core，不是患者起源估计',fontsize=11)
        for ext in ['png','pdf']:
            file=F/f'{slug}_{seed}_early_contact_probability.{ext}';fig.savefig(file,dpi=200);files.append(file.name)
        plt.close(fig)
    (F/'README.md').write_text('\n\n'.join(f'### {file}\n\n在固定2D电极位置上，用颜色显示一个触点在该模式事件中进入最早三个参与质心的频率；未参与触点不进入排序，统计总和为3。每个模型面板只用标明的一条噪声重演，患者为开发FIT，模型为原primary；红圈仅表示模型易激范围。**关注点**：两杆是否在早位触点集合中共同出现，以及具体哪些触点缺失；质心顺序不等于神经元起燃位置，此图不进入损失或独立验证。' for file in files)+'\n')
    print({'output':str(dest),'complete_model_runs':len(model),'files':len(files)},flush=True)


if __name__=='__main__':main()
