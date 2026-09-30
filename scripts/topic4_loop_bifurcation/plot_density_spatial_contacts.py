#!/usr/bin/env python3
"""Native/density spatial events with original geometry and contact slots."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from campaign import ROOT, REPO, read, write, sha
from compare_density_resolution import sources, REPLAY
sys.path.insert(0,str(REPO/'scripts/topic4_zm_runaway_mechanism/frozen_v3'))
from native_readouts import readouts

DISPLAY=['SCL9','SCL8','SCL7','SCL6']+[f'ICL{i}' for i in range(11,0,-1)]


def choose(events,sign):
    pool=[e for e in events if 500<=e['start_ms'] and e['start_ms']+e['duration_ms']<=8000
          and sign*e['direction_axis_mm']>1]
    assert pool, ('No qualified event for direction',sign)
    x=np.array([[e['duration_ms'],e['area_fraction'],e['extent_mm']] for e in pool])
    med=np.median(x,axis=0);iqr=np.diff(np.percentile(x,[25,75],axis=0),axis=0)[0]
    score=np.mean(((x-med)/np.where(iqr>0,iqr,1))**2,axis=1)
    return pool[int(np.argmin(score))],len(pool)


def main():
    compare=read(ROOT/'density_contact_replay/comparison.json');assert compare['status']=='COMPLETE'
    allrows={r['name']:r for r in sources()}
    labels=['native9108401','R2048_num927612','R8192_num927611']
    titles=['Native 8401','Density 2048\nstream 2','Density 8192\nstream 1']
    geo=np.load(REPLAY/'geometry.npz');counts=geo['cell_e_counts'];centers=geo['centers_mm']
    envelopes=np.load(ROOT/'density_contact_replay/envelopes.npz')
    names=envelopes['contact_names'].tolist();order=[names.index(n) for n in DISPLAY]
    reference=envelopes['native9108401__firing']
    scale=np.maximum(np.quantile(reference[250:4000],.995,axis=0),1e-12)
    op=np.load(REPO/'results/topic4_sef_hfo/interictal_brunel_spatial_bifurcation_20260917/operators/g20/geometry.npz')
    data=[]
    for name in labels:
        row=allrows[name];events,_,_,_=readouts(row['t'],row['field'],row['counts'],name)
        if name.startswith('native'):
            zz=None
            for p in sorted((REPLAY/'runs/eta0.0005_s9108401/fields').glob('*.npz')):
                with np.load(p) as z:
                    if 80000 in z['zm_step']:
                        zz=np.bincount(z['cell_e'],weights=z['z'][np.flatnonzero(z['zm_step']==80000)[0]],minlength=400)/counts
                        break
            assert zz is not None
        else:
            with np.load(row['source']) as z:
                ee=z['population_E'];zz=np.bincount(op['group_cell'][ee],weights=z['group_Z'][7999,ee]*z['group_sizes'][ee],minlength=400)/counts
        selections=[choose(events,s) for s in [1,-1]]
        data.append((name,zz,selections,envelopes[f'{name}__firing']/scale))
    vmax=float(np.ceil(max(np.nanmax(e['onset'])-np.nanmin(e['onset']) for _,_,p,_ in data for e,_ in p)/10)*10)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(3,4,figsize=(15,10),gridspec_kw={'width_ratios':[1,1,1,2]},layout='constrained')
    meta=[]
    for j,((name,zz,selected,env),title) in enumerate(zip(data,titles)):
        ax=axs[j,0];imz=ax.imshow(zz.reshape(20,20),origin='lower',extent=(0,20,0,20),vmin=.6,vmax=1,cmap='cividis',interpolation='nearest')
        ax.set_ylabel(f'{title}\ny (mm)');ax.set_title('Z at 8 s')
        for k,(ev,n) in enumerate(selected,1):
            onset=ev['onset']-np.nanmin(ev['onset'])
            im=axs[j,k].imshow(onset.reshape(20,20),origin='lower',extent=(0,20,0,20),vmin=0,vmax=vmax,cmap='viridis',interpolation='nearest')
            direction='A → B' if k==1 else 'B → A'
            axs[j,k].set_title(f'{direction}: {ev["start_ms"]/1000:.3f} s')
            meta.append(dict(model=name,direction=direction,n_eligible=n,selected={a:b for a,b in ev.items() if a!='onset'}))
        for ax in axs[j,:3]:
            for i,(center,color) in enumerate(zip(centers,['#d34e99','#249ac1'])):
                ax.add_patch(Circle(center,1.5,fill=False,lw=1.,edgecolor=color))
                ax.text(center[0],center[1]+1.9,'AB'[i],ha='center',color=color,fontsize=8)
            ax.set(xlabel='x (mm)',xticks=[0,10,20],yticks=[0,10,20])
        ax=axs[j,3]
        imc=ax.imshow(env[1500:2250,order].T,origin='upper',aspect='auto',extent=(3,4.5,14.5,-.5),vmin=0,vmax=1.5,cmap='magma',interpolation='nearest')
        ax.axhline(3.5,c='white',lw=.7);ax.set(yticks=range(15),yticklabels=DISPLAY,xlabel='Time (s)',title='Contact firing envelope')
    fig.colorbar(imz,ax=axs[:,0],shrink=.55,label='Z')
    fig.colorbar(im,ax=axs[:,1:3],shrink=.55,label='Arrival relative to earliest cell (ms)')
    fig.colorbar(imc,ax=axs[:,3],shrink=.55,label='Envelope / native contact q99.5')
    fig.supxlabel('Same geometry, physical core circles and contact order. Each directional example is nearest its own class median duration/area/extent.\nContact window and native-only amplitude scaling are fixed across rows; this is a baseline correspondence diagnostic (G/K off).',fontsize=9)
    out=ROOT/'figures'
    for ext in ['png','svg']:fig.savefig(out/f'density_spatial_contacts.{ext}',dpi=180)
    plt.close(fig)
    write(out/'density_spatial_contacts_metadata.json',dict(producer_sha256=sha(__file__),
        source=str(ROOT/'density_contact_replay/comparison.json'),representative_events=meta,
        event_selection='All complete .5–8s original spatial events in each physical axis direction. Nearest to within-class median(duration,area,extent), scaled by within-classIQR; no matching to another model.',
        fixed_contact_window_s=[3,4.5],contact_scale='Each contact native8401 q99.5 over.5–8s, shared across all rows; no candidate normalization',
        raw_current_observer='Separate comparison JSON; contact image here is the original Gaussian-weighted firing envelope, not a raw-current LFP.',
        native8402='Retained in full quantitative comparison; native8401 supplies the recorded forcing and visual reference.',
        agent_visual_review='PENDING',human_review='PENDING',model_promoted=False,formal_bifurcation=False))
    p=out/'README.md';text=p.read_text();marker='### density_spatial_contacts.png / density_spatial_contacts.svg\n'
    section=marker+'同一空间网络中，比较原生8401与两条数值候选的8秒Z场、两个方向的完整传播事件和固定3–4.5秒触点发放包络。每个方向按该模型全部合格事件自身的持续时间／招募范围／传播距离中位数选例，不按与原生最相似来挑选；原生8402仍保留在定量比较中。触点固定15个槽位，幅度统一用原生8401校准；这里的发放包络与另行计算的原始电流HFO读出分开。\n**关注点**：双核资源与传播是否仅在均值上接近，事件形态和触点幅度／顺序是否随数值分辨率改变；候选未获得空间对应或分岔认证。\n'
    if marker in text:
        before,after=text.split(marker,1);tail=after.find('\n### ');text=before+section+(after[tail:] if tail>=0 else '')
    else:text=text.rstrip()+'\n\n'+section
    p.write_text(text)


if __name__=='__main__':main()
