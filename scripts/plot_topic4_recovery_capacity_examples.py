"""Posthoc most patient-neighbor-compatible examples, distinct from typical events.

This asks about observed capability, not representative distribution recovery.
The existing own-mode-mean figures remain untouched. No simulation or scoring.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
from scripts.paper_figures import plot_topic4_recovery_review as mainfig
review=mainfig.review;an=review.an;rt=review.rt


def main(phase,candidate):
    old,plan,spec,cases=review.stage_cases(phase)
    c=next(c for c in cases if c['base_id']==candidate)
    seeds=sorted({int(s) for cid,t,s in spec['units'] if cid==candidate and int(t)==c['topology']})
    if len(seeds)!=2:raise ValueError('this paired panel requires exactly two replays')
    out=review.night.OUT/('capacity_examples_'+phase)/candidate;F=out/'figures';F.mkdir(parents=True,exist_ok=True)
    patient=mainfig.patient_payloads();canonical,meta,pat,names,porder=patient
    records=[];models={};loaded={};limits=[min(e['tile_lo_ms'] for e in pat.values()),max(e['tile_hi_ms'] for e in pat.values())]
    for seed in seeds:
        path=an.run.result_path(c['output_stage'],candidate,c['topology'],seed)
        unit=an.load_unit(path,old['analysis']['burnin_ms'])
        if unit is None:raise RuntimeError('complete replays required')
        r,a,ids=unit;loaded[seed]=(r,a)
        for lab,label in [('TA',1),('TB',0)]:
            ii=ids[a['event_mode'][ids]==label]
            if not len(ii):continue
            i=int(ii[np.argmin(a['event_distance_modes'][ii,label])]);lo,hi=r['events'][i]['window_ms']
            times=a['centroid_ms'][i];zero=np.nanmin(times);order=[list(a['contact_names']).index(names[k]) for k in porder]
            take=[j for j in order if np.isfinite(times[j])];dt=float(a['contact_envelope_dt_ms'])
            env=a['contact_envelope'][round(lo/dt):round(hi/dt),take].T
            models[(seed,lab)]=dict(event=i,env=env,lo=lo-zero,hi=hi-zero,times=times[take]-zero,labels=a['contact_names'][take],n=len(ii))
            limits=[min(limits[0],lo-zero),max(limits[1],hi-zero)]
            records.append(dict(candidate=candidate,topology=c['topology'],seed=seed,mode=lab,original_primary_mode_n=len(ii),
                patient_neighbor_distance=float(a['event_distance_modes'][i,label]),support=int(a['event_support'][i]),
                arrays_sha256=r['arrays_sha256'],window_ms=[lo,hi],**an.event_timing(r,a,i,a['contact_names'])))
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':9,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    for row,lab in enumerate(['TA','TB']):
        canonical._readout(axes[row,0],pat[lab],dict(names=names),porder,tuple(limits),None,title='患者 '+lab,template=lab)
        axes[row,0].set_box_aspect(None);axes[row,0].set_facecolor('#b6b6b6')
        axes[row,0].set_title(f'患者 {lab} · Fig2C {pat[lab]["event"]}\n真实STFT幅度')
        for col,seed in enumerate(seeds,1):
            ax=axes[row,col];m=models.get((seed,lab))
            if m is None:
                ax.text(.5,.5,'未观测到该类',transform=ax.transAxes,ha='center');continue
            env=m['env']/np.maximum(m['env'].max(1,keepdims=True),1e-20);n=len(env)
            ax.imshow(env,aspect='auto',extent=[m['lo'],m['hi'],n-.5,-.5],cmap='magma',vmin=0,vmax=1,interpolation='nearest')
            ax.plot(m['times'],np.arange(n),'-o',color=an.MODE_COLOR[lab],markerfacecolor='#ffb000',ms=3,lw=.8)
            ax.set(yticks=range(n),yticklabels=m['labels'],xlim=limits,title=f'{lab} · 噪声 {seed} · 事件 {m["event"]}\n从该类{m["n"]}个合格事件中事后择优')
            ax.set_facecolor('#b6b6b6');ax.tick_params(axis='y',labelsize=8)
        for ax in axes[row]:
            ax.axvline(0,c='black',ls='--',lw=.6);ax.set_xlabel('相对最早参与质心 (ms)')
    fig.suptitle(review.display(c)+'｜同一图、同一参数、两条噪声\n事后最小患者同类邻域距离：仅检查曾出现过的能力，不代表常见事件或分布恢复\n原合格孤立窗；患者STFT与模型发放密度信号不同；只平移、不拉伸时间',fontsize=11)
    for ext in ['png','pdf']:fig.savefig(F/f'patient_compatible_extremes.{ext}',dpi=180)
    plt.close(fig)
    # All native activity is retained. Seven complete time bins cover each 250ms window.
    edges=[0,40,80,120,160,200,240,250];native=[]
    for row in records:
        r,a=loaded[row['seed']];lo,hi=row['window_ms'];movie=a['sheet_activity_counts'][round(lo/2):round(hi/2)]
        native.append((row,a,movie,[movie[round(l/2):round(h/2)].mean(0) for l,h in zip(edges[:-1],edges[1:])]))
    vmax=max(float(x.max()) for _,_,_,bins in native for x in bins)
    fig,axes=plt.subplots(len(native),7,figsize=(16,2.5*len(native)),squeeze=False,layout='constrained')
    for row,(record,a,movie,bins) in enumerate(native):
        for col,(lo,hi,field) in enumerate(zip(edges[:-1],edges[1:],bins)):
            ax=axes[row,col];im=ax.imshow(field,origin='lower',extent=[0,20,0,20],cmap='inferno',vmin=0,vmax=vmax,interpolation='nearest')
            mainfig.geometry(ax,c,a);ax.set(xticks=[],yticks=[],xlabel='',ylabel='');ax.set_title(f'{lo}–{hi} ms',fontsize=9)
            if col==0:ax.set_ylabel(f'{record["mode"]} · {record["seed"]}\n事件 {record["event"]}',fontsize=9)
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.6,label='每2ms平均活动E神经元数 / 1mm网格（所标时间区间内平均）')
    fig.suptitle('同一批事后最相容个例：完整原生场\n每格对标注时间区间的全部原生2ms帧求平均，不选亮帧、不删核外活动；四个事件共享色标',fontsize=11)
    for ext in ['png','pdf']:fig.savefig(F/f'native_full_window_bins.{ext}',dpi=160)
    plt.close(fig)
    # A separate native animation displays every original 2ms frame, not bin averages.
    vmax=max(float(movie.max()) for _,_,movie,_ in native);cmap=plt.get_cmap('inferno');frames=[]
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',14)
    for record,a,movie,_ in native:
        for j,frame in enumerate(movie):
            canvas=Image.new('RGB',(460,510),'white');d=ImageDraw.Draw(canvas)
            rgb=(cmap(np.clip(frame[::-1]/vmax,0,1))[:,:,:3]*255).astype('uint8')
            canvas.paste(Image.fromarray(rgb).resize((400,400),Image.Resampling.NEAREST),(30,70))
            d.text((12,5),f'事后择优个例，不代表常见事件；图 {c["topology"]}',font=font,fill='black')
            d.text((12,27),f'{record["mode"]}  噪声 {record["seed"]}  事件 {record["event"]}  +{j*2}ms',font=font,fill='black')
            for xy,rad in zip(c['centers_mm'],c['radii_mm']):
                x,y=30+xy[0]*20,70+(20-xy[1])*20;rr=rad*20;d.ellipse((x-rr,y-rr,x+rr,y+rr),outline='white',width=1)
            for x,y in a['contact_xy_mm']:
                px,py=30+x*20,70+(20-y)*20;d.ellipse((px-2,py-2,px+2,py+2),outline='cyan',width=1)
            d.text((12,480),'原生2ms逐帧；50fps播放；事件之间存在跳转',font=font,fill='black');frames.append(canvas)
    frames[0].save(F/'native_extremes.gif',save_all=True,append_images=frames[1:],duration=20,loop=0)
    rt.write(out/'selection.json',dict(status='POSTHOC_CAPABILITY_DIAGNOSTIC_NOT_ACCEPTANCE',phase=phase,records=records,
        rule='Within each complete replay and assigned mode, choose the original primary event with smallest frozen distance to five FIT neighbors of that mode; no new path/contact filter.',
        interpretation='Development-selected extremes, not representative propagation or held-out validation. Typical examples remain in main_review. Event snippets jump in physical time; continuous movie separate.',
        producer=__file__,producer_sha256=rt.sha(__file__),native_shared_vmax=vmax,patient_cache_sha256=rt.sha(mainfig.CACHE)))
    text=[]
    for file in sorted(F.iterdir()):
        if file.suffix not in ['.png','.pdf','.gif']:continue
        desc='左列真实Fig2C STFT；右列为同一条件两条噪声各自到患者同类邻域距离最小的原合格事件，模型为发放密度包络，未拉伸时间。' if file.stem.startswith('patient') else '显示同一批事后择优事件的全部原生场；静态图为标注时间区间内全部2ms帧平均，GIF为原生2ms逐帧播放，均保留核外活动。'
        text.append(f'### {file.name}\n\n'+desc+'选例是事后能力诊断，既不是该类常见事件，也不是独立验证；各事件的实际观测支持量与选择距离见selection。**关注点**：是否曾出现较相容传播，以及好看的个例与常见事件分布之间仍有多大差距。')
    (F/'README.md').write_text('\n\n'.join(text)+'\n');print(dict(output=str(out),events=len(records)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True);p.add_argument('--candidate',required=True);a=p.parse_args();main(a.phase,a.candidate)
