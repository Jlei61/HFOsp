"""Native rotation diagnostics and patient/model media, independent of BO selection."""
from pathlib import Path
import argparse,fcntl,hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
from scripts import analyze_topic4_three_observable_bo as analysis
run=analysis.run;rt=run.rt;OUT=run.OUT

def rotation():
    import cupy as cp
    from scripts import analyze_topic4_rotation_response as rot
    rot.OUT=OUT;cp.cuda.Device(1).use()
    while True:
        paths=sorted(OUT.glob('*/units/*/*/workers/trajectory.json'));todo=[]
        for p in paths:
            r=rt.read(p);key=hashlib.sha256(str(p).encode()).hexdigest()[:20]
            if r.get('status')=='COMPLETE' and r.get('actual_duration_ms',0)>=20000 and not (OUT/'rotation'/key/'result.json').exists():todo.append(p)
        if todo:
            p=todo[0];rt.write(OUT/'rotation/status.json',dict(status='ANALYZING',source=str(p),gpu=1,time=time.time()))
            rot.analyze(p,1,cp);cp.get_default_memory_pool().free_all_blocks();continue
        rt.write(OUT/'rotation/status.json',dict(status='WAITING_FOR_TRAJECTORIES',complete=len(list((OUT/'rotation').glob('*/result.json'))),gpu=1,time=time.time()))
        if (OUT/'optimization_complete.json').exists() or (OUT/'g3_ready_for_native_review.json').exists():return
        time.sleep(15)

def render_unit(p):
    import numpy as np
    from scripts.analyze_topic4_core_connectivity_search import load_unit
    from scripts.paper_figures import plot_topic4_recovery_review as fr
    from scripts.render_topic4_shape_output_gifs import render
    cid=p.parents[2].name;unit=p.parents[1].name;folder=OUT/'analysis/native_review'/cid/unit;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'manifest.json').exists():return
    r,a,ids=load_unit(p,1500.);c=rt.read(OUT/'candidates'/f'{cid}.json');c['topology']=r['job']['topology_seed']
    physical=rt.read(p.parents[1]/'applied_physics.json');c['_applied_threshold']=physical['threshold'];noise=r['job']['dynamics_seed']
    patient=fr.patient_payloads()
    from src.topic4_pdf_font_guard import install
    import matplotlib.pyplot as plt
    install();plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':9,'pdf.fonttype':3})
    manifest=[fr.four_panel(c,noise,r,a,ids,folder,physical,'primary'),fr.spectral_comparison(c,{noise:(r,a,ids)},[noise],folder,patient,'primary'),render(c,noise,r,a,ids,physical,folder,patient)]
    # Mean patient centroids are a separate statistical panel, never an invented
    # average HFO spectrogram. Render multiple model events with native field.
    render_mean_template_gif(c,noise,r,a,ids,physical,folder)
    rt.write(folder/'manifest.json',dict(source=str(p),source_sha256=rt.sha(p),items=manifest,agent_visual_review='PENDING',human_visual_review='PENDING'))
    pngs=list(folder.glob('*.png'));gifs=list(folder.glob('*.gif'))
    (folder/'README.md').write_text('# 原生传播审阅\n\n'+''.join(f'### {p.name}\n真实轨迹生成，固定SCL/ICL行序与毫秒轴；所有原生活动均显示。GIF采用每模式最早三个合格事件，静态示例靠近本运行模式均值，均不按患者相似度挑选。\n**关注点**：对照患者TA/TB模板及真实Fig2C频谱，区分局部源、顺序与参与；文件解码不代替目视审阅。\n\n' for p in pngs+gifs))

def render_mean_template_gif(c,noise,r,a,ids,physics,folder,display_xlim=None,patient_mean_draw=None,patient_mean_identity=None):
    import numpy as np
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image,ImageDraw
    from scripts.render_topic4_shape_output_gifs import field_tile,save_background
    from scripts.paper_figures import plot_topic4_recovery_review as fr
    ev,names,_=analysis.patient();order=[names.index(n) for n in analysis.DISPLAY]
    chosen=sorted([int(i) for k in [1,0] for i in ids[a['event_mode'][ids]==k][:3]],key=lambda i:r['events'][i]['window_ms'][0])
    if not chosen:return
    vmax=max(float(np.quantile(a['sheet_activity_counts'][750:],.999)),1);frames=[];provenance=[]
    dt=float(a['contact_envelope_dt_ms']);xlim=(-100,200) if display_xlim is None else tuple(display_xlim)
    for i in chosen:
        k=int(a['event_mode'][i]);name='TA' if k==1 else 'TB';p=ev.fit[ev.fit_labels==k]
        centered=p-np.nanmin(p,axis=1,keepdims=True)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning);mean=np.nanmean(centered,0)[order];q=np.nanquantile(centered,[.05,.95],axis=0)[:,order]
        lo,hi=r['events'][i]['window_ms'];t=a['centroid_ms'][i];zero=float(np.nanmin(t));part=np.isfinite(t)[order]
        mass=a['contact_envelope'][round(lo/dt):round(hi/dt),order].T;mass=mass/np.maximum(mass.max(1,keepdims=True),1e-20);mass[~part]=np.nan
        fig,axes=plt.subplots(1,3,figsize=(16,5.4),gridspec_kw={'width_ratios':[1,1,1.1]});fig.subplots_adjust(left=.06,right=.985,bottom=.14,top=.82,wspace=.3)
        ax=axes[0]
        if patient_mean_draw is not None:
            patient_mean_draw(ax,name,xlim,fr.display)
        else:
            ax.fill_betweenx(np.arange(15),q[0],q[1],color='#758399',alpha=.25)
            for ix in [slice(0,4),slice(4,15)]:ax.plot(mean[ix],np.arange(15)[ix],'o-',color='#222222',ms=4)
            ax.set(title=f'患者{name}：平均质心与5–95%范围\n每事件统一减去最早质心，n={len(p)}',xlabel='相对最早参与质心 (ms)',xlim=xlim)
            fr.display.contact_axis(ax)
        axes[1].axis('off');axes[1].set_title('模型：全部原生E活动\n白线为只降阈值的两个core')
        cmap=plt.get_cmap('magma').copy();cmap.set_bad('#777777');ax=axes[2]
        ax.imshow(mass,aspect='auto',extent=[lo-zero,hi-zero,14.5,-.5],cmap=cmap,vmin=0,vmax=1,interpolation='nearest')
        fr.display.centroid_lines(ax,t[order]-zero,color='#00cfff');fr.display.contact_axis(ax);ax.set(xlim=xlim,xlabel='相对最早参与质心 (ms)',title=f'模型{name}：第{i}事件\n发放密度包络，非HFO频谱',facecolor='#b6b6b6')
        for ax in [axes[0],axes[2]]:ax.set_ylim(14.5,-.5);ax.tick_params(labelsize=8)
        fig.canvas.draw();boxfield=axes[1].get_position().bounds;boxread=axes[2].get_position().bounds;bg=save_background(fig)
        for now in np.arange(lo,hi,4):
            canvas=bg.copy();draw=ImageDraw.Draw(canvas);bx,by,bw,bh=boxfield;size=min(int(bw*bg.width),int(bh*bg.height));left=int((bx+bw/2)*bg.width-size/2);top=int((1-by-bh/2)*bg.height-size/2)
            canvas.paste(field_tile(a['sheet_activity_counts'][round(now/2)],c,a,physics,vmax,size),(left,top))
            bx,by,bw,bh=boxread;px=int((bx+(now-zero-xlim[0])/(xlim[1]-xlim[0])*bw)*bg.width)
            if xlim[0]<=now-zero<=xlim[1]:draw.line((px,int((1-by-bh)*bg.height),px,int((1-by)*bg.height)),fill='cyan',width=2)
            draw.text((10,8),f'{c["id"]} | topology {c["topology"]} | noise {noise} | {name} event {i} | t={now:.0f} ms',fill='black')
            frames.append(canvas)
        provenance.append(dict(event=i,mode=name,window_ms=[lo,hi],outside_display_axis=bool(lo-zero<xlim[0] or hi-zero>xlim[1])))
    dest=folder/'patient_mean_native_multievent.gif';temporary=dest.with_suffix('.tmp.gif')
    frames[0].save(temporary,save_all=True,append_images=frames[1:],duration=55,loop=0)
    with Image.open(temporary) as im:
        for i in range(im.n_frames):im.seek(i);im.load()
    temporary.replace(dest);frames[0].save(folder/'patient_mean_native_multievent_preview.png')
    rt.write(folder/'patient_mean_native_multievent.json',dict(events=provenance,selection='first three primary events per frozen mode, chronological',template=patient_mean_identity or 'FIT mean centroid after single per-event earliest-centroid translation; not average spectrum',native_layer='all E cells',xlim_ms=xlim,display_order=analysis.DISPLAY,agent_visual_review='PENDING'))

def media():
    while True:
        ids=[rt.read(OUT/'plan.json')['reference_id']]
        if (OUT/'nomination.json').exists():ids=rt.read(OUT/'nomination.json')['ids']
        elif (OUT/'initial_simulation_complete.json').exists():
            from scripts.control_topic4_three_observable_bo import training_data
            good=[r for r in training_data() if r['scorable']]
            ids+= [r['candidate'] for r in sorted(good,key=lambda r:r['J'])[:2]]
        paths=[p for p in sorted(OUT.glob('*/units/*/*/workers/trajectory.json')) if p.parents[2].name in ids and rt.read(p).get('status')=='COMPLETE']
        for p in paths:render_unit(p)
        rt.write(OUT/'analysis/media_status.json',dict(status='WAITING_FOR_TRAJECTORIES',manifests=len(list((OUT/'analysis/native_review').glob('*/*/manifest.json'))),time=time.time()))
        if (OUT/'optimization_complete.json').exists() or (OUT/'g3_ready_for_native_review.json').exists():return
        time.sleep(20)

def diagnostics():
    from scripts import analyze_topic4_shape_output_response as old
    old.OUT=OUT;old.A=OUT/'analysis/legacy_diagnostics';old.A.mkdir(exist_ok=True)
    old.title=lambda candidate:analysis.point_label(candidate['id'])
    reference=old.load_reference()
    while True:
        for p in sorted(OUT.glob('*/units/*/*/workers/trajectory.json')):
            key=hashlib.sha256(str(p).encode()).hexdigest()[:20]
            if rt.read(p).get('status')=='COMPLETE' and not (old.A/'units'/key/'result.json').exists():old.process(p,reference)
        rows=[]
        for p in (old.A/'units').glob('*/result.json'):
            r=rt.read(p);c=r['counts'];rows.append(dict(candidate=c['candidate'],topology=c['topology'],noise=c['noise'],old_loss=c['L_search'],N=c['primary']))
        rt.write(old.A/'comparison.json',dict(rows=rows,interpretation='Old loss recomputed on the identical event pool; diagnostic only, not optimizer input',time=time.time()))
        if (OUT/'optimization_complete.json').exists() or (OUT/'g3_ready_for_native_review.json').exists():return
        time.sleep(20)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['rotation','media','diagnostics']);arg=p.parse_args()
    with (OUT/f'{arg.action}.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:{'rotation':rotation,'media':media,'diagnostics':diagnostics}[arg.action]()
        except Exception as exc:rt.write(OUT/'analysis'/f'{arg.action}_failure.json',dict(error=repr(exc),time=time.time()));raise

if __name__=='__main__':main()
