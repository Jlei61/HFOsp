"""Review first-noise bridge runs as they complete, without waiting for noise two.

The ordinary full two-noise review remains unchanged. Six predeclared units at
most; this consumer does not nominate conditions or launch physical simulation.
"""
from pathlib import Path
import sys,json,time,hashlib,argparse,fcntl
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from scripts import analyze_topic4_shape_output_response as an
from src.topic4_pdf_font_guard import install

BASE=Path('/data/hfosp/topic4_sef_hfo/core_multiseed_response_curves_20260913')
OUT=BASE/'analysis/first_noise_review'
WINDOW=Path('/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913/window.json')
CIDS=['bridge_circle_out125','bridge_circle_out125_xminus075']
TOPOS=[2511,2711,2712];NOISE=847401

def unit(cid,topo,patient):
    source=BASE/'response/units'/cid/f'{topo}_{NOISE}/workers/trajectory.json';key=hashlib.sha256(str(source).encode()).hexdigest()[:20]
    analyzed=BASE/'analysis/units'/key/'result.json';folder=OUT/f'{cid}_topology{topo}'
    if (folder/'manifest.json').exists():return 'COMPLETE'
    if not source.exists() or not analyzed.exists():return 'WAITING_FOR_COMPLETE_ANALYSIS'
    # Canonical loader checks physical completion/hash, eligible windows and the
    # contact envelope orientation. No event table is synthesized from canaries.
    r,a,ids=an.an.load_unit(source,1500.);summary=an.rt.read(analyzed)
    assert summary['source_sha256']==r['arrays_sha256']
    if summary['counts']['physical_status']!='COMPLETE_NO_RUNAWAY':return 'UNUSABLE_PHYSICAL_STATUS'
    c=an.rt.read(BASE/'candidates'/f'{cid}.json');ap=an.rt.read(source.parents[1]/'applied_physics.json');c['topology']=topo;c['_applied_threshold']=ap['threshold']
    c['display_name']+='｜仅首条新噪声，第二条尚未纳入'
    folder.mkdir(exist_ok=True);fr=an.figreview
    # The canonical comparison has two fixed model columns. Keep the second
    # noise explicitly pending instead of leaving an unlabeled empty axis.
    spectral_candidate=dict(c);spectral_candidate['display_name']+=f'｜网络{topo}'
    records=[fr.spectral_comparison(spectral_candidate,{NOISE:(r,a,ids)},[NOISE,847402],folder,patient,'primary')]
    panel=fr.four_panel(c,NOISE,r,a,ids,folder,ap,'primary');reps=fr.representatives(a,ids)
    fields=[fr.native_timing(r,a,e['event'])[0] for e in reps.values()];values=np.concatenate([f[np.isfinite(f)] for f in fields]) if fields else np.array([])
    panel['native_shared_color_limits_ms']=[float(values.min()),max(float(values.max()),float(values.min())+1)] if len(values) else None;records.append(panel)
    from scripts.render_topic4_shape_output_gifs import render
    records.append(render(c,NOISE,r,a,ids,ap,folder,patient))
    files=[]
    for p in sorted(folder.iterdir()):
        if p.suffix not in ['.png','.gif','.pdf']:continue
        if p.suffix in ['.png','.gif']:
            with Image.open(p) as im:
                for frame in range(getattr(im,'n_frames',1)):im.seek(frame);im.load()
        files.append(dict(file=p.name,sha256=an.rt.sha(p)))
    an.rt.write(folder/'manifest.json',dict(status='COMPLETE_FIRST_NOISE_ONLY_HUMAN_REVIEW_PENDING',source=str(source),source_arrays_sha256=r['arrays_sha256'],candidate=cid,topology=topo,noise=NOISE,counts=summary['counts'],records=records,files=files,selection='Predeclared original-vs-left-shift bridge, all three base topologies, first new noise. Examples closest to own mode mean; multievent GIF uses first three primary events per mode, never patient-nearest.',scope='One stochastic replay per condition/network, not complete two-noise confirmation'))
    (folder/'README.md').write_text(''.join(f'### {p.name}\n基础网络{topo}、首条新噪声847401，原位置／左移桥接条件按预定列表展示；没有按分数挑选。患者固定Fig2C、模型自身均值附近示例、每类前三例全场GIF及连续片段使用固定分杆15行。\n**关注点**：该网络中的TA/TB实际过程；这里只有一条噪声，不能替代完整双噪声复测或患者机制验收。\n\n' for p in sorted(folder.iterdir()) if p.suffix in ['.png','.gif']))
    return 'COMPLETE'

def main(once=False):
    OUT.mkdir(parents=True,exist_ok=True);install();an.plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':9,'pdf.fonttype':3})
    with (OUT/'observer.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);patient=None
        while True:
            rows=[];processed=False;expired=time.time()>=an.rt.read(WINDOW)['review_due_unix']
            for topo in TOPOS:
              for cid in CIDS:
                folder=OUT/f'{cid}_topology{topo}';source=BASE/'response/units'/cid/f'{topo}_{NOISE}/workers/trajectory.json';key=hashlib.sha256(str(source).encode()).hexdigest()[:20]
                if (folder/'manifest.json').exists():status='COMPLETE'
                elif expired:status='NOT_COMPLETE_AT_REVIEW_TIME'
                elif not (BASE/'analysis/units'/key/'result.json').exists():status='WAITING_FOR_COMPLETE_ANALYSIS'
                elif processed:status='QUEUED_FOR_NEXT_PASS'
                elif an.rt.available_gib()<48:status='WAITING_FOR_MEMORY_HEADROOM'
                else:
                    if patient is None:patient=an.figreview.patient_payloads()
                    status=unit(cid,topo,patient);processed=True
                rows.append(dict(candidate=cid,topology=topo,noise=NOISE,status=status))
            completed=sum(r['status']=='COMPLETE' for r in rows)
            an.rt.write(OUT/'status.json',dict(status='COMPLETE' if completed==6 else 'REVIEW_WINDOW_ELAPSED' if expired else 'WAITING_OR_RENDERING_COMPLETED_UNITS',completed=completed,total_predeclared=6,units=rows,updated_unix=time.time()))
            (OUT/'README.md').write_text('# 首条新噪声：逐网络原位置／左移对照\n\n本消费者最多处理6条预定完整运行；不等待同网络第二条噪声即可先审查原生场，不修改正式双噪声分析。'+f'目前完成图件{completed}/6组。\n\n'+''.join(f'- 网络{r["topology"]}，{r["candidate"]}：'+(f'[图与GIF]({r["candidate"]}_topology{r["topology"]}/README.md)' if r['status']=='COMPLETE' else '尚无完整图件')+'\n' for r in rows)+'\n原位、左移为条件比较；每组仍只有首条噪声。新的网络或噪声不自动使当前患者FIT成为独立验证集。\n')
            if once or expired or completed==6:break
            time.sleep(30)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');main(p.parse_args().once)
