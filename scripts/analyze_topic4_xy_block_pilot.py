"""Offline diagnostics and genuine conditional XY response maps; no optimizer input."""
from pathlib import Path
import argparse, csv, json, pickle, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts import run_topic4_xy_block_pilot as run
from scripts import analyze_topic4_three_observable_bo as legacy
from scripts.report_topic4_label_free_dense import describe, metrics
rt, OUT=run.rt, run.OUT


def main(final_media=False):
    p=rt.read(OUT/'plan.json'); scores={f.stem:rt.read(f) for f in (OUT/'scores').glob('*.json')}
    ev,names,_=legacy.patient(); cache=OUT/'analysis/diagnostic_observables.json'
    data=rt.read(cache) if cache.exists() else dict(patient={},runs={})
    modes=[('ALL',None),('TA',1),('TB',0)]
    if not data['patient']:
        for mode,k in modes: data['patient'][mode]=describe(ev.fit if k is None else ev.fit[ev.fit_labels==k],names)
    joint_path=OUT/'analysis/offline_joint_scores.json'
    joint=rt.read(joint_path) if joint_path.exists() else {}
    audit_dir=OUT/'joint_feature_audit'
    joint_ready=(audit_dir/'summary.json').exists()
    if joint_ready:
        from scripts.audit_topic4_joint_observation import JointMap, statistic
        with Path(p['objective']['artifact']).open('rb') as f: objective=pickle.load(f)
        mapping=JointMap(objective)
        with np.load(audit_dir/'feature_map.npz') as z: mapping.W=z['weights'];mapping.phase=z['phases']
        with np.load(audit_dir/'target.npz') as z: target=z['target'];scale=float(z['scale'])
        device=run.dense.gpu_device()
    for cid,score in scores.items():
        if cid in data['runs'] and (not joint_ready or cid in joint): continue
        src=Path(score['source']); r,t,cn=run.dense.load_times(src); assert cn==names
        if joint_ready and cid not in joint:
            if len(t)>=16 and score['J'] is not None:
                value=statistic(mapping.features(objective.raw_features(t),device),target)
                joint[cid]=dict(**value,scaled=value['D_off']/scale,diagnostic_only=True)
            else: joint[cid]=dict(N=len(t),scaled=None,diagnostic_only=True)
        with np.load(src.with_suffix('.npz')) as z:
            ids=[int(i) for i in z['primary_event_indices'] if r['events'][i]['window_ms'][0]>=1500 and r['events'][i]['window_ms'][1]<=r['actual_duration_ms']]
            lab=z['event_mode'][ids]
        data['runs'][cid]={mode:describe(t if k is None else t[lab==k],names) for mode,k in modes}
    rt.write(cache,data)
    if joint_ready:rt.write(joint_path,joint)
    rows=[]
    for cid,score in scores.items():
        c=rt.read(OUT/'candidates'/f'{cid}.json')
        for mode,_ in modes:
            q=data['runs'][cid][mode]; pat=data['patient'][mode]; v=metrics(q,pat,names)
            row=dict(candidate=cid,mode=mode,N=q['N'],J=score['J'],rank_error=v[0],within_rod_order_error=v[1],
                rod_lag_ms=v[2],participation_error=v[3],both_rods=v[4],
                left_x=c['centers_mm'][0][0],left_y=c['centers_mm'][0][1],right_x=c['centers_mm'][1][0],right_y=c['centers_mm'][1][1])
            row['offline_joint_score_ALL_only']=joint.get(cid,{}).get('scaled')
            for shaft in ['SCL','ICL']:
                vals=[abs(z['order_probability']-pat['pairs'][k]['order_probability']) for k,z in q['pairs'].items()
                      if z['shaft']==shaft and z['order_probability'] is not None and pat['pairs'][k]['order_probability'] is not None]
                row[shaft+'_order_error']=float(np.mean(vals)) if vals else None
            rows.append(row)
    with (OUT/'analysis/observables.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    good=[s for s in scores.values() if s['J'] is not None]
    best=min(good,key=lambda s:s['J'])
    rt.write(OUT/'analysis/current.json',dict(best=best,incumbent_J=scores[run.REF]['J'],complete=len(scores),
        scientific_status='PENDING_PROPAGATION_REVIEW',mode_labels='diagnostic only; not read by controller',
        best_diagnostics=[r for r in rows if r['candidate']==best['candidate']]))
    # Colour only actual measured positions; no interpolation or smoothing.
    if sum(not s.get('reused') for s in scores.values())<4: return
    F=OUT/'figures'; F.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Noto Sans CJK JP','DejaVu Sans'],
        'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'legend.frameon':False,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'svg.fonttype':'none','pdf.fonttype':42})
    fig=plt.figure(figsize=(183/25.4,133/25.4))
    panel_x=[.105,.425,.745];panel_w=.205;panel_h=panel_w*183/133
    axes=np.array([[fig.add_axes([x,y,panel_w,panel_h]) for x in panel_x] for y in [.58,.20]])
    byid={r['candidate']:r for r in rows if r['mode']=='ALL'}
    fields=[('rank_error','平均传播顺序误差'),('within_rod_order_error','杆内先后概率误差'),('participation_error','触点参与概率误差')]
    ref=rt.read(OUT/'candidates'/f'{run.REF}.json')
    for j,(field,title) in enumerate(fields):
        # Freeze scales across interim updates. Any overflow is explicitly
        # signalled by the colourbar extension, never silently rescaled.
        vmax=[.25,.30,.30][j]
        for k,grid in enumerate(p['grids']):
            ax=axes[k,j]; arr=np.full((5,5),np.nan)
            for iy,ids in enumerate(grid['ids']):
                for ix,cid in enumerate(ids):
                    if cid in byid and scores[cid]['J'] is not None: arr[iy,ix]=byid[cid][field]
            center=np.array(ref['centers_mm'][k]); edges=np.arange(-.625,.626,.25)
            im=ax.pcolormesh(center[0]+edges,center[1]+edges,np.ma.masked_invalid(arr),cmap='viridis_r',vmin=0,vmax=max(vmax,.01),edgecolors='white',linewidth=.4)
            xx,yy=np.meshgrid(center[0]+np.array(grid['x_offsets_mm']),center[1]+np.array(grid['y_offsets_mm']))
            ax.scatter(xx[np.isfinite(arr)],yy[np.isfinite(arr)],s=2,color='white')
            ax.scatter(*center,marker='*',s=60,color='none',edgecolor='#D55E00',linewidth=1)
            ax.set(aspect='equal',xlabel='x (mm)',ylabel=('左核' if k==0 else '右核')+' y (mm)' if j==0 else '',
                title=title+' ↓' if k==0 else None)
            ax.set_xticks(center[0]+np.array([-.5,0,.5]));ax.set_yticks(center[1]+np.array([-.5,0,.5]))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
            ax.tick_params(labelsize=7.5,pad=2)
        exceeded=any(float(r[field])>vmax for r in byid.values() if r[field] is not None)
        cb=fig.colorbar(im,cax=fig.add_axes([panel_x[j],.092,panel_w,.017]),orientation='horizontal',
            ticks=[0,vmax/2,vmax],extend='max' if exceeded else 'neither')
        cb.ax.tick_params(labelsize=7.5,pad=2,length=2)
    fig.text(.105,.976,'只移动一个核的 XY；另一核、EE 强度与方向固定',ha='left',va='top',fontsize=9)
    grid_ids={cid for grid in p['grids'] for row in grid['ids'] for cid in row}
    completed_grid=sum(cid in scores for cid in grid_ids)
    subtitle=fig.text(.105,.94,f'整体观测，不分 TA/TB；{completed_grid}/49 网格点已有结果；空白待完成，星号为原工作点',ha='left',va='top',fontsize=7.5)
    fig.canvas.draw(); renderer=fig.canvas.get_renderer()
    for label in fig.findobj(matplotlib.text.Text):
        if label.get_visible() and label.get_text():
            box=label.get_window_extent(renderer)
            assert box.x0>=-1 and box.y0>=-1 and box.x1<=fig.bbox.x1+1 and box.y1<=fig.bbox.y1+1,label.get_text()
    assert all(not ax.title.get_window_extent(renderer).overlaps(subtitle.get_window_extent(renderer)) for ax in axes[0])
    fig.savefig(F/'xy_response.png',dpi=400);fig.savefig(F/'xy_response.svg');plt.close(fig)
    (F/'README.md').write_text('### xy_response.png\n上下排分别为左核和右核二维位置响应，列为三个整体观测误差；每个格子对应真实60秒轨迹，空白为尚未完成或不可评分。橙色星为共同参考点，同列共用固定色标；TA/TB未进入此图或优化。\n**关注点**：这是固定另一核及背景的条件响应，不是混合连接参数后的投影；单一拓扑/噪声没有跨网络误差条，亮色表示误差较低。\n')
    rt.write(OUT/'analysis/figure_metadata.json',dict(producer=str(Path(__file__).resolve()),source_csv=str(OUT/'analysis/observables.csv'),
        statistical_unit='one 60s fixed-topology/fixed-noise trajectory per condition',interpolation=False,human_review='pending',formats=['png','editable_svg']))
    if final_media:
        from scripts import report_topic4_label_free_key_results as media
        media.OUT=OUT; media.KEY=OUT/'key_results';media.KEY.mkdir(exist_ok=True)
        if best['candidate']==run.REF:
            existing=rt.read(run.SOURCE/'key_results/current.json')['best']
            assert existing['source_sha256']==best['source_sha256']
            artifact=existing
        else:
            artifact=media.best_media(best,ev,names)
        rt.write(OUT/'key_results/current.json',dict(best=artifact,parameter_figure=str(F/'xy_response.png'),
            scientific_status='PENDING_SCIENTIFIC_AND_VISUAL_REVIEW',labels_used_for_selection=False))
        from scripts.review_topic4_xy_block_pilot import main as review_patterns
        review_patterns()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--final-media',action='store_true')
    main(ap.parse_args().final_media)
