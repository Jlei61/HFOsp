"""Post-score diagnostics; this module is never imported by the optimizer."""
from pathlib import Path
import argparse,csv,json,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scripts import run_topic4_xy_overnight_expansion as run
from scripts import analyze_topic4_three_observable_bo as patient_source
from scripts.report_topic4_label_free_dense import describe,metrics
rt=run.rt
MODES=[('ALL',None),('TA',1),('TB',0)]


def main(out,final_media=False):
    out=Path(out);(out/'analysis').mkdir(exist_ok=True);figdir=out/'key_results/figures';figdir.mkdir(parents=True,exist_ok=True)
    p=rt.read(out/'plan.json');scores={f.stem:rt.read(f) for f in (out/'scores').glob('*.json')}
    ev,names,_=patient_source.patient();cache=out/'analysis/diagnostic_observables.json'
    data=rt.read(cache) if cache.exists() else dict(patient={},runs={})
    if not data['patient']:
        data['patient']={mode:describe(ev.fit if k is None else ev.fit[ev.fit_labels==k],names) for mode,k in MODES}
    for cid,s in scores.items():
        if cid in data['runs']:continue
        src=Path(s['source']);r,t,cn=run.dense.load_times(src);assert cn==names
        with np.load(src.with_suffix('.npz')) as a:
            ids=[int(i) for i in a['primary_event_indices'] if r['events'][i]['window_ms'][0]>=1500 and r['events'][i]['window_ms'][1]<=r['actual_duration_ms']]
            labels=a['event_mode'][ids]
        data['runs'][cid]={mode:describe(t if k is None else t[labels==k],names) for mode,k in MODES}
    rt.write(cache,data);rows=[]
    for cid,s in scores.items():
        c=rt.read(out/'candidates'/f'{cid}.json')
        for mode,_ in MODES:
            q=data['runs'][cid][mode];pat=data['patient'][mode];v=metrics(q,pat,names)
            row=dict(candidate=cid,mode=mode,N=q['N'],J=s['J'],rank_error=v[0],within_rod_order_error=v[1],
                participation_error=v[3],rod_lag_ms=v[2],both_rods=v[4],
                left_x=c['centers_mm'][0][0],left_y=c['centers_mm'][0][1],right_x=c['centers_mm'][1][0],right_y=c['centers_mm'][1][1])
            for shaft in ['SCL','ICL']:
                values=[abs(z['order_probability']-pat['pairs'][k]['order_probability']) for k,z in q['pairs'].items()
                    if z['shaft']==shaft and z['order_probability'] is not None and pat['pairs'][k]['order_probability'] is not None]
                row[shaft+'_order_error']=float(np.mean(values)) if values else None
            rows.append(row)
    with (out/'analysis/observables.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lookup={(r['candidate'],r['mode']):r for r in rows};good=sorted([s for s in scores.values() if s['J'] is not None],key=lambda r:(r['J'],r['candidate']))
    best=good[0];ref=rt.read(out/'candidates'/f"{p['reference_id']}.json");refsource=Path(scores[p['reference_id']]['source'])
    with np.load(refsource.with_suffix('.npz')) as a:contacts=a['contact_xy_mm'];cn=a['contact_names'].astype(str)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
        'xtick.labelsize':8,'ytick.labelsize':8,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(160/25.4,80/25.4));position_source=[]
    for col,(key,cmap,label) in enumerate([('rank_error','Blues','Mean rank error'),('within_rod_order_error','Oranges','Within-rod error')]):
        ax=fig.add_axes([(13+77*col)/160,13/80,54/160,54/80]);ax.set_facecolor('#F0F0F0')
        overflow=False
        for grid in p['grids']:
            core=grid['core'];xs=np.array(grid['x_offsets_mm'])+ref['centers_mm'][core][0];ys=np.array(grid['y_offsets_mm'])+ref['centers_mm'][core][1]
            step=.75;values=np.full((len(ys),len(xs)),np.nan)
            for iy,line in enumerate(grid['ids']):
                for ix,cid in enumerate(line):
                    if (cid,'ALL') in lookup and scores[cid]['J'] is not None:
                        values[iy,ix]=lookup[cid,'ALL'][key]
            extent=[xs[0]-step/2,xs[-1]+step/2,ys[0]-step/2,ys[-1]+step/2]
            im=ax.imshow(values,extent=extent,origin='lower',interpolation='nearest',vmin=0,vmax=.3,cmap=cmap,aspect='equal')
            overflow|=bool(np.any(values>.3));position_source.append(dict(metric=key,core=core,extent=extent,values=values))
        for core,(x,y) in enumerate(ref['centers_mm']):
            ax.add_patch(Circle((x,y),ref['radii_mm'][core],fill=False,color='#555555',lw=.7))
            ax.text(x,y+ref['radii_mm'][core]+.6,'A' if core==0 else 'B',ha='center',fontsize=9)
        for shaft in ['SCL','ICL']:
            ix=sorted(np.flatnonzero(np.char.startswith(cn,shaft)),key=lambda i:int(cn[i][3:]),reverse=True)
            ax.plot(*contacts[ix].T,color='#777777',lw=.7);ax.scatter(*contacts[ix].T,s=7,facecolor='white',edgecolor='#555555',lw=.5,zorder=5)
        ax.set(xlim=(0,20),ylim=(0,20),xticks=[0,10,20],yticks=[0,10,20],xlabel='x (mm)',ylabel='y (mm)' if col==0 else '',aspect='equal')
        ax.tick_params(labelleft=col==0,length=2,pad=2);ax.set_title(label,fontsize=9,pad=8)
        cb=fig.colorbar(im,cax=fig.add_axes([(70+77*col)/160,13/80,1.7/160,54/80]),ticks=[0,.15,.3],extend='max' if overflow else 'neither')
        cb.outline.set_visible(False);cb.ax.tick_params(length=2,pad=2)
    for ext in ['png','svg']:fig.savefig(figdir/f'core_position_response.{ext}',dpi=350)
    plt.close(fig)
    # The ranked loss is a distribution distance, distinct from the displayed means.
    comparison_ids=list(dict.fromkeys([p['reference_id']]+[c for c in ['xy_left_20','xywide_right_41'] if c in scores]+[best['candidate']]))
    summary=dict(best=best,complete=len(scores),new_complete=sum(not s.get('reused') for s in scores.values()),
        comparison_rows=[lookup[c,m] for c in comparison_ids for m,_ in MODES],patient_rod_lag_ms={m:data['patient'][m]['rod_lag_ms']['median'] for m,_ in MODES},
        scientific_status='NOT_ACCEPTED_AS_DUAL_MODE_RECOVERY',label_role='Frozen historical diagnostic labels only; excluded from optimization. This FIT19770 comparison is distinct from paper candidate G/H Timing+Space30049.',
        statistical_unit='One 60s topology2511/noise847401 trajectory per parameter condition',time=time.time())
    rt.write(out/'analysis/current.json',summary);rt.write(out/'analysis/position_map_source.json',dict(grids=position_source,interpolation=False,color_limits=[0,.3]))
    text='# 扩大位置范围：实际结果与科学边界\n\n'
    text+=f"已评分 {len(scores)} 个条件；本批新增 {summary['new_complete']} 条。最低无标签分布损失：`{best['candidate']}`，J={best['J']:.6f}。\n\n"
    text+='|工作点|无标签J↓|总体平均rank误差↓|总体杆内顺序误差↓|总体参与误差↓|TA/TB事件数|TB SCL误差↓|TB杆间中位差ms|\n|---|---:|---:|---:|---:|---|---:|---:|\n'
    for cid in comparison_ids:
        a=lookup[cid,'ALL'];b=lookup[cid,'TB'];ta=lookup[cid,'TA'];fmt=lambda v:f'{v:.3f}' if v is not None else 'NA'
        text+=f"|{cid}|{fmt(a['J'])}|{fmt(a['rank_error'])}|{fmt(a['within_rod_order_error'])}|{fmt(a['participation_error'])}|{ta['N']}/{b['N']}|{fmt(b['SCL_order_error'])}|{fmt(b['rod_lag_ms'])}|\n"
    text+='\n评分包含总体rank、杆内/杆间时序、参与三组完整分布，表中均值误差不是它的三个代数分量。单个最低J不保证所有均值误差同时下降；TB杆间时差改善也不能代替SCL内部顺序恢复。患者历史FIT19770的TB杆间中位差为1.190 ms；该诊断不与Fig4新G/H的Timing+Space30049混为同一数据集。\n\n'
    text+='每点只有一张网络、一段噪声；事件数不当作网络重复。位置移动同时改变core成员和对应局部参数作用的连接集合，不是纯粹平移相同神经元团。无标签loss与TA/TB诊断保持分开。网格是固定背景条件响应，只有后续使用前批分数的BO才是反馈自适应搜索。\n'
    (out/'analysis/scientific_review.md').write_text(text)
    (figdir/'README.md').write_text('### core_position_response.png\n同一固定背景和网络/噪声下，两个core中心的条件XY响应，配准真实20×20 mm平面。蓝色为总体平均rank误差，橙色为杆内顺序误差；灰色为空白/未测，超出0.3色标者显示extend，不插值。\n**关注点**：热格坐标是被测试的core中心，颜色是整段事件的统计误差，不是局部组织活动；每次只移动一个core，联合BO不混进条件热图。同名SVG可编辑。\n')
    if final_media:
        from scripts import report_topic4_geometry_axis_joint_search as media
        media.OUT=out;media.patient_mean.OUT=out/'analysis/patient_display';media.patient_mean.build_envelope()
        sc=dict(best,J_joint=None)
        saved=media.media(sc,names)
        rt.write(out/'key_results/current.json',dict(results=[saved],selection='minimum unchanged label-free J only',patient_template=media.patient_mean.identity(),human_review='PENDING'))
    print(json.dumps(dict(root=str(out),complete=len(scores),best=best['candidate'],J=best['J'])),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=run.OUT);ap.add_argument('--final-media',action='store_true');a=ap.parse_args();main(a.root,a.final_media)
