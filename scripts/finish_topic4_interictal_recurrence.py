#!/usr/bin/env python3
"""Complete the bounded batch's scientific report and measured Fig5 candidates."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import argparse
import json
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import analyze_topic4_interictal_recurrence as audit

OUT=audit.OUT
old=audit.old


def rows():
    protocol=json.loads((OUT/'protocol.json').read_text())
    answer=[]
    for j in protocol['initial_jobs']:
        path=OUT/'analysis'/f"{j['name']}.json"
        if path.exists():answer.append(json.loads(path.read_text()))
    return answer


def plot(row, all_rows):
    folder=Path(row['source']);p=row['primary'];job=row['job'];end=p['observed_s']
    d=old.load(folder);k=old.load(folder,'intrinsic_adaptation_chunks')
    with np.load(OUT/'geometry.npz') as f:g={key:f[key] for key in f.files}
    n=len(d['spikes_1ms'])//10;t=(np.arange(n)+.5)*.01
    rates=d['spikes_1ms'][:n*10].reshape(n,10,2).sum(1)/np.array([320.,80.])
    states=[]
    brief=p['preentry']['brief_events']
    if brief:
        e=brief[len(brief)//2];states.append((e['peak_s'],'Brief event','#277cb3'))
    if p['entries']:
        on=p['entries'][0]['onset_s']
        ep=next((e for e in p['events'] if e['start_s']<=on<e['end_s']),None)
        entry=max(.025,ep['start_s']+.025 if ep else on-.1)
        states.append((entry,'Entry','#d68429'))
        lo=round(on/.01);hi=min(n,round((on+1)/.01))
        states.append((t[lo+np.argmax(rates[lo:hi,0])],'High','#bc2946'))
    post=p['latest_postexit']
    if p['low_activity_exits']:
        pp=next((v for v in p['interhigh_intervals'] if v['temporal_pass']),post)
        if pp and pp['brief_events']:
            es=pp['brief_events'];states.append((es[len(es)//2]['peak_s'],'Brief return','#208677'))
        else:
            states.append((min(end-.05,p['low_activity_exits'][0]['confirmation_s']+.1),'Low activity','#778891'))
    if len(p['entries'])>=2:
        states.append((min(end-.05,p['entries'][1]['confirmation_s']+.3),'High again','#bc2946'))
    if not states:
        states=[(.1,'Initial','#78858c'),(float(t[np.argmax(rates[:,0])]),'Largest event','#d68429'),(end-.1,'Late','#78858c')]
    states=sorted(states,key=lambda x:x[0])
    # State markers refer to the center of the exact 50ms displayed field window.
    snaps=[]
    for tm,label,col in states:
        index=max(0,min(round((tm-.025)/.005),len(d['field_5ms'])-10))
        snaps.append(dict(time_s=index*.005+.025,label=label,color=col,
                          field_Hz=d['field_5ms'][index:index+10].sum(0)/g['cell_e_counts']/.05))
    plt.rcParams.update({'font.size':16,'axes.labelsize':19,'xtick.labelsize':16,'ytick.labelsize':16,
                         'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig=plt.figure(figsize=(24,17))
    gs=fig.add_gridspec(5,2,width_ratios=[1.45,1],height_ratios=[2,1,1.5,.7,1.7],
                        hspace=.52,wspace=.3,left=.075,right=.97,bottom=.065,top=.975)
    it,ix=np.where(d['raster']);rt=it*.0001
    yy=np.r_[np.linspace(0,32,20),np.linspace(36,68,20),np.linspace(73,84,20),np.linspace(89,100,20)]
    a=fig.add_subplot(gs[0,0])
    for low,high,col in [(0,20,'#176ba1'),(20,40,'#168aa2'),(40,60,'#466675'),(60,80,'#c17730')]:
        mask=(ix>=low)&(ix<high);a.scatter(rt[mask],yy[ix[mask]],s=7,c=col,lw=0,rasterized=True)
    for y in [34,70.5,86.5]:a.axhline(y,c='#ccc',lw=.7)
    a.set(xlim=(0,end),ylim=(-2,113),yticks=[16,52,78.5,94.5],yticklabels=['Core A E','Core B E','Other E','I'],xlabel='Time (s)')
    a.text(-.08,1.02,'A',transform=a.transAxes,weight='bold',fontsize=24)
    for i,s in enumerate(snaps,1):
        a.axvline(s['time_s'],c=s['color'],ls=':',lw=1)
        a.text(s['time_s'],103+(i%2)*4,str(i),color=s['color'],ha='center',weight='bold')
    zgs=gs[1,0].subgridspec(1,2,wspace=.32)
    chosen=[next((s for s in snaps if s['label']=='Brief event'),snaps[0]),
            next((s for s in snaps if s['label']=='Brief return'),next((s for s in snaps if s['label']=='Entry'),snaps[-1]))]
    for i,s in enumerate(chosen):
        ax=fig.add_subplot(zgs[i]);lo=max(0,s['time_s']-.1);hi=min(end,lo+.4)
        for low,high,col in [(0,20,'#176ba1'),(20,40,'#168aa2')]:
            mask=(ix>=low)&(ix<high)&(rt>=lo)&(rt<hi)
            ax.scatter(rt[mask],ix[mask],marker='|',s=25,c=col,lw=1.1,rasterized=True)
        ax.set(xlim=(lo,hi),ylim=(-1,40),yticks=[9.5,29.5],yticklabels=['Core A E','Core B E'],xlabel='Time (s)')
        a.add_patch(Rectangle((lo,0),hi-lo,68,fill=False,ec=s['color'],lw=1.8))
        ax.axvline(s['time_s'],ls=':',c=s['color'])
    st=d['slow_time_ms']/1000;b=fig.add_subplot(gs[2,0])
    b.plot(st,d['Z'][:,0],c='#74398f',lw=2,label='Mean Z')
    b.fill_between(st,d['Z'][:,2],d['Z'][:,4],color='#74398f',alpha=.12)
    for index,(label,col) in enumerate([('Core A','#d34e99'),('Core B','#249ac1')]):
        b.plot(st,d['Z'][:,5+index],c=col,lw=.9,label=label)
    bm=b.twinx();bm.plot(st,job['eta_m']*d['M'][:,0],c='#ac732d',lw=1.2,label='Native M')
    bm.set_ylabel(r'$\eta_M M$ (mV equiv.)');bm.spines['right'].set_visible(True)
    b.set(xlim=(0,end),ylim=(0,1.05),ylabel='Resource Z');b.tick_params(labelbottom=False)
    lines=b.get_lines()+bm.get_lines();b.legend(lines,[v.get_label() for v in lines],loc='upper right',fontsize=12)
    b.text(-.08,1.02,'B',transform=b.transAxes,weight='bold',fontsize=24)
    bk=fig.add_subplot(gs[3,0],sharex=b);bk.plot(k['time_ms']/1000,k['sahp_mean_conductance_ratio'],c='#a36329')
    bk.set(ylabel=r'Added $g_K/g_L$',xlabel='Time (s)')
    for s in snaps:
        for ax in [b,bk]:ax.axvline(s['time_s'],ls=':',lw=1,c=s['color'])
    cg=gs[4,0].subgridspec(1,len(snaps)+1,width_ratios=[1]*len(snaps)+[.06],wspace=.25)
    fig.text(.035,gs[4,0].get_position(fig).y1+.012,'C',weight='bold',fontsize=24)
    for i,s in enumerate(snaps):
        ax=fig.add_subplot(cg[i]);im=ax.imshow(s['field_Hz'].reshape(20,20),origin='lower',extent=(0,20,0,20),
                  cmap='magma',norm=PowerNorm(.6,0,500),interpolation='nearest')
        for label,xy in zip('AB',g['centers_mm']):
            ax.add_patch(Circle(xy,float(g['core_radius_mm']),fill=False,ec='#54d9d8',lw=1.5))
            ax.text(*xy,label,color='#54d9d8',fontsize=13,ha='center')
        ax.set(xlabel='x (mm)',xticks=[0,10,20],yticks=[0,10,20]);ax.tick_params(labelsize=13)
        ax.text(.5,1.06,f'{i+1} {s["label"]}\n{s["time_s"]:.2f} s',transform=ax.transAxes,ha='center',color=s['color'],fontsize=13)
        if i==0:
            ax.set_ylabel('y (mm)')
        else:ax.tick_params(labelleft=False)
    cb=fig.colorbar(im,cax=fig.add_subplot(cg[-1]));cb.set_label('E rate (Hz)',fontsize=15)
    dg=gs[:2,1].subgridspec(1,2,width_ratios=[1.8,1],wspace=.32)
    dax=fig.add_subplot(dg[0],projection='3d')
    kval=np.interp(st,k['time_ms']/1000,k['sahp_mean_conductance_ratio'])
    xyz=np.c_[d['Z'][:,0],kval,np.interp(st,t,rates[:,0])]
    lc=Line3DCollection(np.stack([xyz[:-1],xyz[1:]],axis=1),cmap='viridis',norm=Normalize(0,end),linewidth=.8)
    lc.set_array(st[:-1]);dax.add_collection(lc)
    dax.set(xlim=(max(0,xyz[:,0].min()-.03),1.02),ylim=(0,max(.01,xyz[:,1].max()*1.05)),
            zlim=(0,max(30,xyz[:,2].max()*1.05)),xlabel='Mean Z',ylabel=r'$g_K/g_L$',zlabel='E rate (Hz)')
    dax.view_init(23,-54);dax.set_box_aspect((1.3,1,1.2));dax.tick_params(labelsize=11)
    for axis in [dax.xaxis,dax.yaxis,dax.zaxis]:axis.label.set_fontsize(15)
    dax.text2D(-.06,.97,'D',transform=dax.transAxes,weight='bold',fontsize=24)
    bar=fig.colorbar(lc,ax=dax,fraction=.025,pad=.1,shrink=.4);bar.set_label('Time (s)',fontsize=13)
    e=fig.add_subplot(dg[1]);e.axis('off');e.text(0,1,'E',weight='bold',fontsize=24)
    table=[]
    for rr in all_rows:
        j=rr['job'];pp=rr['primary']
        intervals=pp['interhigh_intervals']+([pp['latest_postexit']] if pp['latest_postexit'] else [])
        inter=any(audit.qualifies(v) for v in intervals)
        table.append([f"{j['gamma']:.2f}/{j['sahp_gain']:g}/{j['sahp_tau_s']:g}",
                      str(len(pp['entries'])),str(int(bool(pp['low_activity_exits']))),str(int(inter))])
    tb=e.table(cellText=table,colLabels=['γ / K / τK','High','Exit','IED\nreturn'],loc='center',cellLoc='center',colWidths=[.6,.24,.24,.24])
    tb.auto_set_font_size(False);tb.set_fontsize(10);tb.scale(1,2.4)
    # Fixed Fig3C energy pipeline; no patient reselection or readout fabrication.
    old.OUT=OUT;old.run.OUT=OUT
    summary=dict(job=job,entries=p['entries'],observed_s=end,finite_events=p['events'])
    energy=old.early_energy(job['name'],g,summary,anchor='high_gate')
    if energy:
        from scripts.plot_contact_plane_static import _smooth_rank_field_mm
        ev,frozen,_=energy;fg=gs[2:,1].subgridspec(1,2,wspace=.55)
        for i,(values,label) in enumerate([(ev['model_robust_z'],'Model'),(ev['patient_robust_z'],f"E10 | {ev['patient_public_seizure']}")]):
            ax=fig.add_subplot(fg[i]);values=np.array(values);pts=np.array(frozen['points_mm'])
            xx,yy,field,_,_=_smooth_rank_field_mm(pts[:,0],pts[:,1],values,np.array(frozen['support_a']),
                       frozen['display_xlim_mm'],frozen['display_ylim_mm'],frozen['display_sigma_mm'])
            limit=max(1e-9,float(np.max(np.abs(values))));norm=Normalize(-limit,limit)
            im=ax.imshow(field,origin='lower',extent=[xx.min(),xx.max(),yy.min(),yy.max()],cmap='RdBu',norm=norm)
            ax.scatter(pts[:,0],pts[:,1],c=values,cmap='RdBu',norm=norm,s=45,edgecolors='white')
            ax.set(xlabel='Shared axis (mm)',ylabel='y (mm)' if i==0 else '')
            ax.text(.5,1.03,label,transform=ax.transAxes,ha='center',fontsize=18)
            if i==0:ax.text(-.13,1.16,'F',transform=ax.transAxes,weight='bold',fontsize=24)
            bar=fig.colorbar(im,ax=ax,shrink=.45,pad=.04,fraction=.035);bar.set_label('Power change (robust z)',fontsize=12)
    else:
        ax=fig.add_subplot(gs[2:,1]);ax.axis('off');ax.text(0,1,'F',weight='bold',fontsize=24)
        ax.text(.05,.75,'Early-energy comparison unavailable\nwithout sufficient baseline and entry.',fontsize=16)
    dest=OUT/'figures';dest.mkdir(exist_ok=True);name='fig5_'+job['name']
    for ext in ['png','pdf']:fig.savefig(dest/f'{name}.{ext}',dpi=155)
    plt.close(fig)
    old.write(dest/f'{name}_metadata.json',dict(audit=row,states=snaps,energy=energy[0] if energy else None,
         model_energy_sampling='500Hz observation; provisional, requires dense-current confirmation',
         trajectory='Measured SNN projection, not vector field or bifurcation',human_review='PENDING'))
    return name


def finish():
    rr=rows();protocol=json.loads((OUT/'protocol.json').read_text())
    selected=sorted(rr,key=lambda r:(r['primary']['temporal_loop_pass'],r['primary']['preentry_brief_screen'],
                      bool(r['primary']['low_activity_exits']),len(r['primary']['entries'])),reverse=True)[:2]
    files=[];failures=[]
    for row in selected:
        try:files.append(plot(row,rr))
        except Exception as exc:failures.append(dict(name=row['job']['name'],error=repr(exc)))
    text=['# 闭环筛查结果\n',
          '本轮目标为同一连续轨迹的高活动→反复短间期事件→再次高活动。以下是模型内时序筛查，尚不等于患者传播、发作形态或跨噪声验证。\n',
          '| γ | K倍率 | τK(s) | 时长(s) | 前期短事件 | 高活动次数 | 退出次数 | 分类 |',
          '|---|---|---|---|---|---|---|---|']
    for row in rr:
        p=row['primary'];j=row['job']
        text.append(f"| {j['gamma']:.4g} | {j['sahp_gain']:g} | {j['sahp_tau_s']:g} | {p['observed_s']:g} | {p['preentry']['brief_count']} | {len(p['entries'])} | {len(p['low_activity_exits'])} | {p['classification']} |")
    repair=OUT/'finalization_repair.json'
    if repair.exists():
        text.insert(2,'六组均已停止；四组记录达到60s，两组在56s因原定墙钟时限截尾。末次JSON分析写入曾因NumPy数组序列化失败，已从完整保存的原始计数和checkpoint修复；不改仿真、判据或原日志。[核查记录](finalization_repair.json)。\n')
        text.append('\nγ=1/6、K=0.1、τK=1s：先有67个短事件，20.33s进入高活动，但直到60s无退出；最后5s全E平均500Hz，接近不应期上限，是高率平台，不能自动称为发作样持续振荡。τK=5s则60s内112个短事件、Z末值约0.957，没有进入高活动。两者都没有完成闭环。γ=0.5的τK=1/2s条件在56s前始终未恢复，Z已接近零；只是缩短K衰减未保留原退出作用。\n')
    text.extend(['\n旧27条轨迹按新标准重审，无通过时序闭环者；原46s延长仅有高活动—低活动—高活动，之间没有合格短事件。旧结果没有被覆盖。\n',
                 '若本轮有TEMPORAL_LOOP_PASS，必须继续目视检查原生raster/空间传播，并确认初始短事件保留和不同噪声下的行为；若没有则明确失败或观察窗截尾，不更改阈值、不自动扩大搜索。\n'])
    for file in files:text.append(f'![{file}]({OUT}/figures/{file}.png)\n')
    (OUT/'scientific_review.md').write_text('\n'.join(text))
    if files:
        (OUT/'figures/README.md').write_text('\n'.join(f'### {file}.png / .pdf\n\n同一模型轨迹的连续raster、两核放大、Z/M与新增K、原生空间快照，以及本轮条件的实际分类。D为实测状态投影；F沿用固定Fig3C数据，模型500Hz谱仍为待密集采样确认的结果。\n\n**关注点**：核查高活动终止后是否有多次短传播，再次进入；低活动和间期恢复分开，不把未通过的候选标作完成闭环。\n' for file in files))
    old.write(OUT/'postprocess_complete.json',dict(completed_epoch=time.time(),files=files,failures=failures,
              full_Fig5_acceptance='NOT_ESTABLISHED',human_review='PENDING'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    if args.wait:
        deadline=json.loads((OUT/'protocol.json').read_text())['deadline_epoch']+1200
        while not (OUT/'batch_complete.json').exists() and time.time()<deadline:time.sleep(30)
    finish()
