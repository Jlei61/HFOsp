#!/usr/bin/env python3
"""Fig5-style candidate figures, overview, GIFs and review for the k100 x tau_K batch."""
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
from matplotlib.colors import Normalize, PowerNorm, ListedColormap
from matplotlib.patches import Circle, Rectangle, Patch
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import analyze_topic4_interictal_recurrence as audit
import analyze_topic4_k100_mechanism as mech
import run_topic4_k100_recurrence as run

OUT=run.OUT
old=audit.old
K100=run.K100;TAU=run.TAU


def protocol():return json.loads((OUT/'protocol.json').read_text())


def row_for(name):
    """Final analysis row if present, else the live row; also returns the run folder."""
    p=protocol()
    reused={r['name']:r for r in p['reused_previous_conditions']}
    if name in reused:
        folder=Path(reused[name]['previous_run'])
        row=audit.analyze_folder(folder,OUT/'geometry.npz');row['reused_from']=reused[name]['previous_name'];row['run_status']='OBSERVATION_HORIZON_REACHED'
        return row,folder
    folder=OUT/'runs'/name
    path=OUT/'analysis'/f'{name}.json'
    if path.exists():row=json.loads(path.read_text())
    else:row=old.safe(audit.analyze_folder(folder,OUT/'geometry.npz'))
    return row,folder


def dynamics_label(row):
    p=row['primary'];j=row['job'];end=p['observed_s'];ent=p['entries'];ex=p['low_activity_exits']
    if p['temporal_loop_pass']:return 'loop',f"loop: high {ent[0]['onset_s']:.1f}s, exit {ex[0]['confirmation_s']:.1f}s,\nIED return, high {ent[1]['onset_s']:.1f}s"
    if not ent:return 'no_entry',f"no high in {end:.0f}s\n{p['preentry']['brief_count']} brief events"
    if not ex:return 'stuck',f"high {ent[0]['onset_s']:.1f}s\nno exit by {end:.0f}s"
    post=p['latest_postexit'];n=post['brief_count'] if post else 0
    if len(ent)>=2:return 'high_low_high',f"high {ent[0]['onset_s']:.1f}s, exit {ex[0]['confirmation_s']:.1f}s,\n{n} brief, high {ent[1]['onset_s']:.1f}s"
    return 'exit_no_return',f"high {ent[0]['onset_s']:.1f}s, exit {ex[0]['confirmation_s']:.1f}s,\n{n} brief events after"


CLASS_COLORS={'no_entry':'#dbe5eb','stuck':'#cf6977','exit_no_return':'#e8b26a','high_low_high':'#8fb7d9','loop':'#54a88f'}


def matrix_rows():
    p=protocol();rows=[]
    for j in p['initial_jobs']:
        if not (OUT/'runs'/j['name']/'chunks').exists():continue
        row,folder=row_for(j['name']);rows.append((j,row,folder))
    for r in p['reused_previous_conditions']:
        row,folder=row_for(r['name']);j=dict(row['job'],name=r['name'],k100=r['k100']);rows.append((j,row,folder))
    return rows


def draw_matrix_panel(ax,rows,highlight=None):
    mat=np.full((3,3),np.nan);cells={}
    for j,row,folder in rows:
        if j['seed']!=run.SEED:continue
        iy=K100.index(j['k100']);jx=TAU.index(j['sahp_tau_s']);kind,text=dynamics_label(row)
        cells[iy,jx]=(kind,text,row['primary']['observed_s'],row.get('run_status'));mat[iy,jx]=list(CLASS_COLORS).index(kind)
    ax.imshow(mat,origin='lower',cmap=ListedColormap(list(CLASS_COLORS.values())),vmin=-.5,vmax=4.5,interpolation='nearest',aspect='auto')
    for (iy,jx),(kind,text,obs,status) in cells.items():
        censored=status in ['RUNNING','STOPPED_AT_CHECKPOINT_FOR_TRIAGE','CENSORED_WALL_DEADLINE'] or (obs<run.HORIZON_S and kind in ['no_entry','stuck','exit_no_return'])
        label=text+f"\n[{obs:.0f} s{', censored' if censored else ''}]"
        ax.text(jx,iy,label,ha='center',va='center',fontsize=8.5)
        if highlight and (iy,jx)==highlight:ax.add_patch(Rectangle((jx-.5,iy-.5),1,1,fill=False,ec='k',lw=2.5))
    ax.set(xticks=range(3),xticklabels=[f'{t:g}' for t in TAU],yticks=range(3),yticklabels=[f'{k:g}' for k in K100],xlabel=r'$\tau_K$ (s)',ylabel=r'$k_{100}$')
    ax.yaxis.tick_right();ax.yaxis.set_label_position('right');ax.spines['right'].set_visible(True);ax.spines['left'].set_visible(False)
    ax.tick_params(labelsize=12);ax.xaxis.label.set_fontsize(14);ax.yaxis.label.set_fontsize(14)
    for x in [.5,1.5]:ax.axvline(x,c='w',lw=2);ax.axhline(x,c='w',lw=2)
    handles=[Patch(fc=c,label=l) for l,c in zip(['no high (censored window)','high without exit','high → exit, no IED return','high → low → high','temporal loop'],CLASS_COLORS.values())]
    ax.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2,fontsize=9.5,frameon=False)


def choose_states(p,d,t,rates,end):
    states=[];brief=p['preentry']['brief_events']
    if brief and not p['entries']:
        # stationary interictal record: show an early, a middle and a late brief event
        for lo,hi,label in [(.5,min(20.,end),'Brief event (early)'),(end*.4,end*.6,'Brief event (middle)'),(max(.5,end-20.),end,'Brief event (late)')]:
            sel=[e for e in brief if lo<=e['peak_s']<=hi]
            if sel:
                e=max(sel,key=lambda x:x['peak_Hz']);states.append((e['peak_s'],label,'#277cb3'))
    elif brief:
        e=max(brief,key=lambda x:x['peak_Hz']);states.append((e['peak_s'],'Brief event','#277cb3'))
    if p['entries']:
        on=p['entries'][0]['onset_s'];ep=next((e for e in p['events'] if e['start_s']<=on<e['end_s']),None)
        entry=max(.025,ep['start_s']+.025 if ep else on-.1);states.append((entry,'Entry','#d68429'))
        lo=round(on/.01);hi=min(len(rates),round((on+1)/.01));states.append((t[lo+np.argmax(rates[lo:hi,0])],'High','#bc2946'))
    post=p['latest_postexit']
    if p['low_activity_exits']:
        pp=next((v for v in p['interhigh_intervals'] if v['temporal_pass']),post)
        if pp and pp['brief_events']:
            es=pp['brief_events'];states.append((max(es,key=lambda x:x['peak_Hz'])['peak_s'],'Brief return','#208677'))
        else:states.append((min(end-.05,p['low_activity_exits'][0]['confirmation_s']+.1),'Low activity','#778891'))
    if len(p['entries'])>=2:states.append((min(end-.05,p['entries'][1]['confirmation_s']+.3),'High again','#bc2946'))
    if not states:states=[(.1,'Initial','#78858c'),(float(t[np.argmax(rates[:,0])]),'Largest event','#d68429'),(end-.1,'Late','#78858c')]
    return sorted(states,key=lambda x:x[0])


def plot_fig5(name,rows=None,energy=True):
    rows=rows or matrix_rows();row,folder=row_for(name);p=row['primary'];job=row['job'];end=p['observed_s']
    d=old.load(folder);k=old.load(folder,'intrinsic_adaptation_chunks')
    with np.load(OUT/'geometry.npz') as f:g={key:f[key] for key in f.files}
    n=len(d['spikes_1ms'])//10;t=(np.arange(n)+.5)*.01
    rates=d['spikes_1ms'][:n*10].reshape(n,10,2).sum(1)/np.array([320.,80.])
    states=choose_states(p,d,t,rates,end)
    snaps=[]
    for tm,label,col in states:
        index=max(0,min(round((tm-.025)/.005),len(d['field_5ms'])-10))
        snaps.append(dict(time_s=index*.005+.025,label=label,color=col,field_Hz=d['field_5ms'][index:index+10].sum(0)/g['cell_e_counts']/.05))
    plt.rcParams.update({'font.size':16,'axes.labelsize':19,'xtick.labelsize':16,'ytick.labelsize':16,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig=plt.figure(figsize=(24,17.5))
    gs=fig.add_gridspec(5,2,width_ratios=[1.45,1],height_ratios=[2,1,1.5,.7,1.7],hspace=.52,wspace=.3,left=.075,right=.97,bottom=.06,top=.975)
    it,ix=np.where(d['raster']);rt=it*.0001
    yy=np.r_[np.linspace(0,32,20),np.linspace(36,68,20),np.linspace(73,84,20),np.linspace(89,100,20)]
    a=fig.add_subplot(gs[0,0])
    for low,high,col in [(0,20,'#176ba1'),(20,40,'#168aa2'),(40,60,'#466675'),(60,80,'#c17730')]:
        mask=(ix>=low)&(ix<high);a.scatter(rt[mask],yy[ix[mask]],s=7,c=col,lw=0,rasterized=True)
    for y in [34,70.5,86.5]:a.axhline(y,c='#ccc',lw=.7)
    a.set(xlim=(0,end),ylim=(-2,113),yticks=[16,52,78.5,94.5],yticklabels=['Core A E','Core B E','Other E','I'],xlabel='Time (s)')
    a.text(-.08,1.02,'A',transform=a.transAxes,weight='bold',fontsize=24)
    for i,s in enumerate(snaps,1):
        a.axvline(s['time_s'],c=s['color'],ls=':',lw=1);a.text(s['time_s'],103+(i%2)*4,str(i),color=s['color'],ha='center',weight='bold')
    zgs=gs[1,0].subgridspec(1,2,wspace=.32)
    chosen=[next((s for s in snaps if s['label'].startswith('Brief event')),snaps[0]),
            next((s for s in snaps if s['label']=='Brief return'),next((s for s in snaps if s['label']=='Entry'),next((s for s in snaps if s['label']=='Brief event (late)'),snaps[-1])))]
    for i,s in enumerate(chosen):
        ax=fig.add_subplot(zgs[i]);lo=max(0,s['time_s']-.1);hi=min(end,lo+.4)
        for low,high,col in [(0,20,'#176ba1'),(20,40,'#168aa2')]:
            mask=(ix>=low)&(ix<high)&(rt>=lo)&(rt<hi);ax.scatter(rt[mask],ix[mask],marker='|',s=25,c=col,lw=1.1,rasterized=True)
        ax.set(xlim=(lo,hi),ylim=(-1,40),yticks=[9.5,29.5],yticklabels=['Core A E','Core B E'],xlabel='Time (s)')
        a.add_patch(Rectangle((lo,0),hi-lo,68,fill=False,ec=s['color'],lw=1.8));ax.axvline(s['time_s'],ls=':',c=s['color'])
        ax.set_title(f"{snaps.index(s)+1} {s['label']}",loc='left',color=s['color'],fontsize=14)
    st=d['slow_time_ms']/1000;b=fig.add_subplot(gs[2,0])
    b.plot(st,d['Z'][:,0],c='#74398f',lw=2,label='Mean Z');b.fill_between(st,d['Z'][:,2],d['Z'][:,4],color='#74398f',alpha=.12)
    for index,(label,col) in enumerate([('Core A','#d34e99'),('Core B','#249ac1')]):b.plot(st,d['Z'][:,5+index],c=col,lw=.9,label=label)
    bm=b.twinx();bm.plot(st,job['eta_m']*d['M'][:,0],c='#ac732d',lw=1.2,label='Native M ($\\eta_M M$)');bm.set_ylabel(r'$\eta_M M$ (mV equiv.)');bm.spines['right'].set_visible(True)
    b.set(xlim=(0,end),ylim=(0,1.05),ylabel='Resource Z');b.tick_params(labelbottom=False)
    lines=b.get_lines()+bm.get_lines();b.legend(lines,[v.get_label() for v in lines],loc='lower center',ncol=4,fontsize=11,frameon=True)
    b.text(-.08,1.02,'B',transform=b.transAxes,weight='bold',fontsize=24)
    bk=fig.add_subplot(gs[3,0],sharex=b)
    if k:bk.plot(k['time_ms']/1000,k['sahp_mean_conductance_ratio'],c='#a36329')
    bk.set(ylabel='Added K\n'+r'$g_K/g_L$',xlabel='Time (s)')
    for s in snaps:
        for ax in [b,bk]:ax.axvline(s['time_s'],ls=':',lw=1,c=s['color'])
    cg=gs[4,0].subgridspec(1,len(snaps)+1,width_ratios=[1]*len(snaps)+[.06],wspace=.25)
    fig.text(.035,gs[4,0].get_position(fig).y1+.012,'C',weight='bold',fontsize=24)
    for i,s in enumerate(snaps):
        ax=fig.add_subplot(cg[i]);im=ax.imshow(s['field_Hz'].reshape(20,20),origin='lower',extent=(0,20,0,20),cmap='magma',norm=PowerNorm(.6,0,500),interpolation='nearest')
        for label,xy in zip('AB',g['centers_mm']):
            ax.add_patch(Circle(xy,float(g['core_radius_mm']),fill=False,ec='#54d9d8',lw=1.5));ax.text(*xy,label,color='#54d9d8',fontsize=13,ha='center')
        ax.set(xlabel='x (mm)',xticks=[0,10,20],yticks=[0,10,20]);ax.tick_params(labelsize=13)
        ax.text(.5,1.06,f'{i+1} {s["label"]}\n{s["time_s"]:.2f} s',transform=ax.transAxes,ha='center',color=s['color'],fontsize=13)
        if i==0:ax.set_ylabel('y (mm)')
        else:ax.tick_params(labelleft=False)
    cb=fig.colorbar(im,cax=fig.add_subplot(cg[-1]));cb.set_label('E rate (Hz)',fontsize=15)
    dg=gs[:2,1].subgridspec(1,2,width_ratios=[1,1.1],wspace=.3)
    dax=fig.add_subplot(dg[0],projection='3d')
    kval=np.interp(st,k['time_ms']/1000,k['sahp_mean_conductance_ratio']) if k else np.zeros(len(st))
    xyz=np.c_[d['Z'][:,0],kval,np.interp(st,t,rates[:,0])]
    lc=Line3DCollection(np.stack([xyz[:-1],xyz[1:]],axis=1),cmap='viridis',norm=Normalize(0,end),linewidth=.8);lc.set_array(st[:-1]);dax.add_collection(lc)
    dax.set(xlim=(max(0,xyz[:,0].min()-.03),1.02),ylim=(0,max(.01,xyz[:,1].max()*1.05)),zlim=(0,max(30,xyz[:,2].max()*1.05)),xlabel='Mean Z',ylabel=r'Added $g_K/g_L$',zlabel='E rate (Hz)')
    dax.view_init(23,-54);dax.set_box_aspect((1.3,1,1.2));dax.tick_params(labelsize=11)
    for axis in [dax.xaxis,dax.yaxis,dax.zaxis]:axis.label.set_fontsize(14)
    for i,s in enumerate(snaps,1):
        kk=np.argmin(abs(st-s['time_s']));dax.scatter(*xyz[kk],c='white',edgecolors='#444',s=70,depthshade=False);dax.text(*xyz[kk],str(i),fontsize=11)
    dax.text2D(-.06,.97,'D',transform=dax.transAxes,weight='bold',fontsize=24)
    bar=fig.colorbar(lc,ax=dax,fraction=.035,pad=.01,shrink=.45,location='left');bar.set_label('Time (s)',fontsize=13);bar.ax.tick_params(labelsize=11)
    e=fig.add_subplot(dg[1]);e.text(-.32,1.02,'E',transform=e.transAxes,weight='bold',fontsize=24)
    hl=(K100.index(job['k100']),TAU.index(job['sahp_tau_s'])) if job.get('k100') in K100 and job['sahp_tau_s'] in TAU else None
    draw_matrix_panel(e,rows,highlight=hl)
    energy_out=None
    if energy and p['entries']:
        old.OUT=OUT;old.run.OUT=OUT
        try:
            summary=dict(job=job,entries=p['entries'],observed_s=end,finite_events=p['events'])
            # early_energy expects OUT/'runs'/name; reused runs live elsewhere -> temporary symlink
            link=OUT/'runs'/name;temporary=not link.exists()
            if temporary:link.symlink_to(folder)
            try:energy_out=old.early_energy(name,g,summary,anchor='high_gate')
            finally:
                if temporary:link.unlink()
        except Exception as exc:
            energy_out=None;energy_error=repr(exc)
    if energy_out:
        from scripts.plot_contact_plane_static import _smooth_rank_field_mm
        ev,frozen,_=energy_out;fg=gs[2:,1].subgridspec(1,2,wspace=.55)
        for i,(values,label) in enumerate([(ev['model_robust_z'],'Model (500 Hz observer, provisional)'),(ev['patient_robust_z'],f"E10 | {ev['patient_public_seizure']}")]):
            ax=fig.add_subplot(fg[i]);values=np.array(values);pts=np.array(frozen['points_mm'])
            xx,yy2,field,_,_=_smooth_rank_field_mm(pts[:,0],pts[:,1],values,np.array(frozen['support_a']),frozen['display_xlim_mm'],frozen['display_ylim_mm'],frozen['display_sigma_mm'])
            limit=max(1e-9,float(np.max(np.abs(values))));norm=Normalize(-limit,limit)
            im=ax.imshow(field,origin='lower',extent=[xx.min(),xx.max(),yy2.min(),yy2.max()],cmap='RdBu',norm=norm)
            ax.scatter(pts[:,0],pts[:,1],c=values,cmap='RdBu',norm=norm,s=45,edgecolors='white')
            ax.set(xlabel='Shared axis (mm)',ylabel='y (mm)' if i==0 else '');ax.text(.5,1.03,label,transform=ax.transAxes,ha='center',fontsize=14)
            if i==0:ax.text(-.13,1.16,'F',transform=ax.transAxes,weight='bold',fontsize=24)
            bar=fig.colorbar(im,ax=ax,shrink=.45,pad=.04,fraction=.035);bar.set_label('Power change (robust z)',fontsize=12)
        fig.text(.56,.035,f"F: model {ev['model_positive_contacts']}/15 contacts with power increase; Spearman ρ={ev['contact_rho']:.2f} vs fixed Fig3C patient field.\nRank agreement is not early-energy reproduction; 500 Hz observer, dense confirmation not run.",fontsize=11)
    else:
        ax=fig.add_subplot(gs[2:,1]);ax.axis('off');ax.text(0,1,'F',weight='bold',fontsize=24)
        ax.text(.05,.75,'Patient early-energy comparison not verified:\nno confirmed entry with adequate baseline\nand onset recording in this trajectory.',fontsize=15,va='top')
    dest=OUT/'figures';dest.mkdir(exist_ok=True);fname='fig5_'+name
    for ext in ['png','pdf']:fig.savefig(dest/f'{fname}.{ext}',dpi=150)
    plt.close(fig)
    old.write(dest/f'{fname}_metadata.json',dict(audit=row,states=[{k2:v for k2,v in s.items() if k2!='field_Hz'} for s in snaps],
         energy=old.safe(energy_out[0]) if energy_out else None,model_energy_sampling='500 Hz synaptic-current observer; provisional, dense-current confirmation not run in this batch',
         trajectory='Measured SNN projection (mean Z, mean added gK/gL, all-E rate); not a vector field or bifurcation proof',
         K_label='Added K conductance (new in this line); M is the native adaptation variable',human_review='PENDING',producer_sha256=run.sha(__file__)))
    return fname


def plot_overview():
    rows=matrix_rows();p=protocol()
    plt.rcParams.update({'font.size':13,'axes.labelsize':15,'xtick.labelsize':13,'ytick.labelsize':13,'axes.spines.top':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(3,3,figsize=(20,12),sharex=True,sharey=True)
    fig.subplots_adjust(left=.06,right=.94,top=.93,bottom=.1,wspace=.18,hspace=.4)
    xmax=max(r[1]['primary']['observed_s'] for r in rows)
    for j,row,folder in rows:
        if j['seed']!=run.SEED:continue
        iy=2-K100.index(j['k100']);jx=TAU.index(j['sahp_tau_s']);ax=axes[iy,jx];pp=row['primary']
        d=old.load(folder,keys=['spikes_1ms','slow_time_ms','Z'])
        n=len(d['spikes_1ms'])//10;t=(np.arange(n)+.5)*.01;rate=d['spikes_1ms'][:n*10,0].reshape(n,10).sum(1)/320
        ax.plot(t,rate,c='#176b9c',lw=.8);zax=ax.twinx();zax.plot(d['slow_time_ms']/1000,d['Z'][:,0],c='#8b3d8d',lw=1.1);zax.set(ylim=(0,1.05),yticks=[0,.5,1]);zax.tick_params(axis='y',colors='#8b3d8d')
        if jx==2:zax.set_ylabel('Mean Z',color='#8b3d8d')
        else:zax.tick_params(labelright=False)
        starts=[e['start_s'] for e in pp['preentry']['brief_events']];ax.scatter(starts,np.full(len(starts),-15),s=22,marker='|',c='#209679')
        post=pp['latest_postexit']
        if post:
            starts=[e['start_s'] for e in post['brief_events']];ax.scatter(starts,np.full(len(starts),-15),s=22,marker='|',c='#e07b39')
        for entry in pp['entries']:ax.axvline(entry['onset_s'],c='#b82d46',ls='--',lw=1);ax.text(entry['onset_s']+.5,380,f"{entry['onset_s']:.1f} s",color='#b82d46',fontsize=11)
        for ex in pp['low_activity_exits']:ax.axvline(ex['confirmation_s'],c='#208677',ls='--',lw=1)
        if pp['observed_s']<xmax:ax.axvspan(pp['observed_s'],xmax,facecolor='#eeeeee',edgecolor='#aaaaaa',hatch='///',lw=0)
        kind,_=dynamics_label(row)
        ax.set_title(f"k100 = {j['k100']:g}, τK = {j['sahp_tau_s']:g} s (K gain {j['sahp_gain']:g})"+('  [reused]' if row.get('reused_from') else ''),loc='left',fontsize=13)
        ax.text(.98,.9,f"{pp['preentry']['brief_count']} brief before high | {pp['classification'].replace('_',' ').lower()}",transform=ax.transAxes,ha='right',fontsize=10.5)
        ax.set(xlim=(0,xmax),ylim=(-30,520),yticks=[0,250,500])
    for iy in range(3):axes[iy,0].set_ylabel('Mean E rate (Hz)')
    for jx in range(3):axes[2,jx].set_xlabel('Time (s)')
    fig.legend(handles=[Line2D([],[],c='#176b9c',label='All-E rate (10 ms)'),Line2D([],[],c='#8b3d8d',label='Mean Z'),Line2D([],[],c='#b82d46',ls='--',label='High-activity onset'),Line2D([],[],c='#208677',ls='--',label='Exit confirmed'),
              Line2D([],[],c='#209679',marker='|',ls='none',label='Brief event before first high'),Line2D([],[],c='#e07b39',marker='|',ls='none',label='Brief event after exit'),Patch(facecolor='#eee',edgecolor='#aaa',hatch='///',label='Not observed (censored)')],
              loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,.005),fontsize=12)
    dest=OUT/'figures';dest.mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(dest/f'k100_matrix_overview.{ext}',dpi=140)
    plt.close(fig)
    old.write(dest/'k100_matrix_overview_metadata.json',dict(source=str(OUT),rate_bin_ms=10,Z_sample_ms=20,brief_event_definition=audit.RULE,short_records_hatched_not_imputed=True,human_review='PENDING'))


def branch_rows():
    p=protocol();rows=[]
    for j in p['branch_jobs']:
        folder=OUT/'runs'/j['name']
        if not (folder/'chunks').exists():continue
        path=OUT/'analysis'/f"{j['name']}.json"
        row=json.loads(path.read_text()) if path.exists() else old.safe(audit.analyze_folder(folder,OUT/'geometry.npz'))
        rows.append((j,row,folder))
    return rows


def write_tables():
    """Machine-generated tables for the review: matrix summary, current balance, branches."""
    table=mech.summary_table()
    lines=['## 自动生成表格（由 finish_topic4_k100_recurrence.py write_tables 产生）\n','### 矩阵总表\n',(OUT/'matrix_summary.md').read_text()]
    lines.append('\n### 各条件电流平衡（mV-equivalent，窗口均值；gK_needed 为按分流稳态 V∞<18 mV 反解的 gK/gL）\n')
    lines.append('| 条件 | 窗口 | 全E率 (Hz) | I_E | Z·局部抑制 | 全局分流 | 新增K | 原生M | Z | gK/gL | 需要 gK/gL |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for j,row,folder in matrix_rows()+branch_rows():
        res=mech.analyze(j['name'],folder,write=True)
        if not res:continue
        o=res[0]
        for w,b in o['current_balance'].items():
            lines.append(f"| {j['name']} | {w} | {b['global_E_rate_Hz'] if b['global_E_rate_Hz'] is None else round(b['global_E_rate_Hz'],1)} | {b['I_E']:.1f} | {(b['Z_local_I'] or 0):.1f} | {(b['I_global'] or 0):.1f} | {b['I_K']:.1f} | {b['I_M']:.2f} | {b['Z_mean']:.3f} | {b['g_K']:.3f} | {b['g_K_needed_to_hold_V_below_threshold']:.1f} |")
    br=branch_rows()
    if br:
        lines.append('\n### 因果诊断分支（从保存的完整状态继续，只改一个反馈项；不计入自主闭环）\n')
        lines.append('| 分支 | 来源 | 起点 (s) | 修改 | 观察至 (s) | 末段全E (Hz) | 分支内进入 (s) | 分支内退出确认 (s) | 分支内短事件数 | 末 Z | 末 gK/gL |')
        lines.append('|---|---|---|---|---|---|---|---|---|---|---|')
        for j,row,folder in br:
            p=row['primary'];b=j['branch'];o=mech.analyze(j['name'],folder,write=False)[0];t0=b['start_s']
            exits=[x for x in p['low_activity_exits'] if x['confirmation_s']>t0];entries=[e for e in p['entries'] if e['onset_s']>t0]
            brief=[e for e in p['events'] if e['start_s']>=t0 and .02<=e['duration_s']<=.2]
            lines.append(f"| {j['name']} | {Path(b['source']).name} | {t0:g} | {json.dumps(b['modification'])} | {p['observed_s']:g} | {o['late_all_E_Hz']:.0f} | {entries[0]['onset_s'] if entries else '-'} | {exits[0]['confirmation_s'] if exits else '-'} | {len(brief)} | {o['final_Z']:.3f} | {o['final_gK']:.2f} |")
    (OUT/'review_tables.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


def figure_readme(extra=''):
    dest=OUT/'figures';blocks=['# 图目录说明\n\n候选交付，均待用户目视验收；不替换正式论文图。\n']
    for p in sorted(dest.glob('fig5_*.png')):
        blocks.append(f'### {p.name}\n\n同一条自主轨迹的 Fig.5 式候选图：A 连续 raster（四组各 20 个固定采样细胞）与两核放大；B 原生 Z（均值、10–90% 带、两核）与原生 M；下方为新增 K 电导（与 M 分开标注）；C 与标记时刻严格对应的原生 20×20 场；D 实测状态投影（Z、新增 gK/gL、全 E 率，右侧时间色条）；E k100×τK 参数图（动力学类型、转变时间、观察窗截尾）；F 固定 Fig3C 患者早期能量场对照（500 Hz 观测器，未做密集采样确认）。\n\n**关注点**：看是否有退出后的短事件返回和再次进入；平台期 500 Hz 为不应期上限，不是发作形态；F 的秩相关不等于早期能量增强复现。\n')
    for p in sorted(dest.glob('k100_matrix_overview.png')):
        blocks.append(f'### {p.name}\n\nk100×τK 九格的全 E 率与平均 Z 共用时间轴；绿色短线为首次高活动前的合格短事件，橙色为退出后的短事件；红虚线为高活动起点，绿虚线为退出确认；斜线区为未观察到的时间（截尾，不外推）。\n\n**关注点**：进入时间随 k100 的变化，以及是否有任何格子出现退出与短事件返回。\n')
    for p in sorted((dest/'mechanism').glob('branches_*.png')):
        blocks.append(f'### mechanism/{p.name}\n\n因果诊断分支叠加图：从同一保存的完整状态继续，各分支只改一个反馈项（去 Z 门控 / K 增量倍率 / 续跑）；面板为全 E 率、平均 Z、新增 gK/gL、兴奋输入（实线）与总抑制+适应（虚线）、核 A 平均膜电位；竖虚线为分支起点。\n\n**关注点**：哪一项改动让平台崩溃、崩溃后短事件何时回来；这些是诊断，不是自主闭环。\n')
    for p in sorted((dest/'mechanism').glob('mechanism_*.png')):
        blocks.append(f'### mechanism/{p.name}\n\n机制诊断图：全 E 与两核率；分区 Z；E 兴奋输入、Z 门控局部抑制、Z 门控全局分流、新增 K 电流、原生 M 电流（对数轴）；新增 gK/gL 与全局 gG/gL；分区平均膜电位或 J≥阈值比例。\n\n**关注点**：高活动后 Z 归零使局部与全局两条抑制通路同时消失，只剩 K 与 M；比较 K 电流与兴奋输入的量级。\n')
    for p in sorted(dest.glob('native_*.gif')):
        blocks.append(f'### {p.name}\n\n原生 20×20 场 50 ms 计数动画，与全 E/两核率、Z、新增 gK 同步；画面显示真实仿真时间。\n\n**关注点**：区分局部核内活动、贯通波前与全片平台；不替代患者传播验证。\n')
    for p in sorted(dest.glob('native_storyboard_*.png')):
        blocks.append(f'### {p.name}\n\n原生场八帧快照，统一色标 0–500 Hz，圆圈为物理核半径。\n\n**关注点**：传播是否沿核之间的轴，还是整片同时抬升。\n')
    (dest/'README.md').write_text(''.join(blocks)+extra)


def render_gif(name,t0,t1,frame_ms=50,max_frames=400,label=None):
    """Native 20x20 field animation for [t0,t1] s with synchronized rate / Z / K traces."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    row,folder=row_for(name);d=old.load(folder,keys=['spikes_1ms','regions_1ms','field_5ms','slow_time_ms','Z']);k=old.load(folder,'intrinsic_adaptation_chunks')
    with np.load(OUT/'geometry.npz') as f:g={key:f[key] for key in f.files}
    end=len(d['spikes_1ms'])/1000;t1=min(t1,end);frames_needed=(t1-t0)*1000/frame_ms
    if frames_needed>max_frames:frame_ms=int(np.ceil((t1-t0)*1000/max_frames/5)*5)
    per=frame_ms//5;i0=int(round(t0/.005));nfr=int((t1-t0)*1000/frame_ms)
    fields=np.array([d['field_5ms'][i0+i*per:i0+(i+1)*per].sum(0)/g['cell_e_counts']/(frame_ms*1e-3) for i in range(nfr)])
    ft=t0+(np.arange(nfr)+.5)*frame_ms*1e-3
    n=int(end*100);pop=d['spikes_1ms'][:n*10,0].reshape(n,10).sum(1)/320;regions=d['regions_1ms'][:n*10,:2].reshape(n,10,2).sum(1)/g['region_counts'][:2]/.01;rt=(np.arange(n)+.5)*.01
    plt.rcParams.update({'font.size':12,'axes.spines.right':False,'axes.spines.top':False})
    fig=plt.figure(figsize=(13,5.6));gs=fig.add_gridspec(2,2,width_ratios=[1,1.6],hspace=.45,wspace=.25)
    sp=fig.add_subplot(gs[:,0]);rate_ax=fig.add_subplot(gs[0,1]);z_ax=fig.add_subplot(gs[1,1],sharex=rate_ax)
    im=sp.imshow(fields[0].reshape(20,20),origin='lower',extent=(0,20,0,20),vmin=0,vmax=500,cmap='magma',interpolation='nearest')
    for nm,xy in zip('AB',g['centers_mm']):sp.add_patch(Circle(xy,float(g['core_radius_mm']),fill=False,ec='#55dddd'));sp.text(*xy,nm,c='cyan',ha='center')
    sp.set(xlabel='x (mm)',ylabel='y (mm)');fig.colorbar(im,ax=sp,label=f'E rate in {frame_ms} ms (Hz)',shrink=.7)
    rate_ax.plot(rt,pop,c='black',lw=1,label='All E')
    for i,c in enumerate(['#d24b99','#237ca9']):rate_ax.plot(rt,regions[:,i],c=c,lw=.6,alpha=.8,label='Core '+'AB'[i])
    rate_ax.axhline(200,c='#bc2946',ls=':',lw=.8);rate_ax.set(ylabel='Rate (Hz)',ylim=(0,520));rate_ax.legend(fontsize=9,ncol=3,loc='upper right')
    st=d['slow_time_ms']/1000;z_ax.plot(st,d['Z'][:,0],c='#713399',label='Mean Z');z_ax.set(ylim=(0,1.05),xlabel='Time (s)',ylabel='Mean Z')
    if k:
        k_ax=z_ax.twinx();k_ax.plot(k['time_ms']/1000,k['sahp_mean_conductance_ratio'],c='#b56e22',lw=1);k_ax.set_ylabel('Added $g_K/g_L$',c='#b56e22')
    rate_ax.set(xlim=(t0,t1));cursor=[rate_ax.axvline(t0,c='red',lw=1),z_ax.axvline(t0,c='red',lw=1)]
    title=sp.set_title(f'{ft[0]:.2f} s')
    def update(i):
        im.set_data(fields[i].reshape(20,20));title.set_text(f'{ft[i]:.2f} s')
        for line in cursor:line.set_xdata([ft[i],ft[i]])
        return [im,*cursor,title]
    dest=OUT/'figures';dest.mkdir(exist_ok=True);fname=f"native_{name}{'_'+label if label else ''}.gif"
    FuncAnimation(fig,update,frames=nfr,interval=80,blit=False).save(dest/fname,writer=PillowWriter(fps=12),dpi=80)
    plt.close(fig)
    old.write(dest/(fname.replace('.gif','_metadata.json')),dict(source=str(folder),window_s=[t0,t1],frame_ms=frame_ms,n_frames=nfr,native_counts_no_smoothing=True,human_review='PENDING'))
    return fname


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['fig5','overview','tables','readme','gif']);ap.add_argument('--name');ap.add_argument('--t0',type=float,default=0.);ap.add_argument('--t1',type=float,default=30.);ap.add_argument('--label');ap.add_argument('--no-energy',action='store_true');a=ap.parse_args()
    if a.action=='fig5':print(plot_fig5(a.name,energy=not a.no_energy))
    elif a.action=='overview':plot_overview()
    elif a.action=='tables':write_tables()
    elif a.action=='gif':print(render_gif(a.name,a.t0,a.t1,label=a.label))
    else:figure_readme()
