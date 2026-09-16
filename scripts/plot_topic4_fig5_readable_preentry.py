#!/usr/bin/env python3
"""Readable single transition: honest early spatial snapshots and native raster zooms."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import argparse,hashlib,copy
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle,ConnectionPatch,Rectangle
import plot_topic4_fig5_single_transition as old
import analyze_topic4_fig5_preentry_events as audit
f=old.f
OUT=audit.OUT/'readable_fig5'
COL=['#616873','#267ba8','#dd871c','#ba263c']

def choose_states(a,first,end):
    rate=a['spikes_1ms'][:,0].reshape(-1,10).sum(1)/320
    quiet=[(lo,hi) for lo,hi in f.spans(rate<5) if hi-lo>=2 and hi*.01<first]
    rests=[(lo,hi) for lo,hi in quiet if lo*.01>=.7 and hi*.01<=2.]
    rest=min(rests,key=lambda v:abs((v[0]+v[1])*.005-1.175))
    ev=audit.events(rate,end=first)
    counts=np.vstack([np.zeros(6),a['regions_1ms'].cumsum(0)]);candidates=[];observed=[]
    for t in np.arange(2.025,min(6,first-2)-.025,.005):
        lo=round(t*1000)-25;cnt=counts[lo+50]-counts[lo];hz=cnt/a['region_counts']/.05
        fraction=cnt[:2].sum()/max(1,cnt[:3].sum())
        related=[e for e in ev if e['start_s']-.06<=t<=e['end_s']]
        if hz[:2].max()<20 or not related:continue
        record=dict(time_s=float(round(t,3)),core_neighborhood_spike_fraction=float(fraction),
            regional_rates_Hz=hz[:3],both_cores_over20Hz=bool(min(hz[:2])>=20),event=related[0])
        observed.append(record)
        if fraction>=.65:candidates.append(record)
    assert observed,'No finite core activity observed.'
    locality_pass=bool(candidates)
    if not candidates:candidates=[max(observed,key=lambda c:c['core_neighborhood_spike_fraction'])]
    both=[c for c in candidates if c['both_cores_over20Hz']]
    pool=both or candidates
    chosen=min(pool,key=lambda c:(abs(c['time_s']-end/3),-c['core_neighborhood_spike_fraction']))
    # A symmetric 50-ms window centered on the last established quiet interval's end.
    # It includes the initial recruitment of the episode that does not terminate before high onset.
    entry=quiet[-1][1]*.01
    snaps=[dict(number=i+1,time_s=t,label=label,color=COL[i]) for i,(t,label) in enumerate([
        ((rest[0]+rest[1])*.005,'Resting'),(chosen['time_s'],'Interictal'),(entry,'Entry'),(first+.5,'Ictal')])]
    assert np.all(np.diff([s['time_s'] for s in snaps])>0)
    return snaps,dict(state2=chosen,core_neighborhood_radius_mm=1.75,locality_display_criterion_pass=locality_pass,
        state2_selection='Display-only 50ms snapshot within an early finite event; >=65% E spikes inside the two1.75mm core neighborhoods; at least one core >=20Hz. Prefer both cores active if observed, then closest to first third of shown time.',
        state2_scope='An early phase of a subsequently propagating finite event, not evidence that its entire lifetime is confined to cores. Hard Vth cores retain radius1.5mm. If localization criterion is not observed, retain the most core-weighted measured snapshot and explicitly report failed criterion.',
        entry_selection='Center on the end of the last >=20ms global-E quiet (<5Hz) interval before sustained-high onset; no earlier finite event substituted.',
        entry_episode_start_s=entry,high_onset_s=first)

def full_raster(ax,a,snaps,end):
    it,ix=np.where(a['raster']);tt=it*.0001
    # Fixed neurons and row order, with more vertical room allocated to the two cores.
    mapping=np.r_[np.linspace(0,33,20),np.linspace(36,69,20),np.linspace(72,84,20),np.linspace(87,99,20)]
    for lo,hi,col in [(0,20,f.REGCOL[1]),(20,40,f.REGCOL[2]),(40,60,'#225b7f'),(60,80,'#c17730')]:
        take=(ix>=lo)&(ix<hi)
        ax.scatter(tt[take],mapping[ix[take]],s=7,marker='o',c=col,lw=0,rasterized=True)
    for y in [34.5,70.5,85.5]:ax.axhline(y,c='#bbb',lw=.8)
    ax.set(ylim=(-2,101),yticks=[16.5,52.5,78,93],yticklabels=['Core A E','Core B E','Other E','I'],xlim=(0,end))
    ax.set_title('A  Continuous spike raster',loc='left',weight='bold',pad=16)
    ax.tick_params(axis='x',labelbottom=False)
    for s in snaps:
        ax.text(s['time_s'],.99,str(s['number']),transform=ax.get_xaxis_transform(),ha='center',va='top',
            color=s['color'],weight='bold',fontsize=19,bbox=dict(fc='white',ec='none',alpha=.85,pad=.3))
    return tt,ix,mapping

def zooms(fig,spec,a,snaps,tt,ix):
    gs=spec.subgridspec(1,2,wspace=.30);records=[]
    for i,s in enumerate(snaps[1:3]):
        ax=fig.add_subplot(gs[i]);lo=s['time_s']-.05;hi=lo+.30
        for low,high,col in [(0,20,f.REGCOL[1]),(20,40,f.REGCOL[2])]:
            take=(ix>=low)&(ix<high)&(tt>=lo)&(tt<hi)
            ax.scatter(tt[take],ix[take],s=15,marker='|',c=col,linewidth=.9,rasterized=True)
        ax.axhline(19.5,c='#bbb',lw=.7);ax.axvspan(s['time_s']-.025,s['time_s']+.025,color=s['color'],alpha=.12,lw=0)
        ax.axvline(s['time_s'],color=s['color'],ls=':',lw=1.2)
        ax.set(ylim=(-1,40),xlim=(lo,hi),yticks=[9.5,29.5],yticklabels=['Core A','Core B'],xlabel='Time, t (s)')
        ax.set_title(f'{s["number"]}  {s["label"]}',loc='left',fontsize=18,color=s['color'],pad=9)
        ax.set_xticks([round(lo+.05,2),round(lo+.15,2),round(lo+.25,2)])
        records.append(dict(state=s['number'],time_window_s=[lo,hi],sample_ids=a['sample_ids'][:40],same_spikes_and_order=True))
    return records

def make_grid():
    raw=f.read(old.pilot.OUT/'original_fig5/F_measured_grid.json')
    g={**raw,**{k:np.array(raw[k],float) for k in ['mean','prob','count']}}
    measured=[]
    for s in [9108401,9108402]:
        p=audit.OUT/'runs'/f'eta0.0005_s{s}'/'result.json'
        if p.exists():
            r=f.read(p)
            if r['tracker']['entries']:measured.append(r['tracker']['entries'][0]['confirmation_s'])
    if len(measured)==2:
        g['eta_M']=list(g['eta_M']);g['eta_M'].insert(1,.0005)
        for k,value in [('mean',float(np.mean(measured))),('prob',1.),('count',2.)]:
            values=np.full((1,5),np.nan if k=='mean' else 0.)
            values[0,0]=value;g[k]=np.concatenate([g[k][:1],values,g[k][1:]])
        g['additional_half_M_first_entry_times_s']=measured
        g['half_M_followup']='40s horizon, both exact events observed; both latencies known before censoring. Original180s censored cells unchanged.'
    return g

def draw_grid(fig,spec,g,job):
    ax=fig.add_subplot(spec);cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#eeeeee')
    ny,nx=g['mean'].shape
    im=ax.pcolormesh(np.arange(nx+1),np.arange(ny+1),np.ma.masked_invalid(g['mean']),cmap=cmap,vmin=0,vmax=180,edgecolors='#ffffff88',linewidth=.6)
    ax.set(xticks=np.arange(nx)+.5,xticklabels=[f'{v:g}' for v in g['tau_M_s']],
        yticks=np.arange(ny)+.5,yticklabels=[f'{v:g}' for v in g['eta_M']],xlabel=r'$\tau_M$ (s)',ylabel=r'$\eta_M$')
    ax.set_title('E  M kinetics',loc='left',weight='bold',pad=16)
    for y,x in zip(*np.where(g['count']==2)):
        val=float(g['mean'][y,x]);n=int(round(g['prob'][y,x]*2))
        if n<2:ax.add_patch(Rectangle((x,y),1,1,fc='none',ec='#55555588',lw=0,hatch='///'))
        ax.text(x+.5,y+.5,f'{val:.1f}\n{n}/2',ha='center',va='center',fontsize=13,
            color='white' if val<90 else '#111',bbox=dict(fc=cmap(val/180),ec='none',pad=.2))
    if job['eta_m'] in g['eta_M']:
        i=g['eta_M'].index(job['eta_m']);j=list(g['tau_M_s']).index(job['tau_M_s'])
        ax.add_patch(Rectangle((j,i),1,1,fc='none',ec='black',lw=2))
    cb=fig.colorbar(im,ax=ax,pad=.05);cb.set_label('Restricted mean first-entry time (s)');cb.set_ticks([0,45,90,135,180])


def render(row,grid):
    a,r=audit.load_small(row['source'],keys=('spikes_1ms','regions_1ms','field_1ms','raster','slow_time_ms','Z','M','currents','lfp_time_ms','lfp_raw','time_ms'))
    first=r['tracker']['entries'][0];end=round(first['confirmation_s']+2,2)
    if r['tracker']['restore_s'] is not None:end=min(end,round(r['tracker']['restore_s']-.01,2))
    old.layout.trim_display(a,end)
    assert np.array_equal(a['field_1ms'].sum(1),a['spikes_1ms'][:,0])
    snaps,selection=choose_states(a,first['onset_s'],end)
    comparison=old.compare_families(a,first['onset_s'])
    plt.rcParams.update({'font.size':19,'axes.labelsize':21,'axes.titlesize':22,'xtick.labelsize':18,'ytick.labelsize':18})
    fig=plt.figure(figsize=(30,19))
    outer=fig.add_gridspec(1,2,width_ratios=[1.10,1.08],left=.065,right=.954,top=.955,bottom=.10,wspace=.20)
    left=outer[0].subgridspec(6,1,height_ratios=[2.5,1.0,.16,1.35,.34,1.25],hspace=.32)
    right=outer[1].subgridspec(2,1,height_ratios=[1.10,1],hspace=.33)
    upper=right[0].subgridspec(1,2,width_ratios=[1.18,1],wspace=.70)
    ra=fig.add_subplot(left[0]);tt,ix,mapping=full_raster(ra,a,snaps,end)
    ra.set_title(f'ηM = {row["eta_m"]:g} · τM = 1 s',loc='right',fontsize=17,pad=16)
    zm=fig.add_subplot(left[3],sharex=ra);ma=zm.twinx()
    zoomrecords=zooms(fig,left[1],a,snaps,tt,ix)
    zt=a['slow_time_ms']/1000;z=a['Z'];m=a['M']*row['eta_m']
    zm.fill_between(zt,z[:,2],z[:,4],color=f.REGCOL[0],alpha=.13,lw=0)
    for zi,mi,col in [(0,0,f.REGCOL[0]),(5,1,f.REGCOL[1]),(6,2,f.REGCOL[2])]:
        zm.plot(zt,z[:,zi],c=col,lw=1.8);ma.plot(zt,m[:,mi],c=col,ls='--',lw=1.6)
    zm.set(ylabel='Resource Z',ylim=(0,1.05),xlabel='Time, t (s)');ma.set_ylabel('ηM × M (mV equiv.)',labelpad=12)
    assert m.max()<.5,'Shared effective-M display limit would clip data.'
    ma.set(ylim=(0,.5),yticks=[0,.25,.5])
    ma.spines['right'].set_visible(True);zm.set_title('B  Inhibition resource and adaptation',loc='left',weight='bold',pad=16)
    handles=[Line2D([],[],c=c,label=n) for c,n in zip(f.REGCOL,['All E','Core A','Core B'])]
    handles += [Line2D([],[],c='black',label='Z'),Line2D([],[],c='black',ls='--',label='ηM × M')]
    zm.legend(handles=handles,ncol=5,loc='center left',bbox_to_anchor=(.0,.48),fontsize=14,frameon=True,facecolor='white',edgecolor='none',framealpha=.9)
    for ax in [ra,zm]:
        ax.axvspan(first['onset_s'],end,fc='#ba263c',alpha=.10,lw=0)
        for s in snaps:
            ax.axvspan(s['time_s']-.025,s['time_s']+.025,color=s['color'],alpha=.12,lw=0)
            ax.axvline(s['time_s'],ls=':',lw=1.2,c=s['color'],alpha=.9)
    header=fig.add_subplot(left[4]);header.axis('off');header.text(0,.4,'C  Spatial activity · 50 ms',fontsize=22,weight='bold',bbox=dict(fc='white',ec='none',pad=2))
    mapgrid=left[5].subgridspec(1,4,wspace=.25);maps=[];links=[]
    for i,s in enumerate(snaps):
        ax=fig.add_subplot(mapgrid[i]);lo=round(s['time_s']*1000)-25
        field=a['field_1ms'][lo:lo+50].sum(0)/a['cell_e_counts']/.05
        im=ax.imshow(field.reshape(20,20),origin='lower',extent=[0,20,0,20],cmap='magma',norm=PowerNorm(.6,0,500),interpolation='nearest')
        for j,xy in enumerate(a['centers_mm']):
            ax.add_patch(Circle(xy,float(a['core_radius_mm']),ec='#2dd4cd',fc='none',lw=1.8))
            ax.text(xy[0],xy[1]+2.1,'AB'[j],color='#10656a',fontsize=13,ha='center',weight='bold',bbox=dict(fc='white',ec='none',pad=.25,alpha=.85))
        ax.set(xticks=[0,10,20],yticks=[0,10,20],xlabel='x (mm)')
        if i==0:ax.set_ylabel('y (mm)')
        else:ax.set_yticklabels([])
        ax.set_title(f'{s["number"]}  {s["label"]}\n{s["time_s"]:.3f} s',c=s['color'],fontsize=18,pad=12)
        ax.title.set_bbox(dict(fc='white',ec='none',pad=1))
        connector=ConnectionPatch(xyA=(s['time_s'],-.18),coordsA=zm.get_xaxis_transform(),xyB=(.5,1.29),coordsB=ax.transAxes,
            arrowstyle='-',color=s['color'],lw=1.4,alpha=.8,clip_on=False,zorder=-1)
        fig.add_artist(connector)
        maps.append(dict(number=s['number'],time_window_s=[lo/1000,(lo+50)/1000],rate_Hz=field,
            sheet_fraction_above20Hz=float(np.mean(field>=20))))
        links.append((s,ax))
        if i==0:
            ca=ax.inset_axes([0,-.48,1,.08]);cb=fig.colorbar(im,cax=ca,orientation='horizontal',ticks=[0,250,500]);cb.set_label('E rate (Hz)',fontsize=19)
    before=len(fig.axes);trajectory=old.draw_trajectory(fig,upper[0],a,snaps,end);ta=fig.axes[before];ta.set_title('D  State trajectory',loc='left',weight='bold',fontsize=22,pad=20)
    for axis in [ta.xaxis,ta.yaxis,ta.zaxis]:axis.labelpad=22
    draw_grid(fig,upper[1],grid,r['job'])
    before=len(fig.axes);display=old.draw_e2(fig,right[1],a,comparison)
    e2axes=fig.axes[before:]
    elo,ehi=comparison['early_window_s'];e2axes[4].set_title(f'Early power\n{elo:.2f}–{ehi:.2f} s',fontsize=18)
    for ax in fig.axes[before:]:
        for text in ax.texts:
            if text.get_text().startswith('E2  '):text.set_text('F  Interictal order and early high-state power');text.set_fontsize(22)
    # Uniform larger x/y labels and ticks, including all colorbars and 3D axes.
    allaxes=list(fig.axes)
    for parent in list(fig.axes):allaxes.extend(parent.child_axes)
    for ax in allaxes:
        for axis in [ax.xaxis,ax.yaxis]+([ax.zaxis] if hasattr(ax,'zaxis') else []):axis.label.set_fontsize(21)
        ax.tick_params(labelsize=18)
        if ax.title.get_text() and ax.title.get_fontsize()<18:ax.title.set_fontsize(18)
    fig.canvas.draw()
    # Match each colorbar's height to its square native contact projection.
    for cbidx,mapidx in [(3,2),(5,4)]:
        p=e2axes[cbidx].get_position();mapp=e2axes[mapidx].get_position()
        e2axes[cbidx].set_position([p.x0,mapp.y0,p.width,mapp.height])
    e2axes[3].title.set_fontsize(15)
    ta.set_title('',loc='left')
    fig.text(ta.get_position().x0,upper[0].get_position(fig).y1+.012,'D  State trajectory',fontsize=22,weight='bold')
    state_links=[]
    for s,ax in links:
        rx=ra.get_xaxis_transform().transform((s['time_s'],0))[0];zx=zm.get_xaxis_transform().transform((s['time_s'],0))[0]
        assert abs(rx-zx)<1e-6
        state_links.append(dict(number=s['number'],time_s=s['time_s'],raster_x_px=rx,ZM_x_px=zx))
    dest=OUT/row['name'];figdir=dest/'figures';figdir.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(figdir/f'fig5.{ext}',dpi=150,bbox_inches='tight',pad_inches=.2)
    plt.close(fig)
    meta=dict(source=str(row['source']),job=r['job'],display_time_window_s=[0,end],snapshots=snaps,selection=selection,
        native_maps=maps,state_time_links=state_links,raster_zoom_windows=zoomrecords,
        raster=dict(fixed_neurons=80,core_vertical_fraction=.70,row_order_unchanged=True,marker_area_pt2=7,
            sample_ids=a['sample_ids'],raster_sha256=hashlib.sha256(a['raster'].tobytes()).hexdigest(),row_display_coordinates=mapping),
        panel_mapping={'A':'Continuous raster and two same-spike zooms','B':'Z and effective M','C':'Native50ms spatial fields','D':'Actual SNN trajectory','E':'M first-entry timings','F':'Unchanged reference-definition contact rank and early power'},
        fonts=dict(xyz_axis_labels_pt=21,tick_labels_pt=18,panel_title_pt=22),effective_M_axis_shared_limits=[0,.5],E1=trajectory,F_grid=grid,E2=comparison,E2_display=display,
        event_frequency_audit=str(audit.OUT/'event_audit.json'),event_frequency_increase_claim=False,
        scope='Figure selection is illustrative, not a training target or an independent claim of mode confinement. High rate is not proof of oscillatory clinical seizure.',
        no_simulation_state_change_in_plotting=True,manual_reset_in_display=False,agent_visual_review='PENDING',human_review='PENDING',
        producer=str(Path(__file__).resolve()),producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    f.write(dest/'fig5_metadata.json',f.safe(meta))
    (figdir/'README.md').write_text('### fig5.png / .pdf\n子图重新依左列A–C、右列D–F编号；所有坐标轴标签21pt、刻度18pt。Core A/B在连续raster中占据70%高度，并在下方使用同一神经元同一次放电的300ms窗放大。\n状态2取有限事件的早期局限激活相位，不代表整次间期传播只限于核；状态3取高态前最后一次至少20ms静息结束时刻，四个50ms原生场、时序、轨迹编号同步。M系数及真实高态时间逐版本注明，F保留原未滤波电流代理功率语义。\n**关注点**：事件次数/时长/活动占时的独立统计另见event_audit.json；本图不宣称发作前事件频率递增，空间图禁止把完整传播过程改画成局限事件。候选待人工检查。\n')
    print(row['name'],[(s['label'],s['time_s']) for s in snaps],flush=True)
    return dict(name=row['name'],eta_m=row['eta_m'],seed=row['seed'],figure=str(figdir/'fig5.png'),pdf=str(figdir/'fig5.pdf'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--eta',type=float,nargs='+',default=[.001,.0005]);args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);grid=make_grid()
    rows=[render(row,grid) for row in audit.sources() if row['eta_m'] in args.eta and (row['source']/'result.json').exists()]
    f.write(OUT/'delivery_manifest.json',dict(versions=rows,human_review='PENDING'))
    (OUT/'README.md').write_text('# Fig5 可读性与早期招募修订\n\n'+ '\n'.join(f'- {r["name"]}：[PNG]({r["figure"]}) · [PDF]({r["pdf"]})' for r in rows)+'\n\n事件频率没有被作为绘图前提；原生事件审计与减半M配对结果另存同级。\n')
if __name__=='__main__':main()
