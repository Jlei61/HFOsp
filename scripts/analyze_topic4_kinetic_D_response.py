"""Finite-window response diagram; deliberately no equilibrium/cycle labels."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from topic4_kinetic_D_response import ROOT, OUT, SOURCE, SEED, DS

GEO=dict(np.load(SOURCE/'approx/coarse_20/geometry.npz'))
COUNTS=GEO['count_e']; RC=np.bincount(GEO['g175'],minlength=3)


def safe(x):
    if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [safe(v) for v in x]
    if isinstance(x,np.ndarray):return safe(x.tolist())
    if isinstance(x,np.generic):return safe(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x


def write(p,x):p.write_text(json.dumps(safe(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def stretches(a):
    d=np.diff(np.r_[False,a,False].astype(int))
    return list(zip(np.flatnonzero(d==1),np.flatnonzero(d==-1)))


def summarize(data,start_s,end_s):
    lo,hi=round(start_s*100),round(end_s*100)
    field10=data['field_1ms'].reshape(-1,10,400).sum(1)/COUNTS/.01
    rate=np.average(field10,weights=COUNTS,axis=1)[lo:hi]
    region=data['regions_1ms'].reshape(-1,10,3).sum(1)/RC/.01
    occupied=np.average(field10[lo:hi]>=50,weights=COUNTS,axis=1)
    quiet=[(a,b) for a,b in stretches(rate<5) if b-a>=2]
    events=[(a,b) for (_,a),(b,_) in zip(quiet[:-1],quiet[1:]) if b-a>=2 and rate[a:b].max()>=20]
    broad=bool(np.mean(rate>=200)>=.9 and np.mean(occupied>=.8)>=.9)
    if broad:label='broad_high_activity'
    elif events:label='self_limited_events'
    elif not quiet and np.mean(rate>=5)>=.95:label='persistent_intermediate_activity'
    elif np.mean(rate<5)>=.95:label='low_activity'
    else:label='unresolved'
    duration=[(b-a)*10 for a,b in events]
    return dict(window_s=[start_s,end_s],mean_E_hz=rate.mean(),q10_E_hz=np.quantile(rate,.1),
        q90_E_hz=np.quantile(rate,.9),min_E_hz=rate.min(),max_E_hz=rate.max(),
        quiet_fraction=np.mean(rate<5),quiet_intervals=len(quiet),finite_events=len(events),
        duration_median_ms=np.median(duration) if duration else None,
        high_rate_fraction=np.mean(rate>=200),mean_occupied_E_fraction=occupied.mean(),
        broad_occupied_fraction=np.mean(occupied>=.8),mean_region_hz=region[lo:hi].mean(0),
        core_persistent_fraction=np.mean(region[lo:hi,:2]>=50,axis=0),
        core_quiet_fraction=np.mean(region[lo:hi,:2]<5,axis=0),
        mean_M_E=data['slow_10ms'][lo:hi,2].mean(),end_M_E=data['slow_10ms'][hi-1,2],
        delta_M_E=data['slow_10ms'][hi-1,2]-data['slow_10ms'][lo,2],category=label)


def load():
    rows=[];all_data={}
    for folder in sorted((OUT/'runs').glob('D*')):
        s=json.loads((folder/'status.json').read_text())
        paths=sorted(folder.glob('observations_*.npz'))
        # Completed 4-s observations remain usable while a separately saved
        # extension runs. Never read a partially written observation file.
        if s['status']!='COMPLETE':
            if not (folder/'checkpoint.npz').exists():continue
            paths=[p for p in paths if p.name=='observations_00000_04000.npz']
        if not paths:continue
        chunks=[dict(np.load(p)) for p in paths]
        data={k:np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
        end=len(data['field_1ms'])*.001
        windows={f'{a:g}-{b:g}s':summarize(data,a,b) for a,b in [(0,2),(2,4),(end-2,end)] if b<=end}
        cfg=s['config'];row=dict(name=folder.name,**cfg,duration_s=end,windows=windows,
            terminal=summarize(data,end-2,end),first_terminal=summarize(data,2,4))
        row['terminal_half_means']=[summarize(data,end-2,end-1)['mean_E_hz'],summarize(data,end-1,end)['mean_E_hz']]
        row['category_changed_after_extension']=row['terminal']['category']!=row['first_terminal']['category']
        row['candidate_stationary']=bool(abs(np.diff(row['terminal_half_means'])[0])<max(5,.15*row['terminal']['mean_E_hz']))
        rows.append(row);all_data[folder.name]=data
    return rows,all_data


def transitions(rows):
    result=[]
    for history in (8000,10370):
        group=sorted([r for r in rows if r['history_ms']==history],key=lambda r:r['D'])
        for left,right in zip(group[:-1],group[1:]):
            a,b=left['first_terminal']['category'],right['first_terminal']['category']
            if a!=b:
                result.append(dict(history_ms=history,D_left=left['D'],D_right=right['D'],left=a,right=b,
                    broad_high_onset_bracket=b=='broad_high_activity' and a!='broad_high_activity'))
    return result


def native_anchor_check(rows):
    comparisons=[]
    for history in (8000,10370):
        cand=next((r for r in rows if abs(r['D']-.22884476056519154)<1e-12 and r['history_ms']==history),None)
        if cand is None:continue
        for stream in ('W1','W2'):
            path=SOURCE/f'native/runs/z9420_h{history}_{stream}/readout_arrays.npz'
            with np.load(path) as a:
                rate=a['r10'];field=a['cell_rate10'];region=a['region10']
                assert abs(float(a['t0_s'])-10.37)<1e-12
                assert np.allclose(np.average(field,weights=COUNTS,axis=1),rate,atol=1e-10)
            for lo,hi in ((2,4),(6,8)):
                if cand['duration_s']<hi:continue
                x=rate[lo*100:hi*100]
                quiet=[(a,b) for a,b in stretches(x<5) if b-a>=2]
                events=[(a,b) for (_,a),(b,_) in zip(quiet[:-1],quiet[1:]) if b-a>=2 and x[a:b].max()>=20]
                nat=dict(mean_E_hz=x.mean(),quiet_fraction=np.mean(x<5),quiet_intervals=len(quiet),
                    finite_events=len(events),mean_occupied_E_fraction=np.average(field[lo*100:hi*100]>=50,weights=COUNTS,axis=1).mean(),
                    mean_region_hz=region[lo*100:hi*100,:3].mean(0))
                candidate={k:cand['first_terminal' if lo==2 else 'terminal'][k] for k in nat}
                comparisons.append(dict(history_ms=history,native_input=stream,window_s=[lo,hi],native=nat,candidate=candidate,
                    comparison='paired future input' if stream=='W1' else 'native noise context, candidate remains W1'))
    write(OUT/'native_anchor_comparison.json',dict(
        scope='Same frozen9.42s field and histories. W1 is paired; native W2 adds development noise context. Not an equivalence test.',pairs=comparisons))


def select_maps(rows,data):
    low=sorted([r for r in rows if r['history_ms']==8000],key=lambda r:r['D'])
    wide=[r for r in low if r['first_terminal']['category'] in
          ('broad_high_activity','persistent_intermediate_activity')
          and r['first_terminal']['mean_occupied_E_fraction']>=.5]
    near=lambda d:min(low,key=lambda r:abs(r['D']-d))
    candidates=[low[0],near(.20),near(.22884476056519154),wide[0] if wide else low[-1]]
    picks=[]
    for r in candidates:
        d=data[r['name']];start=2000
        counts=d['field_1ms'][start:4000]
        blocks=counts.reshape(-1,50,400).sum(1)/COUNTS/.05
        rates=np.average(blocks,weights=COUNTS,axis=1)
        # Representative block nearest median (D=0), or empirical 90th percentile.
        q=.5 if r is candidates[0] else .9
        choose=int(np.argmin(abs(rates-np.quantile(rates,q))))
        picks.append(dict(name=r['name'],D=r['D'],history_ms=r['history_ms'],
            category=r['first_terminal']['category'],window_s=[start*.001+choose*.05,start*.001+(choose+1)*.05],
            selection_quantile=q,field_E_hz=blocks[choose],
            state_selection=['D0 baseline','D.20 self-limited reference','exact point-3 field',
               'earliest sampled persistent low-history response with mean occupied fraction at least .5'][len(picks)]))
    return picks


def plot(rows,data):
    dest=OUT/'figures';dest.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':12})
    fig=plt.figure(figsize=(12.7,7.6))
    gs=fig.add_gridspec(2,4,width_ratios=[1.25,1.25,1,1],height_ratios=[1,1],
        left=.085,right=.945,bottom=.12,top=.93,wspace=.65,hspace=.45)
    ax=fig.add_subplot(gs[0,:2]);zx=fig.add_subplot(gs[1,:2])
    colors={8000:'#282828',10370:'#bb6929'};markers={8000:'o',10370:'^'}
    for history in (8000,10370):
        subset=sorted([r for r in rows if r['history_ms']==history],key=lambda r:r['D'])
        x=np.array([r['D'] for r in subset]);y=np.array([r['first_terminal']['mean_E_hz'] for r in subset])
        qlo=np.array([r['first_terminal']['q10_E_hz'] for r in subset]);qhi=np.array([r['first_terminal']['q90_E_hz'] for r in subset])
        for panel in (ax,zx):
            # No solid/dashed stability conventions: these are sampled responses.
            panel.vlines(x,qlo,qhi,color=colors[history],lw=1,alpha=.6)
            panel.scatter(x,y,c=colors[history],marker=markers[history],s=35,zorder=4)
    ax.set(xlim=(-.015,1.015),ylim=(-5,510),xlabel=r'$D=1-\langle Z_E\rangle$',ylabel='Global E rate (Hz / neuron)')
    zx.set(xlim=(.19,.26),ylim=(-5,270),xlabel=r'$D=1-\langle Z_E\rangle$',ylabel='Global E rate (Hz / neuron)')
    ax.set_xticks([0,.2,.4,.6,.8,1]);zx.set_xticks([.20,.22,.24,.26])
    handles=[Line2D([],[],color=colors[h],marker=markers[h],ls='',label=lab) for h,lab in
        [(8000,'Mean, 2–4 s · low history'),(10370,'Mean, 2–4 s · high history')]]
    handles.append(Line2D([],[],color='#666666',marker='|',markersize=14,ls='',label='10–90% over time'))
    ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(-.015,1.3),ncol=2,frameon=False,fontsize=9.5)
    ax.text(-.16,1.04,'A',transform=ax.transAxes,weight='bold',fontsize=18)
    zx.text(-.16,1.04,'B',transform=zx.transAxes,weight='bold',fontsize=18)
    picks=select_maps(rows,data);maps=[]
    for j,p in enumerate(picks):
        panel=fig.add_subplot(gs[j//2,2+j%2]);maps.append(panel)
        im=panel.imshow(p['field_E_hz'].reshape(20,20),origin='lower',extent=[0,20,0,20],
            cmap='magma',norm=PowerNorm(.6,0,500),interpolation='nearest')
        for k,xy in enumerate(GEO['centers_mm']):
            panel.add_patch(Circle(xy,1.5,fc='none',ec='#21d4d0',lw=1.1))
            panel.text(xy[0],xy[1]+2.1,'AB'[k],ha='center',fontsize=8,color='#12666a',bbox=dict(fc='white',ec='none',pad=.2))
        panel.set(xticks=[0,10,20],yticks=[0,10,20],xlabel='x (mm)',ylabel='y (mm)' if j%2==0 else '')
        panel.text(0,1.06,f'{j+1}   D = {p["D"]:.3f}',transform=panel.transAxes)
        if j==0:panel.text(-.31,1.24,'C',transform=panel.transAxes,weight='bold',fontsize=18)
        if j%2:panel.set_yticklabels([])
        target=next(r for r in rows if r['name']==p['name'])
        for aa in (ax,zx):
            if aa.get_xlim()[0]<=p['D']<=aa.get_xlim()[1]:
                aa.annotate(str(j+1),(p['D'],target['first_terminal']['mean_E_hz']),xytext=(5,8),textcoords='offset points',fontsize=10)
    fig.canvas.draw();box=maps[-1].get_position();top=maps[0].get_position().y1
    cax=fig.add_axes([.962,box.y0,.011,top-box.y0])
    fig.colorbar(im,cax=cax,ticks=[0,250,500],label='E rate (Hz)')
    for ext in ('png','pdf','svg'):fig.savefig(dest/f'fig_kinetic_fixed_D_response.{ext}',dpi=210,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    write(OUT/'spatial_panels.json',picks)
    # Companion makes all temporal results inspectable without crowding main figure.
    ds=sorted(set(r['D'] for r in rows));f,axs=plt.subplots(len(ds),2,figsize=(12,1.4*len(ds)),squeeze=False,sharey=True)
    for i,d in enumerate(ds):
        for j,h in enumerate((8000,10370)):
            r=next((r for r in rows if r['D']==d and r['history_ms']==h),None)
            a=axs[i,j]
            if r is None:a.axis('off');continue
            obs=data[r['name']];rate=obs['field_1ms'].reshape(-1,10,400).sum((1,2))/320
            a.plot((np.arange(len(rate))+.5)*.01,rate,c=colors[h],lw=.65)
            a.axvspan(2,4,color=colors[h],alpha=.08)
            if r['duration_s']>=8:a.axvspan(6,8,color=colors[h],alpha=.08)
            a.set(ylim=(0,510),xlim=(0,r['duration_s']),ylabel=f'D={d:.4g}\nE Hz')
            if i==0:a.set_title('Low history' if h==8000 else 'High history')
            if i==len(ds)-1:a.set_xlabel('Time after clamping Z (s)')
    f.tight_layout()
    for ext in ('png','pdf'):f.savefig(dest/f'fig_all_fixed_D_traces.{ext}',dpi=140,bbox_inches='tight')
    plt.close(f)
    extended=[r for r in rows if r['duration_s']>=8]
    if extended:
        f,axs=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
        for j,h in enumerate((8000,10370)):
            rr=sorted([r for r in extended if r['history_ms']==h],key=lambda r:r['D'])
            for key,label,marker in [('first_terminal','2–4 s','o'),('terminal','6–8 s','D')]:
                axs[j].scatter([r['D'] for r in rr],[r[key]['mean_E_hz'] for r in rr],
                    color=colors[h],marker=marker,facecolors=colors[h] if key=='terminal' else 'none',label=label)
            axs[j].set(xlabel=r'$D=1-\langle Z_E\rangle$',ylabel='Global E time mean (Hz)',
                title='Low history' if h==8000 else 'High history')
            axs[j].legend(frameon=False)
        for ext in ('png','pdf'):f.savefig(dest/f'fig_duration_check.{ext}',dpi=180,bbox_inches='tight')
        plt.close(f)
    return picks


def main(plots=True):
    rows,data=load();tr=transitions(rows)
    native_anchor_check(rows)
    input_qa={}
    if rows:
        with np.load(SOURCE/'approx/input/W1.npz') as f:expected=f['global_rate_per_ms']
        for name,obs in data.items():
            x=obs['global_external_rate']
            input_qa[name]=bool(np.array_equal(x,expected[:len(x)]))
        assert all(input_qa.values()),input_qa
        write(OUT/'input_pairing_qa.json',input_qa)
    disagreements=[]
    for d in sorted(set(r['D'] for r in rows)):
        pair=[r for r in rows if r['D']==d]
        if len(pair)==2 and pair[0]['first_terminal']['category']!=pair[1]['first_terminal']['category']:
            disagreements.append(dict(D=d,categories={str(r['history_ms']):r['first_terminal']['category'] for r in pair}))
    result=dict(model='g40_mean',scope='Finite-window fixed-Z response, single external stream, M dynamic.',
        primary_common_window_s=[2,4],extension_window_s=[6,8],
        classical_bifurcation_type='NOT_ESTABLISHED',runs=rows,transition_intervals=tr,history_disagreements=disagreements)
    write(OUT/'response_summary.json',result)
    flat=[]
    for r in rows:
        flat.append(dict(D=r['D'],history_ms=r['history_ms'],duration_s=r['duration_s'],
            **{k:v for k,v in r['first_terminal'].items() if not isinstance(v,(list,np.ndarray))}))
    if flat:
        with (OUT/'response_summary.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(safe(flat))
    if plots and len(rows)>=2:plot(rows,data)
    print(json.dumps(safe(dict(completed=len(rows),transitions=tr,history_disagreements=disagreements)),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--no-plots',action='store_true');a=p.parse_args();main(not a.no_plots)
