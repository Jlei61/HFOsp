#!/usr/bin/env python3
"""Evidence figures: exact delayed stability, recruitment and conditional nullclines."""
from analyze_topic4_corrected_bifurcation import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import json

PREV=ROOT/'results/topic4_sef_hfo/corrected_rate_high_activity_screen_v1'
FIG=OUT/'figures'
COL={'low':'#3975a5','burst':'#a72d79','tonic':'#cc7819','fold':'#6b6b6b'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})


def mean(fields,s,pop=0):return np.average(fields[:,pop],axis=1,weights=s.m.count_e if pop==0 else s.m.count_i)


def gather(s):
    rows=[]
    for folder in [PREV/'runs',OUT/'runs',OUT/'crossings']:
        for path in sorted(folder.glob('*.json')):
            d=read(path)
            if d.get('grid',10)!=10 or d.get('dt_ms',.1)!=.1:continue
            a=np.load(path.with_suffix('.npz'));e=a['fields_hz'][:,0];tail=e[len(e)//2:];g=np.average(tail,axis=1,weights=s.m.count_e)
            rec={str(th):{'max_simultaneous':float(np.max(np.average(tail>th,axis=1,weights=s.m.count_e))),
                'late_window_union':float(np.average(tail.max(axis=0)>th,weights=s.m.count_e))} for th in [5,20,50]}
            # Event-level union must not be replaced by the union over the full recording.
            active=g>5;starts=np.flatnonzero(np.diff(np.r_[False,active].astype(int))==1);ends=np.flatnonzero(np.diff(np.r_[active,False].astype(int))==-1)+1
            events=[]
            for left,right in zip(starts,ends):
                if left==0 or right>=len(tail):continue
                chunk=tail[left:right];first=np.argmax(chunk>20,axis=0);mask=np.max(chunk,axis=0)>20
                events.append({'start_late_ms':int(left),'end_late_ms':int(right),'spatial_union20':float(np.average(mask,weights=s.m.count_e)),
                    'max_simultaneous20':float(np.max(np.average(chunk>20,axis=1,weights=s.m.count_e))),
                    'first_crossing_span_ms':float(np.ptp(first[mask])) if mask.any() else None})
            rows.append({'name':d['name'],'path':str(path),'q':d.get('q',d.get('qend')),'tau_gaba_ms':d.get('tau_gaba_ms',s.tau),
                'initial':d['initial'],'duration_ms':d['duration_ms'],'mean_e_hz':float(g.mean()),'min_e_hz':float(g.min()),'max_e_hz':float(g.max()),
                'local_max_hz':float(tail.max()),'recruitment':rec,'events':events,'diagnostics':d['diagnostics']})
    write(OUT/'state_and_recruitment.json',{'thresholds_hz':[5,20,50],'statistical_unit':'Deterministic trajectory and complete burst interval; no patient inference or inferential p-values.',
        'event_boundary':'Contiguous global E >5 Hz in late half; clipped intervals excluded; 20 Hz cell crossing measured within this interval.',
        'spatial_weighting':'Cells weighted by actual E neuron counts. Activity is the coarse cell mean, not the fraction of individually spiking neurons.',
        'union_caution':'Late-window union and per-event union are different observables; table retains both.', 'rows':rows})
    return rows


def main_figure(s,rows):
    fig,ax=plt.subplots(2,3,figsize=(15,8.6),layout='constrained')
    hopfs=sorted([read(p) for p in (OUT/'hopf').glob('*dt0.1.json')],key=lambda d:d['tau_gaba_ms'])
    folds=[read(p) for p in (OUT/'folds').glob('*.json')]
    a=ax[0,0]
    a.plot([x['q'] for x in hopfs],[x['tau_gaba_ms'] for x in hopfs],'-o',ms=3,c=COL['burst'],label='Tracked complex-pair crossing')
    for branch,style in [('low','--'),('high',':')]:
        f=sorted([x for x in folds if x['branch']==branch],key=lambda d:d['tau_gaba_ms'])
        a.plot([x['q'] for x in f],[x['tau_gaba_ms'] for x in f],style,c=COL['fold'],label=f'{branch.capitalize()} fixed-point fold')
    # Actual measured points only; no painted phase regions inferred from failed roots.
    seen=set()
    for d in rows:
        if 'qstart' in read(d['path']):continue
        if (d['q'],d['tau_gaba_ms']) in seen:continue
        seen.add((d['q'],d['tau_gaba_ms']));label='Low' if d['max_e_hz']<1 else 'Tonic' if d['max_e_hz']-d['min_e_hz']<1 else 'Variable'
        c={'Low':COL['low'],'Tonic':COL['tonic'],'Variable':COL['burst']}[label]
        a.scatter(d['q'],d['tau_gaba_ms'],c=c,s=15,marker='s',alpha=.65,zorder=1)
    a.set(xlabel='Global GABA jump multiplier q',ylabel='GABA decay (ms)',title='A  Parameter boundaries and sampled states',xlim=(.23,1.27),ylim=(7,44))
    a.legend(fontsize=7,loc='upper right');a.text(.98,.04,'Squares: low / variable / tonic trajectories\nCurves: fixed-point bifurcations, not basin boundaries',transform=a.transAxes,ha='right',fontsize=7)
    a=ax[0,1];h=read(OUT/'hopf/tau20.6116_dt0.1.json');tr=h['critical_pair_track']
    a.plot([x['q'] for x in tr],[x['real_per_s'] for x in tr],'-o',c=COL['burst']);a.axhline(0,color='.5',lw=.8)
    a.axvline(h['q'],color='.6',ls=':',lw=.8);a.set(xlabel='q',ylabel='Re(critical eigenvalue), s$^{-1}$',title='B  A conjugate pair crosses the boundary')
    a.set_ylim(-.65,.65)
    a.text(.98,.98,f"q = {h['q']:.6f}; f = {h['frequency_hz']:.3f} Hz\nUnstable roots: 2  →  0 as q increases\nFull delayed tangent: 72,600 states",transform=a.transAxes,va='top',ha='right',fontsize=8,bbox={'facecolor':'white','edgecolor':'none','alpha':.85})
    a.text(.05,.05,'Native dt: 0.1 ms; fixed-area limit:\nq = 0.737617, f = 3.962 Hz',transform=a.transAxes,fontsize=8)
    a=ax[0,2];fp=read(OUT/'fixed_points.json')['rows']
    for branch in ['high','low']:
        pts=[p for p in fp if p['valid'] and p['branch']==branch];a.plot([p['q'] for p in pts],[p['mean_e_hz'] for p in pts],c='.4',lw=1,label='Fixed points' if branch=='high' else None)
        arc=np.load(OUT/f'arc_{branch}.npz')
        for direction in [-1,1]:
            mask=arc['direction']==direction;a.plot(arc['q'][mask],np.average(arc['r'][mask,:100],axis=1,weights=s.m.count_e)*1000,c='.65',ls=':',lw=.6)
    scan={}
    for d in rows:
        if abs(d['tau_gaba_ms']-s.tau)<1e-5 and d['initial']=='high':
            if d['q'] not in scan or d['duration_ms']>scan[d['q']]['duration_ms']:scan[d['q']]=d
    for d in scan.values():
        a.plot([d['q']]*2,[d['min_e_hz'],d['max_e_hz']],c=COL['burst'],alpha=.7,lw=1)
        a.scatter([d['q']]*2,[d['min_e_hz'],d['max_e_hz']],c=COL['burst'],s=10)
    a.axvline(h['q'],c=COL['burst'],ls='--',lw=.8);a.set(xlabel='q',ylabel='Mean E rate (Hz; symlog)',title='C  Fixed points and late trajectory ranges',ylim=(-.002,550),xlim=(.24,1.01));a.set_yscale('symlog',linthresh=.01);a.legend(fontsize=7)
    a.text(.05,.05,'Magenta endpoints: simulated extrema\nThey are not continued periodic-orbit branches.',transform=a.transAxes,fontsize=7)
    a=ax[1,0]
    for name,label,c in [('q0.76_to_0.76_pulse2','Hold q=0.76 + one local pulse',COL['low']),('q0.76_to_0.72_pulse2','q: 0.76 → 0.72 + same pulse',COL['burst'])]:
        p=OUT/'crossings'/f'{name}.npz'
        if p.exists():
            fields=np.load(p)['fields_hz'];a.plot(np.arange(1,len(fields)+1)/1000,mean(fields,s),lw=.8,label=label,c=c)
    a.axvline(2,color='.5',ls=':',lw=.8);a.set(xlabel='Time (s)',ylabel='Mean E rate (Hz)',title='D  Self-termination versus recurrent bursting');a.legend(fontsize=7,loc='upper right')
    a=ax[1,1];scan=sorted(scan.values(),key=lambda x:x['q'])
    for th,ls in [('5',':'),('20','-'),('50','--')]:
        burst=[d for d in scan if d['max_e_hz']-d['min_e_hz']>1]
        a.plot([d['q'] for d in burst],[d['recruitment'][th]['max_simultaneous']*100 for d in burst],ls+'o',ms=3,label=f'Cell rate >{th} Hz',c=COL['burst'],alpha=1 if th=='20' else .5)
        stationary=[d for d in scan if d not in burst]
        a.scatter([d['q'] for d in stationary],[d['recruitment'][th]['max_simultaneous']*100 for d in stationary],s=14,color=COL['fold'],alpha=.5)
    a.set(xlabel='q',ylabel='Maximum simultaneous recruitment (%)',title='E  Spatial recruitment is a separate observable',ylim=(-3,105),xlim=(.24,1.01));a.axvline(h['q'],c='.6',ls=':',lw=.8);a.legend(fontsize=7,loc='lower left')
    a.text(.97,.95,'E-cell means, weighted by E counts',transform=a.transAxes,ha='right',va='top',fontsize=7)
    a=ax[1,2];p=OUT/'crossings/q0.76_to_0.72_pulse0.npz'
    if p.exists():
        fields=np.load(p)['fields_hz'];tail=fields[4000:,0];g=np.average(tail,axis=1,weights=s.m.count_e);peak=int(np.argmax(g));e=tail[peak]
        im=a.imshow(e.reshape(10,10),origin='lower',extent=[0,20,0,20],cmap='magma',vmin=0,vmax=500,interpolation='nearest');fig.colorbar(im,ax=a,shrink=.8,label='E cell rate (Hz)')
        for x,y,label in [(3.06391,18.71986,'A'),(12.82353,15.09091,'B')]:a.add_patch(plt.Circle((x,y),1.75,fill=False,color='cyan',lw=.8));a.text(x,y,label,color='cyan',ha='center',va='center',fontsize=7)
    a.set(xlabel='x (mm)',ylabel='y (mm)',title='F  Crossing Hopf does not recruit the whole sheet')
    fig.suptitle('Corrected spatial rate: oscillatory instability, state switching and recruitment',fontsize=14)
    fig.savefig(FIG/'bifurcation_and_recruitment.png',dpi=180);fig.savefig(FIG/'bifurcation_and_recruitment.pdf');plt.close(fig)


def nullclines(s):
    fig,ax=plt.subplots(2,2,figsize=(11,8.3),layout='constrained');archive=np.load(OUT/'fixed_points.npz')
    numerical={}
    for j,q in enumerate([.76,.72]):
        r,err,ok=s.solve(q,archive[f'low_{q:.6f}']);assert ok
        E=float(np.average(r[:100],weights=s.m.count_e)*1000);I=float(np.average(r[100:],weights=s.m.count_i)*1000)
        xx=np.linspace(E*.6,E*1.4,81);yy=np.linspace(I*.65,I*1.35,81);X,Y=np.meshgrid(xx,yy);FE=np.empty_like(X);FI=np.empty_like(X)
        for index in np.ndindex(X.shape):
            rr=r*np.r_[np.full(100,X[index]/E),np.full(100,Y[index]/I)];f=s.F(rr,q)/s.tr*1e6
            FE[index]=np.average(f[:100],weights=s.m.count_e);FI[index]=np.average(f[100:],weights=s.m.count_i)
        a=ax[0,j];a.contour(X,Y,FE,levels=[0],colors=[COL['burst']],linewidths=2);a.contour(X,Y,FI,levels=[0],colors=[COL['low']],linewidths=2)
        a.plot([],[],color=COL['burst'],label='Projected dE/dt = 0');a.plot([],[],color=COL['low'],label='Projected dI/dt = 0');a.scatter([E],[I],color='black',s=28,zorder=5)
        a.set(xlabel='Mean E rate (Hz)',ylabel='Mean I rate (Hz)',title=f'{chr(65+j)}  q={q:.2f}: fixed point remains present')
        a.text(.02,.97,'Full delayed model: '+('stable' if j==0 else 'oscillatorily unstable'),transform=a.transAxes,va='top',fontsize=9);a.legend(fontsize=7,loc='lower right')
        numerical[f'q{q}']={'mean_e_hz':E,'mean_i_hz':I,'stationary_residual':err}
        np.savez_compressed(OUT/f'conditional_nullcline_q{q}.npz',X=X,Y=Y,FE=FE,FI=FI,equilibrium=r)
    a=ax[1,0]
    for q,color in [(.76,COL['low']),(.72,COL['burst'])]:
        path=OUT/'crossings'/f'q0.76_to_{q:g}_pulse2.npz'
        if not path.exists():continue
        fields=np.load(path)['fields_hz'];e=mean(fields,s);i=mean(fields,s,1)
        a.plot(e[2000:4000],i[2000:4000],color=color,lw=.9,label=f'q={q:g}, after one pulse')
    a.set(xlabel='Mean E rate (Hz)',ylabel='Mean I rate (Hz)',title='C  Actual full-state trajectories, projected to E–I');a.legend(fontsize=8)
    a=ax[1,1];p=OUT/'crossings/q0.76_to_0.72_pulse0.npz'
    if p.exists():
        fields=np.load(p)['fields_hz'];e=mean(fields,s);i=mean(fields,s,1);a.plot(e[5000:],i[5000:],color=COL['burst'],lw=1)
        for k in [5100,5200,5300]:
            a.annotate('',xy=(e[k+8],i[k+8]),xytext=(e[k],i[k]),arrowprops={'arrowstyle':'->','color':COL['burst'],'lw':1})
    a.set(xlabel='Mean E rate (Hz)',ylabel='Mean I rate (Hz)',title='D  Repeated late loops without repeated stimulation')
    fig.suptitle('Nullclines are conditional slices; stability belongs to the full delayed model',fontsize=13)
    fig.supxlabel('Top: E/I spatial shapes fixed to each equilibrium; filters and delay history set to stationary input values.\nBottom: actual trajectories retain evolving filters and history, so the top vector field does not generate these loops.',fontsize=8)
    fig.savefig(FIG/'conditional_nullclines_and_phase_portraits.png',dpi=180);fig.savefig(FIG/'conditional_nullclines_and_phase_portraits.pdf');plt.close(fig)
    write(OUT/'nullcline_definition.json',{'definition':'For each q, scale the equilibrium E and I spatial rate shapes independently. Slave every synaptic filter and delay bin to stationary expectations of the scaled rates. Project instantaneous rate derivative with E/I cell-count weights.',
        'coordinates':'Mean E and mean I rates in Hz; projected derivatives in Hz per second.',
        'scope':'Intersections include the exact full equilibrium by construction. This conditional stationary-input plane is not an invariant manifold, a fitted autonomous two-variable model, or a proof of delayed stability.', 'equilibria':numerical})


def bistability(s):
    fig,ax=plt.subplots(2,2,figsize=(11,7.7),layout='constrained')
    for pulse,color in [(0,COL['low']),(5,COL['burst'])]:
        fields=np.load(OUT/'crossings'/f'q0.75_to_0.75_pulse{pulse}.npz')['fields_hz'];t=np.arange(1,len(fields)+1)/1000;e=mean(fields,s)
        ax[0,0].plot(t,e,color=color,lw=.8,label='No pulse' if pulse==0 else 'One 20-ms local pulse')
        ax[0,1].plot(t[4000:],e[4000:],color=color,lw=.8)
    ax[0,0].axvline(2,color='.5',ls=':');ax[0,0].set(xlabel='Time (s)',ylabel='Mean E rate (Hz)',title='A  q=0.75 held fixed: finite stimulus switches state');ax[0,0].legend(fontsize=8)
    ax[0,1].set(xlabel='Time (s)',ylabel='Mean E rate (Hz)',title='B  Late activity: rest and bursting both persist')
    sources=[(OUT/'crossings/q0.76_to_0.72_pulse0.npz','C  q=0.72: one late burst'),(PREV/'runs/q0.5_gaba20.6116_grid10_dt0.1_high_6000ms.npz','D  q=0.50: one late burst')]
    for a,(path,title) in zip(ax[1],sources):
        fields=np.load(path)['fields_hz'];e=fields[:,0];g=mean(fields,s);peaks,_=find_peaks(g,prominence=10,distance=150);p=int(peaks[peaks>len(e)//2][-2]);left=p-120;right=p+120;chunk=e[left:right];mask=chunk.max(axis=0)>20
        first=np.argmax(chunk>20,axis=0).astype(float);first[~mask]=np.nan;first-=np.nanmin(first)
        cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#eeeeee');im=a.imshow(first.reshape(10,10),origin='lower',extent=[0,20,0,20],cmap=cmap,vmin=0,vmax=160,interpolation='nearest')
        fig.colorbar(im,ax=a,shrink=.75,label='First 20-Hz crossing relative to earliest (ms)')
        a.set(xlabel='x (mm)',ylabel='y (mm)',title=title)
        a.text(.02,.98,f'Recruited cell weight: {np.average(mask,weights=s.m.count_e)*100:.1f}%',transform=a.transAxes,va='top',fontsize=8,bbox={'facecolor':'white','edgecolor':'none','alpha':.8})
    fig.suptitle('State switching also depends on the basin; wider recruitment has its own readout',fontsize=12)
    fig.supxlabel('Maps: one 240-ms window around a late burst peak; gray cells never cross 20 Hz in that window.\nReference substrate and equations fixed. These are model propagation patterns, not validated patient TA/TB events.',fontsize=8)
    fig.savefig(FIG/'bistability_and_spatial_propagation.png',dpi=180);fig.savefig(FIG/'bistability_and_spatial_propagation.pdf');plt.close(fig)


if __name__=='__main__':
    FIG.mkdir(exist_ok=True);s=System();rows=gather(s);main_figure(s,rows);nullclines(s);bistability(s);print('figures complete',flush=True)
