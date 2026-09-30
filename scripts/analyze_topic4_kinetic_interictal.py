"""Read-only upstream audit of early interictal dynamics in topology 6101.

The g40 mean candidate is a single autonomous trajectory from zero, not the
8 s transplanted runs and not topology 2511. No simulation or parameter fitting.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from audit_topic4_fig5_native_reduction_correspondence import finite_events, runs
from analyze_topic4_spatial_kinetic_candidate import event_comparison

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results/topic4_sef_hfo'
SOURCE = BASE/'fig5_current_network_z_state_v1'
CANDIDATE = BASE/'fig5_spatial_kinetic_full_trajectory_20260916'
OUT = BASE/'fig5_kinetic_interictal_audit_20260916'
WINDOWS = [(0.5, 8.), (0.5, 3.), (3., 5.5), (5.5, 8.), (8., 9.42)]
NAMES = ['Native 9108401', 'Native 9108402', 'Candidate 9108401']
COLORS = ['#222222', '#999999', '#0072B2']


def clean(x):
    if isinstance(x, dict): return {k:clean(v) for k,v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if isinstance(x, np.generic): return clean(x.item())
    if isinstance(x, float) and not np.isfinite(x): return None
    return x


def write(name, x):
    (OUT/name).write_text(json.dumps(clean(x), ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def load():
    geo = dict(np.load(SOURCE/'replay/geometry.npz'))
    cg = dict(np.load(SOURCE/'approx/coarse_20/geometry.npz'))
    counts, regions = cg['count_e'], np.bincount(cg['g175'], minlength=3)
    models = []
    for seed in [9108401, 9108402]:
        pieces = []
        end = 0
        for p in sorted((SOURCE/f'replay/runs/eta0.0005_s{seed}/chunks').glob('*.npz')):
            with np.load(p) as z:
                assert int(z['start_step']) == end
                end = int(z['end_step'])
                pieces.append({k:z[k] for k in ['spikes_1ms','regions_1ms','field_1ms','raster','Z','M','slow_time_ms']})
        assert end == 125000
        models.append({k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]})
    with np.load(CANDIDATE/'run/trajectory.npz') as z:
        assert int(z['start_ms']) == 0
        assert np.array_equal(z['sample_ids'], geo['sample_ids'])
        assert np.array_equal(z['region_counts'], regions)
        models.append({k:z[k] for k in models[0]})
    for z in models:
        assert z['spikes_1ms'].shape == (12500,2)
        assert np.array_equal(z['field_1ms'].sum(1), z['spikes_1ms'][:,0])
        assert np.array_equal(z['regions_1ms'][:,:3].sum(1), z['spikes_1ms'][:,0])
        z['rate10'] = z['spikes_1ms'][:,0].reshape(-1,10).sum(1)/320
        z['field_rate'] = z['field_1ms']/counts/.001
        z['core_rate'] = z['regions_1ms'][:,:3]/regions/.001
    return models, geo, counts


def trailing5(x):
    cs = np.concatenate([np.zeros((1,x.shape[1])),np.cumsum(x,axis=0)])
    right = np.arange(len(x))+1
    left = np.maximum(0,right-5)
    return (cs[right]-cs[left])/(right-left)[:,None]


def descriptors(z, lo, hi, counts):
    r = z['rate10'][round(lo*100):round(hi*100)]
    _, events = finite_events(r,start=lo)
    field, core = trailing5(z['field_rate']), trailing5(z['core_rate'])
    rows, maps = [], []
    for e in events:
        a,b = round(e['start_s']*1000),round(e['end_s']*1000)
        censored = np.any(field[max(0,a-10):a]>=50,axis=0)
        hit = field[a:b]>=50
        cumulative = np.vstack([np.zeros((1,400),int),np.cumsum(hit,axis=0)])
        sustained = cumulative[5:]-cumulative[:-5] >= 5
        valid = np.any(sustained,axis=0)&~censored
        arrival = np.full(400,np.nan)
        arrival[valid] = np.argmax(sustained[:,valid],axis=0)
        early = z['core_rate'][a:min(a+30,b),:2].mean(0)
        label = 'A' if early[0]>2*early[1] else 'B' if early[1]>2*early[0] else 'balanced'
        corehits = []
        for k in [0,1]:
            # Participation is sustained activity, not an assertion of initiation.
            corehits.append(any(j-i>=5 for i,j in runs(core[a:b,k]>=50)))
        row = dict(**e,early_core_activity=label,early_core_A_hz=early[0],early_core_B_hz=early[1],
                   core_A_participates=corehits[0],core_B_participates=corehits[1],
                   recruited_E_fraction=counts[valid].sum()/counts.sum(),
                   sustained_active_fraction_including_left_censored=counts[np.any(sustained,axis=0)].sum()/counts.sum(),
                   peak_active_E_fraction=np.max((field[a:b]>=50)@counts/counts.sum()),
                   left_censored_E_fraction=counts[censored].sum()/counts.sum(),
                   arrival_10_90_ms=np.diff(np.quantile(arrival[valid],[.1,.9]))[0] if valid.sum()>1 else None)
        rows.append(row); maps.append(arrival)
    intervals=np.diff([e['start_s'] for e in rows])*1000
    durations=[e['duration_ms'] for e in rows]
    zv=[z['Z'][np.argmin(abs(z['slow_time_ms']-t*1000)),0] for t in (lo,hi)]
    lv=3*np.mean(((intervals[:-1]-intervals[1:])/(intervals[:-1]+intervals[1:]))**2) if len(intervals)>1 else None
    summary=dict(window_s=[lo,hi],n_events=len(rows),n_intervals=len(intervals),
                 mean_global_E_hz=r.mean(),quiet_fraction=np.mean(r<5),
                 duration_median_ms=np.median(durations) if rows else None,
                 duration_range_ms=[min(durations),max(durations)] if rows else None,
                 interval_median_ms=np.median(intervals) if len(intervals) else None,
                 interval_mean_ms=np.mean(intervals) if len(intervals) else None,
                 interval_CV=np.std(intervals)/np.mean(intervals) if len(intervals)>1 else None,
                 adjacent_interval_local_variation=lv,
                 interval_range_ms=[min(intervals),max(intervals)] if len(intervals) else None,
                 early_core_counts={k:sum(e['early_core_activity']==k for e in rows) for k in ['A','B','balanced']},
                 core_participation={k:np.mean([e[f'core_{k}_participates'] for e in rows]) for k in ['A','B']},
                 recruited_fraction_median=np.median([e['recruited_E_fraction'] for e in rows]),
                 sustained_active_fraction_median=np.median([e['sustained_active_fraction_including_left_censored'] for e in rows]),
                 left_censored_fraction_median=np.median([e['left_censored_E_fraction'] for e in rows]),
                 peak_active_E_fraction_median=np.median([e['peak_active_E_fraction'] for e in rows]),
                 propagation_span_median_ms=np.median([e['arrival_10_90_ms'] for e in rows if e['arrival_10_90_ms'] is not None]),
                 Z_E_endpoints=zv,intervals_ms=intervals,events=rows)
    return summary,maps


def save(fig, name):
    for ext in ['png','svg']:
        fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)


def plot(models, stats, geo, arrival):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13,7.4),layout='constrained')
    for z,ss,name,c in zip(models,stats,NAMES,COLORS):
        t=(np.arange(750)+.5)*.01+.5
        axes[0,0].plot(t,z['rate10'][50:800],color=c,lw=1,alpha=.8,label=name)
        take=(z['slow_time_ms']>=500)&(z['slow_time_ms']<=8000)
        axes[1,0].plot(z['slow_time_ms'][take]/1000,z['Z'][take,0],color=c,lw=1.6)
        for j,(key,label) in enumerate([('intervals_ms','Onset interval (ms)'),('events','Event duration (ms)')]):
            values=np.sort(ss[0][key] if key!='events' else [e['duration_ms'] for e in ss[0][key]])
            axes[j,1].step(values,np.arange(1,len(values)+1)/len(values),where='post',color=c,lw=2)
            axes[j,1].set(xlabel=label,ylabel='Fraction of observations',ylim=(0,1.03))
    axes[0,0].set(xlim=(.5,8),ylabel='All E rate (Hz)',xlabel='Time (s)')
    axes[1,0].set(xlim=(.5,8),ylabel='Mean E resource Z',xlabel='Time (s)')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,1),ncol=3)
    save(fig,'01_interictal_timing_and_drift')
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for col,mi in enumerate([0,1,2]):
        vals=np.array([[stats[mi][j]['early_core_counts'][k] for k in ['A','B','balanced']] for j in [1,2,3]])
        floor=np.zeros(3)
        for k,c in enumerate(['#D55E00','#0072B2','#999999']):
            axes[0,col].bar(range(3),vals[:,k],bottom=floor,color=c,label=['A dominant','B dominant','Balanced'][k]);floor+=vals[:,k]
        axes[0,col].set(title=NAMES[mi],xticks=range(3),xticklabels=['0.5–3','3–5.5','5.5–8'],xlabel='Window (s)',ylim=(0,17),ylabel='Complete events')
        z=models[mi];left,right=3000,4500
        rt=z['raster'][left*10:right*10];it,ix=np.where(rt)
        colors=np.array(['#D55E00']*20+['#0072B2']*20+['#555555']*20+['#999999']*20)
        axes[1,col].scatter(left/1000+it*.0001,ix,s=9,marker='|',c=colors[ix],linewidths=.6)
        for y in [19.5,39.5,59.5]:axes[1,col].axhline(y,color='#cccccc',lw=.6)
        axes[1,col].set(xlim=(3,4.5),ylim=(80,-1),yticks=[9.5,29.5,49.5,69.5],yticklabels=['A E','B E','Other E','I'],xlabel='Time (s)')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,1),ncol=3)
    save(fig,'02_core_modes_and_real_rasters')
    fig,axes=plt.subplots(3,3,figsize=(10,10),layout='constrained')
    for row in range(3):
        for col,mode in enumerate(['A','B','balanced']):
            idx=[i for i,e in enumerate(stats[row][0]['events']) if e['early_core_activity']==mode]
            maps=np.array(arrival[row])[idx]
            participation=np.isfinite(maps).mean(0)
            values=np.array([np.median(v[np.isfinite(v)]) if np.any(np.isfinite(v)) else np.nan for v in maps.T])
            values[participation<.5]=np.nan
            ax=axes[row,col]
            im=ax.imshow(np.ma.masked_invalid(values.reshape(20,20)),origin='lower',extent=(0,20,0,20),cmap='viridis',vmin=0,vmax=120)
            for xy in geo['centers_mm']:ax.add_patch(Circle(xy,1.5,fill=False,ec='#e06645',lw=1.2))
            ax.scatter(*geo['contact_xy'].T,s=9,facecolors='none',edgecolors='#333333',lw=.5)
            ax.set(title=f'{mode if mode=="balanced" else mode+" dominant"}, n={len(idx)}',xticks=[0,10,20],yticks=[0,10,20])
            if row==2:ax.set_xlabel('x (mm)')
            if col==0:ax.set_ylabel(NAMES[row]+'\ny (mm)')
    fig.colorbar(im,ax=axes,label='Median arrival from event onset (ms)',fraction=.022,pad=.02)
    save(fig,'03_conditional_2d_arrival_patterns')


def main():
    OUT.mkdir(exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
    models,geo,counts=load()
    stats=[];arrival=[]
    for z in models:
        ss=[]
        for lo,hi in WINDOWS:
            row,maps=descriptors(z,lo,hi,counts);ss.append(row)
            if (lo,hi)==WINDOWS[0]:arrival.append(maps)
        stats.append(ss)
    # Existing full-trajectory statistics are an independent readback check.
    prev=json.loads((CANDIDATE/'figure_metadata.json').read_text())['interictal_statistics']['early_interictal']
    for ix,key in [(0,'native'),(2,'candidate')]:
        assert stats[ix][0]['n_events']==prev[key]['finite_events']
        assert np.isclose(stats[ix][0]['interval_CV'],prev[key]['onset_interval_CV'],rtol=0,atol=1e-12)
    comparisons={}
    for label,i,j in [('native_noise_reference',0,1),('candidate_vs_native_same_input',0,2),('candidate_vs_native_other_input',1,2)]:
        comparisons[label]=event_comparison(stats[i][0]['events'],arrival[i],stats[j][0]['events'],arrival[j])
    write('analysis.json',dict(models={k:v for k,v in zip(NAMES,stats)},spatial_comparisons=comparisons,
          numerical_checks='count conservation; zero-start and sample identity; reproduce previous event count/CV',
          statistical_unit='one autonomous candidate trajectory and two native input trajectories on one topology; events/pairs are descriptive',
          scope='dynamic Z/M, early finite interictal-like epoch, not a stationary attractor or bifurcation',
          seeg_three_observables='NOT_EVALUATED: candidate has no saved matching contact readout; coarse field is not a substitute',
          source_paths=[str(SOURCE),str(CANDIDATE)]))
    flat=[]
    for name,ss in zip(NAMES,stats):
        for s in ss:
            for e in s['events']:flat.append(dict(model=name,window=str(s['window_s']),**e))
    with (OUT/'events.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    np.savez_compressed(OUT/'early_event_arrivals.npz',**{f'arrival_ms_{i}':v for i,v in enumerate(arrival)},counts=counts)
    plot(models,stats,geo,arrival)
    print(json.dumps(clean(dict(early={name:{k:v for k,v in ss[0].items() if k not in ['events','intervals_ms']} for name,ss in zip(NAMES,stats)},
        spatial={k:{a:b for a,b in v.items() if a!='pairs'} for k,v in comparisons.items()})),indent=2))


if __name__=='__main__':main()
