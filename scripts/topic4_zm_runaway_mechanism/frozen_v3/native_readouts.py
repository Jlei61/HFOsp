"""Native reference readouts (milestone definitions) and 2-D propagation metrics for stage A4.

All-E rate per 1 ms (cell counts weighted by cell size), 10-ms smoothing, quiet <5 Hz for >=20 ms,
events >=20 ms with peak >=20 Hz, high >=200 Hz for 200 ms, expansion = neuron-weighted fraction
of 20x20 cells above 50 Hz. Propagation: per event, cell onset time = first 1-ms bin where the
cell's 5-ms smoothed rate exceeds half its event peak (cells that reach >=20 Hz); direction =
projection of (late centroid - early centroid) on the core A->B axis.
"""
from common_v3 import *
from scipy.ndimage import uniform_filter1d
import glob
geo=dict(np.load(OPERATORS/'g20/geometry.npz'));centers=geo['centers_mm']
GRID=20
def cell_xy():
    i=np.arange(GRID*GRID);return np.c_[i%GRID+.5,i//GRID+.5]   # mm, cell index = x + 20*y (checked against positions)
def cell_index_order():
    # operator cell index convention: verify using group positions
    cell=geo['group_cell'];pos=geo['positions'];xy=cell_xy()
    err=np.abs(pos-xy[cell]).max();assert err<.6,err
    return True
cell_index_order()
AXIS=(centers[1]-centers[0]);AXIS=AXIS/np.linalg.norm(AXIS)
core_cells=[np.flatnonzero(np.linalg.norm(cell_xy()-c,axis=1)<=1.5) for c in centers]

def load_native(run=NATIVE):
    files=sorted(glob.glob(str(run/'chunks/*.npz')));field=[];t=[]
    for f in files:
        z=np.load(f);field.append(z['field_1ms'].astype(np.float32));t.append(z['time_ms'])
    field=np.concatenate(field);t=np.concatenate(t)
    counts=np.load(run.parent.parent/'geometry.npz')['cell_e_counts']
    rate=field/np.maximum(counts,1)*1000   # Hz per neuron per cell per 1 ms
    return t,rate,counts

def readouts(t,rate,counts,label):
    """rate: (T,400) Hz per cell per 1-ms bin. Returns event table and milestone summary."""
    w=counts/counts.sum();allE=rate@w;sm=uniform_filter1d(allE,10,mode='nearest')
    quiet=sm<5;edges=np.diff(np.r_[0,quiet.astype(int),0]);qs=np.flatnonzero(edges==1);qe=np.flatnonzero(edges==-1)
    quiet_runs=[(a,b) for a,b in zip(qs,qe) if b-a>=20]
    active=~quiet;edges=np.diff(np.r_[0,active.astype(int),0]);es=np.flatnonzero(edges==1);ee=np.flatnonzero(edges==-1)
    events=[]
    for a,b in zip(es,ee):
        if b-a<20 or sm[a:b].max()<20:continue
        seg=rate[a:b];peak_cell=uniform_filter1d(seg,5,axis=0,mode='nearest')
        part=peak_cell.max(0)>=20;onset=np.full(GRID*GRID,np.nan)
        for c in np.flatnonzero(part):
            onset[c]=a+np.argmax(peak_cell[:,c]>=.5*peak_cell[:,c].max())
        ok=np.isfinite(onset);xy=cell_xy()
        if ok.sum()>=4:
            order=np.argsort(onset[ok]);cells=np.flatnonzero(ok)[order];k=max(1,len(cells)//4)
            early=xy[cells[:k]].mean(0);late=xy[cells[-k:]].mean(0);vec=late-early
            direction=float(vec@AXIS);extent=float(np.linalg.norm(vec))
        else:direction=np.nan;extent=np.nan
        coreA=peak_cell[:,core_cells[0]].max()>=20;coreB=peak_cell[:,core_cells[1]].max()>=20
        surround=part.copy();surround[core_cells[0]]=False;surround[core_cells[1]]=False
        events.append(dict(start_ms=float(t[a]),duration_ms=float(b-a),peak_hz=float(sm[a:b].max()),n_cells=int(part.sum()),
            area_fraction=float(w[part].sum()),coreA=bool(coreA),coreB=bool(coreB),surround_cells=int(surround.sum()),
            direction_axis_mm=direction,extent_mm=extent,onset=onset))
    high=sm>=200;edges=np.diff(np.r_[0,high.astype(int),0]);hs=np.flatnonzero(edges==1);he=np.flatnonzero(edges==-1)
    high_onset=next((float(t[a]) for a,b in zip(hs,he) if b-a>=200),None)
    expansion=(rate>50)@w
    summary=dict(label=label,duration_ms=float(t[-1]-t[0]+1),n_events=len(events),quiet_fraction=float(quiet.mean()),
        high_onset_ms=high_onset,mean_allE_hz=float(allE.mean()),
        expansion_first_ms={q:(float(t[np.argmax(expansion>=q)]) if (expansion>=q).any() else None) for q in [.25,.5,.75]})
    return events,summary,allE,sm

def window_stats(events,t0,t1):
    ev=[e for e in events if t0<=e['start_ms']<t1]
    if not ev:return dict(n=0)
    d=np.array([e['duration_ms'] for e in ev]);gap=np.diff([e['start_ms'] for e in ev]) if len(ev)>1 else np.array([np.nan])
    dirs=np.array([e['direction_axis_mm'] for e in ev]);dirs=dirs[np.isfinite(dirs)]
    return dict(n=len(ev),median_duration_ms=float(np.median(d)),iqr_duration_ms=[float(x) for x in np.percentile(d,[25,75])],
        median_gap_ms=float(np.nanmedian(gap)),both_cores=int(sum(e['coreA'] and e['coreB'] for e in ev)),
        coreA_only=int(sum(e['coreA'] and not e['coreB'] for e in ev)),coreB_only=int(sum(e['coreB'] and not e['coreA'] for e in ev)),
        median_area=float(np.median([e['area_fraction'] for e in ev])),median_surround_cells=float(np.median([e['surround_cells'] for e in ev])),
        forward=int((dirs>1).sum()),reverse=int((dirs<-1).sum()),undirected=int((abs(dirs)<=1).sum()),median_extent_mm=float(np.median([e['extent_mm'] for e in ev if np.isfinite(e['extent_mm'])] or [np.nan])))

if __name__=='__main__':
    out={}
    for label,run in [('seed9108401',NATIVE),('seed9108402',NATIVE2)]:
        t,rate,counts=load_native(run);events,summary,allE,sm=readouts(t,rate,counts,label)
        summary['windows']={f'{a}-{b}':window_stats(events,a,b) for a,b in [(1000,4000),(4000,8000),(8000,9420),(1000,9420)]}
        out[label]=summary;np.savez_compressed(DEST/f'native_reference/{label}_readouts.npz',t=t,allE=allE,smoothed=sm,rate_cells=rate.astype(np.float32),
            onsets=np.array([e['onset'] for e in events]),event_start=[e['start_ms'] for e in events])
        print(label,json.dumps(clean({k:v for k,v in summary.items()}),indent=1))
    write(DEST/'native_reference/summary.json',out)
