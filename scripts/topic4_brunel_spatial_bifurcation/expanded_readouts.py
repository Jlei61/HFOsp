"""Frozen contact observer and three original contact statistics on exact replays."""
from common import *
import importlib.util
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks,butter,sosfiltfilt,hilbert
from scipy.stats import rankdata
import warnings

DEST=OUT/'expanded'
OLD=ROOT/'results/topic4_sef_hfo/interictal_spatial_surrogate_6101_20260916'
SOURCE=ROOT/'.worktrees/topic4-continuous-core-state-r1/src/topic4_observation_repaired.py'
spec=importlib.util.spec_from_file_location('frozen_contact_observer',SOURCE);observer=importlib.util.module_from_spec(spec);spec.loader.exec_module(observer)

def smooth2(x):
    k=np.exp(-np.arange(-8,9)**2/(2*2.5**2));k/=k.sum()
    return np.stack([np.convolve(y,k,mode='same') for y in x.T],axis=1)

def describe(t,names):
    t=np.asarray(t,float).reshape(-1,15);valid=np.isfinite(t);r=np.full_like(t,np.nan)
    for i,row in enumerate(t):
        assert valid[i].sum()>=2
        r[i,valid[i]]=(rankdata(row[valid[i]],method='average')-1)/(valid[i].sum()-1)
    order=np.full((15,15),np.nan);support=np.zeros((15,15),int)
    for i in range(15):
        for j in range(15):
            if i==j or names[i][:3]!=names[j][:3]:continue
            d=t[:,j]-t[:,i];d=d[np.isfinite(d)];support[i,j]=len(d)
            if len(d):order[i,j]=np.mean((d>0)+.5*(d==0))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        mr=np.nanmean(r,axis=0) if len(t) else np.full(15,np.nan)
    return dict(N=len(t),mean_rank=mr,participation=valid.mean(0) if len(t) else np.full(15,np.nan),
                within_shaft_order_probability=order,within_shaft_pair_support=support)

def compare(a,b,names):
    if not a['N'] or not b['N']:return None
    values=[]
    for key in ['mean_rank','within_shaft_order_probability','participation']:
        d=abs(np.array(a[key],float)-np.array(b[key],float));parts=[]
        for shaft in ['SCL','ICL']:
            ix=np.array([i for i,n in enumerate(names) if n.startswith(shaft)])
            z=d[np.ix_(ix,ix)] if d.ndim==2 else d[ix];z=z[np.isfinite(z)]
            if len(z):parts.append(float(z.mean()))
        values.append(float(np.mean(parts)) if len(parts)==2 else None)
    return dict(zip(['mean_rank_difference','within_shaft_order_difference','participation_difference'],values))

def main():
    contracts={kind:read(OLD/f'observer_{kind}.json') for kind in ['firing','current_hfo']}
    names=contracts['firing']['contact_names'];records=[]
    folders=sorted((DEST/'native').glob('J*'))+sorted((DEST/'native_high').glob('J*'))
    for folder in folders:
        if not (folder/'exact_readout.json').exists():continue
        tag=folder.name+('_high' if folder.parent.name=='native_high' else '')
        dest=DEST/'readouts'/tag;dest.mkdir(parents=True,exist_ok=True)
        meta=read(folder/'contract.json');J=meta['J_EE_core'];z=np.load(folder/'trajectory.npz');x=np.load(folder/'exact_readout.npz')
        assert x['contact_names'].tolist()==names
        # Frozen observer uses weighted spike counts per two-ms frame, not Hz.
        firing=smooth2(x['contact_rate_hz'].reshape(-1,2,15).sum(1)/1000)
        band=sosfiltfilt(butter(4,[80,250],fs=2000,output='sos',btype='bandpass'),x['lfp_raw'],axis=0)
        current=smooth2(abs(hilbert(band,axis=0)).reshape(-1,4,15).mean(1))
        rec={}
        for key,env in [('firing',firing),('current_hfo',current)]:
            ob=observer.observe(env.T,2.,contracts[key]);mu=np.asarray(ob['centroid_ms'],float).reshape(-1,15)
            ids=np.asarray(ob['primary_event_indices'],int)
            ids=ids[[ob['events'][i]['window_ms'][0]>=500 and ob['events'][i]['window_ms'][1]<=10000 for i in ids]]
            rec[key]=dict(summary=describe(mu[ids],names),all_detected_summary=describe(mu,names),
                observation=ob,centroids_ms=mu,primary_event_indices=ids)
        rates=gaussian_filter1d(z['regional_rates_hz'],5,axis=0);dyn=[]
        for k in range(2):
            y=rates[500:,k];osc,_=find_peaks(y,height=10,prominence=10,distance=60);osc=osc+500
            # A burst must return to the same low-activity threshold. Peaks on
            # a tonic background are oscillatory fluctuations, not burst IEIs.
            intervals=observer.runs(y>=5);merged=[]
            for a,b in intervals:
                if merged and a-merged[-1][1]<10:merged[-1][1]=b
                else:merged.append([a,b])
            bursts=[(a,b) for a,b in merged if a>0 and b<len(y) and b-a>=4 and y[a:b].max()>=10]
            p=np.array([500+a+int(np.argmax(y[a:b])) for a,b in bursts],int);iei=np.diff(p)
            dyn.append(dict(mean_rate_hz=float(z['regional_rates_hz'][500:,k].mean()),
                quiet_fraction=float(np.mean(y<5)),high_rate_fraction=float(np.mean(y>200)),
                peak_times_ms=p+1,peak_count=len(p),IEI_ms=iei,
                burst_windows_ms=[[a+500,b+500] for a,b in bursts],oscillation_peak_count=len(osc),
                event_definition='Self-limited excursions above 5 Hz, peak >=10 Hz, >=4 ms; gaps <10 ms joined; both boundaries observed',
                IEI_CV=float(np.std(iei,ddof=1)/np.mean(iei)) if len(iei)>=2 else None,
                median_IEI_ms=float(np.median(iei)) if len(iei) else None))
        rec.update(J_EE_core=J,contact_names=names,dynamics=dyn,trajectory=str(folder/'trajectory.npz'),exact_readout=str(folder/'exact_readout.npz'),initial_state=meta['initial_state'],tag=tag)
        np.savez_compressed(dest/'envelopes.npz',firing=firing,current_hfo=current)
        write(dest/'result.json',rec)
        row=dict(J_EE_core=J,source=str(dest/'result.json'),dynamics=dyn,initial_state=meta['initial_state'],tag=tag,
            firing=rec['firing']['summary'],current_hfo=rec['current_hfo']['summary'],
            firing_detected=rec['firing']['observation']['n_groups'],current_detected=rec['current_hfo']['observation']['n_groups'])
        records.append(row);print(J,[(a['peak_count'],a['IEI_CV'],a['quiet_fraction']) for a in dyn],rec['firing']['summary']['N'],rec['current_hfo']['summary']['N'],flush=True)
    ref=next((r for r in records if r['J_EE_core']==.88),None)
    if ref:
        for r in records:
            r['difference_from_J0p88']={k:compare(r[k],ref[k],names) for k in ['firing','current_hfo']}
    write(DEST/'readouts/result.json',dict(rows=records,contact_names=names,
        observer_sources={k:str(OLD/f'observer_{k}.json') for k in contracts},
        statistical_unit='Each fixed topology/seed trajectory; events describe within-trajectory variation',
        comparison='Three shaft-balanced absolute differences from J=0.88, not patient fitting scores or model-equivalence errors',
        sampling='1 ms exact firing rate converted to weighted spikes/2 ms; actual 0.5 ms current proxy filtered 80-250 Hz',
        selection='Original fixed thresholds and isolated-window eligibility; lack of eligible events is not zero error or zero activity'))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,default=DEST);args=parser.parse_args()
    DEST=args.base;main()
