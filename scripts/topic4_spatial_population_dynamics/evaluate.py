"""Matched observers and spatial-event summaries; no acceptance by appearance."""
from shared import *
from analyze import dynamics,metric_distance,coverage
from scipy.ndimage import gaussian_filter1d
from scipy.stats import wasserstein_distance
import itertools,csv

def spatial_events(z,ob,ids,positions,sizes):
    xy=np.minimum(positions[:int(sizes[:3].sum())].astype(int),19)
    n=np.bincount(xy[:,1]*20+xy[:,0],minlength=400).reshape(20,20)
    field=gaussian_filter1d(z['field_counts']/np.maximum(n,1)/.002,2.5,axis=0)
    core=gaussian_filter1d(z['six_counts'][:,:2]/sizes[:2]/.002,2.5,axis=0)
    rows=[];maps=[]
    for event in ids:
        start,end=ob['events'][int(event)]['window_ms'];lo=max(0,int(np.ceil(start/2)));hi=min(len(field),int(np.floor(end/2)))
        active=field[lo:hi]>20.;hit=np.any(active,axis=0);arrival=np.where(hit,(np.argmax(active,axis=0)+lo)*2+1,np.nan)
        fraction=float(hit.mean());valid=arrival[np.isfinite(arrival)]
        times=[]
        for a in range(2):
            inds=np.flatnonzero(core[lo:hi,a]>20.);times.append(float((lo+inds[0])*2+1) if len(inds) else None)
        rows.append(dict(event=int(event),area_fraction=fraction,peak_active_fraction=float(active.sum((1,2)).max()/400),
            field_onset_span_ms=float(np.quantile(valid,.95)-np.quantile(valid,.05)) if len(valid)>1 else None,
            core_B_minus_A_ms=times[1]-times[0] if all(t is not None for t in times) else None,
            window_ms=[start,end]))
        if len(valid)>1:
            rank=np.full_like(arrival,np.nan);rank[hit]=(arrival[hit]-np.min(valid))/max(2.,np.max(valid)-np.min(valid));maps.append(rank)
    mean_map=np.full((20,20),np.nan)
    if maps:
        x=np.array(maps);den=np.isfinite(x).sum(0);mean_map=np.divide(np.nansum(x,axis=0),den,out=mean_map,where=den>0)
    return rows,mean_map

def summarize(name,z,key,positions,sizes):
    ob,ids,mu,q=observations(z[key]);space,spatial_map=spatial_events(z,ob,ids,positions,sizes)
    out=dict(name=name,N=len(ids),summary=q,coverage=coverage(q),event_ids=ids,centroid_ms=mu[ids],
        dynamics=dynamics(z['six_counts'],sizes),spatial_events=space,mean_spatial_onset_map=spatial_map)
    write(OUT/'summaries'/f'{name}.json',out);write(OUT/'observations'/f'{name}.json',ob)
    return out

def main():
    model=np.load(OUT/'model.npz');sizes=np.bincount(model['region'],minlength=6);positions=model['positions']
    native={};rows=[]
    for seed in (848101,848102,848103):
        name=f'native_s{seed}';path=OUT/'summaries'/f'{name}.json'
        native[seed]=read(path) if path.exists() else summarize(name,np.load(PRIOR/f'native/{seed}/trajectory.npz'),'contact_envelope',positions,sizes)
    pairs=[dict(a=a,b=b,errors=metric_distance(native[a]['summary'],native[b]['summary'])) for a,b in itertools.combinations(native,2)]
    spread=np.max([p['errors'] for p in pairs],axis=0);models={}
    for folder in sorted((OUT/'runs').iterdir()):
        if not (folder/'result.json').exists():continue
        config=read(folder/'result.json');z=np.load(folder/'trajectory.npz')
        assert np.array_equal(z['nu_core'],np.load(PRIOR/f'native/{config["seed"]}/trajectory.npz')['nu_core'])
        for key,readout in [('contact_envelope','neuron'),('group_contact_envelope','population')]:
            name=folder.name+'_'+readout
            q=summarize(name,z,key,positions,sizes);errors=np.array([metric_distance(q['summary'],native[s]['summary']) for s in native])
            q.update(config=config,readout=readout,errors_to_native=errors,native_pair_max=spread,
                robust_exceedance=np.all(errors>spread[None,:],axis=0),
                status='NO_VALID_EVENTS' if q['N']==0 else ('PROPAGATION_MISMATCH' if np.any(np.all(errors>spread[None,:],axis=0)) else 'NO_CLEAR_THREE_METRIC_FAILURE_NOT_ACCEPTED'))
            write(OUT/'summaries'/f'{name}.json',q);models[name]=q
            own=list(native).index(config['seed']);d=q['dynamics']
            rows.append(dict(run=folder.name,readout=readout,seed=config['seed'],N=q['N'],
                AE_mean_hz=d['AE']['mean_hz'],BE_mean_hz=d['BE']['mean_hz'],AE_low_fraction=d['AE']['low_rate_fraction'],BE_low_fraction=d['BE']['low_rate_fraction'],
                rank_error=errors[own,0],order_error=errors[own,1],participation_error=errors[own,2],status=q['status']))
    for name,data in [('comparison.csv',rows)]:
        if data:
            with (OUT/name).open('w') as f:
                w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(safe(data))
    write(OUT/'comparison.json',dict(native_pairs=pairs,native_pair_max=spread,rows=rows,
        status='EXPLORATORY_NOT_ACCEPTED',bifurcation_allowed=False,
        spatial_definition='1 mm field, 5 ms Gaussian smoothing, 20 Hz/cell first crossing in each unchanged observer event window; full sheet and both cores',
        statistical_unit='same topology, three native noise seeds; events nested within runs'))
    print(json.dumps(safe(rows),indent=2),flush=True)

if __name__=='__main__':main()
