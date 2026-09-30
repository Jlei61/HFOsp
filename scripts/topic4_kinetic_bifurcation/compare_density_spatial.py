"""Same-observable spatial comparisons at the two transition-side D values."""
from analyze_qualification import *


def extract(folder,D,left_ms=1000,right_ms=4000):
    cfg=json.load(open(folder/'config.json'));initial=cfg.get('initial_ms',0.)
    stats=summarize(folder,left_ms,right_ms)
    with np.load(folder/'trajectory.npz') as z:
        rates=z['rate_1ms'];raw=z['field_1ms'];counts=z['count_e']
    count=counts.reshape(20,2,20,2).sum((1,3)).ravel()
    field=(raw*counts).reshape(-1,20,2,20,2).sum((2,4)).reshape(-1,400)/np.maximum(count,1)
    left=int(left_ms-initial);right=int(right_ms-initial)
    if D==.25:
        image=field[left:right].mean(0);maps=image[None,:];lags=[]
        definition=f'Mean spatial rate in the common {left_ms:g}--{right_ms:g} ms window'
    else:
        r10=rates.reshape(-1,10,4).mean(1);maps=[];lags=[]
        for event in stats['events']:
            a=int((event['start_ms']-initial)/10);b=int((event['end_ms']-initial)/10)
            peak=10*(a+np.argmax(r10[a:b,0]))+5
            maps.append(field[peak-25:peak+25].mean(0))
            # Include the preceding globally quiet interval: a core can start
            # before the global rate reaches 5 Hz. Starting the search at the
            # global crossing would censor such leads to an artificial zero.
            lo=a
            while lo>0 and r10[lo-1,0]<5.:lo-=1
            cross=[]
            for k in (1,2):
                active=[(s,t) for s,t in stretches(r10[lo:b,k]>=20.) if t-s>=2]
                if not active or active[0][0]<2:cross.append(np.nan)
                else:cross.append(float((lo+active[0][0])*10))
            lags.append(cross[1]-cross[0])
        maps=np.asarray(maps);image=np.median(maps,axis=0)
        definition='Median across complete events of the 50-ms spatial rate centered on the global 10-ms rate peak; core lag uses a sustained 20-Hz crossing from the preceding global quiet interval and excludes left-censored crossings'
    return dict(source=str(folder),D=D,definition=definition,event_count=len(stats['events']),
        core_B_minus_A_crossing_ms=lags,image=image,individual_maps=maps,count=count)


def compare(a,b):
    w=a['count']/a['count'].sum();x=a['image'];y=b['image'];dx=x-w@x;dy=y-w@y
    corr=np.sum(w*dx*dy)/np.sqrt(np.sum(w*dx*dx)*np.sum(w*dy*dy))
    occupied_x=x>=50;occupied_y=y>=50
    return dict(weighted_spatial_correlation=float(corr),
        weighted_RMSE_hz=float(np.sqrt(np.sum(w*(x-y)**2))),
        occupied_E_jaccard=float(np.sum(w*(occupied_x&occupied_y))/max(np.sum(w*(occupied_x|occupied_y)),1e-30)),
        occupied_E_fraction_a=float(w@occupied_x),occupied_E_fraction_b=float(w@occupied_y))


def main():
    rows=[];arrays={};comparisons=[]
    for D in (.225,.25):
        density=OUT/'qualification/selected_g40'/f'D{D:.6f}_degree6_dv0.125_4000ms'
        assert json.load(open(density/'status.json'))['status']=='COMPLETE'
        collected={'density':extract(density,D)}
        for seed in (1901,1902):
            folder=OUT/'particle_controls/selected_g40'/f'D{D:.6f}_Nscale1_seed{seed}_4000ms_microscopic'
            collected[f'particle_{seed}']=extract(folder,D)
        for name,data in collected.items():
            key=f'D{D:g}_{name}';arrays[key]=data['image'].reshape(20,20)
            rows.append(dict(model=name,**{k:v for k,v in data.items() if k not in ('image','individual_maps','count')}))
        c=dict(D=D,density_vs_particle_1901=compare(collected['density'],collected['particle_1901']),
            density_vs_particle_1902=compare(collected['density'],collected['particle_1902']),
            between_particle_seeds=compare(collected['particle_1901'],collected['particle_1902']))
        comparisons.append(c);print(c,flush=True)
    np.savez_compressed(OUT/'density_spatial_comparison.npz',**arrays)
    report=dict(status='DESCRIPTIVE_SPATIAL_COMPARISON_COMPLETE',rows=rows,comparisons=comparisons,
        statistical_unit='Trajectory / input seed; spatial bins are descriptors and are not independent repetitions',
        scope='Selected g40 with stationary private Poisson input and common OU zero; not full native SNN acceptance',
        limitation='Two particle seeds provide a descriptive noise comparison, not a confidence bound. Propagation timing and common-OU correspondence remain incomplete.')
    (OUT/'density_spatial_comparison.json').write_text(json.dumps(safe(report),indent=2)+'\n')


if __name__=='__main__':main()
