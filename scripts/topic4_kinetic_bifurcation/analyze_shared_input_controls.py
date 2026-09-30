"""Original shared-noise bridge, without interpreting it as a bifurcation."""
from compare_density_spatial import *


def main():
    folder=OUT/'population_pair_replication/native_shared_input_controls';base=OUT/'particle_controls/selected_g40'
    assert json.load(open(folder/'status.json'))['status']=='EXECUTION_COMPLETE'
    rows=[];maps={};mean_maps={};input_pairs=[]
    for D in (.225,.25):
        duration=12000 if D==.225 else 4000
        for seed in (1901,1902):
            for name,suffix in [('individual','_microscopic')]+([('grouped','')] if D==.225 else []):
                for mode,tag in [('zero',''),('native','_native_shared_OU')]:
                    source=base/f'D{D:.6f}_Nscale1_seed{seed}_{duration}ms{suffix}{tag}'
                    assert json.load(open(source/'status.json'))['status']=='COMPLETE'
                    windows=[summarize(source,*w) for w in (((1000,4000),(4000,8000),(8000,12000)) if D==.225 else ((1000,4000),))]
                    left,right=windows[-1]['window_ms']
                    with np.load(source/'trajectory.npz') as z:
                        count=z['count_e'];raw=z['field_1ms'][int(left):int(right)]
                    coarse_count=count.reshape(20,2,20,2).sum((1,3)).ravel()
                    image=(raw*count).reshape(-1,20,2,20,2).sum((2,4)).reshape(-1,400).mean(0)/np.maximum(coarse_count,1)
                    mean_maps[(D,seed,name,mode)]=dict(image=image,count=coarse_count)
                    if D==.25 or windows[-1]['finite_events']:
                        maps[(D,seed,name,mode)]=extract(source,D,left,right)
                    rows.append(dict(D=D,seed=seed,model=name,shared_noise=mode,source=str(source.resolve()),windows=windows,
                        core_B_minus_A_ms=maps[(D,seed,name,mode)]['core_B_minus_A_crossing_ms'] if (D,seed,name,mode) in maps else []))
            if D==.225:
                paths=[base/f'D0.225000_Nscale1_seed{seed}_12000ms{suffix}_native_shared_OU/shared_input.npz' for suffix in ('','_microscopic')]
                with np.load(paths[0]) as a,np.load(paths[1]) as b:qa={k:bool(np.array_equal(a[k],b[k])) for k in a.files}
                assert all(qa.values());input_pairs.append(dict(seed=seed,identical_shared_input_records=qa))
    short=base/'D0.225000_Nscale1_seed1901_1000ms_microscopic_native_shared_OU'
    long=base/'D0.225000_Nscale1_seed1901_12000ms_microscopic_native_shared_OU';prefix=[]
    for filename in ('trajectory.npz','shared_input.npz'):
        with np.load(short/filename) as a,np.load(long/filename) as b:
            for k in a.files:
                v=b[k] if k=='count_e' else b[k][:len(a[k])]
                error=float(np.max(abs(a[k]-v)));bitwise=bool(np.array_equal(a[k],v))
                passed=bitwise or (k=='field_1ms' and error<1e-10)
                assert passed,(filename,k,error)
                prefix.append(dict(file=filename,key=k,bitwise=bitwise,maximum_error=error))
    contrasts=[]
    for D in (.225,.25):
        collections=[('all_time_mean',mean_maps)]+([('complete_event_median',maps)] if D==.225 else [])
        for kind,collection in collections:
            for seed in (1901,1902):
                key=(D,seed,'individual')
                if all((*key,mode) in collection for mode in ('zero','native')):
                    contrasts.append(dict(D=D,seed=seed,spatial_observable=kind,contrast='individual shared-zero versus shared-native',**compare(collection[(*key,'zero')],collection[(*key,'native')])))
                if D==.225 and all((D,seed,name,'native') in collection for name in ('individual','grouped')):
                    contrasts.append(dict(D=D,seed=seed,spatial_observable=kind,contrast='grouping under identical native shared input',**compare(collection[(D,seed,'individual','native')],collection[(D,seed,'grouped','native')])))
    result=dict(status='SHARED_INPUT_BRIDGE_ANALYZED',rows=rows,spatial_contrasts=contrasts,
        paired_input_records=input_pairs,prefix_replay=prefix,
        statistical_unit='Two private/shared-input realizations per condition, with paired models; events and spatial cells are nested.',
        scope='Original global and spatial OU laws in the selected g40 finite model, compared with their zero-field autonomous reference. New streams do not reproduce the original Fig5 realization.',
        interpretation='Assess whether shared input changes which side is occupied and spatial propagation. A discrepancy is evidence about the autonomous-to-stochastic bridge, not a failed numerical root or a directly classified bifurcation.')
    (folder/'analysis.json').write_text(json.dumps(safe(result),indent=2)+'\n')
    for x in rows:
        y=x['windows'][-1];print(x['D'],x['seed'],x['model'],x['shared_noise'],y['category'],y['mean_E_hz'],y['quiet_fraction'],y['finite_events'],flush=True)


if __name__=='__main__':main()
