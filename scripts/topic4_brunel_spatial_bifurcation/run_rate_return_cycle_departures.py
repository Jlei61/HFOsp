"""Test both transverse unstable sides of the exact first H1 return cycle.

Uses the full Poincare eigenvector, including delay history. These are finite
nonlinear departure experiments, not a certified heteroclinic connection.
"""
from run_rate_sameJ_basin_bridge import *
from scipy.interpolate import interp1d


def initial_pair(s,dt,amplitude):
    path=RATE_OUT/'periodic_completion/orbits/connection_J0942_H1_middle_N128.npz'
    folder=RATE_OUT/'periodic_completion/poincare_floquet'
    source=folder/(path.stem+'_dt0.05.npz');m=np.load(source)
    verdict=read(folder/(path.stem+'_step_check.json'))
    assert verdict['status']=='UNSTABLE'
    ix=int(np.argmax(abs(m['multipliers'])));mu=m['multipliers'][ix]
    assert abs(mu.imag)<1e-9 and mu.real>1
    local=m['local_vectors'][:,ix].real.reshape(9,s.P)
    vh=m['history_vectors'][:,ix].real.reshape(-1,s.P)
    vd=float(m['dt']);vr=s.output(local)
    y,h,meta=recover_cpu(s,path,dt,phase_index=0)
    times=np.arange(len(h))*dt
    sampled=interp1d(np.arange(len(vh)+1)*vd,np.vstack([vr,vh]),axis=0)(times)
    scale=amplitude/1000/max(abs(sampled).max(),abs(vr).max())
    dh=np.empty_like(h);dh[(-np.arange(len(h)))%len(h)]=sampled*scale
    dy=local*scale
    pair=[]
    for sign in [-1,1]:
        yy=y+sign*dy;hh=h+sign*dh
        assert max(abs(s.output(yy)-hh[0]))<1e-12
        assert min(yy[:2].min(),hh.min())>0
        pair.append((sign,yy,hh))
    weights=s.geo['group_size']*s.E;energy=weights*abs(vr)**2
    regional=[float(energy[s.geo['group_region']==k].sum()/energy.sum()) for k in [0,1,2]]
    meta.update(mode_source=str(source),multiplier=mu,simulation_dt_ms=dt,
        requested_initial_rate_amplitude_Hz=amplitude,
        amplitude_definition='Maximum absolute output perturbation over current state and full physical-delay history.',
        direction='Leading real eigenvector of the transverse Poincare return map, at the original BVP phase.',
        mode_output_energy_fractions_at_this_phase=regional,
        history_interpolation='Linear interpolation from Floquet time grid to simulation time grid; includes exact current output.')
    return pair,meta


def run(s,sign,y,h,meta,dt,duration,device):
    folder=DEST/'departures'/f'sign{sign:+d}_dt{dt:g}_a{meta["requested_initial_rate_amplitude_Hz"]:g}'
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():return
    check=read(RATE_OUT/'periodic_completion'/f'long_graph_check_dt{dt:g}.json');assert check['status']=='PASS'
    g=GraphIntegrator(s,.942,dt,device,y,h);chunks=[];begin=time.time()
    blocks=int(np.ceil(duration/g.samples))
    contract=meta|dict(sign=sign,actual_duration_ms=blocks*g.samples,sampled_noise=False,
        interpretation='Finite nonlinear departure along a transverse unstable Floquet direction. No global connection is presumed.')
    write(folder/'contract.json',contract)
    np.savez_compressed(folder/'initial.npz',state=y,history=h,dt_ms=dt)
    for k in range(blocks):
        chunks.append(g.block().astype('float32'))
        assert bool(g.cp.isfinite(g.engine.y).all()) and float(g.engine.y[:2].min())>-1e-10
        if k%5==0 or k+1==blocks:
            write(folder/'status.json',dict(status='RUNNING',pid=os.getpid(),time_ms=g.elapsed_ms,total_ms=blocks*g.samples))
            print('DEPARTURE',sign,g.elapsed_ms,'/',blocks*g.samples,flush=True)
    activity=np.concatenate(chunks);regional=np.array([s.regional_rates(x/1000) for x in activity])
    np.savez_compressed(folder/'trajectory.npz',time_ms=np.arange(len(activity))+1,group_rate_hz=activity,
        regional_rates_hz=regional,contact_rate_hz=activity@s.geo['contact_rate_weights'],
        final_state=g.engine.y.get(),final_history=g.engine.history.get(),dt_ms=dt)
    manifest=read(RATE_OUT/'periodic_completion/composite_case_resolution.json')
    matches={letter:template_match(s,activity,next(q['orbit'] for q in manifest['rows'] if q['case']==letter)) for letter in ['b','c']}
    best=min(matches,key=lambda key:matches[key]['relative_full_group_RMS_error'])
    result=dict(status='COMPLETE',template_matches=matches,
        finite_window_match=best if matches[best]['relative_full_group_RMS_error']<.05 else 'UNRESOLVED',
        contract=contract,seconds=time.time()-begin)
    write(folder/'result.json',result);write(folder/'status.json',dict(status='COMPLETE'))
    print('RESULT',sign,result,flush=True)
    del g;gc.collect()
    import cupy as cp
    cp.get_default_memory_pool().free_all_blocks()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dt',type=float,default=.05)
    p.add_argument('--amplitude',type=float,default=.005);p.add_argument('--duration',type=int,default=8000)
    p.add_argument('--device',type=int,default=1);a=p.parse_args();s=RateField()
    pair,meta=initial_pair(s,a.dt,a.amplitude)
    for sign,y,h in pair:run(s,sign,y,h,meta,a.dt,a.duration,a.device)
