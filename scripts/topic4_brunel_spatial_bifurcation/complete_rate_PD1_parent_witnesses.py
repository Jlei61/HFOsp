"""Physical same-branch witnesses on both sides of the accepted first PD."""
from complete_rate_positive_stability import *
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0)
    p.add_argument('--after-pid',type=int,nargs='*',default=[])
    p.add_argument('--min-free-gib',type=float,default=8.)
    p.add_argument('--stream-harmonics',action='store_true')
    a=p.parse_args();folder=DEST/'PD1_parent_witnesses';folder.mkdir(parents=True,exist_ok=True)
    worker=folder/'worker.json';rows=[]
    def status(stage,**kw):
        write(worker,dict(status=stage,pid=os.getpid(),rows=rows,**kw))
    deps={pid:Path(f'/proc/{pid}/cmdline').read_bytes() for pid in a.after_pid if Path(f'/proc/{pid}/cmdline').exists()}
    while deps:
        for pid,identity in list(deps.items()):
            f=Path(f'/proc/{pid}/cmdline')
            if not f.exists() or f.read_bytes()!=identity:deps.pop(pid)
        if deps:status('WAITING_DEPENDENCIES',dependencies=list(deps));time.sleep(30)
    parent=read(PERIODIC_OUT/'PD_double_low_validation.json')
    assert parent.get('full_acceptance',False)
    sources=['PD_double_low_eval_J0.93871143923_N2048',
             'PD_double_low_eval_J0.93872871472_N2048']
    s=RateField();weights=s.geo['group_size']/s.geo['group_size'].sum()
    def gate(mesh=0,stage='PROFILE'):
        required=max(a.min_free_gib,6. if stage=='NEWTON' and mesh>=4096 else 0.)
        while True:
            free=float(subprocess.check_output(['nvidia-smi','-i',str(a.device),
                '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
            if free>=required*1024:return
            status('WAITING_GPU_RESOURCE',free_mib=free,required_free_gib=required,
                   N=mesh,next_stage=stage);time.sleep(30)
    def resource():
        release(a.device);gate()
    for side,name in zip(['below','above'],sources):
        result=folder/(side+'.json')
        if result.exists() and read(result).get('status') in ['NUMERICALLY_STABLE','UNSTABLE']:
            rows.append(read(result));continue
        path=PERIODIC_OUT/'orbits'/(name+'.npz');resource()
        status('PROFILE_CHECK',side=side,orbit=str(path))
        meta=read(path.with_suffix('.json'))
        start=best_cached(dict(orbit=str(path),J_EE_core=meta['J_EE_core'],T_ms=meta['T_ms']))
        actual,accuracy=prepare(start,a.device,max_N=8192,check_filter_states=True,
            harmonic_chunk_size=32,adaptive_memory=True,stream_harmonics=a.stream_harmonics,
            host_krylov=a.stream_harmonics,before_mesh=gate)
        if accuracy['status']=='RESOLUTION_CHECKED' and accuracy['maximum_group_defect_Hz']>=.001:
            actual,accuracy=prepare(actual,a.device,max_N=8192,min_N=2*accuracy['N'],
                check_filter_states=True,harmonic_chunk_size=32,adaptive_memory=True,
                stream_harmonics=a.stream_harmonics,host_krylov=a.stream_harmonics,before_mesh=gate)
        if accuracy['status']!='RESOLUTION_CHECKED' or accuracy['maximum_group_defect_Hz']>=.001:
            write(result,dict(status='PROFILE_UNRESOLVED',resolution=accuracy));continue
        before,after=np.load(path),np.load(actual);N=max(len(before['r']),len(after['r']))
        x,y=[resample(z['r']*1000,N,axis=0) for z in [before,after]]
        distance,phase=distances(x[:,None,:],y,weights)
        scale=np.sqrt(np.mean(np.sum((x-x.mean(0))**2*weights,axis=1)))
        drift=float(distance[0]/scale);period_drift=abs(float(after['T'])/float(before['T'])-1)
        assert drift<.02 and period_drift<.01
        J=float(after['J']);assert abs(J-float(before['J']))<1e-12
        assert (J<parent['J_EE_core'])==(side=='below')
        attempts=[]
        for nev,steps in [(2,[.1,.05]),(6,[.05,.025]),(8,[.025,.0125])]:
            spectra=[];files=[]
            for dt in steps:
                resource();tag=f'PD1_parent_{side}_physical_k{nev}_{actual.stem}'
                target=PERIODIC_OUT/'poincare_floquet'/f'{tag}_dt{dt:g}.json'
                status('POINCARE',side=side,nev=nev,dt_ms=dt,orbit=str(actual))
                q=read(target) if target.exists() else compute_return(actual,dt,nev,
                    a.device,max(10,2*nev+4),stream_harmonics=True,output_label=tag)
                assert Path(q['orbit']).resolve()==actual.resolve()
                spectra.append(q);files.append(str(target))
            verdict=paired_modes(*spectra)
            attempts.append(dict(sources=files,classification=verdict))
            if verdict['status']!='UNRESOLVED':break
        row=dict(side=side,status=verdict['status'],J_EE_core=J,T_ms=float(after['T']),
            original_orbit=str(path),analyzed_orbit=str(actual),resolution=accuracy,
            relative_waveform_refinement_change=drift,relative_period_change=period_drift,
            classification=verdict,attempts=attempts)
        write(result,row);rows.append(row)
    by={q['side']:q for q in rows};passed=False
    if set(by)=={'below','above'}:
        low=by['below']['classification'];mu=values(low);margin=np.array(low['per_mode_margin'])
        reliable=np.array(low.get('reliable_mode_mask',np.zeros(len(mu),bool)))
        negative=bool(np.any(reliable&(abs(mu.imag)<1e-6)&(mu.real<-1-margin)))
        passed=(low['status']=='UNSTABLE' and negative and by['above']['status']=='NUMERICALLY_STABLE')
    q=dict(status='PARENT_SIDES_CHECKED' if passed else 'PARENT_SIDES_UNRESOLVED',
        accepted_parent_source=str(PERIODIC_OUT/'PD_double_low_validation.json'),
        critical_J=parent['J_EE_core'],rows=rows,
        scope='Two physical corrected cycles support the parent stability orientation at PD1. This is not an exhaustive crossing count between samples and does not independently establish nonlinear PD criticality.')
    write(folder/'result.json',q);status(q['status'])


if __name__=='__main__':
    try:main()
    except Exception as exc:
        path=DEST/'PD1_parent_witnesses/worker.json'
        previous=read(path) if path.exists() else {}
        write(path,dict(**{k:v for k,v in previous.items() if k not in ['status','error']},
            status='COMPUTATION_FAILED',error=repr(exc)))
        raise
