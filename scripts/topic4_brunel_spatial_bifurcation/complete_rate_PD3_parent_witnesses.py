"""Recheck PD3 crossing orientation on physically corrected parent cycles."""
from complete_rate_positive_stability import *
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1)
    p.add_argument('--after-pid',type=int,nargs='*',default=[])
    p.add_argument('--min-free-gib',type=float,default=6.)
    a=p.parse_args();folder=DEST/'PD3_parent_witnesses';folder.mkdir(exist_ok=True)
    worker=folder/'worker.json';rows=[]
    def status(stage,**kw):
        write(worker,dict(status=stage,pid=os.getpid(),timestamp=time.time(),**kw))
    def resource(*unused):
        release(a.device)
        while True:
            free=float(subprocess.check_output(['nvidia-smi','-i',str(a.device),
                '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
            if free>=a.min_free_gib*1024:return
            status('WAITING_GPU_RESOURCE',free_mib=free);time.sleep(30)
    try:
        deps={pid:Path(f'/proc/{pid}/cmdline').read_bytes() for pid in a.after_pid
              if Path(f'/proc/{pid}/cmdline').exists()}
        while deps:
            for pid,identity in list(deps.items()):
                proc=Path(f'/proc/{pid}/cmdline')
                if not proc.exists() or proc.read_bytes()!=identity:deps.pop(pid)
            if deps:status('WAITING_DEPENDENCIES',dependencies=list(deps));time.sleep(30)
        s=RateField();weights=s.geo['group_size']/s.geo['group_size'].sum()
        validation=PERIODIC_OUT/'PD_A_return_validation.json';parent=read(validation)
        assert parent['status']=='VALIDATED_PD'
        for witness in parent['parent_crossing_witnesses']:
            old=read(Path(witness['source']));origin=Path(old['orbit'])
            side='below' if old['J_EE_core']<parent['J_EE_core'] else 'above'
            output=folder/f'{side}.json'
            if output.exists() and read(output).get('status')=='PHYSICAL_NEGATIVE_MODE_CHECKED':
                rows.append(read(output));continue
            resource();status('PARENT_PHYSICAL_REFINEMENT',side=side,source=str(origin))
            actual,check=prepare(origin,a.device,max_N=8192,host_krylov=True,
                check_filter_states=True,harmonic_chunk_size=64,stream_harmonics=True,
                before_mesh=resource)
            assert check['status']=='RESOLUTION_CHECKED' and check['filter_state_check']['positive']
            oldz,newz=np.load(origin),np.load(actual);N=max(len(oldz['r']),len(newz['r']))
            x,y=[resample(z['r']*1000,N,axis=0) for z in [oldz,newz]]
            distance,_=distances(x[:,None,:],y,weights)
            scale=np.sqrt(np.mean(np.sum((x-x.mean(0))**2*weights,axis=1)))
            drift=float(distance[0]/scale);period_drift=abs(float(newz['T'])/float(oldz['T'])-1)
            assert drift<.02 and period_drift<.01 and abs(float(newz['J'])-float(oldz['J']))<1e-12
            attempts=[];accepted=False
            for steps in [[.05,.025],[.025,.0125],[.0125,.00625]]:
                spectra=[];sources=[];modes=[];residuals=[]
                for dt in steps:
                    resource();tag=f'PD3_parent_{side}_physical_{actual.stem}'
                    path=PERIODIC_OUT/'poincare_floquet'/f'{tag}_dt{dt:g}.json'
                    status('PARENT_POINCARE',side=side,dt_ms=dt,orbit=str(actual))
                    q=read(path) if path.exists() else compute_return(actual,dt,6,a.device,16,
                        stream_harmonics=True,output_label=tag)
                    assert Path(q['orbit']).resolve()==actual.resolve()
                    vals=values(q);ii=np.flatnonzero((vals.real<0)&(abs(vals.imag)<1e-8))
                    assert len(ii),'No real negative mode returned'
                    i=int(ii[np.argmin(abs(vals[ii]-witness['multiplier']))])
                    assert abs(vals[i]-witness['multiplier'])<.1,'Mode identity changed after correction'
                    modes.append(float(vals[i].real));residuals.append(q['residuals'][i]/max(1,abs(vals[i])))
                    spectra.append(q);sources.append(str(path))
                change=abs(modes[0]-modes[1]);margin=max(2e-5,6*change,
                    4*max(q['phase_tangent_relative_defect'] for q in spectra))
                accepted=(max(residuals)<1e-6 and abs(abs(modes[-1])-1)>margin)
                expected=modes[-1]<-1-margin if side=='below' else -1+margin<modes[-1]<0
                attempts.append(dict(sources=sources,multipliers=modes,normalized_residuals=residuals,
                    paired_change=change,margin=margin,critical_direction_checked=bool(accepted and expected)))
                if accepted and expected:break
            assert accepted and expected,attempts
            row=dict(status='PHYSICAL_NEGATIVE_MODE_CHECKED',side=side,source=str(origin),orbit=str(actual),
                J_EE_core=float(newz['J']),physical=check,relative_waveform_change=drift,
                relative_period_change=period_drift,negative_multiplier=modes[-1],attempts=attempts,
                scope='Matched negative real mode in two-step Poincare spectra of the same physically corrected cycle. Other inherited unstable modes do not determine the PD crossing direction.')
            write(output,row);rows.append(row)
        by={q['side']:q for q in rows}
        assert set(by)=={'below','above'} and by['below']['negative_multiplier']<-1<by['above']['negative_multiplier']<0
        result=dict(status='PHYSICAL_PARENT_CROSSING_CHECKED',rows=rows,
            parent_validation=str(validation),critical_J=parent['J_EE_core'],
            scope='Orientation of the independently located PD3 crossing on corrected positive cycles. Child criticality still requires its separate physical profiles, radial mode and inherited instability.')
        write(folder/'result.json',result)
        current=read(validation);current['parent_crossing_witness_status']=result['status']
        current['physical_parent_crossing_source']=str(folder/'result.json');write(validation,current)
        status(result['status'])
    except Exception as exc:
        status('COMPUTATION_FAILED',error=repr(exc));raise


if __name__=='__main__':main()
