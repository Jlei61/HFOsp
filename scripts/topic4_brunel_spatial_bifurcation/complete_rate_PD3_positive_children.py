"""Refine PD3 children with a fixed nonzero doubled-period component.

Near PD3, fixing J leaves an almost neutral radial direction in Newton's
system. The amplitude border instead solves for J and retains the child;
the corrected parameter, departure coefficient and physical states are all
checked again. No old child criticality is inherited.
"""
from complete_rate_positive_stability import *
from rate_periodic_accuracy import defect
from audit_rate_filter_states import filter_state_minima
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1)
    p.add_argument('--min-free-gib',type=float,default=5.5);a=p.parse_args()
    folder=DEST/'PD3_child_followup';scripts=Path(__file__).parent
    worker=folder/'worker_profiles.json';result=folder/'physical_profiles.json'
    def status(state,**kw):
        write(worker,dict(status=state,pid=os.getpid(),timestamp=time.time(),**kw))
        print(state,kw,flush=True)
    def gate(stage,N):
        release(a.device)
        required=max(a.min_free_gib,8. if N>=8192 else a.min_free_gib)
        if stage=='PHYSICAL_CHECK':required=1.5  # bounded harmonic defect; no Newton basis
        while True:
            free=float(subprocess.check_output(['nvidia-smi','-i',str(a.device),
                '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
            if free>=required*1024:return
            status('WAITING_GPU_RESOURCE',stage=stage,N=N,free_mib=free);time.sleep(30)
    try:
        s=RateField();v=read(PERIODIC_OUT/'PD_A_return_validation.json')
        parent=v['filter_state_followup'];assert parent['status']=='REFINED_PARENT_AND_MODE_CHECKED'
        assert parent['continuous_orbit_check']['filter_state_check']['positive']
        root=folder/'accepted_parent_for_child_switch.json'
        write(root,dict(orbit=parent['orbit'],J_EE_core=parent['J_EE_core'],mode=parent['mode']))
        z=np.load(parent['orbit']);u0=np.load(parent['mode'])['u']
        weights=s.geo['group_size']/s.geo['group_size'].sum()
        names=['PDreturnchild_a0.20000_N2048','PDreturnchild_a0.50000_N2048',
            'PDreturnchild_a1.00000_N2048','PDreturnextension_a60_refined_N2048']
        rows=[]
        for index,name in enumerate(names):
            origin=PERIODIC_OUT/'orbits'/f'{name}.npz';old=np.load(origin)
            out=folder/f'{name}_physical.json'
            if out.exists() and read(out).get('status')=='PASS':rows.append(read(out));continue
            for N in [4096,8192]:
                r=resample(old['r'],N,axis=0)
                base=np.tile(resample(z['r'],N//2,axis=0),(2,1))
                u=resample(np.r_[u0,-u0],N,axis=0);u/=np.max(abs(u))
                amp=float(np.sum((r-base)*1000*u)/np.sum(u*u));assert amp>0
                label=f'PD3_physical_amplitude_{index}_20260920'
                branch=PERIODIC_OUT/f'{label}_branch_N{N}.json'
                if not branch.exists():
                    gate('AMPLITUDE_BORDERED_REFINEMENT',N)
                    log=folder/f'{label}_N{N}.log'
                    command=[sys.executable,'-u',str(scripts/'rate_period_doubled_branch.py'),
                        str(root),parent['mode'],'--N',str(N//2),'--device',str(a.device),
                        '--label',label,'--amplitudes',str(amp),'--from-orbit',str(origin),
                        '--linear-normalize','--host-krylov','--stream-harmonics','--tol','2e-11']
                    with log.open('w') as f:
                        child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT)
                        status('AMPLITUDE_BORDERED_REFINEMENT',source=str(origin),N=N,
                            child_pid=child.pid,log=str(log));code=child.wait()
                    if code:raise RuntimeError(f'Child refinement failed: {code}; {log}')
                q=read(branch)[-1];path=Path(q['orbit']);gate('PHYSICAL_CHECK',N)
                physical=defect(s,path,a.device,harmonic_chunk_size=64,stream_harmonics=True)
                new=np.load(path);physical['filter_state_check']=filter_state_minima(s,new['r'],float(new['T']))
                good=(physical['filter_state_check']['positive'] and physical['minimum_rate_Hz']>=-1e-9
                    and physical['maximum_group_defect_Hz']<1e-5)
                if not good:
                    write(folder/f'{name}_N{N}_physical_failed.json',physical);continue
                x=resample(old['r']*1000,N,axis=0);y=new['r']*1000
                delta,_=distances(x[:,None,:],y,weights)
                rms=np.sqrt(np.mean(np.sum((x-x.mean(0))**2*weights,axis=1)))
                drift=float(delta[0]/rms);period_change=abs(float(new['T'])/float(old['T'])-1)
                parameter_change=float(new['J'])-float(old['J'])
                assert drift<.02 and period_change<.01 and abs(parameter_change)<1e-7
                assert float(new['J'])<parent['J_EE_core'] and q['half_period_relative_mismatch']>1e-7
                row=dict(status='PASS',source=str(origin),orbit=str(path),physical=physical,
                    relative_waveform_refinement_change=drift,relative_period_change=period_change,
                    J_change_from_source=parameter_change,amplitude_constraint_Hz=amp,
                    continuation_source=str(branch),
                    scope='Same doubled-period branch corrected with a nonzero amplitude border; J is corrected rather than fixed. Physical profiles only, no stability or criticality inherited.')
                write(out,row);rows.append(row);break
            else:raise RuntimeError(f'No positive resolved child through N8192: {origin}')
            write(result,dict(status='RUNNING',rows=rows))
        write(result,dict(status='ALL_FOUR_PROFILES_PASS',rows=rows,
            parent_validation=str(PERIODIC_OUT/'PD_A_return_validation.json'),
            scope='Four positive amplitude-constrained corrected children; temporal profiles and branch identity checked. Mode/stability acceptance remains separate.'))
        status('ALL_FOUR_PROFILES_PASS')
    except Exception as exc:
        status('COMPUTATION_FAILED',error=repr(exc));raise


if __name__=='__main__':main()
