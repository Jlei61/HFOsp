"""Locate and verify the negative-multiplier crossing after the H2 cycle fold.

The two witnesses are on the same upper Core-B-mean branch. Keep them
distinct from the other branch at almost the same J, and do not infer a
flip from the sign of the dominant multiplier at unrelated cycles.
"""
from complete_rate_positive_stability import *
from rate_periodic_accuracy import defect
from audit_rate_filter_states import filter_state_minima
import subprocess


LABEL='PD_H2_after_LPC13'


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0)
    p.add_argument('--min-free-gib',type=float,default=3.)
    a=p.parse_args();folder=DEST/'H2_local_PD';folder.mkdir(exist_ok=True)
    scripts=Path(__file__).parent;worker=folder/'worker.json'
    def status(state,**kw):
        write(worker,dict(status=state,pid=os.getpid(),timestamp=time.time(),**kw))
    def gate(stage):
        release(a.device)
        while True:
            free=float(subprocess.check_output(['nvidia-smi','-i',str(a.device),
                '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
            if free>=a.min_free_gib*1024:return
            status('WAITING_GPU_RESOURCE',stage=stage,free_mib=free);time.sleep(30)
    def run(command,stage):
        gate(stage);log=folder/f'{stage}_{time.time_ns()}.log'
        with log.open('w') as output:
            child=subprocess.Popen([sys.executable,'-u',*map(str,command)],stdout=output,stderr=subprocess.STDOUT)
            status(stage,child_pid=child.pid,log=str(log));code=child.wait()
        if code:raise RuntimeError(f'{stage}: exit {code}; {log}')
    try:
        first=PERIODIC_OUT/'orbits/LPC_B2_local_+0.00100_N256.npz'
        second=PERIODIC_OUT/'orbits/LPC_B2_local_+0.00300_N256.npz'
        seed=PERIODIC_OUT/'antiperiodic/LPC_B2_local_+0.00100_N256_N256.npz'
        info=read(seed.with_suffix('.json'));vals=np.asarray(info['eigenvalues'])
        vals=vals[:,0]+1j*vals[:,1]
        index=int(np.argmin(abs(vals)))
        assert abs(vals[index].imag)<1e-10 and info['residuals'][index]<1e-7
        roots=[];profiles=[];s=RateField()
        for N in [256,512]:
            path=PERIODIC_OUT/f'{LABEL}_N{N}.json'
            if not path.exists():
                command=[scripts/'rate_period_doubling.py',first,second,'--seed',seed,
                    '--label',LABEL,'--N',N,'--device',a.device,'--low-memory',
                    '--linear-normalize','--nearest-orbit-seed','--stream-harmonics',
                    '--null-root-tol','1e-9','--slope-step','2e-9']
                if N==256:
                    command+=['--seed-index',index,'--cache-glob','LPC_B2_local_+0.*_N256.npz']
                    prior=PERIODIC_OUT/f'{LABEL}_scan_N256.json'
                    if prior.exists():
                        previous=read(prior)
                        archive=folder/'roundoff_plateau_before_resume.json'
                        if archive.exists():previous+=read(archive)['scan']
                        candidates=[q for q in previous if q['antiperiodic_relative_residual']<1e-9]
                        if candidates:
                            candidate=min(candidates,key=lambda q:q['antiperiodic_relative_residual'])
                            center=candidate['J_EE_core']
                            command[1:3]=[candidate['orbit'],candidate['orbit']]
                            command+=['--parameter-bracket',center-2e-10,center+2e-10]
                else:
                    center=roots[-1]['J_EE_core']
                    command+=['--parameter-bracket',center-3e-8,center+3e-8]
                run(command,f'ROOT_N{N}')
            root=read(path);assert root['antiperiodic_relative_residual']<1e-7
            gate('PROFILE_CHECK');status('PROFILE_CHECK',N=N)
            check=defect(s,Path(root['orbit']),a.device,harmonic_chunk_size=64,stream_harmonics=True)
            z=np.load(root['orbit']);check['filter_state_check']=filter_state_minima(s,z['r'],float(z['T']))
            w=s.geo['group_size']*s.E*(s.geo['group_region']==1);w/=w.sum()
            mean=float(z['r'].mean(0)@w*1000)
            fold=read(PERIODIC_OUT/'LPC_B2_validation.json')['mesh_checks'][-1]
            offset=mean-fold['coordinate_value']
            assert .001<offset<.003,('Wrong side of the cycle fold',offset)
            assert check['filter_state_check']['positive'] and check['minimum_rate_Hz']>=-1e-9
            assert check['maximum_group_defect_Hz']<1e-6
            check.update(core_B_mean_Hz=mean,offset_from_fold_core_B_mean_Hz=offset)
            write(folder/f'profile_N{N}.json',check);profiles.append(check);roots.append(root)
            first=second=Path(root['orbit']);seed=PERIODIC_OUT/f'{LABEL}_mode_N{N}.npz'
        lo,root=roots;dj=abs(lo['J_EE_core']-root['J_EE_core']);assert dj<1e-9
        checks=[]
        for dt in [.05,.025,.0125,.00625,.003125]:
            path=PERIODIC_OUT/f'{LABEL}_monodromy_check_N512_dt{dt:g}.json'
            if not path.exists():
                run([scripts/'verify_rate_PD_monodromy.py',PERIODIC_OUT/f'{LABEL}_N512.json',seed,
                    '--dt',dt,'--device',a.device,'--label',LABEL,'--stream-harmonics'],f'MONODROMY_dt{dt:g}')
            q=read(path);assert Path(q['orbit']).resolve()==first.resolve()
            checks.append(q)
            errors=np.array([q['minus_one_relative_defect'] for q in checks])
            if len(checks)>=3 and errors[-1]<1e-4 and np.all(errors[:-1]/errors[1:]>3):
                break
        errors=np.array([q['minus_one_relative_defect'] for q in checks])
        assert errors[-1]<1e-4 and np.all(errors[:-1]/errors[1:]>3),errors
        # Check simplicity in the antiperiodic operator, independently of
        # its bordered scalar zero and the propagated minus-one vector.
        spectral=PERIODIC_OUT/'antiperiodic'/f'{first.stem}_N512.json'
        if not spectral.exists():
            run([scripts/'rate_antiperiodic.py',first,'--N',512,'--device',a.device],'NULLSPACE_SIMPLICITY')
        anti=read(spectral);ev=np.asarray(anti['eigenvalues']);ev=ev[:,0]+1j*ev[:,1]
        order=np.argsort(abs(ev));assert abs(ev[order[0]])<1e-6 and abs(ev[order[1]])>1e-3
        assert max(anti['residuals'])<1e-6
        u=np.load(seed)['u'];energy=np.mean(abs(u)**2,axis=0)*s.geo['group_size']*s.E
        by=[float(energy[s.geo['group_region']==k].sum()/energy.sum()) for k in range(3)]
        result=dict(status='VALIDATED_PD',label='PD4',internal_label=LABEL,N=512,lower_mesh_N=256,
            J_EE_core=root['J_EE_core'],T_ms=root['T_ms'],mesh_J_difference=dj,
            antiperiodic_relative_residual=root['antiperiodic_relative_residual'],
            crossing_border_slope=root['dborder_dJ'],mesh_roots=roots,
            continuous_orbit_check=profiles[-1],full_physical_profile_status='PASS',full_acceptance=True,
            accepted_parent_orbit=root['orbit'],accepted_mode=str(seed),direct_monodromy_checks=checks,
            minus_one_error_reduction=errors[:-1]/errors[1:],simple_antiperiodic_nullspace_source=str(spectral),
            E_rate_mode_energy_A_B_surround=by,criticality='NOT_COMPUTED',parent_stability='ROOT_SPECTRUM_PENDING',
            scope='Two meshes, positive constituent filters, off-grid equations, simple antiperiodic zero, nonzero border slope, and full-state/history minus-one propagation at successively halved steps. No child criticality or global connection claim.')
        write(PERIODIC_OUT/f'{LABEL}_validation.json',result);status('CRITICAL_POINT_VALIDATED')
        pair=[]
        for dt in [.05,.025]:
            gate('ROOT_SPECTRUM');status('ROOT_SPECTRUM',dt_ms=dt)
            tag=f'{LABEL}_root';path=PERIODIC_OUT/'poincare_floquet'/f'{tag}_dt{dt:g}.json'
            q=read(path) if path.exists() else compute_return(first,dt,6,a.device,16,
                stream_harmonics=True,output_label=tag)
            assert Path(q['orbit']).resolve()==first.resolve();pair.append(q)
        result['root_spectrum']=paired_modes(*pair)
        if result['root_spectrum']['status']=='UNSTABLE':result['parent_stability']='ALREADY_UNSTABLE'
        write(PERIODIC_OUT/f'{LABEL}_validation.json',result);status('ROOT_SPECTRUM_CHECKED')
        child_label=LABEL+'_child'
        branch=PERIODIC_OUT/f'{child_label}_branch_N1024.json'
        if not branch.exists() or len(read(branch))<3:
            run([scripts/'rate_period_doubled_branch.py',PERIODIC_OUT/f'{LABEL}_N512.json',seed,
                '--N',512,'--label',child_label,'--device',a.device,'--amplitudes','.02','.04','.08',
                '--linear-normalize','--stream-harmonics','--tol','2e-11'],'CHILD_BRANCH')
        childchecks=[]
        for row in read(branch):
            gate('CHILD_PROFILE');q=defect(s,Path(row['orbit']),a.device,harmonic_chunk_size=64,stream_harmonics=True)
            z=np.load(row['orbit']);q['filter_state_check']=filter_state_minima(s,z['r'],float(z['T']))
            assert q['filter_state_check']['positive'] and q['maximum_group_defect_Hz']<1e-6
            childchecks.append(dict(**row,physical_check=q))
        write(folder/'physical_children.json',dict(rows=childchecks,criticality='NOT_COMPUTED',
            scope='Physical new doubled cycles only. Crossing direction and child Floquet checks remain separate.'))
        status('ROOT_AND_PHYSICAL_CHILDREN_FINISHED')
    except Exception as exc:
        status('COMPUTATION_FAILED',error=repr(exc));raise


if __name__=='__main__':main()
