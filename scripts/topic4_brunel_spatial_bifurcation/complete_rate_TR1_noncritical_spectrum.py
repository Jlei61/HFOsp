"""Check the other Floquet modes at the already validated first torus root.

The critical conjugate pair is matched to its independently checked mode.
Its unit modulus is not counted as numerical instability. This check does
not infer finite-amplitude torus stability or a global branch connection.
"""
from complete_rate_positive_stability import *
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0)
    p.add_argument('--min-free-gib',type=float,default=4.)
    a=p.parse_args();folder=DEST/'TR1_root_spectrum';folder.mkdir(exist_ok=True)
    worker=folder/'worker.json'
    def status(state,**kw):
        write(worker,dict(status=state,pid=os.getpid(),timestamp=time.time(),**kw))
        print(state,kw,flush=True)
    def gate():
        release(a.device)
        while True:
            free=float(subprocess.check_output(['nvidia-smi','-i',str(a.device),
                '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
            if free>=a.min_free_gib*1024:return
            status('WAITING_GPU_RESOURCE',free_mib=free);time.sleep(30)
    try:
        source=PERIODIC_OUT/'TR_A_B_validation.json';validated=read(source)
        assert validated['status']=='LOCALLY_SUBCRITICAL_TORUS_SUPPORTED'
        root=validated['critical_point'];path=Path(root['orbit'])
        target=complex(*root['multiplier']);targets=np.array([target,target.conjugate()])
        assert abs(target-1)>1e-3 and abs(abs(target)-1)<1e-7
        gate();status('PHYSICAL_ROOT_CHECK',orbit=str(path))
        actual,physical=prepare(path,a.device,max_N=256,check_filter_states=True,
            stream_harmonics=True,host_krylov=True,adaptive_memory=True)
        assert physical['status']=='RESOLUTION_CHECKED'
        assert physical['filter_state_check']['positive']
        old,new=np.load(path),np.load(actual)
        assert abs(float(new['J'])-root['J_EE_core'])<1e-12
        assert abs(float(new['T']/old['T'])-1)<1e-7
        before=resample(old['r'],len(new['r']),axis=0)
        drift=float(np.linalg.norm(new['r']-before)/np.linalg.norm(before-before.mean(0)))
        assert drift<1e-6
        attempts=[]
        for nev,steps in [(6,[.05,.025]),(10,[.025,.0125])]:
            spectra=[];sources=[]
            for dt in steps:
                gate();tag=f'TR1_root_noncritical_k{nev}_20260920'
                dest=PERIODIC_OUT/'poincare_floquet'/f'{tag}_dt{dt:g}.json'
                status('ROOT_SPECTRUM',nev=nev,dt_ms=dt,orbit=str(actual))
                q=read(dest) if dest.exists() else compute_return(actual,dt,nev,a.device,
                    2*nev+4,stream_harmonics=True,output_label=tag)
                assert Path(q['orbit']).resolve()==actual.resolve()
                spectra.append(q);sources.append(str(dest))
            paired=paired_modes(*spectra);mu=values(paired)
            _,critical=linear_sum_assignment(abs(targets[:,None]-mu[None,:]))
            matching_error=float(max(abs(targets-mu[critical])))
            assert matching_error<1e-3 and np.all(np.abs(mu[critical].imag)>1e-3)
            # Matching the same independently checked pair at both steps
            # prevents a weakly growing noncritical mode from being removed.
            coarse_mu=values(spectra[0])
            _,coarse_critical=linear_sum_assignment(abs(targets[:,None]-coarse_mu[None,:]))
            assert max(abs(targets-coarse_mu[coarse_critical]))<1e-3
            noncritical=np.ones(len(mu),bool);noncritical[critical]=False
            reliable=np.asarray(paired['reliable_mode_mask'],bool)
            margin=np.asarray(paired['per_mode_margin'])
            inside=reliable&(abs(mu)<1-margin)
            outside=reliable&(abs(mu)>1+margin)
            resolved=bool(paired['filter_coverage'] and paired['section_projection_checked']
                and np.all((inside|outside)[noncritical]))
            count=int(outside[noncritical].sum()) if resolved else None
            result=dict(status='NONCRITICAL_SPECTRUM_RESOLVED' if resolved else 'NONCRITICAL_SPECTRUM_PENDING',
                source_validation=str(source),orbit=str(actual),J_EE_core=root['J_EE_core'],
                physical=physical,relative_root_waveform_change=drift,sources=sources,
                critical_mode_target=targets,matched_critical_indices=critical,
                critical_matching_error=matching_error,matched_critical_multipliers=mu[critical],
                noncritical_multipliers=mu[noncritical],noncritical_unstable_dimension=count,
                other_modes_numerically_stable=bool(resolved and count==0),paired_spectrum=paired,
                scope='At the validated TR1 root, exclude only its separately checked non-real conjugate pair. Numerical count of all other full-history Poincare multipliers, conditional on numerical Arnoldi coverage; no rigorous enclosure, finite-torus Lyapunov spectrum or interval-wide completeness claim.')
            attempts.append(result);write(folder/'result.json',dict(**result,attempts=attempts))
            if resolved:break
        release(a.device);status('BATCH_FINISHED',scientific_status=result['status'],
            noncritical_unstable_dimension=count)
    except Exception as exc:
        status('COMPUTATION_FAILED',error=repr(exc));raise


if __name__=='__main__':main()
