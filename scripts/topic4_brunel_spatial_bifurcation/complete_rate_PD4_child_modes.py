"""Local PD4 child classification on the same physical spatial rate branch."""
from complete_rate_positive_stability import *
from rate_floquet import compute as compute_full
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0)
    p.add_argument('--after-pid',type=int,nargs='*',default=[])
    p.add_argument('--child-index',type=int,default=0)
    p.add_argument('--min-free-gib',type=float,default=3.5);a=p.parse_args()
    folder=DEST/'H2_local_PD';worker=folder/'worker_child_modes.json';scripts=Path(__file__).parent
    label='PD_H2_after_LPC13';tag=f'PD4_physical_child_radial_index{a.child_index}_20260920'
    def status(state,**kw):
        write(worker,dict(status=state,pid=os.getpid(),timestamp=time.time(),**kw))
        print(state,kw,flush=True)
    def gate(stage):
        release(a.device)
        while True:
            free=float(subprocess.check_output(['nvidia-smi','-i',str(a.device),
                '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
            if free>=a.min_free_gib*1024:return
            status('WAITING_GPU_RESOURCE',stage=stage,free_mib=free);time.sleep(30)
    def run(command,stage):
        gate(stage);log=folder/f'{stage}_{time.time_ns()}.log'
        with log.open('w') as out:
            child=subprocess.Popen([sys.executable,'-u',*map(str,command)],stdout=out,stderr=subprocess.STDOUT)
            status(stage,child_pid=child.pid,log=str(log));code=child.wait()
        if code:raise RuntimeError(f'{stage}: {code}; {log}')
    try:
        deps={pid:Path(f'/proc/{pid}/cmdline').read_bytes() for pid in a.after_pid
              if Path(f'/proc/{pid}/cmdline').exists()}
        while deps:
            for pid,identity in list(deps.items()):
                proc=Path(f'/proc/{pid}/cmdline')
                if not proc.exists() or proc.read_bytes()!=identity:deps.pop(pid)
            if deps:status('WAITING_PHYSICAL_CHILDREN',dependencies=list(deps));time.sleep(30)
        validation=PERIODIC_OUT/f'{label}_validation.json';parent=read(validation)
        assert parent['full_acceptance'] and parent['status']=='VALIDATED_PD'
        witnesses=[read(DEST/'H2_fold_neighborhood'/f'offset_{v}.json')
                   for v in ['+0.00100_refined','+0.00300']]
        assert witnesses[0]['J_EE_core']<parent['J_EE_core']<witnesses[1]['J_EE_core']
        assert witnesses[0]['coordinate_value']<parent['continuous_orbit_check']['core_B_mean_Hz']<witnesses[1]['coordinate_value']
        critical_negative=[]
        for witness in witnesses:
            assert witness['physical']['filter_state_check']['positive']
            q=witness['classification'];vals=values(q)
            ii=np.flatnonzero((vals.real<0)&(abs(vals.imag)<1e-8))
            i=int(ii[np.argmin(abs(vals[ii]+1))]);assert q['reliable_mode_mask'][i]
            critical_negative.append(vals[i].real)
        assert critical_negative[0]<-1<critical_negative[1]<0
        children=read(folder/'physical_children.json')['rows'];assert len(children)>=3
        coeff=np.array([q['J_shift_over_amplitude_squared'] for q in children])
        assert np.all(coeff>0) or np.all(coeff<0)
        spread=float(np.ptp(coeff)/abs(np.mean(coeff)));assert spread<.1
        for q in children:
            assert q['physical_check']['filter_state_check']['positive']
            assert q['physical_check']['maximum_group_defect_Hz']<1e-6
            assert witnesses[0]['J_EE_core']<q['J_EE_core']<witnesses[1]['J_EE_core']
            assert q['half_period_relative_mismatch']>1e-7
            assert abs(q['T_ms']/(2*parent['T_ms'])-1)<.01
        selected=children[a.child_index]
        target=Path(selected['orbit']);N=len(np.load(target)['r']);seed=Path(parent['accepted_mode'])
        # The autonomous phase makes the bordered Newton system singular
        # at exactly zero growth. Start near the local flip prediction on
        # the smallest checked child; this is only a numerical seed.
        slope=(-1-critical_negative[0])/(parent['J_EE_core']-witnesses[0]['J_EE_core'])
        predicted_mu=1+4*slope*selected['J_shift']
        assert predicted_mu>0 and abs(predicted_mu-1)>1e-8
        modes=[];growth=float(np.log(predicted_mu)/selected['T_ms'])
        write(folder/'radial_initialization.json',dict(child_index=a.child_index,orbit=str(target),
            prediction_only=True,initial_growth_per_ms=growth,initial_multiplier=predicted_mu,
            reason='Avoid the independently demonstrated zero-growth autonomous phase nullspace. Use the nearest physical parent witness and the local flip normal form only to initialize Newton.'))
        for mesh in [N,2*N]:
            path=PERIODIC_OUT/f'{tag}_N{mesh}.json'
            if not path.exists():
                command=[scripts/'rate_real_floquet_mode.py',target,'--seed',seed,
                    '--N',mesh,f'--growth={growth}','--device',a.device,'--label',tag,'--stream-harmonics']
                if not modes:command.append('--antiperiodic-seed')
                run(command,f'CHILD_RADIAL_N{mesh}')
            q=read(path);assert Path(q['orbit']).resolve()==target.resolve()
            assert q['status']=='EIGENPAIR_CONVERGED_CHECKS_PENDING' and q['residual']<2e-9
            modes.append(q);growth=q['lambda_per_ms'];seed=PERIODIC_OUT/f'{tag}_mode_N{mesh}.npz'
        mu=modes[-1]['multiplier'];change=abs(mu-modes[0]['multiplier'])
        identity_file=folder/f'{tag}_identity_N{modes[-1]["N"]}.json'
        run([scripts/'check_rate_PD4_child_mode_identity.py','--N',modes[-1]['N'],'--label',tag],
            'RADIAL_MODE_IDENTITY')
        identity=read(identity_file)
        assert identity['status']=='RADIAL_MODE_IDENTITY_PASS'
        assert abs(mu-1)>100*max(change,max(q['residual'] for q in modes))
        supercritical=bool(np.all(coeff<0) and mu<1)
        subcritical=bool(np.all(coeff>0) and mu>1)
        assert supercritical or subcritical,('Departure and radial multiplier require further study',coeff,mu)
        checked=False;mesh=modes[-1]['N']
        checkfile=PERIODIC_OUT/f'{tag}_segmented_checks_N{mesh}.json'
        for steps in [[.05,.025,.0125],[.025,.0125,.00625]]:
            if not checkfile.exists() or read(checkfile).get('status')!='COMPLETE' or (
                read(checkfile)['checks'][-1]['dt_ms']>steps[-1]*(1+1e-10)):
                run([scripts/'check_rate_segmented_real_mode.py','--label',tag,'--N',mesh,
                    '--device',a.device,'--segments',16,'--dt',*steps,'--stream-harmonics'],
                    f'CHILD_SEGMENTED_FLOW_{steps[-1]:g}')
            seg=read(checkfile);assert Path(seg['orbit']).resolve()==target.resolve()
            e=np.array([q['maximum_mode_relative_defect'] for q in seg['checks']])
            checked=bool(e[-1]<min(1e-4,abs(mu-1)/10) and np.all(e[:-1]/e[1:]>3)
                and min(q['minimum_negative_control_defect'] for q in seg['checks'])>.01)
            if checked:break
        assert checked,'Independent radial-mode propagation not yet accurate enough'
        spectra=[]
        for dt in [.05,.025]:
            gate('CHILD_INHERITED_INSTABILITY');status('CHILD_INHERITED_INSTABILITY',dt_ms=dt)
            path=PERIODIC_OUT/'floquet'/f'{tag}_dominant_dt{dt:g}.json'
            q=read(path) if path.exists() else compute_full(target,dt,1,a.device,
                ncv=10,output_label=tag+'_dominant',stream_harmonics=True)
            assert Path(q['orbit']).resolve()==target.resolve()
            v=values(q)[0];assert abs(v)>2 and q['residuals'][0]/abs(v)<1e-6
            spectra.append(q)
        relative_change=float(abs(values(spectra[0])[0]/values(spectra[1])[0]-1));assert relative_change<.01
        result=dict(status='LOCALLY_CHECKED_PD_CHILD',label='PD4',parent_validation=str(validation),
            physical_children_source=str(folder/'physical_children.json'),rows=children,
            criticality='SUPERCRITICAL_PD' if supercritical else 'SUBCRITICAL_PD',
            full_physical_child_checks=True,child_orbit=str(target),child_stability='UNSTABLE',
            departure_coefficient_relative_spread=spread,radial_modes=modes,
            radial_mode_identity_source=str(identity_file),
            radial_multiplier_mesh_difference=change,independent_segmented_source=str(checkfile),
            inherited_instability_sources=spectra,inherited_multiplier_step_change=relative_change,
            parent_crossing_sources=[str(DEST/'H2_fold_neighborhood'/f'offset_{v}.json')
                for v in ['+0.00100_refined','+0.00300']],
            scope='Local criticality along the flip direction of an already unstable parent; inherited instability keeps the checked child unstable. No stable burst onset, global connection, irregular attractor, or propagation-template switch is inferred.')
        write(PERIODIC_OUT/f'{label}_child_validation.json',result)
        current=read(validation);current.update(criticality=result['criticality'],
            child_stability='UNSTABLE',child_classification_source=str(PERIODIC_OUT/f'{label}_child_validation.json'))
        write(validation,current);status('LOCAL_CHILD_CLASSIFICATION_FINISHED',criticality=result['criticality'])
    except Exception as exc:
        status('COMPUTATION_FAILED',error=repr(exc));raise


if __name__=='__main__':main()
