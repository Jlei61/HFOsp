"""Check PD3 child modes again on positive corrected temporal profiles.

Retain the earlier calculation as historical evidence. This batch writes a
separate review result; it does not automatically restore canonical criticality.
"""
from complete_rate_positive_stability import *
from rate_floquet import compute as compute_full
from rate_periodic_accuracy import defect
from audit_rate_filter_states import filter_state_minima
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1)
    p.add_argument('--after-pid',type=int,nargs='*',default=[])
    p.add_argument('--min-free-gib',type=float,default=6.)
    a=p.parse_args();folder=DEST/'PD3_child_followup';folder.mkdir(exist_ok=True)
    worker=folder/'worker_modes.json';scripts=Path(__file__).parent
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
            if deps:status('WAITING_PROFILES',dependencies=list(deps));time.sleep(30)
        source=folder/'physical_profiles.json';profiles=read(source)
        assert profiles['status']=='ALL_FOUR_PROFILES_PASS'
        rows=profiles['rows'];assert len(rows)==4
        s=RateField();weights=s.geo['group_size']*s.E;weights/=weights.sum()
        parent=read(PERIODIC_OUT/'PD_A_return_validation.json')
        assert parent['status']=='VALIDATED_PD'
        geometry=[]
        for row in rows:
            path=Path(row['orbit'])
            physical=row['physical']
            assert Path(physical['orbit']).resolve()==path.resolve()
            assert path.resolve().stat().st_mtime<=source.stat().st_mtime
            z=np.load(path);r=z['r'];N=len(r)
            assert physical['N']==N
            assert abs(physical['J_EE_core']-float(z['J']))<1e-12
            assert abs(physical['T_ms']-float(z['T']))<1e-9
            assert physical['filter_state_check']['positive'] and physical['minimum_rate_Hz']>=-1e-9
            assert physical['maximum_group_defect_Hz']<1e-5
            half=(r[:N//2]-r[N//2:])*500
            amp=float(np.sqrt(np.mean(half*half,axis=0)@weights))
            shift=float(z['J'])-parent['J_EE_core'];assert amp>0 and shift<0
            geometry.append(dict(orbit=str(path),N=N,physical=physical,
                antisymmetric_E_RMS_Hz=amp,J_EE_core=float(z['J']),J_shift=shift,
                J_shift_over_squared_amplitude=shift/amp**2,T_ms=float(z['T'])))
        geometry.sort(key=lambda q:q['antisymmetric_E_RMS_Hz'])
        coeff=np.array([q['J_shift_over_squared_amplitude'] for q in geometry[:3]])
        spread=float(np.ptp(coeff)/abs(np.mean(coeff)));assert spread<.1
        target=Path(geometry[-1]['orbit']);target_N=geometry[-1]['N']
        seed=PERIODIC_OUT/'PD3child_a60_radial_mode_N2048.npz'
        growth=read(PERIODIC_OUT/'PD3child_a60_radial_N2048.json')['lambda_per_ms']
        mode_rows=[];tag='PD3child_physical_radial_20260920'
        for N in [target_N,2*target_N]:
            path=PERIODIC_OUT/f'{tag}_N{N}.json'
            if not path.exists():
                run([scripts/'rate_real_floquet_mode.py',target,'--seed',seed,
                    '--N',N,f'--growth={growth}','--device',a.device,'--label',tag,
                    '--stream-harmonics'],f'RADIAL_MODE_N{N}')
            q=read(path);assert Path(q['orbit']).resolve()==target.resolve()
            assert q['status']=='EIGENPAIR_CONVERGED_CHECKS_PENDING'
            assert q['residual']<2e-9 and 0<q['multiplier']<1
            mode_rows.append(q);growth=q['lambda_per_ms']
            seed=PERIODIC_OUT/f'{tag}_mode_N{N}.npz'
        change=abs(mode_rows[0]['multiplier']-mode_rows[1]['multiplier'])
        assert change<1e-5 and 1-mode_rows[-1]['multiplier']>100*change
        N=mode_rows[-1]['N'];checkfile=PERIODIC_OUT/f'{tag}_segmented_checks_N{N}.json'
        if not checkfile.exists() or read(checkfile)['status']!='COMPLETE':
            run([scripts/'check_rate_segmented_real_mode.py','--label',tag,'--N',N,
                '--device',a.device,'--dt','.1','.05','.025','--segments',16,
                '--stream-harmonics'],'INDEPENDENT_SEGMENTED_FLOW')
        segmented=read(checkfile);checks=segmented['checks']
        assert Path(segmented['orbit']).resolve()==target.resolve()
        for key in ['maximum_mode_relative_defect','maximum_phase_relative_defect']:
            err=np.array([q[key] for q in checks]);assert err[-1]<1e-4 and np.all(err[:-1]/err[1:]>3)
        assert min(q['minimum_negative_control_defect'] for q in checks)>.01
        spectra=[]
        for dt in [.1,.05]:
            gate('INHERITED_INSTABILITY');status('INHERITED_INSTABILITY',dt_ms=dt)
            path=PERIODIC_OUT/'floquet'/f'{tag}_dominant_dt{dt:g}.json'
            q=read(path) if path.exists() else compute_full(target,dt,1,a.device,
                ncv=10,output_label=tag+'_dominant',stream_harmonics=True)
            assert Path(q['orbit']).resolve()==target.resolve()
            mu=values(q)[0];assert abs(mu)>2 and q['residuals'][0]/abs(mu)<1e-6
            spectra.append(q)
        multiplier_change=float(abs(values(spectra[0])[0]/values(spectra[1])[0]-1))
        assert multiplier_change<.01
        result=dict(status='PHYSICAL_CHILD_MODES_CHECKED_REVIEW_PENDING',
            source_profiles=str(source),physical_geometry=geometry,
            departure_coefficient_relative_spread=spread,radial_modes=mode_rows,
            radial_multiplier_mesh_difference=change,segmented_checks_source=str(checkfile),
            dominant_spectra=spectra,dominant_relative_step_change=multiplier_change,
            proposed_local_criticality='SUPERCRITICAL_PD',child_stability='UNSTABLE',
            full_physical_child_checks=True,
            scope='Positive corrected child profiles, quadratic local departure, two temporal mode meshes, full-state/history segment checks, and independent inherited instability. Canonical promotion additionally requires review of matched physical parent-side crossing witnesses; no global connection or stable burst onset is inferred.')
        write(folder/'physical_mode_result.json',result);status(result['status'])
    except Exception as exc:
        status('COMPUTATION_FAILED',error=repr(exc));raise


if __name__=='__main__':main()
