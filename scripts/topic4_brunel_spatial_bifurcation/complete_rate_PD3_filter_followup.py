"""Close PD3's physical-filter review on its independently refined parent."""
from rate_periodic import *
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1);a=p.parse_args()
    label='PD_A_return';source=PERIODIC_OUT/(label+'_filter_refined_mode_check.json')
    q=read(source);assert q['status']=='REFINED_PROFILE_NULL_MODE_CHECKED'
    assert q['antiperiodic_relative_residual']<1e-7
    worker=PERIODIC_OUT/'PD3_filter_state_followup_worker.json';checks=[]
    for dt in [.05,.025,.0125]:
        output=PERIODIC_OUT/f'{label}_filter_refined_monodromy_check_N1024_dt{dt:g}.json'
        if not output.exists():
            log=PERIODIC_OUT/f'{label}_filter_refined_dt{dt:g}.log'
            with log.open('w') as f:
                child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).with_name('verify_rate_PD_monodromy.py')),
                    str(source),q['mode'],'--dt',str(dt),'--device',str(a.device),'--label',label+'_filter_refined'],stdout=f,stderr=subprocess.STDOUT)
                write(worker,dict(status='RUNNING',pid=os.getpid(),child_pid=child.pid,dt_ms=dt,log=str(log)))
                code=child.wait()
            if code:
                write(worker,dict(status='STOPPED_WITH_ERROR',pid=os.getpid(),exit_code=code,log=str(log)));raise RuntimeError(log)
        v=read(output);assert Path(v['orbit']).resolve()==Path(q['orbit']).resolve();checks.append(v)
    errors=np.array([v['minus_one_relative_defect'] for v in checks]);phase=np.array([v['phase_relative_defect'] for v in checks])
    assert errors[-1]<1e-4 and phase[-1]<1e-4,(errors,phase)
    assert np.all(errors[:-1]/errors[1:]>3) and np.all(phase[:-1]/phase[1:]>3)
    repair=read(q['profile_refinement_source']);row=next(v for v in repair['rows'] if v['label']==label)
    assert row['resolution']['filter_state_check']['positive']
    result=dict(status='REFINED_PARENT_AND_MODE_CHECKED',**{k:q[k] for k in ['orbit','mode','N','J_EE_core','T_ms','antiperiodic_relative_residual']},
        continuous_orbit_check=row['resolution'],direct_monodromy_checks=checks,
        profile_refinement_source=q['profile_refinement_source'],
        scope='Same PD3 location; positive finer parent and independent full-state/history minus-one propagation. Original two-mesh root and crossing evidence retained. No new critical point or whole child-branch acceptance.')
    path=PERIODIC_OUT/(label+'_filter_state_followup.json');write(path,result)
    validation=PERIODIC_OUT/(label+'_validation.json');prior=read(validation)
    history=PERIODIC_OUT/'stability_coverage/attempt_history'/f'PD3_before_filter_state_acceptance_{time.time_ns()}.json';write(history,prior)
    prior.update(filter_state_followup=dict(source=str(path),**result),accepted_parent_orbit=q['orbit'])
    write(validation,prior);write(worker,dict(status='COMPLETE',pid=os.getpid(),source=str(path)))
    print('PD3 FILTER FOLLOWUP',errors,phase,flush=True)


if __name__=='__main__':main()
