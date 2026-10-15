"""Engineering canaries only; no formal final-B selection or dispatch."""
import copy
import subprocess
import time
import numpy as np
import psutil
from scripts import run_topic4_core_ou_correlation_recovery as worker
n=worker.night;rt=worker.rt


def prepare():
    if worker.PLAN.exists():return rt.read(worker.PLAN)
    plan=copy.deepcopy(rt.read(n.OUT/'tonic_input_plan.json'))
    plan['version']='core_connectivity_v2_core_OU_correlation_v1'
    plan['status']='ENGINEERING_CANARY_PREPARED_NO_FORMAL_DISPATCH'
    plan['physics']['version']=plan['version']
    plan['physics']['shared_core_OU']='mixture coefficient sqrt(rho); rho=1 is the exact inherited shared drive'
    plan['physics']['external_input']='Only E-core: shared OU loading sqrt(rho), plus independent core-wide OU loading sqrt(1-rho), same marginal OU law. Outside E and all I retain deterministic expected arrivals.'
    plan['physics']['state']='Only stochastic input processes; no neuronal Z/M, slow-I state, kick, patient labels or route-specific drive.'
    plan['source_snapshot'][str(worker.SCRIPT)]=rt.sha(worker.SCRIPT)
    driver=worker.ROOT/'src/topic4_core_ou_correlation.py';plan['source_snapshot'][str(driver)]=rt.sha(driver)
    plan['hypothesis']='Both TA-like and some TB-like events occur rarely under fixed parameters. Shared core-wide input may restrict access to distinct native propagation responses; correlation change is a hypothesis, not an explanation already established.'
    plan['correlation_axis']=dict(name='core_ou_correlation',values=[1.,.5,0.],
        equation='eta_k=sqrt(rho)*xi_global+sqrt(1-rho)*xi_k; independent xi_k follow the same tau150ms, sigma_n3.3 OU recurrence and zero initialization',
        invariant='Same individual-core marginal OU mean, variance and time constant before rate clipping; no new spatial field outside cores.',
        realization='Across rho, shared seeds do not imply identical stepwise Poisson/global innovations. Within-rho weight or threshold pairs can separately verify actual input identity.')
    parent=copy.deepcopy(plan['canary']['candidates'][0]);candidates=[];stage='core_OU_correlation_canary_20260911'
    for rho in [1.,.5,0.]:
        c=copy.deepcopy(parent);c.update(id='coreOU_canary_rho'+str(rho).replace('.','p'),stage=stage,core_ou_correlation=rho,
            parent_id='tonic_canary_1p0',changed_parameter='core_ou_correlation',changed_value=rho,changed_parameters={'core_ou_correlation':rho})
        rt.write(n.old.OUT/'candidates'/f'{c["id"]}.json',c);candidates.append(c)
    plan['canary']=dict(stage=stage,candidates=candidates,topology_seed=2511,dynamics_seed=847101,duration_ms=500.,max_new=3,
        tests=['rho1 matches all inherited500ms arrays','static graph and thresholds unchanged','actual core mixture rate at every step','outside E/all I rates unchanged','no intermediate clipping'],
        resources='Run at most one canary while <=7 current formal workers remain; same RSS and available-memory guards.',
        budget='Three additional500ms engineering units only, separately reported; no addition to formal48-run budget.')
    plan['formal']='NOT_SELECTED. Final A complete propagation review must precede any final B freeze; canaries alone never nominate a model.'
    rt.write(worker.PLAN,plan)
    return plan


def run_canaries():
    plan=prepare();can=plan['canary'];resource=rt.read(n.OUT/'plan.json')['resources'];records=[]
    for c in can['candidates']:
        path=n.old.result_path(can['stage'],c['id'],2511,847101)
        if n.old.complete(path):continue
        while True:
            status=rt.read(n.OUT/'status.json');active=[p for p in status.get('active',[]) if psutil.pid_exists(p['pid'])]
            if len(active)<8 and rt.available_gib()>resource['min_available_gib']+resource['reserve_per_worker_gib']:break
            if time.time()>rt.read(n.OUT/'plan.json')['stop_new_dispatch_unix']:raise RuntimeError('canary dispatch deadline')
            time.sleep(10)
        log=n.OUT/'logs'/f'{c["id"]}.log'
        with log.open('a') as f:
            proc=subprocess.Popen([rt.PYTHON,'-u',str(worker.SCRIPT),'worker','--stage',can['stage'],'--candidate',c['id'],
                '--topology','2511','--seed','847101','--duration','500'],cwd=worker.ROOT,env=rt.ENV,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            peak=0.
            while proc.poll() is None:
                try:
                    p=psutil.Process(proc.pid);rss=sum(q.memory_info().rss for q in [p]+p.children(recursive=True))/2**30;peak=max(peak,rss)
                    if rss>resource['kill_tree_rss_gib'] or rt.available_gib()<resource['min_available_gib']:
                        n.stop_tree(proc.pid);raise RuntimeError('canary resource guard')
                except psutil.NoSuchProcess:pass
                rt.write(n.OUT/'core_ou_correlation_canary_status.json',dict(status='CANARY_RUNNING',candidate=c['id'],pid=proc.pid,peak_tree_rss_gib=peak,formal_selection='NOT_SELECTED'))
                time.sleep(5)
            if proc.returncode or not n.old.complete(path):raise RuntimeError(f'canary failed: {log}')
        records.append(dict(candidate=c['id'],peak_tree_rss_gib=peak))
    reference=n.old.result_path('tonic_canary_20260911','tonic_canary_1p0',2511,847101)
    comparison=n.old.result_path(can['stage'],can['candidates'][0]['id'],2511,847101)
    with np.load(reference.with_suffix('.npz')) as a,np.load(comparison.with_suffix('.npz')) as b:
        assert set(a.files)==set(b.files)
        parity={k:bool(np.array_equal(a[k],b[k],equal_nan=True)) if np.issubdtype(a[k].dtype,np.number) else bool(np.array_equal(a[k],b[k])) for k in a.files}
    assert all(parity.values())
    checks=[]
    ref_identity=rt.read(reference)['static_array_identity']
    for c in can['candidates']:
        p=n.old.result_path(can['stage'],c['id'],2511,847101);r=rt.read(p);ap=rt.read(p.parent.parent/'applied_physics.json')
        m=r['core_ou_mixture_audit']
        assert r['actual_duration_ms']==500 and r['static_array_identity']==ref_identity
        assert ap['threshold']['n_raised']==0 and r['maximum_outside_rate_deviation']==0
        assert m['maximum_I_rate_error']==0 and max(m['maximum_core_rate_error'].values())<1e-12 and m['intermediate_global_clip_steps']==0
        checks.append(dict(candidate=c['id'],rho=c['core_ou_correlation'],arrays_sha256=r['arrays_sha256'],actual_input_audit=m))
    rt.write(n.OUT/'core_ou_correlation_canary_audit.json',dict(status='PASS_ENGINEERING_ONLY',rho1_exact_array_parity=parity,checks=checks,resources=records,
        no_science_claim='500ms canaries validate implementation only. No formal correlation experiment dispatched or patient propagation conclusion.'))
    rt.write(n.OUT/'core_ou_correlation_canary_status.json',dict(status='CANARY_PASS_NO_FORMAL_DISPATCH',formal_selection='NOT_SELECTED'))
    print('CORE_OU_CANARIES_PASS; NO_FORMAL_DISPATCH',flush=True)


if __name__=='__main__':run_canaries()
