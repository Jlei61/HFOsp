#!/usr/bin/env python3
"""Resume unchanged physical trials; use only the repaired observer for reports."""
from pathlib import Path
import fcntl,json,os,shutil,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_topic4_multidimensional_pilot as pilot
from scripts.resume_topic4_multidimensional_pilot import prepare_resume
from scripts.qualify_topic4_observation_repair import OUT,OLD,read,write,sha


def verified_lock():
    path=OUT/'resume_lock.json'
    if not path.exists():
        q=read(OUT/'qualification.json');audit=read(OUT/'reobserved_workers.json');tests=read(OUT/'implementation_verification.json')
        if not all(q['checks'].values()) or not all(audit[k] for k in ('all_prefix_checks_pass','all_future_peak_checks_pass')) or not tests['pass']:
            raise RuntimeError('repair qualification has not passed')
        sources=['src/topic4_observation_repaired.py','src/topic4_interictal_repaired_evaluation.py',
                 'scripts/qualify_topic4_observation_repair.py','scripts/report_topic4_observation_repair.py',
                 'scripts/resume_topic4_repaired_observation.py','tests/test_topic4_observation_repaired.py']
        inputs=[OUT/n for n in ('observation_contract.json','qualification.json','evaluator.pkl','implementation_verification.json')]
        write(path,{'source_hashes':{p:sha(ROOT/p) for p in sources},'input_hashes':{str(p):sha(p) for p in inputs},
              'created_unix':time.time(),'simulation_design_changed':False,'development_only':True,
              'initial_completed_jobs':len(audit['workers']),'periodic_analysis_after_at_least_new_jobs':8})
    lock=read(path)
    pilot.base.verify_sources(lock['source_hashes'])
    for p,h in lock['input_hashes'].items():
        if sha(p)!=h:raise RuntimeError('frozen repair input changed: '+p)
    return lock


def state(name,**kw):
    write(OUT/'status.json',{'status':name,'updated_unix':time.time(),'raw_execution_status':str(OLD/'status.json'),
          'final_substrate_frozen':False,'fig5_released':False,**kw})


def analyze():
    guard=open(OUT/'analysis.lock','a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verified_lock();env={**pilot.base.ENV,'LD_LIBRARY_PATH':'/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib'}
    for script,args in [('qualify_topic4_observation_repair.py',['--rescore-only']),('report_topic4_observation_repair.py',[])]:
        subprocess.run([pilot.base.PYTHON,str(ROOT/'scripts'/script),*args],cwd=ROOT,env=env,check=True)
    audit=read(OUT/'reobserved_workers.json')
    if not audit['all_prefix_checks_pass'] or not audit['all_future_peak_checks_pass']:raise RuntimeError('new trajectory violated observer invariants')
    n=len(audit['workers']);folder=OUT/'milestones'/f'{n:03d}_completed';folder.mkdir(parents=True,exist_ok=True)
    for name in ('before_after_parameter_comparison.json','paired_parameter_comparison.json','actual_N_patient_reference.json','candidate_metrics.csv'):
        shutil.copy2(OUT/name,folder/name)
    shutil.copy2(OUT/'figures/observation_repair_comparison.png',folder/'observation_repair_comparison.png')
    write(OUT/'latest_analysis.json',{'status':'COMPLETE','n_completed_jobs':n,'updated_unix':time.time(),'snapshot':str(folder),
          'full_model_acceptance_qualified':False,'automatic_next_round':False})


def main():
    if '--analyze' in sys.argv:analyze();return
    guard=open(OLD/'controller.lock','a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    physical=prepare_resume();repair=verified_lock()
    if '--prepare-only' in sys.argv:
        print(json.dumps({'status':'REPAIRED_OBSERVATION_RESUME_READY','completed_reused':repair['initial_completed_jobs']}));return
    design=read(OLD/'design.json');cp,mp,sp=pilot.prepare_phase('paired_round1',design['candidates'],design['network_and_dynamics_seeds'],design['duration_ms'],physical['source_hashes'])
    folder=cp.parent;workers=folder/'workers';logs=folder/'run_logs'
    def output(job):return workers/f'{job[0]}_seed_{job[1]}.json'
    jobs=[(r['candidate_id'],s) for r in design['candidates'] for s in design['network_and_dynamics_seeds']]
    pending=[j for j in jobs if not pilot.worker_complete(output(j),sp)];complete=len(jobs)-len(pending)
    active={};failures=[];analysis=None;analysis_log=None;last=complete
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    env={**pilot.base.ENV,'LD_LIBRARY_PATH':'/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib','CUDA_VISIBLE_DEVICES':''}
    write(OLD/'observation_repair_resumption.json',{'status':'RESUMED_WITH_REPAIRED_ANALYSIS','time':time.time(),
          'repaired_output':str(OUT),'old_pause_record_preserved':True,'raw_completed_reused':complete})
    while pending or active:
        for j,(proc,log) in list(active.items()):
            code=proc.poll()
            if code is None:continue
            log.close();del active[j]
            if code or not pilot.worker_complete(output(j),sp):failures.append({'job':j,'exit_code':code})
            else:complete+=1
        if analysis is not None and analysis.poll() is not None:
            if analysis.returncode:failures.append({'analysis_exit_code':analysis.returncode})
            analysis_log.close();analysis=None
        if not failures:
            try:pilot.base.verify_sources(physical['source_hashes']);verified_lock()
            except Exception as exc:failures.append({'contract_error':repr(exc)})
        if failures:
            state('FAILURE_DRAINING_NO_NEW_DISPATCH',failures=failures,complete=complete,running=len(active))
            pilot.status('FAILURE_DRAINING_NO_NEW_DISPATCH',failures=failures,complete=complete,running=len(active))
            if not active:break
            time.sleep(10);continue
        if analysis is None and complete-last>=8:
            last=complete;analysis_log=open(OUT/f'analysis_after_{complete:03d}.log','w')
            analysis=subprocess.Popen(['/usr/bin/prlimit',f'--as={8*1024**3}','--',pilot.base.PYTHON,__file__,'--analyze'],cwd=ROOT,env=env,stdout=analysis_log,stderr=subprocess.STDOUT)
        allowance=max(0,int((pilot.base.available_gib()-40-pilot.outstanding_snn_reserve())/18))
        slots=min(design['maximum_workers']-len(active),allowance,len(pending)) if shutil.disk_usage(OLD).free>30*1024**3 else 0
        for _ in range(slots):
            j=pending.pop(0);log=open(logs/f'{j[0]}_seed_{j[1]}.log','w')
            cmd=['/usr/bin/prlimit',f'--as={18*1024**3}','--',pilot.base.PYTHON,str(pilot.WORKER),'--config',str(cp),
                 '--candidate-id',j[0],'--seed',str(j[1]),'--expected-commit',commit,'--runtime-manifest',str(sp),
                 '--artifact-root',str(pilot.base.ART),'--out-json',str(output(j)),'--out-npz',str(output(j).with_suffix('.npz'))]
            active[j]=(subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT),log)
        kw={'total':len(jobs),'complete':complete,'running':len(active),'pending':len(pending),'analysis_running':analysis is not None,
            'memory_available_gib':pilot.base.available_gib(),'active':[{'candidate_id':j[0],'seed':j[1],'pid':p.pid} for j,(p,f) in active.items()]}
        name='RUNNING_REPAIRED_PAIRED_ROUND1' if active else 'WAITING_RESOURCE_ADMISSION'
        pilot.status(name,**kw);state(name,**kw)
        if pending or active:time.sleep(10)
    if analysis is not None:
        code=analysis.wait();analysis_log.close()
        if code:failures.append({'analysis_exit_code':code})
    if failures:raise RuntimeError(str(failures))
    write(folder/'completion.json',{'status':'COMPLETE','jobs':len(jobs),'runtime_sha256':sha(sp)})
    state('ANALYZING_COMPLETED_ROUND',complete=complete)
    latest=read(OUT/'latest_analysis.json') if (OUT/'latest_analysis.json').exists() else {}
    if latest.get('n_completed_jobs')!=complete:analyze()
    pilot.status('ROUND1_COMPLETE_REPAIRED_ANALYSIS_PENDING_SCIENTIFIC_REVIEW',complete=complete,analysis_output=str(OUT))
    state('ROUND1_COMPLETE_PENDING_SCIENTIFIC_REVIEW',complete=complete,automatic_next_round=False)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if '--analyze' not in sys.argv:state('ERROR_PAUSED',error=repr(exc))
        raise
