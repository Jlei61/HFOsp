#!/usr/bin/env python3
"""Read-only simulation audit; repair final analysis/status, never the trajectory."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import json
import pickle
import shutil
import time
from pathlib import Path
import numpy as np
import analyze_topic4_interictal_recurrence as audit

ROOT=audit.OUT


def main():
    protocol=json.loads((ROOT/'protocol.json').read_text())
    archive=ROOT/'before_finalization_repair';archive.mkdir(exist_ok=True)
    for name in ['status.json','summary.json','batch_complete.json','scientific_review.md','postprocess_complete.json']:
        if (ROOT/name).exists() and not (archive/name).exists():shutil.copy2(ROOT/name,archive/name)
    rows=[];checks=[]
    for job in protocol['initial_jobs']:
        folder=ROOT/'runs'/job['name'];result=json.loads((folder/'result.json').read_text())
        end_step=0
        for file in sorted((folder/'chunks').glob('*.npz')):
            if '.tmp.' in file.name:continue
            with np.load(file) as a:
                assert int(a['start_step'])==end_step;end_step=int(a['end_step'])
                assert a['spikes_1ms'][:,0].sum()==a['regions_1ms'][:,:3].sum()==a['field_5ms'].sum()
                assert a['spikes_1ms'][:,1].sum()==a['regions_1ms'][:,3:].sum()
                for key in ['Z','M','currents','regional_currents']:
                    assert np.isfinite(a[key]).all(),(job['name'],file.name,key)
        with (folder/'checkpoint.pkl').open('rb') as h:saved=pickle.load(h)
        assert saved['engine']['step']==end_step==round(result['end_s']*10000)
        assert saved['job']==job
        log=(ROOT/'logs'/f"{job['name']}.log").read_text()
        assert 'TypeError: Object of type ndarray is not JSON serializable' in log
        assert "fixed.carrier.base.write(OUT/'analysis'/f'{name}.json',row)" in log
        row=audit.analyze_folder(folder,ROOT/'geometry.npz')
        assert row['primary']['observed_s']==result['end_s']
        horizon_reached=result['end_s']>=job['horizon_s']
        row.update(observation_horizon_reached=horizon_reached,
                   finalization_repaired=True,simulation_data_unchanged=True)
        d=audit.old.load(folder,keys=['spikes_1ms','slow_time_ms','Z','M'])
        k=audit.old.load(folder,'intrinsic_adaptation_chunks')
        durations=[e['duration_s'] for e in row['primary']['preentry']['brief_events']]
        row['state_summary']=dict(late_E_Hz=float(d['spikes_1ms'][-5000:,0].sum()/32000/5),
               final_Z=float(d['Z'][-1,0]),final_K=float(k['sahp_mean_conductance_ratio'][-1]),
               median_preentry_brief_ms=float(np.median(durations)*1000) if durations else None)
        # Regression check of the repaired writer with a real ndarray-bearing result.
        audit.old.write(ROOT/'analysis'/f"{job['name']}.json",row)
        reread=json.loads((ROOT/'analysis'/f"{job['name']}.json").read_text())
        assert reread['exact_all_E_zero_interval_s']==row['exact_all_E_zero_interval_s'].tolist()
        pp=row['primary']
        summary=dict(name=job['name'],status='OBSERVATION_HORIZON_REACHED' if horizon_reached else 'WALL_TIME_CENSORED',
              runtime_status=result['status'],observed_s=pp['observed_s'],classification=pp['classification'],
              preentry_brief=pp['preentry']['brief_count'],temporal_pass=pp['temporal_loop_pass'],
              gamma=job['gamma'],K=job['sahp_gain'],tau_K_s=job['sahp_tau_s'],
              entries=pp['entries'],exits=pp['low_activity_exits'],**row['state_summary'])
        rows.append(summary)
        checks.append(dict(name=job['name'],continuous_chunks=True,spike_count_conservation=True,
                     finite_slow_states=True,checkpoint_matches_recorded_end=True,
                     observation_horizon_reached=horizon_reached,observed_s=pp['observed_s']))
    report=dict(repaired_epoch=time.time(),status='PASS',rows=checks,
          failure_location='Final analysis JSON write after trajectory/checkpoint/result had already been saved.',
          fix='Use existing NumPy-safe analysis writer; recompute all final analyses from committed data.',
          original_logs_and_results_preserved=True,no_simulation_rerun=True,
          no_criterion_change=True,n_horizon_reached=sum(c['observation_horizon_reached'] for c in checks),
          n_short_wall_censored=sum(not c['observation_horizon_reached'] for c in checks))
    audit.old.write(ROOT/'finalization_repair.json',report)
    audit.old.write(ROOT/'summary.json',dict(updated_epoch=time.time(),rows=rows,
         scope='Six conditions, one fixed topology and noise. Four60s records, two56s wall-time-censored records.',
         full_Fig5_acceptance='NOT_ESTABLISHED',finalization_repaired=True))
    prior=json.loads((archive/'status.json').read_text())
    audit.old.write(ROOT/'status.json',dict(updated_epoch=time.time(),running={},queued=[],
          finished=6,completed_observation_horizon=report['n_horizon_reached'],
          wall_time_censored=report['n_short_wall_censored'],completed=report['n_horizon_reached'],
          failures=[],historical_postprocessing_failures=prior['failures'],finalization_repaired=True,
          temporal_loop_passes=sum(r['temporal_pass'] for r in rows),no_new_rounds=True,
          deadline_epoch=protocol['deadline_epoch']))
    audit.old.write(ROOT/'batch_complete.json',dict(updated_epoch=time.time(),rows=rows,
          finished=6,all_six_complete=False,n_observation_horizon_reached=report['n_horizon_reached'],
          n_wall_time_censored=report['n_short_wall_censored'],finalization_repaired=True,
          full_Fig5_acceptance='NOT_ESTABLISHED'))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
