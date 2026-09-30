#!/usr/bin/env python3
"""Paired existing native trajectories: early K accumulation versus G carryover."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import shutil,time
import numpy as np
from campaign import ROOT,read,write,sha
from analyze_feedback_tail import SOURCE,load
from run_topic4_recovery_window import assert_same_state
import run_topic4_loop_zk_conditional as native

OUT=ROOT/'feedback_build_up_review_v2'


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    write(OUT/'contract.json',dict(status='REGISTERED_BEFORE_EXISTING_PAIRED_BUILDUP_AUDIT',created_epoch=time.time(),
        question='Does changingG response time affect K accumulation before lowactivity, in addition to leaving a Gtail afterward?',
        design='Read the two existingpairednative120s seeds8402/8403, G30 tauG0 versus0.5. Verifyjob differences are onlyname/tauG and20s exogenousstate plus recordedinputidentity. Compare firstentry-aligned0.5,1,2s windows (one,two,four feedbacktimeconstants), then the common window ending at delayedcondition firstsustainedR<=5. Original1ms causalR/G/K; no input reconstruction ornewnative simulation.',
        interpretation='PairedtauG intervention establishes a total effect; temporalbuildup and Gtail are candidate mediators. Do not identify their individual necessity from this descriptive comparison or transfer conditionalKthresholds across spatialfields. Separate onsetovershoot, delayedqoffGcarryover and subsequentZrecovery.',
        unit='Two reusednative seeds, not newreplicates. Windowmeans andpeaks are withintrajectory descriptions, not seed uncertainty.',
        prefix_QA='Use R(t)<=R(nextsample)*exp(1ms/15ms), from nonnegativespikeincrements, to certify the entire prefeedbackprefix. The firstsample withq>0 canalreadycontain a feedbackeffect between0.1msupdates; the earlier assertion wronglyincluded it. Originalfailure andproducer preserved; no numericaltolerance is loosened.',
        producer_sha256=sha(__file__),formal_bifurcation_allowed=False))
    tail=read(ROOT/'feedback_tail_mechanism/analysis.json');allrows=[]
    for seed in [9108402,9108403]:
        names=[f'G30_response{tau:g}_s{seed}' for tau in [0.,.5]]
        folders=[SOURCE/'runs'/n for n in names];jobs=[read(p/'result.json')['job'] for p in folders]
        differences={k:[jobs[0].get(k),jobs[1].get(k)] for k in jobs[0].keys()|jobs[1].keys() if jobs[0].get(k)!=jobs[1].get(k)}
        assert set(differences)=={'name','global_tau_s'}
        inputs=[load(p,'chunks',['inputs'])['inputs'] for p in folders];assert np.array_equal(*inputs)
        states=[native.read_pickle(p/'states/t20s.pkl')['engine'] for p in folders]
        fields=['rng_state','xi','external_drive']
        assert_same_state({k:states[0][k] for k in fields},{k:states[1][k] for k in fields})
        paths=[]
        for name,p in zip(names,folders):
            m=load(p,'mechanism_chunks',['time_ms','global_E_rate_Hz','global_raw_conductance_ratio'])
            a=load(p,'intrinsic_adaptation_chunks',['time_ms','sahp_mean_conductance_ratio'])
            g=load(p,'global_response_chunks',['time_ms','q','G_raw'])
            assert np.array_equal(m['time_ms'],a['time_ms']) and np.array_equal(m['time_ms'],g['time_ms'])
            t=m['time_ms']/1000;R=m['global_E_rate_Hz'];G=m['global_raw_conductance_ratio'];K=a['sahp_mean_conductance_ratio'];q=g['q']
            assert np.array_equal(G,g['G_raw'])
            paths.append(dict(t=t,R=R,G=G,K=K,q=q))
        old=[next(q for q in tail['rows'] if q['name']==name) for name in names]
        onset=old[0]['onset_s'];assert onset==old[1]['onset_s']
        upper=np.maximum(paths[0]['R'][1:],paths[1]['R'][1:])*np.exp(.001/.015)
        begin=int(np.flatnonzero(upper>=200)[0])+1
        assert (upper[:begin-1]<200).all()
        assert np.array_equal(paths[0]['R'][:begin],paths[1]['R'][:begin])
        assert np.array_equal(paths[0]['K'][:begin],paths[1]['K'][:begin])
        low=old[1]['first_R_below5_for100ms_s'];assert low is not None
        intervals=[(onset,onset+v) for v in [.5,1.,2.]]+[(onset,low)]
        summaries=[]
        for name,d in zip(names,paths):
            windows=[]
            for lo,hi in intervals:
                m=(d['t']>=lo)&(d['t']<hi);i=np.searchsorted(d['t'],lo);j=np.searchsorted(d['t'],hi)
                windows.append(dict(absolute_window_s=[lo,hi],relative_to_entry_s=[lo-onset,hi-onset],
                    R_mean_peak_Hz=[float(d['R'][m].mean()),float(d['R'][m].max())],
                    G_mean_peak=[float(d['G'][m].mean()),float(d['G'][m].max())],
                    K_start_end_peak=[float(d['K'][i]),float(d['K'][j]),float(d['K'][m].max())],
                    gate_mean=float(d['q'][m].mean()),G_with_q_zero_duration_s=float(((d['G']>0)&(d['q']==0)&m).sum()*.001)))
            summaries.append(dict(name=name,windows=windows,
                at_delayed_first_R5=dict(R_Hz=float(d['R'][np.searchsorted(d['t'],low)]),G=float(d['G'][np.searchsorted(d['t'],low)]),K=float(d['K'][np.searchsorted(d['t'],low)]))))
            mask=(d['t']>=onset-.5)&(d['t']<=low+2)
            np.savez_compressed(OUT/f'{name}.npz',absolute_time_s=d['t'][mask],relative_to_entry_s=d['t'][mask]-onset,
                causal_R_Hz=d['R'][mask],Graw=d['G'][mask],mean_K=d['K'][mask],q=d['q'][mask])
        allrows.append(dict(seed=seed,job_differences=differences,recorded_future_input_bitwise=True,twenty_second_exogenous_state_bitwise=True,
            complete_certified_feedback_off_prefix_R_K_bitwise=True,certified_feedback_off_prefix_end_s=float(paths[0]['t'][begin-1]),onset_s=onset,delayed_first_R5_s=low,conditions=summaries))
    result=dict(status='COMPLETE_EXISTING_PAIRED_FEEDBACK_BUILDUP_AUDIT',rows=allrows,formal_bifurcation_allowed=False,
        scope=read(OUT/'contract.json')['interpretation'],producer_sha256=sha(__file__))
    write(OUT/'result.json',result);shutil.copy2(__file__,OUT/'producer.py');print(result,flush=True)


if __name__=='__main__':main()
