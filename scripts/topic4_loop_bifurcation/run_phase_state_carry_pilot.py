#!/usr/bin/env python3
"""Three source-phase updates carrying replica V/ref/M and their phase."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,shutil,subprocess,time
from pathlib import Path
import numpy as np
from campaign import ROOT,PYTHON,read,write,sha
import run_phase_source_pilot as implementation
import phase_history_sampler

OUT=ROOT/'high_history_phase_state_carry_pilot'
SOURCE=ROOT/'high_history_phase_source_closure'


def configure():
    contract=read(OUT/'contract.json')
    assert contract['map_sha256']==sha(implementation.__file__)
    assert contract['history_sampler_sha256']==sha(phase_history_sampler.__file__)
    implementation.OUT=OUT;implementation.CARRY_PHASE_HISTORY=True


def prepare():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    assert read(SOURCE/'pilot_result.json')['status']=='COMPLETE_THREE_BOUNDED_PHASE_MAP_UPDATES'
    for name in ['phase_replica_averaging_review','phase_replica_averaging_reset_control']:
        assert read(ROOT/name/'result.json')['status']=='COMPLETE_REPLICA_PHASE_AVERAGING_DECOMPOSITION'
    write(OUT/'contract.json',dict(status='REGISTERED_BEFORE_THREE_PHASE_HISTORY_UPDATES',created_epoch=time.time(),
        question='Does preserving phase-conditioned V/ref/M history retain the native coherent state under self-generated source feedback?',
        evidence='Paired80target localresponse atinputphase0 changesonlyinitialV/ref/M: E phaseprofileRMS frommatchednative10s is0.0103 fortruenativehistory versus0.0715 forreset, despite almostidenticalrates. Withinreplica phasevariance agreeswithnative; unconditionalresetting/averaging cannot be assumed tohave forgottenhistory.',
        design='Exactly3 source-statistic mapupdates,40000targets x64 numericalreplicas, fixed22step candidateperiod. Initialphase0 with true72s jointV/ref/M replicated acrossnumericalcopies. Eachsubsequentupdate carries the preceding outputreplicaV/ref/M and exactnextinputphase, plus self-generatedsourcephaseprobability andresidualPSD. No additional nativeforcing, parameter/period fitting or physicalequationchange.',
        remaining_approximation='Recurrentresidualinput remains stationaryindependent-sourceGaussian. Externalfilter initializedatstationarymean eachlocalassay then3-4sburn; this is not fullnativeengine continuation. Source-statistic iteration is notphysicaltime. Period/frequency/stability stilluncertified.',
        validation='Newhistorykernel must reproduce unchangedphasekernel everyspike/stat when supplied the sameuniforminitialparameters. Preserve phasealignment across eachrecord endpoint; count/exposure andresidualParseval verified eachbatch. Local80pairedhistorytest provides motivation, not allcellacceptance.',
        stop='At most3updates. Stop ifphase-specificcausalR crosses200, matchednativefieldRMS>10Hz oranycoremeandrift>10Hz. No automaticroot/periodsearch/frequencyresponse/nativeextra.',
        producer_sha256=sha(__file__),map_sha256=sha(implementation.__file__),history_sampler_sha256=sha(phase_history_sampler.__file__),
        formal_bifurcation_allowed=False))
    for name in ['operators','parameters.npz','filter_power.npy','original_ampa_jump.npz','original_gaba_jump.npz']:
        (OUT/name).symlink_to(SOURCE/name)
    first=OUT/'generation_0';first.mkdir()
    for name in ['phase_spike_probability.npy','source_rate_Hz.npy','residual_PSD.npy','current_mean.npy','current_phase_wave.npy']:
        (first/name).symlink_to(SOURCE/'generation_0'/name)
    raw=np.load(OUT/'parameters.npz');state=np.stack([raw['initial_V'],raw['initial_ref'],raw['initial_M']],axis=1)
    a=np.broadcast_to(state[:,None,:],(40000,implementation.R,3)).copy()
    np.save(first/'replica_V_ref_M.npy',a);np.save(first/'replica_initial_phase.npy',np.zeros((40000,implementation.R),dtype='i4'))
    assert np.array_equal(a[:,0,0],raw['initial_V']) and np.array_equal(a[:,0,1],raw['initial_ref']) and np.array_equal(a[:,0,2],raw['initial_M'])
    write(OUT/'initialization_qa.json',dict(status='PASS',actual72s_V_ref_M_exact=True,all_initial_phases_zero=True,
        input_source_statistics_identical_to_previous_pilot=True))
    for source,name in [(Path(__file__),'producer.py'),(Path(implementation.__file__),'map_producer.py'),(Path(phase_history_sampler.__file__),'history_sampler_producer.py')]:shutil.copy2(source,OUT/name)


def supervise():
    configure();assert not (OUT/'supervisor.json').exists()
    write(OUT/'supervisor.json',dict(status='RUNNING',pid=os.getpid(),completed_updates=0,updated_epoch=time.time()))
    R=implementation.R;N=implementation.N;L=implementation.L;results=[];stop=None
    raw=np.load(OUT/'parameters.npz');initial=np.load(OUT/'generation_0/source_rate_Hz.npy')
    for generation in range(1,4):
        folder=OUT/f'generation_{generation}';folder.mkdir()
        for name,shape,dtype in [('residual_PSD',(40000,N//2+1),'f8'),('source_rate_Hz',(40000,),'f8'),
            ('phase_spike_probability',(40000,L),'f8'),('replica_V_ref_M',(40000,R,3),'f8'),('replica_initial_phase',(40000,R),'i4')]:
            a=np.lib.format.open_memmap(folder/(name+'.npy'),mode='w+',dtype=dtype,shape=shape);a[:]=np.nan if dtype=='f8' else -1;a.flush();del a
        jobs=[]
        for part in range(2):
            log=(folder/f'worker_part{part}.log').open('w')
            process=subprocess.Popen([PYTHON,__file__,'worker','--generation',str(generation),'--part',str(part)],stdout=log,stderr=subprocess.STDOUT)
            jobs.append((process,log))
        codes=[]
        for process,log in jobs:codes.append(process.wait());log.close()
        if any(codes):
            write(OUT/'supervisor.json',dict(status='FAILED_WORKER',generation=generation,codes=codes));raise RuntimeError(codes)
        state=np.load(folder/'replica_V_ref_M.npy',mmap_mode='r');phase=np.load(folder/'replica_initial_phase.npy')
        assert np.isfinite(state).all() and phase.min()>=0 and phase.max()<L
        result=implementation.collect(generation)
        result.update(replica_V_ref_M_and_phase_carried=True,initial_native_state_supplied_only_once=True)
        write(folder/'complete.json',result);results.append(result)
        rate=np.load(folder/'source_rate_Hz.npy')
        if not result['G0_segment_consistent']:stop='PHASE_G_SEGMENT_CHANGED'
        if result['matched_native_field_RMS_Hz']>10:stop='NATIVE_SPATIAL_RELEVANCE_LOST'
        for q in [0,1]:
            mask=(np.arange(40000)<32000)&(raw['region']==q)
            if abs((rate-initial)[mask].mean())>10:stop='NATIVE_CORE_RELEVANCE_LOST'
        write(OUT/'supervisor.json',dict(status='RUNNING' if stop is None else 'STOPPED_AT_RELEVANCE_GUARD',pid=os.getpid(),completed_updates=generation,stop_reason=stop,updated_epoch=time.time()))
        if stop:break
    status='COMPLETE_THREE_PHASE_HISTORY_UPDATES' if stop is None else 'COMPLETE_EARLY_STOP_AT_RELEVANCE_GUARD'
    write(OUT/'result.json',dict(status=status,updates=results,stop_reason=stop,root_certified=False,physical_stability_established=False,formal_bifurcation_allowed=False))
    write(OUT/'supervisor.json',dict(status=status,completed_updates=len(results),stop_reason=stop,updated_epoch=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','supervise','worker']);parser.add_argument('--generation',type=int);parser.add_argument('--part',type=int)
    args=parser.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='supervise':supervise()
    else:configure();implementation.worker(args.generation,args.part)
