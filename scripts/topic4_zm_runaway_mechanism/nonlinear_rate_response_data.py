"""Independent local-LIF waveform calibration for a finite-state rate response.

No spatial network, Z/M path, old candidate-cycle waveform or native future
spike train supplies a training target. Validation profiles use separate
generator/noise seeds and are not opened by the fitting code.
"""
from common import OUT, BASE, np, read, write, log
from native_cycle_waveform_response import simulate
from lif_mc import condition
from pathlib import Path
from datetime import datetime
import argparse
import hashlib

DEST = OUT / 'nonlinear_rate_response'


def register():
    DEST.mkdir(exist_ok=True); path=DEST/'contract.json'; assert not path.exists()
    write(path,dict(created_local=datetime.now().astimezone().isoformat(),
          question='Can a differentiable finite-state nonlinear rate response retain local LIF reset/recovery dynamics and frequency response, before using it in the unchanged spatial network?',
          model=dict(history='Three normalized input moments drive12linear history states each:tau=[1,4,16,64]ms,orders1..3.36states per group,strictly stable local filter poles.',
                     normalization='u=[asinh((mu-Vreset)/(theta-Vreset)),log1p(vE/(theta-Vreset)^2),log1p(vI/(theta-Vreset)^2)]',
                     output='Smooth bounded nonlinear readout of u and h-u, separate E/I calibrations. Two64unit tanh layers; static LIF transfer supplies a baseline, with a separately calibrated static residual. All response functions are explicit and differentiable.',
                     interpretation='Approximate LIF-calibrated nonlinear response, not an exact derivation or a result asserted by the Brunel paper. No neuron particles or density grid are dynamic network states.',
                     spatial_invariants='Same spatial group projection, graph, threshold field, AMPA/GABA mean/variance operators, propagation delays, external-input definitions and Z/M equations. No fitting of onset times, Z/M trajectories or propagation targets.'),
          stage='LOCAL_CALIBRATION_ONLY; no autonomous spatial or bifurcation launch by this data script',
          source_linear_calibration=str(BASE/'dynamic_assay/rows.json'),
          source_static_calibration=str(OUT/'frozen_data/transfer_table'),
          new_data=dict(per_population=dict(train_periodic=96,train_startup=16,validation_periodic=24,validation_startup=8),
                        wave_samples=2048,phase_bins=128,dt_ms=.1,threshold_mv=18.,reset_mv=11.,
                        train_replicates=4096,validation_replicates=8192,
                        periodic_periods_ms=[100.,200.,400.],burn_cycles=3,record_cycles=8,startup_duration_ms=500.,
                        train_generator_seed=920041,validation_generator_seed=920042,
                        train_noise_seed=920043,validation_noise_seed=920044),
          families=['smooth multisine,low/moderate mean','positive burst then inhibition/recovery','smooth alternating plateaus','startup from reset and zero current'],
          sampling_unit='Independent local noise paths; input profiles are designed conditions. Common streams across profiles are retained, so frequency/profile counts are not independent biological samples.',
          training_budget='One fixed architecture and one optimizer schedule, random seed fixed before fitting. Calibrate only original training tables and train profiles. Do not fit original24waveforms,32validationworkpoints,64newvalidationprofiles or native network trajectories.',
          validation=dict(original_waveforms='All24retain waveformL2<=0.15 andmeanerror<=0.10',
                          original_linear='Original146eligibleAC:max14fail; original76eligibleDC retain per-row0.15. Signs and numerical checks separate. No relaxation to fit this candidate.',
                          fresh_waveforms='At least90percent of64newindependentprofiles satisfy the same waveform andmean gates; report all failures and near-zero-rate profiles.',
                          other_requirements='Verify history-state/derivative implementation, time-step/feature quadrature and full static response before any network acceptance. Original autonomous interictal/propagation/ZM gates remain required.'),
          stop='One bounded local candidate. A fit or waveform match does not accept the spatial model. If local validation fails, report the actual failure; no automatic architecture enlargement or onset fitting.'))


def make_wave(rng, family, T, W, startup=False):
    phase=np.arange(W)/W; t=phase*T
    if family==0:
        x=np.full(W,rng.uniform(-1.2,3.5))
        amplitude=np.exp(rng.uniform(np.log(.2),np.log(3.5)))
        for harmonic in range(1,5):
            x+=amplitude/harmonic**1.5*np.sin(2*np.pi*harmonic*phase+rng.uniform(0,2*np.pi))
    elif family==1:
        x=np.full(W,rng.uniform(-.3,3.))
        center=rng.uniform(.1,.4)*T
        distance=lambda center: (t-center+.5*T)%T-.5*T
        width=rng.uniform(.025,.10)*T
        x+=rng.uniform(8,90)*np.exp(-.5*(distance(center)/width)**2)
        center2=center+rng.uniform(.12,.35)*T
        x-=rng.uniform(3,40)*np.exp(-.5*(distance(center2)/(width*rng.uniform(.8,2.)))**2)
    else:
        low=rng.uniform(-30.,2.); high=rng.uniform(3.,90.)
        center=rng.uniform(-.5,.5); sharpness=rng.uniform(2.,8.)
        x=low+(high-low)*(.5+.5*np.tanh(sharpness*(np.sin(2*np.pi*phase+rng.uniform(0,2*np.pi))-center)))
    variances=[]
    for lo,hi in [(.12**2,3.2**2),(.02**2,6.**2)]:
        logv=np.full(W,rng.uniform(np.log(lo),np.log(hi)))
        for harmonic in [1,2,3]:
            logv+=rng.uniform(0,.7)/harmonic*np.sin(2*np.pi*harmonic*phase+rng.uniform(0,2*np.pi))
        # Smooth bounded log variance; bounds define the stimulus family,
        # not a state clamp in the candidate model.
        midpoint=.5*(np.log(lo)+np.log(hi));radius=.5*(np.log(hi)-np.log(lo))
        logv=midpoint+radius*np.tanh((logv-midpoint)/radius)
        variances.append(49.*np.exp(logv))
    if startup:
        # A finite trace, beginning at a constant operating point before a
        # smooth pulse. The acquired kernel uses only its first500ms cycle.
        baseline=rng.uniform(-.8,4.); center=rng.uniform(100.,300.);width=rng.uniform(10.,50.)
        x=baseline+rng.uniform(2.,50.)*np.exp(-.5*((t-center)/width)**2)
        if rng.uniform()<.5:x-=rng.uniform(2.,20.)*np.exp(-.5*((t-center-80.)/(1.5*width))**2)
    return np.array([11.+7.*x,*variances])


def prepare():
    c=read(DEST/'contract.json');meta=DEST/'profiles.json';assert not meta.exists()
    d=c['new_data'];rows=[];waves=[];pars=[]
    for split in ['train','validation']:
        rng=np.random.default_rng(d[split+'_generator_seed'])
        for pop in 'EI':
            for startup in [False,True]:
                count=d['per_population'][split+('_startup' if startup else '_periodic')]
                for number in range(count):
                    T=d['startup_duration_ms'] if startup else d['periodic_periods_ms'][(number//3)%3]
                    family=3 if startup else number%3
                    wave=make_wave(rng,family,T,d['wave_samples'],startup)
                    assert np.isfinite(wave).all() and wave[1:].min()>0
                    index=len(rows);rows.append(dict(id=index,split=split,pop=pop,startup=startup,
                        family=family,period_ms=T,replicates=d[split+'_replicates'],
                        burn_steps=0 if startup else round(d['burn_cycles']*T/d['dt_ms']),
                        # Exclude the endpoint of a finite startup trace:
                        # the periodic kernel would wrap t=T back to t=0.
                        record_steps=round((1 if startup else d['record_cycles'])*T/d['dt_ms'])-int(startup)))
                    waves.append(wave);pars.append(condition(0.,18.,1.,1.,pop,dt=d['dt_ms']))
    assert len(rows)==288 and sum(r['split']=='train' for r in rows)==224
    np.savez_compressed(DEST/'prepared.npz',wave=waves,pars=pars)
    write(meta,dict(status='REGISTERED_BEFORE_ACQUISITION',rows=rows,
                   training_profiles=224,validation_profiles=64,
                   source='Fresh synthetic inputs only; no currentSNN/candidatecyclewaveformcopied'))
    log('NONLINEAR RATE PROFILES PREPARED',len(rows))


def acquire(device):
    c=read(DEST/'contract.json');d=c['new_data'];info=read(DEST/'profiles.json');data=np.load(DEST/'prepared.npz')
    assert read(OUT/'native_cycle_waveform_response/implementation_check.json')['status']=='PASS'
    dest=DEST/'local_data';dest.mkdir(exist_ok=True)
    batches={}
    for row in info['rows']:
        key=(row['split'],row['period_ms'],row['burn_steps'],row['record_steps'],row['replicates'])
        batches.setdefault(key,[]).append(row)
    progress=dict(status='RUNNING',expected=288,completed=[])
    assert not (DEST/'acquisition.json').exists()
    write(DEST/'acquisition.json',progress)
    for key,rows in batches.items():
        split,T,burn,steps,R=key;dt=d['dt_ms'];B=d['phase_bins']
        phase=((np.arange(steps)+1)*dt/T)%1
        bins=np.minimum((phase*B).astype(int),B-1);exposure=np.bincount(bins,minlength=B)*dt
        assert exposure.min()>0
        for start in range(0,len(rows),8):
            chosen=rows[start:start+8];indices=[r['id'] for r in chosen]
            counts=simulate(data['pars'][indices],data['wave'][indices],R,T,dt,burn,steps,d[split+'_noise_seed'],B,device)
            for row,observed in zip(chosen,counts):
                index=row['id'];prefix=dest/f'profile{index:03d}';assert not prefix.with_suffix('.npz').exists()
                rates=observed/exposure[None,:]*1000
                # uint32 is retained, rather than relying on an assumed
                # per-bin count upper bound for a compressed integer type.
                np.savez_compressed(prefix.with_suffix('.npz'),counts=observed,
                    rate_hz=rates.mean(0),sem_hz=rates.std(0,ddof=1)/np.sqrt(R),exposure_ms=exposure,
                    period_ms=T,dt_ms=dt,record_steps=steps,burn_steps=burn,profile_id=index)
                progress['completed'].append(index)
            write(DEST/'acquisition.json',progress);log('LOCAL DATA',len(progress['completed']),288,split)
    progress['status']='COMPLETE';write(DEST/'acquisition.json',progress)
    write(DEST/'validation_separation.json',dict(status='ACQUIRED_NOT_USED_FOR_FITTING',
        train_ids=[r['id'] for r in info['rows'] if r['split']=='train'],
        validation_ids=[r['id'] for r in info['rows'] if r['split']=='validation'],
        scope='Do not inspect validation targets or select training settings from them. Lock model weights and predictions before scoring. Original24waveforms and32workpoints remain external validation.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--register',action='store_true');parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--device',type=int,default=0);a=parser.parse_args()
    if a.register:register()
    elif a.prepare:prepare()
    else:acquire(a.device)
