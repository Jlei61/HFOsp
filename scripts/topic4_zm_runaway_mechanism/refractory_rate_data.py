"""Bounded additional stimulus coverage, independent of spatial onset targets."""
from refractory_rate_response import DEST,OUT,np,read,write,log
from nonlinear_rate_response_data import make_wave
from local_flux_recorder import capture,check_availability
from native_cycle_waveform_response import simulate
from lif_mc import condition
from datetime import datetime
import argparse,hashlib,os

PARENT=OUT/'nonlinear_rate_response'

def register():
    assert read(DEST/'implementation_check.json')['status']=='PASS'
    path=DEST/'contract.json';assert not path.exists()
    write(path,dict(created_local=datetime.now().astimezone().isoformat(),
        status='LOCKED_BEFORE_ADDITIONAL_TARGETS_AND_FIT',
        model='Continuous refractory renewal flux r=rho*(1-integral of r over tref),rho=exp(ell)/0.1ms.6exact current-covariance states,36linear input-history states,39-to64tanh-to64tanh-to1 residual log-intensity readout, separate E/I. No particle or voltage grid in network state.',
        numerical_bridge='Implicit Euler: p=sigmoid(ell+log(dt/0.1)); flux=p times own available fraction/dt. At0.1ms this agrees with native decrement-before-membrane refractory bookkeeping. It is an approximate conditional LIF response, not an exact derivation.',
        training_sources='Original static and dynamic calibration plus224existing TRAIN profiles and256additional TRAIN profiles. Observed availability used only as local conditional-flux calibration labels. No native spatial future firing, onset time, Z/M or propagation targets.',
        validation_sources='Original24waveforms and32workpoints; previous64validationprofiles are reused diagnostics and no longer novel.64additional unseen profiles are independent validation. Predictions and weights locked before scoring.',
        new_data=dict(train_periodic_per_pop=120,train_startup_per_pop=8,validation_periodic_per_pop=24,validation_startup_per_pop=8,
            W=2048,B=128,dt=.1,periods=[100.,200.,400.],burn_cycles=3,record_cycles=8,
            train_R=4096,validation_R=8192,train_generator_seed=920061,validation_generator_seed=920062,
            train_noise_seed=920063,validation_noise_seed=920064,
            mean='1.3 times existing normalized-mean waveform about reset; approximate range -50to120 in normalized units',
            variance='Smooth sigma_E plateaus0.2to4; sigma_I between [0,.001,.01,.1] and[1,3,5,8], balanced deterministic assignment; random phase,width and smoothness independent of targets.'),
        training=dict(seed=920065,steps_per_pop=12000,threads=2,optimizer='AdamW,weight_decay1e-6,clipnorm10',
            learning_rate=[[0,.001],[4000,.0003],[8000,.0001]],
            sample='2048 equally spaced record steps/profile; all profiles equally represented; no validation checkpoint selection',
            batch=dict(flux=512,static=128,linear=64),
            loss='10*mean((available*softplus(ell)-fired*ell-empirical_entropy)/max(profile_mean_fired,.001)) + static squared asinh(rate/.1)/.1 difference + original normalized complex gain error. Static/linear evaluated on same readout, covariance and refractory dynamics.',
            boundary='Fixed schedule and architecture; no automatic extra training or architecture enlargement after validation.'),
        acceptance=dict(original_waveforms='24/24 waveformL2<=.15,meanerror<=.10',
            fresh_waveforms='At least58/64 with same thresholds',numerical='Per-profile0.1to0.05ms prediction difference<=.02',
            linear='Original finite-amplitude and per-row DC/AC gates unchanged; analytic gains only preliminary diagnostics',
            spatial='Original interictal,propagation,Z/M-path gates remain required; local pass does not promote model or certify bifurcation.'),
        invariant='Spatial graph,geometry,thresholds,mean/variance synapses,delays,external-input definition and Z/M laws remain fixed.'))

def prepare():
    d=read(DEST/'contract.json')['new_data'];assert not (DEST/'profiles.json').exists()
    rows=[];waves=[];pars=[]
    for split in ['train','validation']:
        rng=np.random.default_rng(d[split+'_generator_seed'])
        for pop in 'EI':
            for startup in [False,True]:
                count=d[split+('_startup' if startup else '_periodic')+'_per_pop']
                for j in range(count):
                    T=500. if startup else d['periods'][(j//3)%3];family=3 if startup else j%3
                    wave=make_wave(rng,family,T,d['W'],startup);wave[0]=11+1.3*(wave[0]-11)
                    phase=np.arange(d['W'])/d['W']
                    for channel in [1,2]:
                        gate=.5+.5*np.tanh(rng.uniform(1.5,5.)*(np.sin(2*np.pi*phase+rng.uniform(0,2*np.pi))-rng.uniform(-.5,.5)))
                        if channel==1:lo=rng.uniform(.2,.6);hi=rng.uniform(1.5,4.)
                        else:lo=[0.,.001,.01,.1][j%4];hi=[1.,3.,5.,8.][(j//4)%4]
                        wave[channel]=49*(lo+(hi-lo)*gate)**2
                    index=len(rows);rows.append(dict(id=index,split=split,pop=pop,startup=startup,family=family,
                        period_ms=T,replicates=d[split+'_R'],burn_steps=0 if startup else round(3*T/d['dt']),
                        record_steps=round((1 if startup else 8)*T/d['dt'])-int(startup)))
                    waves.append(wave);pars.append(condition(0.,18.,1.,1.,pop,dt=d['dt']))
    assert len(rows)==320 and sum(r['split']=='train' for r in rows)==256
    np.savez_compressed(DEST/'prepared.npz',wave=waves,pars=pars)
    write(DEST/'profiles.json',dict(rows=rows,status='STIMULI_LOCKED_BEFORE_ACQUISITION'))

def acquire(device):
    d=read(DEST/'contract.json')['new_data'];dest=DEST/'local_data';dest.mkdir(exist_ok=True)
    # Shared helper must reproduce the already checked observer exactly.
    original=np.load(PARENT/'prepared.npz');meta=read(PARENT/'profiles.json')['rows'];checks=[]
    for index in [0,1,2,83,112,113,114,204]:
        row=meta[index];a,b,c=capture(original['pars'][index:index+1],original['wave'][index:index+1],4096,
            row['period_ms'],.1,row['burn_steps'],row['record_steps'],920043,device)
        z=np.load(OUT/f'local_refractory_flux/training_data/profile{index:03d}.npz')
        assert np.array_equal(a[0],z['phase_counts']) and np.array_equal(b[0],z['spike_counts']) and np.array_equal(c[0],z['available_counts'])
        checks.append(index)
    write(DEST/'recorder_helper_check.json',dict(status='PASS',bitwise_profiles=checks))
    rows=read(DEST/'profiles.json')['rows'];data=np.load(DEST/'prepared.npz')
    progress=dict(status='RUNNING',expected=320,completed=[],pid=os.getpid());assert not (DEST/'acquisition.json').exists()
    write(DEST/'acquisition.json',progress)
    for row in rows:
        k=row['id'];T=row['period_ms'];R=row['replicates'];burn=row['burn_steps'];steps=row['record_steps'];dt=d['dt']
        count,fire,avail=capture(data['pars'][k:k+1],data['wave'][k:k+1],R,T,dt,burn,steps,d[row['split']+'_noise_seed'],device)
        count,fire,avail=count[0],fire[0],avail[0]
        check_availability(fire,avail,int(data['pars'][k,19]),R)
        if k in [0,127,128,255,256,287,288,319]:
            vanilla=simulate(data['pars'][k:k+1],data['wave'][k:k+1],R,T,dt,burn,steps,d[row['split']+'_noise_seed'],128,device)[0]
            assert np.array_equal(vanilla,count)
        bins=np.minimum(((((np.arange(steps)+1)*dt/T)%1)*128).astype(int),127)
        exposure=np.bincount(bins,minlength=128)*dt;rate=count/exposure[None,:]*1000
        path=dest/f'profile{k:03d}.npz';assert not path.exists()
        np.savez_compressed(path,phase_counts=count,spike_counts=fire,available_counts=avail,
            rate_hz=rate.mean(0),sem_hz=rate.std(0,ddof=1)/np.sqrt(R),exposure_ms=exposure,dt_ms=dt,
            period_ms=T,burn_steps=burn,record_steps=steps,replicates=R,profile_id=k,nref_steps=int(data['pars'][k,19]))
        progress['completed'].append(k);write(DEST/'acquisition.json',progress)
        if (k+1)%16==0:log('REFRACTORY ADDITIONAL DATA',k+1,320,row['split'])
    progress['status']='COMPLETE';write(DEST/'acquisition.json',progress)
    write(DEST/'recording_qa.json',dict(status='PASS',availability_identity_all320=True,vanilla_bitwise_sentinels=[0,127,128,255,256,287,288,319],validation_used_for_fit=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','prepare','acquire']);p.add_argument('--device',type=int,default=0);a=p.parse_args()
    {'register':register,'prepare':prepare,'acquire':lambda:acquire(a.device)}[a.command]()
