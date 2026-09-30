"""New independent validation only, with the parent's fixed stimulus family."""
from conditioned_refractory_rate import DEST,np,read,write,log
from nonlinear_rate_response_data import make_wave
from local_flux_recorder import capture,check_availability
from native_cycle_waveform_response import simulate
from lif_mc import condition
import argparse,os

def prepare():
    d=read(DEST/'contract.json')['new_data'];assert not (DEST/'profiles.json').exists();rng=np.random.default_rng(d['validation_generator_seed'])
    rows=[];waves=[];pars=[];W=2048;dt=.1
    for pop in 'EI':
        for startup in [False,True]:
            for j in range(8 if startup else 24):
                T=500. if startup else [100.,200.,400.][(j//3)%3];family=3 if startup else j%3
                wave=make_wave(rng,family,T,W,startup);wave[0]=11+1.3*(wave[0]-11);phase=np.arange(W)/W
                for ch in [1,2]:
                    gate=.5+.5*np.tanh(rng.uniform(1.5,5.)*(np.sin(2*np.pi*phase+rng.uniform(0,2*np.pi))-rng.uniform(-.5,.5)))
                    if ch==1:lo=rng.uniform(.2,.6);hi=rng.uniform(1.5,4.)
                    else:lo=[0.,.001,.01,.1][j%4];hi=[1.,3.,5.,8.][(j//4)%4]
                    wave[ch]=49*(lo+(hi-lo)*gate)**2
                k=len(rows);rows.append(dict(id=k,split='validation',pop=pop,startup=startup,family=family,period_ms=T,replicates=d['replicates'],
                    burn_steps=0 if startup else round(3*T/dt),record_steps=round((1 if startup else 8)*T/dt)-int(startup)))
                waves.append(wave);pars.append(condition(0.,18.,1.,1.,pop,dt=dt))
    np.savez_compressed(DEST/'prepared.npz',wave=waves,pars=pars)
    write(DEST/'profiles.json',dict(rows=rows,status='FRESH_VALIDATION_STIMULI_LOCKED_BEFORE_TARGETS',training_profiles=0))

def acquire(device):
    c=read(DEST/'contract.json')['new_data'];rows=read(DEST/'profiles.json')['rows'];inputs=np.load(DEST/'prepared.npz');dest=DEST/'local_data';dest.mkdir(exist_ok=True)
    progress=dict(status='RUNNING',expected=64,completed=[],pid=os.getpid());assert not (DEST/'acquisition.json').exists();write(DEST/'acquisition.json',progress)
    for row in rows:
        k=row['id'];T=row['period_ms'];R=row['replicates'];burn=row['burn_steps'];steps=row['record_steps'];dt=.1
        counts,fired,available=capture(inputs['pars'][k:k+1],inputs['wave'][k:k+1],R,T,dt,burn,steps,c['validation_noise_seed'],device)
        counts,fired,available=counts[0],fired[0],available[0];check_availability(fired,available,int(inputs['pars'][k,19]),R)
        if k in [0,31,32,63]:
            vanilla=simulate(inputs['pars'][k:k+1],inputs['wave'][k:k+1],R,T,dt,burn,steps,c['validation_noise_seed'],128,device)[0];assert np.array_equal(vanilla,counts)
        bins=np.minimum(((((np.arange(steps)+1)*dt/T)%1)*128).astype(int),127);exposure=np.bincount(bins,minlength=128)*dt;rate=counts/exposure[None,:]*1000
        path=dest/f'profile{k:03d}.npz';assert not path.exists()
        np.savez_compressed(path,phase_counts=counts,spike_counts=fired,available_counts=available,rate_hz=rate.mean(0),sem_hz=rate.std(0,ddof=1)/np.sqrt(R),
            exposure_ms=exposure,period_ms=T,dt_ms=dt,burn_steps=burn,record_steps=steps,replicates=R,nref_steps=int(inputs['pars'][k,19]))
        progress['completed'].append(k);write(DEST/'acquisition.json',progress)
        if len(progress['completed'])%8==0:log('CONDITIONED FRESH DATA',len(progress['completed']),64)
    progress['status']='COMPLETE';write(DEST/'acquisition.json',progress)
    write(DEST/'recording_qa.json',dict(status='PASS',profiles=64,availability_identity=True,vanilla_bitwise_sentinels=[0,31,32,63],training_data_added=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','acquire']);p.add_argument('--device',type=int,default=0);a=p.parse_args();{'prepare':prepare,'acquire':lambda:acquire(a.device)}[a.command]()
