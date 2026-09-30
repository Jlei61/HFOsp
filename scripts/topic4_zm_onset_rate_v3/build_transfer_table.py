"""Static colored-noise transfer table phi(x, sigma_E, sigma_I) per population, CRN Monte Carlo.

x=(mu-V_reset)/(theta-V_reset); sigma_c = sqrt(v_c)/(theta-V_reset) with v_c the diffusion
variance tau_m sum J^2 nu (mV^2). Threshold scaling invariance was verified exactly
(lif_mc.py). Table built at theta=18 (theta-V_reset=7 mV). Common random numbers across
conditions (same replicate -> same stream) keep the sampled surface smooth.
"""
from lif_mc import *
import argparse

def x_grid():
    u=np.r_[np.arange(np.arcsinh(-25.),-1.5-.1,.2),np.arange(-1.5,2.2,.08),np.arange(2.2+.08,np.arcsinh(200.)+.2,.2)]
    return np.sinh(u)
SIGMA_E={'E':np.array([0,.05,.1,.15,.2,.25,.3,.35,.4,.5,.6,.7,.85,1.,1.2,1.4,1.7,2.,2.4,2.8,3.3,3.9,4.6,5.5,6.5]),
         'I':np.array([0,.05,.1,.15,.2,.25,.3,.35,.4,.5,.6,.7,.85,1.,1.2,1.4,1.7,2.,2.4,2.8,3.3,3.9])}
SIGMA_I={'E':np.array([0,.05,.1,.15,.2,.25,.3,.35,.4,.5,.6,.7,.85,1.,1.2,1.4,1.7,2.,2.4,2.8,3.3,3.9,4.6,5.5,6.5,7.5,8.5]),
         'I':np.array([0,.05,.1,.15,.2,.25,.3,.35,.4,.5,.6,.7,.85,1.,1.2,1.4,1.7,2.,2.4,2.8,3.3,3.9,4.6,5.5,6.5])}
THETA=18.;VR=11.;SCALE=THETA-VR

def conditions(pop):
    X=x_grid();SE=SIGMA_E[pop];SI=SIGMA_I[pop];pars=[]
    for x in X:
        for se in SE:
            for si in SI:
                pars.append(condition(VR+SCALE*x,THETA,(SCALE*se)**2,(SCALE*si)**2,pop))
    return X,SE,SI,np.array(pars)

def main(a):
    folder=DEST/'transfer_table';folder.mkdir(parents=True,exist_ok=True)
    contract=folder/'contract.json'
    if not contract.exists():
        write(contract,dict(status='REGISTERED_BEFORE_RUN',registered=time.strftime('%Y-%m-%d %H:%M:%S'),
            question='Stationary rate of the native-discretised LIF under Gaussian-diffusion AMPA/GABA input as a smooth function of (mean, AMPA variance, GABA variance) per population',
            readout='spikes per replicate over duration_ms after burn_ms; rate = mean count / duration',
            statistical_unit='independent noise path (replicate); CRN across conditions',
            replicates=a.replicates,duration_ms=a.duration,burn_ms=a.burn,seed=a.seed,dt_ms=DT,
            grids=dict(x=x_grid(),sigma_E=SIGMA_E,sigma_I=SIGMA_I,theta=THETA,V_reset=VR),
            validation_plan='Held-out workpoints not on the grid (the 8 v2-assayed points + random draws) compared with the spline interpolant; tolerance registered in validation_contract.json',
            no_fitting='Table values are raw MC means; the interpolant is fitted only to be a smooth representation of the same data'))
    for pop in a.pops:
        out=folder/f'table_{pop}.npz'
        if out.exists():log('exists',out);continue
        X,SE,SI,pars=conditions(pop);P=len(pars);log('conditions',pop,P)
        counts=np.zeros((P,a.replicates),np.uint16);done=np.zeros(P,bool)
        ck=folder/f'checkpoint_{pop}.npz'
        if ck.exists():
            z=np.load(ck);counts[:]=z['counts'];done[:]=z['done'];log('resumed',done.sum())
        batch=a.batch;started=time.time()
        for start in range(0,P,batch):
            idx=np.arange(start,min(P,start+batch))
            if done[idx].all():continue
            obs=run(pars[idx],a.replicates,a.duration,a.burn,a.seed,crn=True,device=a.device)
            counts[idx]=obs[:,:,2].astype(np.uint16);done[idx]=True
            np.savez(ck,counts=counts,done=done)
            log(pop,f'{done.sum()}/{P}',f'{time.time()-started:.0f}s')
        rate=counts.mean(axis=1)/a.duration*1000;sem=counts.std(axis=1)/np.sqrt(a.replicates)/a.duration*1000
        np.savez_compressed(out,x=X,sigma_E=SE,sigma_I=SI,rate_hz=rate.reshape(len(X),len(SE),len(SI)),
                            sem_hz=sem.reshape(len(X),len(SE),len(SI)),counts=counts,replicates=a.replicates,duration_ms=a.duration,burn_ms=a.burn,seed=a.seed,theta=THETA,v_reset=VR,pop=pop)
        ck.unlink();log('WROTE',out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pops',nargs='+',default=['E','I']);p.add_argument('--replicates',type=int,default=512)
    p.add_argument('--duration',type=float,default=2000.);p.add_argument('--burn',type=float,default=200.);p.add_argument('--seed',type=int,default=20260918)
    p.add_argument('--batch',type=int,default=1024);p.add_argument('--device',type=int,default=0);main(p.parse_args())
