"""Coarse-grid linear frequency response for mean / AMPA-variance / GABA-variance modulation.

Paired +/- sinusoidal modulation with common random numbers (v2 assay protocol), demodulated
to the complex response per replicate. Output per (pop, x, sE, sI, channel, frequency):
mean complex response (Hz per mV or Hz per mV^2 of diffusion variance) and SEM.
"""
from lif_mc import *
import argparse
XG={'E':np.array([-3.,-1.5,-.8,-.3,0.,.3,.6,.8,1.,1.3,1.8,2.5,4.,7.,15.,40.]),
    'I':np.array([-3.,-1.5,-.8,-.3,0.,.3,.6,.8,1.,1.3,1.8,2.5,4.,7.,15.])}
SEG={'E':np.array([.15,.35,.7,1.4,2.8,5.]),'I':np.array([.15,.35,.7,1.4,2.8])}
SIG={'E':np.array([0.,.3,.7,1.4,2.8,5.5]),'I':np.array([0.,.3,.7,1.4,2.8,5.])}
FREQ=[0.,2.,5.,10.,20.,40.,80.]
THETA=18.;VR=11.;SCALE=THETA-VR;MEAN_AMP=.15;VAR_REL=.05

def main(a):
    folder=DEST/'dynamic_assay';folder.mkdir(parents=True,exist_ok=True)
    contract=folder/'contract.json'
    if not contract.exists():
        write(contract,dict(status='REGISTERED_BEFORE_RUN',registered=time.strftime('%Y-%m-%d %H:%M:%S'),
            question='Linear response shape (normalised by its zero-frequency value) of the same LIF to modulation of input mean, AMPA diffusion variance and GABA diffusion variance, as a function of workpoint',
            readout='paired +/- demodulated complex rate response per replicate; mean and SEM over replicates',
            statistical_unit='independent noise path (replicate), paired across +/- and CRN across conditions',
            replicates=a.replicates,duration_ms=a.duration,burn_ms=a.burn,seed=a.seed,frequencies_hz=FREQ,
            mean_amplitude_mv=MEAN_AMP,variance_relative_amplitude=VAR_REL,grids=dict(x=XG,sigma_E=SEG,sigma_I=SIG),
            use='fit finite-dimensional response filters; linearity checked by half amplitude at registered points'))
    rows=[];pars=[]
    for pop in a.pops:
        for x in XG[pop]:
            for se in SEG[pop]:
                for si in SIG[pop]:
                    for ch in [0,1,2]:
                        if ch==2 and si==0:continue
                        for f in FREQ:
                            amp=MEAN_AMP if ch==0 else VAR_REL
                            rows.append(dict(pop=pop,x=x,sigma_E=se,sigma_I=si,channel=ch,frequency_hz=f,amplitude=amp))
                            pars.append(condition(VR+SCALE*x,THETA,(SCALE*se)**2,(SCALE*si)**2,pop,amplitude=amp,freq_hz=f,channel=ch))
    pars=np.array(pars);P=len(pars);log('dynamic conditions',P)
    out=folder/'raw.npz';ck=folder/'checkpoint.npz'
    resp=np.zeros((P,2));sem=np.zeros(P);rate=np.zeros(P);done=np.zeros(P,bool)
    if ck.exists():
        z=np.load(ck);resp[:]=z['resp'];sem[:]=z['sem'];rate[:]=z['rate'];done[:]=z['done'];log('resumed',done.sum())
    started=time.time()
    for start in range(0,P,a.batch):
        idx=np.arange(start,min(P,start+a.batch))
        if done[idx].all():continue
        obs=run(pars[idx],a.replicates,a.duration,a.burn,a.seed,crn=True,device=a.device)
        for j,i in enumerate(idx):
            p=pars[i];amp=p[4] if p[20]==0 else p[4]*p[2 if p[20]==1 else 3]
            est=(obs[j,:,0]+1j*obs[j,:,1])/(a.duration*amp)*1000
            m=est.mean();resp[i]=[m.real,m.imag];sem[i]=np.sqrt(np.mean(abs(est-m)**2)/a.replicates)
            rate[i]=(obs[j,:,2]+obs[j,:,3]).mean()/2/a.duration*1000
        done[idx]=True;np.savez(ck,resp=resp,sem=sem,rate=rate,done=done);log(f'{done.sum()}/{P}',f'{time.time()-started:.0f}s')
    for i,r in enumerate(rows):r.update(response=[resp[i,0],resp[i,1]],sem=sem[i],rate_hz=rate[i])
    np.savez_compressed(out,resp=resp,sem=sem,rate=rate,pars=pars);write(folder/'rows.json',dict(status='COMPLETE',rows=rows));ck.unlink();log('WROTE',out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pops',nargs='+',default=['E','I']);p.add_argument('--replicates',type=int,default=1024)
    p.add_argument('--duration',type=float,default=4000.);p.add_argument('--burn',type=float,default=200.);p.add_argument('--seed',type=int,default=20260919)
    p.add_argument('--batch',type=int,default=512);p.add_argument('--device',type=int,default=0);main(p.parse_args())
