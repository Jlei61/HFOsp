"""Identify rate relaxation from original-LIF pulse responses; heldout pulse."""
from common import *
from calibrate_transfer import CODE
from model import transfer_eval
import cupy as cp
from scipy.ndimage import gaussian_filter1d

def main():
    cp.cuda.Device(0).use();prep=read(KIN/'coarse_40/prepared.json');p=prep['params'];nu=prep['nu_ext_per_ms'];dest=BASE/'dynamic_calibration';dest.mkdir(exist_ok=True)
    if (dest/'result.json').exists():return
    durations=8000;R=8192;t=(np.arange(durations)+1)*DT
    drive=np.zeros(durations)
    for start,amp in [(100,1.),(300,6.),(500,24.)]:drive[(t>start)&(t<=start+20)]=amp
    candidates=np.array([.1,.25,.5,1,2,3,5,8,12,20.]);rows=[]
    write(dest/'contract.json',dict(source='Original LIF pulse responses with original stationary private Poisson input',theta_E=[14.5,16.5,18.],
        fit_pulses_mv=[1,6],heldout_pulse_mv=24,pulse_duration_ms=20,tau_candidates_ms=candidates,
        purpose='Identify response dynamics for explicit rate equations; no spatial or patient fitting'))
    for pop in ['E','I']:
        theta=np.array([14.5,16.5,18.]) if pop=='E' else np.array([18.]);P=len(theta);N=P*R
        code=CODE.replace('I[i]+drive[g]','I[i]+drive[row*'+str(P)+'+g]').replace('counts+g','counts+row*'+str(P)+'+g')
        kernel=cp.RawKernel(code,'calibrate',options=('--fmad=false',))
        q=cp.zeros(N);I=q.copy();v=cp.full(N,11.);ref=cp.zeros(N,cp.int32);counts=cp.zeros((50,P),cp.uint32)
        tr=p['tau_r_AMPA'];td=p['tau_d_AMPA'];tm=p[f'tau_m_{pop}'];nr=round(p[f'tau_ref_{pop}']/DT)
        pars=tuple(np.float64(z) for z in [np.exp(-DT/tr),np.exp(-DT/td),np.exp(-DT/tm),tm/tr*p[f'J_ext_{pop}']])+(np.int32(nr),)
        rng=cp.random.RandomState(199719+int(pop=='I'));native=[]
        for k in range(-3000,durations,50):
            dd=cp.asarray(np.zeros((50,P)) if k<0 else np.repeat(drive[k:k+50,None],P,axis=1));ext=rng.poisson(nu*DT,size=(50,N)).astype(cp.uint32);counts.fill(0)
            for j in range(50):kernel(((N+255)//256,),(256,),(ext,q,I,v,ref,counts,cp.asarray(theta),dd,np.int32(N),np.int32(R),np.int32(j),np.int32(k>=0),*pars))
            if k>=0:native.append(counts.get()/R/DT)
        native=np.concatenate(native);tab=read(BASE/f'transfer/{pop}.json');pred=np.empty((len(candidates),durations,P))
        f=np.stack([transfer_eval(drive,np.full(len(drive),th),np.full(len(drive),nu),np.array(tab['theta']),np.array(tab['nu']),
            np.array(tab['input_knots']),np.array(tab['coefficients']),tab['ref_ms'])[0] for th in theta],axis=1)
        for j,tau in enumerate(candidates):
            state=f[0].copy();dec=np.exp(-DT/tau)
            for k in range(durations):state=dec*state+(1-dec)*f[k];pred[j,k]=state
        observed=gaussian_filter1d(native,20,axis=0);expected=gaussian_filter1d(pred,20,axis=1)
        errors=[]
        for start in [100,300,500]:
            sl=slice(int(start/DT),int((start+120)/DT));den=np.mean(observed[sl]**2,axis=0)
            errors.append(np.mean(np.mean((expected[:,sl]-observed[sl])**2,axis=1)/np.maximum(den,1e-10),axis=1))
        best=int(np.argmin(np.sum(errors[:2],axis=0)));row=dict(population=pop,tau_rate_ms=candidates[best],fit_normalized_mse=float(np.mean(errors[:2],axis=0)[best]),
            heldout_normalized_mse=float(errors[2][best]),all_errors=errors)
        np.savez_compressed(dest/f'{pop}.npz',time_ms=t,native_rate_per_ms=native,rate_predictions_per_ms=pred,theta=theta,drive_mv=drive,tau_ms=candidates)
        rows.append(row);print(pop,row['tau_rate_ms'],row['fit_normalized_mse'],row['heldout_normalized_mse'],flush=True)
    write(dest/'result.json',dict(status='COMPLETE',rows=rows,scope='Open-loop rate relaxation calibration; full spatial dynamics and frequency-dependent response remain to verify'))

if __name__=='__main__':main()

