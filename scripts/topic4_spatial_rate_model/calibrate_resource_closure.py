"""Reduce the native inhibitory-current threshold rule using observed dispersion.

Fit the probability of I_G below the original fixed threshold from within-cell
current moments. Do not fit entry time, Z time constant, or the threshold itself.
"""
from common import *
from scipy.optimize import nnls,least_squares
from scipy.special import ndtr
import time


def collect(seed):
    geo=dict(np.load(GRID/'geometry.npz'));cell=geo['cell_e'];count=geo['count_e']
    ith=float(read(ROOT/'results/topic4_sef_hfo/fig5_preentry_event_audit_20260914/protocol.json')['I_th'])
    rows=[];times=[]
    for path in sorted((SOURCE/f'replay/runs/eta0.0005_s{seed}/fields').glob('*.npz')):
        with np.load(path) as z:
            keep=(z['zm_step']*.1>=500)&(z['zm_step']*.1<8000)
            if not keep.any():continue
            for time,values in zip(z['zm_step'][keep]*.1,z['ii'][keep]):
                mean=np.bincount(cell,weights=values,minlength=400)/count
                var=np.maximum(np.bincount(cell,weights=values.astype(float)**2,minlength=400)/count-mean**2,0.)
                fraction=np.bincount(cell,weights=values<ith,minlength=400)/count
                rows.append(np.array([mean,var,fraction]));times.append(time)
    return np.array(rows),np.array(times),ith,count


def probability(mean,a,b,ith):
    mean=np.maximum(mean,1e-8);cv2=a/mean+b
    sigma=np.sqrt(np.log1p(cv2))
    return ndtr((np.log(ith/mean)+.5*sigma**2)/sigma)


def main():
    folder=OUT/'resource_closure';folder.mkdir(exist_ok=True)
    write(folder/'contract.json',dict(training_seed=9108401,validation_seed=9108402,window_ms=[500,8000],
        observable='Within1mm cell fraction of original E neurons with I_G below the fixed native threshold',
        unchanged=['I_th','tau_Z=5000ms','eta_M=.0005','tau_M=1000ms','native graph and threshold field'],
        candidates='Measured mean/variance with positive lognormal moment closure; calibrate two global dispersion parameters',
        prohibited_targets='Entry time, D threshold, future native Z or rate as simulator input',
        sampling_unit='Whole input realization; cells and times are calibration observations, not independent biological replicates'))
    train,times,ith,count=collect(9108401);mean,var,p=train.transpose(1,0,2)
    x=mean.ravel();y=var.ravel();target=p.ravel();weights=np.tile(count,len(mean)).astype(float)
    # Fit conditional variance in relative units, so plateau currents do not
    # overwhelm threshold-region observations.
    scale=np.maximum(x,10.);A=np.c_[x,x*x]/scale[:,None]**2
    initial=nnls(A,y/scale**2)[0];initial=np.maximum(initial,[1e-4,1e-6])
    good=(x>ith/5)&(x<ith*5);w=np.sqrt(weights[good]/weights[good].mean())
    fit=least_squares(lambda p:(probability(x[good],*np.exp(p),ith)-target[good])*w,np.log(initial),
        bounds=(np.log([1e-4,1e-6]),np.log([500.,10.])),max_nfev=100)
    coefficients=np.exp(fit.x);rows=[]
    for seed,data in [(9108401,train),(9108402,collect(9108402)[0])]:
        mu,va,actual=data.transpose(1,0,2);pred=probability(mu,*coefficients,ith)
        old=ndtr((ith-mu)/5.)
        rows.append(dict(seed=seed,role='training' if seed==9108401 else 'held_out_input',
            rmse_old=float(np.sqrt(np.mean((old-actual)**2))),rmse_new=float(np.sqrt(np.mean((pred-actual)**2))),
            mean_bias_old=float(np.mean(old-actual)),mean_bias_new=float(np.mean(pred-actual))))
    np.savez_compressed(folder/'training_moments.npz',moments=train,time_ms=times)
    write(folder/'result.json',dict(status='CONDITIONAL_RESOURCE_CLOSURE_FITTED',variance_a_mv=coefficients[0],variance_b=coefficients[1],
        native_I_th_mv=ith,rows=rows,optimizer_success=bool(fit.success),
        network_validation='PENDING; local probability improvement does not establish autonomous dynamics'))
    print(coefficients,rows,flush=True)


if __name__=='__main__':main()
