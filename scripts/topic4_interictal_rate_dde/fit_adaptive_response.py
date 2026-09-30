"""Fit state-dependent relaxation on existing weak/moderate LIF pulses only."""
from common import *
from model import transfer_eval
from numba import njit
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d

@njit(cache=True)
def predict(f,drive,tau0,c):
    out=np.empty_like(f);r=f[0].copy()
    for k in range(len(f)):
        tau=.1+(tau0-.1)/(1+(drive[k]/c)**2);a=np.exp(-DT/tau)
        r=a*r+(1-a)*f[k];out[k]=r
    return out

def main():
    dest=BASE/'dynamic_calibration';rows=[]
    for pop in ['E','I']:
        z=np.load(dest/f'{pop}.npz');tab=read(BASE/f'transfer/{pop}.json');drive=z['drive_mv'];theta=z['theta'];nu=read(KIN/'coarse_40/prepared.json')['nu_ext_per_ms']
        f=np.stack([transfer_eval(drive,np.full(len(drive),th),np.full(len(drive),nu),np.array(tab['theta']),np.array(tab['nu']),np.array(tab['input_knots']),np.array(tab['coefficients']),tab['ref_ms'])[0] for th in theta],axis=1)
        native=gaussian_filter1d(z['native_rate_per_ms'],20,axis=0)
        def errors(v):
            out=gaussian_filter1d(predict(f,drive,*np.exp(v)),20,axis=0);values=[]
            for start in [100,300,500]:
                sl=slice(int(start/DT),int((start+120)/DT));den=np.mean(native[sl]**2,axis=0)
                values.append(float(np.mean(np.mean((out[sl]-native[sl])**2,axis=0)/np.maximum(den,1e-10))))
            return values
        opt=minimize(lambda v:np.mean(errors(v)[:2]),np.log([8.,5.]),method='Nelder-Mead',bounds=[(np.log(.11),np.log(100)),(np.log(.1),np.log(100))],options={'maxiter':150,'xatol':1e-5,'fatol':1e-7})
        tau0,c=np.exp(opt.x);err=errors(opt.x)
        row=dict(population=pop,tau0_ms=tau0,current_scale_mv=c,tau_min_ms=.1,fit_normalized_mse=np.mean(err[:2]),heldout_normalized_mse=err[2],optimizer_success=bool(opt.success),iterations=int(opt.nit));rows.append(row);print(row,flush=True)
    write(dest/'adaptive_response.json',dict(status='COMPLETE',rows=rows,equation='tau(D)=0.1+(tau0-0.1)/(1+(D/c)^2)',
        fit_source='Only same weak1mV and moderate6mV pulses; strong24mV pulse excluded from fitting',
        purpose='Resolve observed faster native response at stronger drive; smooth state-dependent rate dynamics with analytic tangent'))

if __name__=='__main__':main()
