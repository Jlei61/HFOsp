"""Measure smooth rate transfer from original LIF responses, not hazard fits.

Microscopic cells are used only in this offline calibration. The spatial model
has no particles. No spatial output or patient target is used for calibration.
"""
from common import *
import cupy as cp
from scipy.interpolate import PchipInterpolator

CODE=r'''
extern "C" __global__ void calibrate(const unsigned int* ext,double* q,double* I,double* v,int* ref,
 unsigned int* counts,const double* theta,const double* drive,int N,int R,int row,int record,
 double ar,double ad,double am,double jump,int nr){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;int g=i/R;
 q[i]=ar*q[i]+jump*ext[(long long)row*N+i];I[i]=ad*I[i]+(1-ad)*q[i];ref[i]=max(0,ref[i]-1);
 if(ref[i]==0){v[i]=am*v[i]+(1-am)*(I[i]+drive[g]);if(v[i]>=theta[g]){v[i]=11.;ref[i]=nr;if(record)atomicAdd(counts+g,1);}}
 else v[i]=11.;
}
'''

def coefficients(x,values):
    """C2 monotone quintic interpolation with analytic first derivatives."""
    slopes=PchipInterpolator(x,values).derivative()(x);sec=np.diff(values)/np.diff(x)
    for j in range(len(x)):
        adj=sec[max(0,j-1):min(len(sec),j+1)]
        if np.any(adj==0) or np.any(adj*slopes[j]<0):slopes[j]=0
        else:slopes[j]=np.sign(slopes[j])*min(abs(slopes[j]),2*abs(adj).min())
    out=[]
    for j in range(len(x)-1):
        h=x[j+1]-x[j];d=values[j+1]-values[j];a=slopes[j]*h;b=slopes[j+1]*h
        out.append([6*d-3*a-3*b,-15*d+8*a+7*b,10*d-6*a-4*b,0,a,values[j]])
    return out

def main():
    destination=BASE/'transfer';destination.mkdir(parents=True,exist_ok=True)
    cp.cuda.Device(0).use();prep=read(KIN/'coarse_40/prepared.json');p=prep['params'];nu0=prep['nu_ext_per_ms']
    drives=np.array([-128,-64,-32,-16,-8,-4,-2,-1,0,1,2,4,6,8,12,16,24,32,48,64,96,128,192,256,384,512,768,1024,2048.])
    nus=nu0*np.array([.7,1.,1.3]);R=512;duration=2000.;burn=300.;kernel=cp.RawKernel(CODE,'calibrate',options=('--fmad=false',))
    write(destination/'contract.json',dict(question='Rate transfer and state derivatives for a spatial rate DDE',
        source='Original private-Poisson LIF equations and physical parameters, isolated constant recurrent drive',
        population_members_only_offline=R,duration_ms=duration,burn_ms=burn,drives_mv=drives,input_rates_per_ms=nus,
        calibration_uses_spatial_targets=False,calibration_uses_patient_targets=False,
        interpolation='C2 monotone quintic in asinh(drive/8); convex threshold and nu interpolation; rates constrained by original refractory maximum',
        source_literature='https://doi.org/10.1371/journal.pcbi.1005545',
        relation_to_literature='Empirically calibrated rate cascade, not a claim of an exact Fokker-Planck derivation'))
    for pop in ['E','I']:
        path=destination/f'{pop}.json'
        if path.exists():continue
        th=np.array([14.2,14.5,15,15.5,16,16.5,17.25,18.]) if pop=='E' else np.array([18.])
        shape=(len(th),len(nus),len(drives));tt,nn,dd=np.meshgrid(th,nus,drives,indexing='ij')
        theta,nu,drive=[x.ravel() for x in [tt,nn,dd]];N=len(theta)*R;P=len(theta)
        q=cp.zeros(N);cur=q.copy();v=cp.full(N,11.);ref=cp.zeros(N,cp.int32);count=cp.zeros(P,cp.uint32)
        tm=p[f'tau_m_{pop}'];tr=p['tau_r_AMPA'];td=p['tau_d_AMPA'];nr=round(p[f'tau_ref_{pop}']/DT)
        pars=tuple(np.float64(x) for x in [np.exp(-DT/tr),np.exp(-DT/td),np.exp(-DT/tm),tm/tr*p[f'J_ext_{pop}']])+(np.int32(nr),)
        rng=cp.random.RandomState(199711+int(pop=='I'));lam=cp.asarray(np.repeat(nu,R)*DT)
        dtheta=cp.asarray(theta);ddrive=cp.asarray(drive);started=time.time()
        for step in range(-round(burn/DT),round(duration/DT),50):
            ext=rng.poisson(lam,size=(50,N)).astype(cp.uint32)
            for row in range(50):
                kernel(((N+255)//256,),(256,),(ext,q,cur,v,ref,count,dtheta,ddrive,np.int32(N),np.int32(R),np.int32(row),np.int32(step>=0),*pars))
            if step%2000==0:print(pop,'calibration ms',step*DT,'elapsed',round(time.time()-started,1),flush=True)
        rates=count.get().reshape(shape)/(R*duration);raw=rates.copy()
        # Reuse the original independent stationary fit-half estimates where
        # available; preserve the old reserved-repeat half for validation.
        old=read(HAZARD/f'calibration_steady/{pop}/result.json');nold=len(old['rates_hz'])//2
        for (t,d),hz in zip(old['parameter_pairs'][:nold],old['rates_hz'][:nold]):
            if t in th and d in drives:rates[list(th).index(t),1,list(drives).index(d)]=hz/1000
        rates=np.maximum.accumulate(rates,axis=-1);rates=np.minimum(rates,(1-1e-7)/p[f'tau_ref_{pop}'])
        x=np.arcsinh(drives/8);isi=np.log(1/np.maximum(rates,1e-15)-p[f'tau_ref_{pop}'])
        co=np.array([[coefficients(x,row) for row in arr] for arr in isi])
        write(path,dict(status='COMPLETE',population=pop,theta=th,nu=nus,drive=drives,input_scale_mv=8.,input_knots=x,
            rate_per_ms=rates,raw_rate_per_ms=raw,coefficients=co,ref_ms=p[f'tau_ref_{pop}'],duration_ms=duration,R=R,seconds=time.time()-started,
            maximum_monotone_adjustment_hz=float(np.max(abs(raw-rates))*1000)))
        print('TRANSFER COMPLETE',pop,flush=True)

if __name__=='__main__':main()
