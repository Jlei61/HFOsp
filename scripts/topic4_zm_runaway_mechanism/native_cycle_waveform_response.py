"""Open-loop colored-LIF response to actual candidate-cycle input waveforms."""
from native_path import *
from large_quadrature_periodic import LargeQuadratureGalerkin
from scipy.fft import rfft,irfft
from lif_mc import condition,run as static_run
import argparse

DEST=OUT/'native_cycle_waveform_response'
CODE=r'''
#include <curand_kernel.h>
extern "C" __global__ void waveform(const double* pars,const double* wave,unsigned int* counts,
 int P,int R,int W,int B,int steps,int burn,double dt,double T,unsigned long long seed){
 int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=P*R)return;int g=id/R,k=id%R;
 const double* p=pars+g*24;curandStatePhilox4_32_10_t rng;curand_init(seed,k,0,&rng);
 double qa=0,ia=0,qg=0,ig=0,v=p[21];int ref=0;
 for(int t=-burn;t<steps;t++){
  double phase=fmod((t+1)*dt/T,1.);if(phase<0)phase+=1.;
  double pos=phase*W;int lo=(int)floor(pos);double a=pos-lo;lo%=W;int hi=(lo+1)%W;
  const double* x=wave+g*3*W;
  double mu=(1-a)*x[lo]+a*x[hi];
  double ve=(1-a)*x[W+lo]+a*x[W+hi],vi=(1-a)*x[2*W+lo]+a*x[2*W+hi];
  float4 n=curand_normal4(&rng);
  double na=p[11]*n.x,nb=p[12]*n.x+p[13]*n.y,nc=p[14]*n.z,nd=p[15]*n.z+p[16]*n.w;
  double af=sqrt(ve),gf=sqrt(vi);
  ia=p[7]*qa+p[8]*ia+af*nb;qa=p[6]*qa+af*na;
  ig=p[9]*qg+p[10]*ig+gf*nd;qg=p[17]*qg+gf*nc;
  double cur=mu+ia-ig;bool fired=false;ref=max(0,ref-1);
  if(ref==0){v=p[18]*v+(1-p[18])*cur;if(v>=p[1]){v=p[21];ref=(int)p[19];fired=true;}}
  else v=p[21];
  if(t>=0&&fired){int bin=min((int)(phase*B),B-1);counts[id*B+bin]++;}
 }
}
'''


def simulate(pars,wave,R,T,dt,burn_steps,steps,seed,bins,device):
    import cupy as cp
    cp.cuda.Device(device).use();pars=np.asarray(pars);wave=np.ascontiguousarray(wave)
    P=len(pars);W=wave.shape[-1]
    assert wave.shape==(P,3,W) and np.all(wave[:,1:]>=0) and np.isfinite(wave).all()
    counts=cp.zeros((P,R,bins),dtype=cp.uint32)
    kernel=cp.RawKernel(CODE,'waveform',options=('--fmad=false',))
    kernel(((P*R+127)//128,),(128,),(cp.asarray(pars),cp.asarray(wave),counts,
        np.int32(P),np.int32(R),np.int32(W),np.int32(bins),np.int32(steps),np.int32(burn_steps),
        float(dt),float(T),np.uint64(seed)))
    return counts.get()


def check(device):
    DEST.mkdir(exist_ok=True)
    pars=[condition(20.,18.,1.,1.,'E'),condition(16.,14.5,1.,1.,'I')]
    wave=np.ones((2,3,2));wave[:,0]=np.array([20.,16.])[:,None]
    old=static_run(pars,256,400.,100.,7918,device=device)[:,:,2]
    new=simulate(pars,wave,256,37.7,.1,1000,4000,7918,13,device).sum(axis=2)
    assert old.sum()>0 and np.array_equal(old,new),(old.sum(),new.sum(),np.max(abs(old-new)))
    q=dict(status='PASS',constant_wave_per_replicate_count_bitwise=True,
        replicates=256,conditions=2,counts=int(new.sum()),duration_ms=400,burn_ms=100)
    write(DEST/'implementation_check.json',q);log('WAVEFORM IMPLEMENTATION',q)


def prepare(device):
    c=read(OUT/'native_cycle_waveform_response_contract.json');z=np.load(OUT/c['source'])
    s=model();attach_native_path(s);s.set_D(float(z['D']))
    assert np.array_equal(s.Z,z['Z']) and float(z['residual'])<2e-8
    r=z['r'];N=len(r);T=float(z['T']);v=z['tangent'][:r.size].reshape(r.shape)
    phase=irfft(2j*np.pi*np.arange(N//2+1)[:,None]*rfft(r*1000,axis=0),n=N,axis=0)
    w=s.sizes*s.E;projection=np.sum(v*phase*w)/np.sum(phase**2*w)
    energy=np.mean((v-projection*phase)**2,axis=0)*s.sizes
    masks=[s.E&(s.geo['group_region']==j) for j in range(3)]+[~s.E]
    groups=[int(np.flatnonzero(mask)[np.argmax(energy[mask])]) for mask in masks]
    assert len(set(groups))==4
    o=LargeQuadratureGalerkin(s,N,65536,device);o.cache_mean_operators=False
    cp=o.cp;cp.fft.config.get_plan_cache().set_size(0)
    h=o.harmonics(o.linear_inputs,cp.asarray(r),T,s.Z)
    selected=h[:3,:,cp.asarray(groups)].get();del h
    # linear_inputs stores filtered variance. Undo that one-pole filter to
    # recover the driving intensity; the LIF assay applies its own synapses.
    lam=2j*np.pi*np.arange(N//2+1)/T
    selected[1]*=(1+lam*s.tau[0]/2)[:,None]
    selected[2]*=(1+lam*s.tau[1]/2)[:,None]
    W=c['wave_samples'];wave=irfft(selected,n=W,axis=1)*(W/N)
    wave[0]+=s.private_mu[groups][None,:];wave[1]+=s.private_ve[groups][None,:]
    wave=wave.transpose(2,0,1).copy()
    assert np.min(wave[:,1:])>=0,float(np.min(wave[:,1:]))
    prediction=irfft(rfft(r[:,groups],axis=0),n=W,axis=0)*(W/N)*1000
    pars=np.array([condition(0.,s.theta[g],1.,1.,'E' if s.E[g] else 'I') for g in groups])
    info=[dict(label=label,group=g,pop='E' if s.E[g] else 'I',theta_mv=float(s.theta[g]),
        original_cells=int(s.sizes[g]),Z=float(s.Z[g]),family_energy=float(energy[g]),
        minimum_variance_intensity=float(wave[j,1:].min()))
        for j,(label,g) in enumerate(zip(['Core A E','Core B E','Surround E','I'],groups))]
    np.savez_compressed(DEST/'prepared.npz',wave=wave,prediction_hz=prediction,
        pars=pars,groups=groups,T_ms=T,D=float(z['D']))
    write(DEST/'preparation.json',dict(status='PASS',source=str(OUT/c['source']),N=N,
        wave_samples=W,groups=info,phase_projection_removed=float(projection),
        drive='Raw delayed variance drive; already-filtered mean current including imposed cycle M',
        scope='Selected groups of the1mm conditional cycle, not whole-network validation.'))
    log('WAVEFORM PREPARED',info)


def main(device):
    c=read(OUT/'native_cycle_waveform_response_contract.json')
    if not (DEST/'implementation_check.json').exists():check(device)
    assert read(DEST/'implementation_check.json')['status']=='PASS'
    if not (DEST/'prepared.npz').exists():prepare(device)
    z=np.load(DEST/'prepared.npz');R=c['replicates'];dt=c['dt_ms'];T=float(z['T_ms']);B=c['phase_bins']
    steps=round(c['record_cycles']*T/dt);burn=round(c['burn_cycles']*T/dt)
    counts=simulate(z['pars'],z['wave'],R,T,dt,burn,steps,c['seed'],B,device)
    times=(np.arange(steps)+1)*dt;phase=(times/T)%1;bins=np.minimum((phase*B).astype(int),B-1)
    occupancy=np.bincount(bins,minlength=B)*dt;assert occupancy.min()>0
    rates=counts/occupancy[None,None,:]*1000
    measured=rates.mean(axis=1);sem=rates.std(axis=1,ddof=1)/np.sqrt(R)
    W=z['wave'].shape[-1];pos=phase*W;lo=np.floor(pos).astype(int)%W;hi=(lo+1)%W;a=pos-np.floor(pos)
    sampled=(1-a[:,None])*z['prediction_hz'][lo]+a[:,None]*z['prediction_hz'][hi]
    predicted=np.array([np.bincount(bins,weights=sampled[:,j],minlength=B)/(occupancy/dt) for j in range(len(measured))])
    rows=[];groups=read(DEST/'preparation.json')['groups']
    for j,group in enumerate(groups):
        error=float(np.linalg.norm(predicted[j]-measured[j])/max(np.linalg.norm(measured[j]),np.sqrt(B)))
        model_mean=float(np.average(predicted[j],weights=occupancy))
        mc_means=counts[j].sum(axis=1)/(steps*dt)*1000;mean=float(mc_means.mean())
        bias=abs(model_mean-mean)/max(mean,1.)
        split=float(np.linalg.norm(rates[j,:R//2].mean(0)-rates[j,R//2:].mean(0))/max(np.linalg.norm(measured[j]),np.sqrt(B)))
        passed=error<=c['acceptance']['normalized_waveform_RMSE_max'] and bias<=c['acceptance']['relative_cycle_mean_error_max']
        rows.append(dict(**group,model_cycle_mean_hz=model_mean,MC_cycle_mean_hz=mean,
            MC_cycle_mean_SEM_hz=float(mc_means.std(ddof=1)/np.sqrt(R)),normalized_waveform_RMSE=error,
            relative_cycle_mean_error=bias,MC_split_half_difference=split,passed=bool(passed)))
    np.savez_compressed(DEST/'response.npz',counts=counts,measured_hz=measured,sem_hz=sem,
        predicted_hz=predicted,occupancy_ms=occupancy,T_ms=T,phase_centres=(np.arange(B)+.5)/B)
    q=dict(status='COMPLETE',verdict='PASS_SELECTED_OPEN_LOOP_GROUPS' if all(r['passed'] for r in rows) else 'FAIL_SELECTED_OPEN_LOOP_GROUPS',
        rows=rows,replicates=R,steps=steps,burn_steps=burn,dt_ms=dt,T_ms=T,
        scope=c['scope'],statistical_unit=c['statistical_unit'])
    write(DEST/'result.json',q);log('WAVEFORM RESPONSE',q)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1)
    p.add_argument('--check-only',action='store_true');a=p.parse_args()
    if a.check_only:check(a.device)
    else:main(a.device)
