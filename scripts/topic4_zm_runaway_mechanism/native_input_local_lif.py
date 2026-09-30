"""Fixed local LIF reference for teacher-forced native group input histories.

This is an input-response validation experiment, not a replacement network.
"""
from common import OUT,np,read,write
from lif_mc import condition,run as static_run
from datetime import datetime
import argparse,math,time

DEST=OUT/'native_input_bridge';LOCAL=DEST/'local_lif'
CODE=r'''
#include <curand_kernel.h>
extern "C" __global__ void drive(const double* pars,const double* wave,const double* thresholds,
 const int* replicates,unsigned int* counts,int P,int R,int W,int factor,int burn,int B,int stepsperbin,unsigned long long seed){
 int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=P*R)return;int g=id/R,k=id%R;if(k>=replicates[g])return;
 const double* p=pars+24*g;const double* x=wave+(long long)g*4*W;
 curandStatePhilox4_32_10_t rng;curand_init(seed,k,0,&rng);
 double qa=0,ia=0,qg=0,ig=0,v=p[21];int ref=0;
 for(int t=0;t<W*factor;t++){
  int ix=t/factor;float4 n=curand_normal4(&rng);double af=sqrt(x[W+ix]),gf=sqrt(x[2*W+ix]);
  double na=p[11]*n.x,nb=p[12]*n.x+p[13]*n.y,nc=p[14]*n.z,nd=p[15]*n.z+p[16]*n.w;
  ia=p[7]*qa+p[8]*ia+af*nb;qa=p[6]*qa+af*na;
  ig=p[9]*qg+p[10]*ig+gf*nd;qg=p[17]*qg+gf*nc;
  double current=x[ix]+ia-x[3*W+ix]*ig;bool fired=false;ref=max(0,ref-1);
  if(ref==0){v=p[18]*v+(1-p[18])*current;if(v>=thresholds[id]){v=p[21];ref=(int)p[19];fired=true;}}
  else v=p[21];
  int bin=(t-burn)/stepsperbin;if(t>=burn && bin<B && fired)counts[(long long)id*B+bin]++;
 }
}'''


def simulate(pars,wave,theta,nrep,dt,burn_ms,bins,bin_ms,seed,device):
    import cupy as cp
    cp.cuda.Device(device).use();P,R=theta.shape;W=wave.shape[-1];factor=round(.1/dt)
    assert abs(factor*dt-.1)<1e-12 and wave.shape==(P,4,W)
    assert np.isfinite(wave).all() and wave[:,1:3].min()>=0
    kernel=cp.RawKernel(CODE,'drive',options=('--fmad=false',))
    output=cp.zeros((P,R,bins),dtype=cp.uint32)
    kernel(((P*R+127)//128,),(128,),(cp.asarray(pars),cp.asarray(wave),cp.asarray(theta),cp.asarray(nrep,dtype=cp.int32),output,
        np.int32(P),np.int32(R),np.int32(W),np.int32(factor),np.int32(round(burn_ms/dt)),np.int32(bins),np.int32(round(bin_ms/dt)),np.uint64(seed)))
    return output.get()


def register():
    LOCAL.mkdir(exist_ok=True);assert not (LOCAL/'contract.json').exists()
    write(LOCAL/'contract.json',dict(created_local=datetime.now().astimezone().isoformat(),
        status='REGISTERED_BEFORE_LOCAL_LIF_EXECUTION',groups=read(DEST/'contract.json')['selected_groups'],
        question='Does locked rate response agree with independent GaussianLIF under actual native groupcount-derived input? Does keeping original within-group threshold heterogeneity change that comparison?',
        input='Reconstructed AMPA/GABA means plus observed pre-step groupZ/M; delayed privatePoisson raw variance forcing, exact two-pole OU noise, Z multiplies GABA current after filtering.',
        conditions=['group_mean_threshold','original_cell_threshold_mixture'],
        min_replicates_per_case=8192,replicates='Round up to multiple of originalgroupN for equal originalthreshold weights and exact finiteN synthetic batches.',
        seed=920110,dt_ms=[.1,.05],history_ms=[8000,9000],primary_ms=[9000,10350],bin_ms=50,
        initialization='Reset voltage,zero noise currents,refractory0 at8s;1s preparation excluded. Native inputs are held on0.1ms intervals in both time steps.',
        comparisons='Same-clock50ms totalcounts; MCmean and central95percent finiteN predictive range under conditional independentGaussianinput. Acrossgroup/time bins are descriptive repeated observations. No fitted shifts.',
        numerical_gate='Matcheddt binnedmean relativeL2<=.05 or <=3timesMCsplitL2; exact constant-input counts vs originalLIF kernel required first.',
        interpretation='LocalGaussianLIF vs lockedrate local discrepancy diagnoses response only within this assumed input law. Agreement between them with native mismatch supports missing input/heterogeneity closure. Predictive range is model-based, not a calibrated native confidence interval.',
        scope='Fixed12conditions at2steps; no rate fitting or newwhole-network simulation, no model promotion or bifurcation.'))


def check(device):
    pars=np.array([condition(20.,18.,1.,1.,'E'),condition(16.,14.5,1.,1.,'I')]);R=256
    wave=np.ones((2,4,5000));wave[:,0]=np.array([20.,16.])[:,None]
    theta=np.broadcast_to(pars[:,1,None],(2,R)).copy()
    old=static_run(pars,R,400.,100.,7918,device=device)[:,:,2]
    new=simulate(pars,wave,theta,np.array([R,R]),.1,100.,8,50.,7918,device).sum(2)
    assert old.sum()>0 and np.array_equal(old,new)
    write(LOCAL/'implementation_check.json',dict(status='PASS',constant_input_counts_bitwise=True,groups=2,replicates=R,total_count=int(new.sum())))


def run(device):
    c=read(LOCAL/'contract.json');check(device)
    z=np.load(DEST/'selected_input_history.npz');raw=np.load(DEST/'selected_raw_variance_forcing.npz')
    assert np.array_equal(z['time_ms'],raw['time_ms']) and np.array_equal(z['groups'],raw['groups'])
    m={key:z['moments'][:,j] for j,key in enumerate(z['moment_names'])};r=z['reconstructed'];groups=z['groups'];G=len(groups)
    mu=r[:,0]-m['z']*r[:,1]-m['mcurrent']
    wave=np.stack([mu,raw['raw_variances'][:,2],raw['raw_variances'][:,3],m['z']],axis=1).transpose(2,1,0).copy()
    wave=np.concatenate([wave,wave],axis=0);n=z['group_size'].astype(int)
    replicates=np.ceil(c['min_replicates_per_case']/n).astype(int)*n;nrep=np.tile(replicates,2);R=int(nrep.max())
    theta=np.empty((2*G,R));geo=np.load(DEST/'membership.npz')
    for j,g in enumerate(groups):
        original=geo['actual_threshold_mv'][geo['cell_group']==g];assert len(original)==n[j]
        theta[j]=z['theta'][j];theta[G+j]=np.resize(original,R)
    for dt in c['dt_ms']:
        out=LOCAL/f'dt{dt:g}.npz';assert not out.exists()
        pars=np.array([condition(0.,z['theta'][j%G],1.,1.,'E' if z['population'][j%G]==0 else 'I',dt=dt) for j in range(2*G)])
        started=time.time();counts=simulate(pars,wave,theta,nrep,dt,1000.,27,50.,c['seed'],device)
        np.savez_compressed(out,counts=counts,replicates=nrep,group_sizes=np.tile(n,2),groups=np.tile(groups,2),dt_ms=dt)
        print('LOCAL INPUT LIF COMPLETE',dt,'seconds',time.time()-started,flush=True)
    fine=np.load(LOCAL/'dt0.05.npz');coarse=np.load(LOCAL/'dt0.1.npz');fixed=np.load(DEST/'fixed_readout.npz');rows=[]
    for j in range(2*G):
        jg=j%G;N=int(n[jg]);nr=int(nrep[j]);a=fine['counts'][j,:nr].astype(float);b=coarse['counts'][j,:nr].astype(float)
        rate=a.mean(0)/.05;old=b.mean(0)/.05;split=(a[:nr//2].mean(0)-a[nr//2:].mean(0))/.05
        norm=max(np.linalg.norm(rate),np.sqrt(len(rate)));err=float(np.linalg.norm(rate-old)/norm);noise=float(np.linalg.norm(split)/norm)
        samples=a.reshape(-1,N,27).sum(1);low,high=np.quantile(samples,[.025,.975],axis=0)
        target=fixed['native_counts'][:,jg];pred=fixed['counts_projected_private'][:,jg]
        rows.append(dict(group=int(groups[jg]),threshold_case=c['conditions'][j//G],N=N,replicates=nr,
            observed_native_count=int(target.sum()),MC_mean_count=float(samples.sum(1).mean()),fixed_rate_count=float(pred.sum()),
            native_windows_outside_model_95percent=int(((target<low)|(target>high)).sum()),windows=27,
            numerical_L2=err,MCsplit_L2=noise,numerical_pass=bool(err<=max(.05,3*noise)),
            rate_to_MC_L2=float(np.linalg.norm(pred-samples.mean(0))/max(np.linalg.norm(samples.mean(0)),1)),
            native_to_MC_L2_descriptive=float(np.linalg.norm(target-samples.mean(0))/max(np.linalg.norm(samples.mean(0)),1))))
    write(LOCAL/'result.json',dict(status='COMPLETE',rows=rows,numerical_all_pass=all(r['numerical_pass'] for r in rows),
        scope=c['scope'],predictive_bounds=c['interpretation'],model_promoted=False))
    print(rows,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','run']);p.add_argument('--device',type=int,default=1);a=p.parse_args()
    register() if a.command=='register' else run(a.device)
