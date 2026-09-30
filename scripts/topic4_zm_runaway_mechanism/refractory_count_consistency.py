"""Bounded finite-population rate diagnostic with count-consistent refractoriness.

No membrane particles. n[k]~Binomial(N-sum(previous refractory counts),p[k]).
The conditional expected flux is the existing local rate equation evaluated at
that available fraction. The deterministic skeleton and frozen weights remain.
The inherited Poisson private-variance approximation is explicitly NOT repaired.
"""
from common import OUT,np,read,write,log
import refractory_external_filter_pair as pair
from datetime import datetime
import argparse

DEST=OUT/'conditioned_refractory_count_consistency'

BINOMIAL=r'''
__device__ int draw_binomial(int n,double p,double u){
 if(n==0 || p<=0.)return 0;if(p>=1.)return n;
 bool flip=p>.5;double q=flip?1.-p:p;
 double mass=exp(n*log1p(-q)),cdf=mass;int k=0;
 while(u>cdf && k<n){mass*=((double)(n-k)/(k+1))*q/(1.-q);cdf+=mass;k++;}
 return flip?n-k:k;
}
extern "C" __global__ void sample_binomial(const int* n,const double* p,int* out,int repetitions,unsigned long long seed){
 int i=blockIdx.x*blockDim.x+threadIdx.x,condition=i/repetitions;if(condition>=48)return;
 curandStatePhilox4_32_10_t rng;curand_init(seed,i,0,&rng);out[i]=draw_binomial(n[condition],p[condition],curand_uniform_double(&rng));
}
extern "C" __global__ void quantiles(const int* n,const double* p,const double* u,int* out,int samples){
 int i=blockIdx.x*blockDim.x+threadIdx.x,c=i/samples;if(c<48)out[i]=draw_binomial(n[c],p[c],u[i%samples]);
}
'''

def count_code(P):
    code=pair.corrected_code(P).replace('#include <curand_kernel.h>','#include <curand_kernel.h>\n'+BINOMIAL)
    old='if(pars[20*P+g]>.5)syn[4*P+g]=em*syn[4*P+g]+(1.-em)*.5*E*r;'
    assert code.count(old)==1;code=code.replace(old,'')
    old='if(noise){curandStatePhilox4_32_10_t rng;curand_init(seed,g,(unsigned long long)tick,&rng);double n=pars[21*P+g];unsigned int count=curand_poisson(&rng,fmax(r,0.)*n*dt);r=(double)count/(n*dt);}'
    assert code.count(old)==1
    new=r'''if(noise){
 int N=(int)llround(pars[21*P+g]),nref=(int)llround(pars[P+g]/dt),occupied=0;
 for(int j=1;j<nref;j++){int slot=(tick-j)%depth;if(slot<0)slot+=depth;occupied+=(int)llround(history[(long long)slot*P+g]*N*dt);}
 int available=N-occupied;double probability=available>0?r*N*dt/available:0.;
 if(available<0 || available>N || probability< -1e-10 || probability>1.+1e-10){r=nan("");}
 else{curandStatePhilox4_32_10_t rng;curand_init(seed,g,(unsigned long long)tick,&rng);
 int count=draw_binomial(available,probability,curand_uniform_double(&rng));r=(double)count/(N*dt);}
 }
 if(pars[20*P+g]>.5)syn[4*P+g]=em*syn[4*P+g]+(1.-em)*.5*E*r;'''
    return code.replace(old,new)

class CountEngine(pair.FilteredEngine):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.count_module=self.cp.RawModule(code=count_code(self.s.P),options=('--fmad=false',),name_expressions=['finish','sample_binomial','quantiles'])
        self.k['finish']=self.count_module.get_function('finish')
        if self.noise:
            # local_rate writes the conditional mean into the new slot, then
            # finish overwrites it with the actual count; earlier slots stay counts.
            self.local.history=self.transport.history

def register():
    assert read(pair.DEST/'jobs.json')['status']=='COMPLETE'
    audit=read(OUT/'conditioned_refractory_spatial_diagnostic/emitted_refractory_bound_audit.json')
    assert any(x['violations']>0 for x in audit['rows'])
    DEST.mkdir(exist_ok=True);assert not (DEST/'contract.json').exists()
    c=read(pair.DEST/'contract.json');c.update(created_local=datetime.now().astimezone().isoformat(),
        status='REGISTERED_BEFORE_COUNT_CONSISTENCY_DIAGNOSTIC',
        question='Do physical finite-count refractoriness andcount-drivenM change thefailednative dynamics under the same frozenrate response?',
        only_change='Finite output:Binomial(actual available groupcount,p) replacesPoisson(expected groupcount); the same emittedcounts driveownrefractoryhistory,synapticdelays andM. Conditional rate usesactualavailablefraction. Deterministicrate equations,physicalweights,externalAMPAfilter,Z andresponseweights unchanged.',
        reason='SavedPoisson counts violate the finite2msE/1msI population firingbound; filteredseed1 has4906E and2018I alignedgroup-windowviolations. This proves aphysicalinconsistency, notthatitcaused theonseterror.',
        baseline=str(pair.DEST/'recorded_drive_poisson_seed1'),
        runs=[dict(label='recorded_drive_binomial_seed1',drive='seed9108401',noise=True,seed=1)],
        noise='One conditionalBinomial draw pergroup/timestep with originalgroupNandabsolute refractorytime. Noindividualmembranevariables orparticle simulation.',
        limitations='Private diffusion still inherits stationaryPoisson shared/private subtraction; this is one finite-output consistency diagnostic, not an exact mesoscopic renewal-input closure. Newlocalvalidation unchangedFAIL. Noadaptivefit, extra seeds, parametersearch or bifurcationpromotion.',
        implementation_gate='ExactBinomialinverseCDF checks includingp endpoints; fixedsamplingmean/variance; finitecounts,allrefractorywindows,conditionalavailableflux,andMupdates checkedbefore fullrun.')
    write(DEST/'contract.json',c)

def check(device):
    from scipy.stats import binom
    e=CountEngine(noise=True,device=device);cp=e.cp;s=e.s
    ns=np.repeat(np.array([0,1,5,20,80,105]),8);ps=np.tile([0.,.0001,.01,.1,.5,.8,.99,1.],6);R=20000
    uniforms=np.array([1e-10,.0001,.01,.1,.5,.9,.99,.9999,1-1e-10]);qout=cp.empty((48,len(uniforms)),dtype=cp.int32)
    e.count_module.get_function('quantiles')(((qout.size+127)//128,),(128,),(cp.asarray(ns,dtype=cp.int32),cp.asarray(ps),cp.asarray(uniforms),qout,np.int32(len(uniforms))))
    # Use the same complement convention: at a discrete CDF jump, ppf(1-u,p)
    # selects the opposite endpoint from N-ppf(u,1-p). The latter is intended.
    oracle=np.array([n-binom.ppf(uniforms,n,1-p) if .5<p<1 else binom.ppf(uniforms,n,p) for n,p in zip(ns,ps)])
    assert np.array_equal(qout.get(),oracle)
    output=cp.empty((48,R),dtype=cp.int32);kernel=e.count_module.get_function('sample_binomial')
    kernel(((48*R+127)//128,),(128,),(cp.asarray(ns,dtype=cp.int32),cp.asarray(ps),output,np.int32(R),np.uint64(920082)))
    observed=output.get();rows=[]
    for i,(n,p) in enumerate(zip(ns,ps)):
        assert observed[i].min()>=0 and observed[i].max()<=n
        mean=n*p;var=n*p*(1-p);em=float(observed[i].mean()-mean);ev=float(observed[i].var(ddof=1)-var)
        if var:
            sem=np.sqrt(var/R);fourth=3*var*var+var*(1-6*p*(1-p))
            # Exact finite-R variance of the unbiased sample variance. The
            # asymptotic term alone vanishes for Bernoulli p=.5.
            sev=np.sqrt((fourth-(R-3)/(R-1)*var*var)/R)
            assert abs(em)<7*sem and abs(ev)<7*sev,(n,p,em,ev,sem,sev)
        else:assert em==0 and ev==0
        rows.append(dict(N=int(n),p=float(p),mean_error=em,variance_error=ev))
    e.graph();e.chunk();e.chunk();assert e.local.history.data.ptr==e.transport.history.data.ptr
    # Active prefix: compare available fraction / output count / M for every step.
    worst_integer=0.;worst_m=0.;steps=500
    for _ in range(steps):
        tick=int(e.local.clock.get()[0]);history=e.transport.history.get();oldm=e.syn[4].get();available=np.empty(s.P)
        for pop,mask in [('E',s.E),('I',~s.E)]:
            nref=round(float(s.ref[mask][0])/e.dt);counts=history[(tick+1-np.arange(1,nref))%len(history)][:,mask]*s.sizes[mask]*e.dt
            worst_integer=max(worst_integer,float(abs(counts-np.rint(counts)).max()))
            available[mask]=s.sizes[mask]-np.rint(counts).sum(0)
        assert available.min()>=0 and np.all(available<=s.sizes)
        e.step();emitted=e.emitted.get();counts=emitted*s.sizes*e.dt
        worst_integer=max(worst_integer,float(abs(counts-np.rint(counts)).max()))
        assert counts.min()>=0 and np.all(counts<=available+1e-10)
        expected=oldm*np.exp(-e.dt/1000)+(1-np.exp(-e.dt/1000))*.5*s.E*emitted
        worst_m=max(worst_m,float(abs(expected-e.syn[4].get()).max()))
        assert np.array_equal(e.transport.history[(tick+1)%len(history)].get(),emitted)
    assert worst_integer<1e-10 and worst_m<1e-12
    # Noise-off dynamics must remain bitwise unchanged, including M.
    base=pair.FilteredEngine(noise=False,device=device);new=CountEngine(noise=False,device=device);base.graph();new.graph()
    for _ in range(2):
        x=base.chunk();y=new.chunk();assert np.array_equal(x,y)
        assert np.array_equal(base.syn.get(),new.syn.get()) and np.array_equal(base.local.state.get(),new.local.state.get())
    write(DEST/'implementation_check.json',dict(status='PASS',binomial_conditions=rows,repetitions=R,exact_quantile_checks=int(qout.size),
        group_steps_checked=steps*s.P,maximum_count_integer_error=worst_integer,M_update_max_error=worst_m,
        expected_skeleton_20ms_bitwise=True,all_count_bounds=True,scope='Implementation only; whole-network andprivatevarianceclosure validation stillrequired.'))
    log('COUNT CONSISTENCY IMPLEMENTATION PASS',worst_integer,worst_m)

def run(device):
    pair.DEST=DEST;pair.FilteredEngine=CountEngine;pair.run(device)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','check','run']);p.add_argument('--device',type=int,default=0);a=p.parse_args();{'register':register,'check':lambda:check(a.device),'run':lambda:run(a.device)}[a.command]()
