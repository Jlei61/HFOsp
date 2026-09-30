"""Autonomous CPU integration and independent checks for reset-memory rate."""
from reset_memory_rate import *
from validate_refractory_rate_response import evaluate as evaluate_parent
from scipy.special import expit,logit
from scipy.optimize import brentq


@njit
def integrate(preactivation,qweight,W2,b2,W3,b3,base,dt,ref,tau):
    nref=round(ref/dt);history=np.zeros(nref);occupied=0.;q=0.;a=np.exp(-dt/tau)
    rates=np.empty(len(base));qs=np.empty(len(base));minimum=1.
    for k in range(len(base)):
        slot=k%nref;occupied-=history[slot];available=1-occupied
        if available < -1e-10 or available > 1+1e-10:raise ValueError('Unphysical available mass')
        qs[k]=q;h=np.tanh(preactivation[k]+qweight*(np.log1p(q)/3));v=np.tanh(W2@h+b2)
        ell=base[k]+W3@v+b3+np.log(dt/DT0)
        probability=1/(1+np.exp(-ell)) if ell>=0 else np.exp(ell)/(1+np.exp(ell))
        fired=available*probability;history[slot]=fired;occupied+=fired
        rates[k]=1000*fired/dt;q=a*(q+fired);minimum=min(minimum,1-occupied)
    return rates,qs,minimum


def direct(net,f,base,dt):
    # Fuse the fixed invertible coordinates; q itself remains endogenous.
    weights={k:v.detach().numpy() for k,v in net.state_dict().items()}
    W=weights['network.0.weight']@weights['transform'];b=weights['network.0.bias']-W@weights['center']
    pre=np.ascontiguousarray(f@W[:,:39].T+b)
    return integrate(pre,np.ascontiguousarray(W[:,39]),weights['network.2.weight'],weights['network.2.bias'],weights['network.4.weight'][0],float(weights['network.4.bias'][0]),base,dt,net.ref,net.tau)


def evaluate(net,base,wave,T,dt,burn,steps,theta=18.):
    f=features(wave,T,dt,burn,steps,net.pop,theta,include_burn=True)
    b=base.evaluate(physical_from_features(f,theta),theta);r,q,m=direct(net,f,b,dt)
    assert np.isfinite(r).all() and r.min()>=0 and np.isfinite(q).all()
    return r[burn:],m


def equilibria(net,base,physical,theta=18.,dt=DT0):
    """Find all sign-bracketed equilibria; do not assert global completeness."""
    f=np.zeros((1,39));f[0,:3]=normalized_input(np.asarray(physical)[None],theta)[0]/SCALE
    b=float(base.evaluate(np.asarray(physical)[None],theta)[0]);K0=float(reset_kernel(net.pop,[0],dt)[0].real)
    W={k:v.detach().numpy() for k,v in net.state_dict().items()}
    def residual(rate):
        ff=np.r_[f[0],np.log1p(K0*rate/1000)/3];h=np.tanh(W['network.0.weight']@((ff-W['center'])@W['transform'].T)+W['network.0.bias'])
        ell=b+float(W['network.4.weight']@np.tanh(W['network.2.weight']@h+W['network.2.bias'])+W['network.4.bias'])
        return rate-net.maximum*expit(ell+np.log(net.ref/DT0))
    grid=np.unique(np.r_[0,np.geomspace(1e-10,net.maximum,512)]);y=np.array([residual(x) for x in grid]);roots=[]
    for i in range(len(grid)-1):
        if y[i]*y[i+1]<0:roots.append(brentq(residual,grid[i],grid[i+1],xtol=1e-11))
    if abs(y[0])<1e-12:roots.insert(0,0.)
    return roots,residual


@njit
def scalar_response(ell0,beta,frequency,amplitude,dt,ref,tau,burn,steps):
    nref=round(ref/dt);h=np.zeros(nref);q=0.;occupied=0.;a=np.exp(-dt/tau);r=np.empty(steps)
    for k in range(burn+steps):
        slot=k%nref;occupied-=h[slot]
        ell=ell0+beta*np.log1p(q)/3+amplitude*np.cos(2*np.pi*frequency*(k+1)*dt/1000)
        p=1/(1+np.exp(-ell));fired=(1-occupied)*p;h[slot]=fired;occupied+=fired;q=a*(q+fired)
        if k>=burn:r[k-burn]=1000*fired/dt
    return r


def check():
    torch.set_num_threads(2);parents,bases,_=load_parent();rows=[]
    for pop in 'EI':
        net=embed(pop,parents[pop]);d=np.load(DEST/f'training_arrays/{pop}.npz');idx=np.arange(0,len(d['flux_fired']),733)
        ff=torch.tensor(d['flux_features'][idx].astype(float));b=torch.tensor(d['flux_logits'][idx].astype(float))
        error=float(abs(net.logits(ff,b)-parents[pop].logits(ff[:,:39],b)).max().detach());assert error<1e-8
        idx=np.arange(0,len(d['linear_target']),193);args=[torch.tensor(d[k][idx].astype('complex128' if np.iscomplexobj(d[k]) else ('int64' if k=='linear_channel' else 'float64'))) for k in ['linear_features','linear_logits','linear_base_gradient','linear_input_gradient','linear_channel','linear_bank','linear_cov','linear_K']]
        rate=net.rate_given_q(args[0],args[1]).detach();Kq=torch.tensor(d['linear_Kq'][idx].astype('c16'))
        newer=net.linear_at_equilibrium(*args,Kq,rate).detach().numpy();oldargs=[args[0][:,:39]]+args[1:]
        older=parents[pop].linear_response(*oldargs).detach().numpy();gainerr=float(abs(newer-older).max());assert gainerr<1e-8
        profiles=read(OUT/'refractory_rate_response/profiles.json')['rows'];info=next(r for r in profiles if r['split']=='train' and r['pop']==pop)
        inp=np.load(OUT/'refractory_rate_response/prepared.npz');wave=inp['wave'][info['id']];T=info['period_ms'];steps=round(T/.1);burn=round(T/.1)
        expected,_=evaluate_parent(parents[pop],bases[pop],wave,T,.1,burn,steps)
        actual,_=evaluate(net,bases[pop],wave,T,.1,burn,steps);err=float(abs(expected-actual).max());assert err<1e-7,(pop,err)
        impulse=np.zeros(32);impulse[0]=1;q=reset_trace(impulse,.1,net.tau);ref=np.r_[0,np.exp(-np.arange(1,32)*.1/net.tau)]
        assert np.max(abs(q-ref))<1e-14
        # Nonzero feedback check against actual recurrence, independently of NN.
        for beta in [-2.,1.]:
            rate0=20.;K0=float(reset_kernel(pop,[0])[0].real);q0=K0*rate0/1000
            p=rate0*DT0/1000/(1-(round(net.ref/DT0)-1)*rate0*DT0/1000)
            ell0=logit(p)-beta*np.log1p(q0)/3
            for freq in [5.,25.]:
                steps=round(4000/DT0);burn=round(2000/DT0);amp=1e-4
                r=scalar_response(ell0,beta,freq,amp,DT0,net.ref,net.tau,burn,steps)
                t=(np.arange(burn,burn+steps)+1)*DT0;measured=2*np.mean(r*np.exp(-2j*np.pi*freq*t/1000))/amp
                _,_,K=transfer_factors(pop,[freq]);Kq=reset_kernel(pop,[freq]);ellq=beta/(3*(1+q0))
                theory=rate0*(1-p)/((1-p)+p*K[0]/DT0-rate0/1000*(1-p)*ellq*Kq[0])
                errgain=float(abs(measured-theory)/max(abs(theory),1e-10));assert errgain<1e-5,(pop,beta,freq,errgain)
                rows.append(dict(pop=pop,beta=beta,frequency_hz=freq,relative_gain_error=errgain))
        rows.append(dict(pop=pop,embedding_logit_error=error,parent_gain_error=gainerr,parent_autonomous_rate_error_hz=err))
    write(DEST/'implementation_check.json',dict(status='PASS',rows=rows,causal_impulse_check=True,nonzero_feedback_gain_check=True,
        scope='Exact embedding, autonomous implementation and feedback derivative checks only; no scientific validation.'))
    log('RESET MEMORY CHECK PASS',rows)


if __name__=='__main__':check()
