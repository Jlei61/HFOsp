"""Autonomous voltage-memory integration and independent feedback checks."""
from voltage_memory_rate import *
from validate_refractory_rate_response import evaluate as evaluate_parent
from scipy.special import expit, logit
from scipy.optimize import brentq


@njit
def integrate(pre, vweight, W2, b2, W3, b3, base, mu, dt, ref, tau, theta):
    nref=round(ref/dt); history=np.zeros(nref); occupied=0.; v=VR; a=np.exp(-dt/tau)
    rates=np.empty(len(base)); voltages=np.empty(len(base)); minimum=1.
    for k in range(len(base)):
        slot=k%nref; occupied-=history[slot]; available=1-occupied
        if available < -1e-10 or available > 1+1e-10: raise ValueError('Unphysical refractory mass')
        vpre=a*v+(1-a)*(VR+(mu[k]-VR)*available); voltages[k]=vpre
        h=np.tanh(pre[k]+vweight*(np.arcsinh((vpre-VR)/(theta-VR))/3))
        ell=base[k]+W3@np.tanh(W2@h+b2)+b3+np.log(dt/DT0)
        p=1/(1+np.exp(-ell)) if ell>=0 else np.exp(ell)/(1+np.exp(ell))
        fired=available*p; history[slot]=fired; occupied+=fired
        rates[k]=1000*fired/dt; v=vpre-(theta-VR)*fired; minimum=min(minimum,1-occupied)
    return rates,voltages,minimum


def direct(net,f,base,dt,theta=18.):
    w={k:v.detach().numpy() for k,v in net.state_dict().items()}
    W=w['network.0.weight']@w['transform']; b=w['network.0.bias']-W@w['center']
    pre=np.ascontiguousarray(f@W[:,:39].T+b); mu=physical_from_features(f,theta)[:,0].copy()
    return integrate(pre,np.ascontiguousarray(W[:,39]),w['network.2.weight'],w['network.2.bias'],
        w['network.4.weight'][0],float(w['network.4.bias'][0]),base,mu,dt,net.ref,net.tau,theta)


def evaluate(net,base,wave,T,dt,burn,steps,theta=18.):
    f=features(wave,T,dt,burn,steps,net.pop,theta,include_burn=True)
    b=base.evaluate(physical_from_features(f,theta),theta); r,v,m=direct(net,f,b,dt,theta)
    assert np.isfinite(r).all() and r.min()>=0 and np.isfinite(v).all()
    return r[burn:],m


def equilibria(net,base,physical,theta=18.,dt=DT0):
    f=np.zeros((1,39)); f[0,:3]=normalized_input(np.asarray(physical)[None],theta)[0]/SCALE
    b=float(base.evaluate(np.asarray(physical)[None],theta)[0]); w={k:v.detach().numpy() for k,v in net.state_dict().items()}
    def residual(rate):
        v=stationary_voltage(net.pop,physical[0],rate,theta,dt)
        ff=np.r_[f[0],voltage_feature(v,theta)]
        h=np.tanh(w['network.0.weight']@((ff-w['center'])@w['transform'].T)+w['network.0.bias'])
        ell=b+float(w['network.4.weight']@np.tanh(w['network.2.weight']@h+w['network.2.bias'])+w['network.4.bias'])
        return rate-net.maximum*expit(ell+np.log(net.ref/DT0))
    grid=np.unique(np.r_[0,np.geomspace(1e-10,net.maximum,512)]); y=np.array([residual(x) for x in grid]); roots=[]
    for i in range(len(grid)-1):
        if y[i]*y[i+1]<0: roots.append(brentq(residual,grid[i],grid[i+1],xtol=1e-11))
    if abs(y[0])<1e-12: roots.insert(0,0.)
    return roots,residual


@njit
def scalar_response(ell0,beta,gamma,mu0,frequency,amplitude,dt,ref,tau,burn,steps):
    nref=round(ref/dt); h=np.zeros(nref); occupied=0.; v=VR; a=np.exp(-dt/tau); r=np.empty(steps)
    for k in range(burn+steps):
        slot=k%nref; occupied-=h[slot]; avail=1-occupied
        mu=mu0+amplitude*np.cos(2*np.pi*frequency*(k+1)*dt/1000)
        vpre=a*v+(1-a)*(VR+(mu-VR)*avail)
        ell=ell0+beta*np.arcsinh((vpre-VR)/7)/3+gamma*(mu-mu0)
        p=1/(1+np.exp(-ell)); fired=avail*p; h[slot]=fired; occupied+=fired; v=vpre-7*fired
        if k>=burn: r[k-burn]=1000*fired/dt
    return r


def check():
    torch.set_num_threads(2); parents,bases,_=load_parent(); rows=[]
    for pop in 'EI':
        net=embed(pop,parents[pop]); d=np.load(DEST/f'training_arrays/{pop}.npz'); idx=np.arange(0,len(d['flux_fired']),733)
        ff=torch.tensor(d['flux_features'][idx].astype(float)); b=torch.tensor(d['flux_logits'][idx].astype(float))
        err=float(abs(net.logits(ff,b)-parents[pop].logits(ff[:,:39],b)).max().detach()); assert err<1e-8
        idx=np.arange(0,len(d['linear_target']),193)
        keys=['linear_features','linear_logits','linear_base_gradient','linear_input_gradient','linear_channel','linear_bank','linear_cov','linear_K']
        args=[torch.tensor(d[k][idx].astype('complex128' if np.iscomplexobj(d[k]) else ('int64' if k=='linear_channel' else 'float64'))) for k in keys]
        rate=net.rate_given_voltage(args[0],args[1]).detach()
        newer=net.linear_at_equilibrium(*args,torch.tensor(d['linear_Umu'][idx].astype('c16')),torch.tensor(d['linear_Urate'][idx].astype('c16')),rate).detach().numpy()
        older=parents[pop].linear_response(args[0][:,:39],*args[1:]).detach().numpy()
        ge=float(abs(newer-older).max()); assert ge<1e-8
        profiles=read(OUT/'refractory_rate_response/profiles.json')['rows']; info=next(r for r in profiles if r['split']=='train' and r['pop']==pop)
        wave=np.load(OUT/'refractory_rate_response/prepared.npz')['wave'][info['id']]; T=info['period_ms']; steps=round(T/.1); burn=round(T/.1)
        expected,_=evaluate_parent(parents[pop],bases[pop],wave,T,.1,burn,steps)
        actual,_=evaluate(net,bases[pop],wave,T,.1,burn,steps); re=float(abs(expected-actual).max()); assert re<1e-7
        # Independent scalar recurrence at constant flux verifies the DC state.
        for mu in [-30.,16.,60.]:
            rate0=20.; dt=.1; fired=np.full(10000,rate0*dt/1000)
            v=voltage_trace(np.full(len(fired),mu),fired,dt,net.tau,net.ref)
            steady=float(stationary_voltage(pop,mu,rate0)); assert abs(v[-1]-steady)<1e-10
        # Both signs of nonzero feedback, not merely the zero-weight embedding.
        for beta in [-1.,1.]:
            rate0=20.; mu=16.; p=rate0*DT0/1000/(1-(net.ref-DT0)*rate0/1000)
            v0=stationary_voltage(pop,mu,rate0); ell0=logit(p)-beta*voltage_feature(v0)
            ellv=beta/(21*np.cosh(3*voltage_feature(v0))); gamma=.1
            for freq in [5.,25.]:
                steps=40000; burn=20000; amp=1e-4
                r=scalar_response(ell0,beta,gamma,mu,freq,amp,DT0,net.ref,net.tau,burn,steps)
                t=(np.arange(burn,burn+steps)+1)*DT0; measured=2*np.mean(r*np.exp(-2j*np.pi*freq*t/1000))/amp
                _,_,K=transfer_factors(pop,[freq]); Umu,Ur=voltage_factors(pop,[freq],mu,rate0)
                theory=rate0*(1-p)*(gamma+ellv*Umu[0])/((1-p)+p*K[0]/DT0-rate0/1000*(1-p)*ellv*Ur[0])
                error=float(abs(measured-theory)/max(abs(theory),1e-10)); assert error<1e-5,(pop,beta,freq,error)
                rows.append(dict(pop=pop,beta=beta,frequency_hz=freq,relative_closed_gain_error=error))
        rows.append(dict(pop=pop,parent_logit_error=err,parent_gain_error=ge,parent_trajectory_error_hz=re))
    write(DEST/'implementation_check.json',dict(status='PASS',rows=rows,static_voltage_check=True,
        scope='Numerical embedding, physical-coordinate recurrence and closed-feedback derivative checks, not scientific response or spatial acceptance.'))
    log('VOLTAGE MEMORY CHECK PASS',rows)


if __name__=='__main__': check()
