"""Independent derivative and finite-history checks before candidate fitting."""
from nonlinear_rate_response import *
from scipy.integrate import solve_ivp


def main():
    torch.set_num_threads(2);torch.manual_seed(920046);rng=np.random.default_rng(920046)
    rows=[]
    for pop in 'EI':
        net=RateReadout(pop).double();base=Baseline(pop)
        # Exercise all derivatives, rather than the initialized zero residual.
        with torch.no_grad():net.network[-1].weight.normal_(0,.1);net.network[-1].bias.fill_(.2)
        for theta in [14.2,18.]:
            count=20;physical=np.column_stack([11+(theta-11)*rng.uniform(.8,8,count),(theta-11)**2*rng.uniform(.2,3,count),(theta-11)**2*rng.uniform(.2,3,count)])
            features=np.zeros((count,39));features[:,:3]=normalized_input(physical,theta)/SCALE
            logits,bgrad=base.evaluate(physical,theta,True);igrad=normalized_jacobian(physical,theta)
            for channel in range(3):
                for freq in [0.,5.,80.]:
                    pred=net.linear_response(torch.tensor(features),torch.tensor(logits),torch.tensor(bgrad),torch.tensor(igrad),torch.full((count,),freq,dtype=torch.float64),torch.full((count,),channel,dtype=torch.long)).detach().numpy()
                    H=(1+2j*np.pi*freq*np.repeat(TAUS,3)/1000)**(-np.tile([1,2,3],4))-1
                    eps=1e-4;gain=[]
                    for part in ['real','imag']:
                        # Real perturbation includes the instantaneous input;
                        # imaginary component only perturbs the filter history.
                        direction=np.zeros((count,39));direction[:,3+12*channel:3+12*(channel+1)]=getattr(H,part)[None,:]*igrad[:,channel,None]
                        direction[:,channel]=igrad[:,channel] if part=='real' else 0
                        values=[]
                        for sign in [-1,1]:
                            shifted=features+sign*eps*direction
                            bp=physical.copy()
                            if part=='real':bp[:,channel]+=sign*eps
                            with torch.no_grad():values.append(net(torch.tensor(shifted),torch.tensor(base.evaluate(bp,theta))).numpy())
                        gain.append((values[1]-values[0])/(2*eps))
                    fd=gain[0]+1j*gain[1];error=float(np.max(abs(fd-pred)/(1+abs(pred))))
                    assert error<2e-7,(pop,theta,channel,freq,error)
                    rows.append(dict(pop=pop,theta=theta,channel=channel,frequency=freq,normalized_derivative_error=error))
    # Constant input from zero history: independent continuous ODE solution.
    wave=np.repeat(np.array([[23.],[100.],[200.]]),2,axis=1)
    u=normalized_input(wave[:,0])/SCALE;dt=.1;steps=1000
    f=history_features(wave,1000.,dt,0,steps)
    A=np.zeros((36,36));B=np.zeros((36,3))
    for c in range(3):
        for j,tau in enumerate(TAUS):
            first=12*c+3*j
            for k in range(3):
                A[first+k,first+k]=-1/tau
                if k:A[first+k,first+k-1]=1/tau
                else:B[first,c]=1/tau
    truth=solve_ivp(lambda t,h:A@h+B@u,[0,100.],np.zeros(36),rtol=2e-11,atol=2e-12,t_eval=np.arange(1,steps+1)*dt).y.T
    observed=f[:,3:]+np.repeat(f[:,:3],12,axis=1)
    error=float(np.max(abs(observed-truth)));assert error<2e-10,error
    # Sinusoidal forcing against exact continuous frequency response.
    T=200.;W=16384;t=np.arange(W)*T/W
    un=np.array([.5+.1*np.sin(2*np.pi*t/T),np.full(W,.4),np.full(W,.3)]).T
    physical=physical_from_features(np.column_stack([un,np.zeros((W,36))])).T
    errors=[]
    for dt in [.1,.05]:
        n=round(T/dt);features=history_features(physical,T,dt,round(2000/dt),n)
        time=np.arange(1,n+1)*dt
        pred_h=features[:,3:15]+features[:,0,None]
        H=(1+2j*np.pi*np.repeat(TAUS,3)/T)**(-np.tile([1,2,3],4))
        truth=.5+.1*np.imag(np.exp(2j*np.pi*time/T)[:,None]*H)
        errors.append(float(np.max(abs(pred_h-truth))))
    assert errors[1]<.55*errors[0] and errors[0]<.00016,errors
    result=dict(status='PASS',static_and_frequency_derivative_checks=rows,
        maximum_derivative_error=max(r['normalized_derivative_error'] for r in rows),
        constant_history_ODE_error=error,sinusoid_errors_dt01_dt005=errors,
        scope='Numerical implementation only; no fitted weights or validation data were used. Finite-input quadrature and learned model accuracy remain separate requirements.')
    write(DEST/'implementation_check.json',result);log('RATE IMPLEMENTATION', {k:v for k,v in result.items() if k!='static_and_frequency_derivative_checks'})


if __name__=='__main__':main()
