"""Analytic reconstruction of a cycle-fold tangent in physical delay time.

Differentiate the linear harmonic filters and each physical edge delay exactly.
The nonlinear periodic BVP tangent is supplied independently. This removes
subtraction error in the reconstructed initial vector; it does not change the
orbit, the tangent, or the full variational integration used to test them.
"""
from rate_periodic import *


def harmonic_tangent(s, r, T, J, tangent):
    dr=tangent[:-2].reshape(r.shape)/1000
    dT=T*tangent[-2];dJ=tangent[-1]/1000
    cf=np.fft.rfft(r,axis=0)/len(r)
    dc=np.fft.rfft(dr,axis=0)/len(r)
    states=np.empty((9,len(cf),s.P),complex)
    derivatives=np.empty_like(states)

    def divided(value, derivative, lam, dlam, tau):
        den=1+lam*tau
        return value/den, derivative/den-value*tau*dlam/den**2

    for k,(v,dv) in enumerate(zip(cf,dc)):
        lam=2j*np.pi*k/T;dlam=-lam*dT/T
        phase=np.exp(-lam*s.delays)
        arrivals=[];changes=[]
        for kind,(row,col,mask,delay) in enumerate(s.raw):
            power=1 if kind==0 else 2 if kind==2 else 0
            base=delay@phase
            mult=np.where(mask,J**power,1.) if power else np.ones(len(row))
            change=(delay@(-s.delays*phase))*dlam*mult
            if power:change+=base*np.where(mask,power*J**(power-1)*dJ,0.)
            def apply(a,x):
                return sparse.coo_matrix((a,(row,col)),shape=(s.P,s.P)).tocsr()@x
            arrivals.append(apply(base*mult,v))
            changes.append(apply(base*mult,dv)+apply(change,v))
        a,b,aa,bb=arrivals;da,db,daa,dbb=changes
        H=s.filter_response(lam)
        dH=(-s.alpha*s.tf/(1+lam*s.tf)**2
            -(1-s.alpha)*s.ts/(1+lam*s.ts)**2)*dlam
        drive=v/H;ddrive=dv/H-v*dH/H**2
        xf,dxf=divided(drive,ddrive,lam,dlam,s.tf)
        xs,dxs=divided(drive,ddrive,lam,dlam,s.ts)
        qa,dqa=divided(s.tm*s.area[0]*a,s.tm*s.area[0]*da,lam,dlam,s.rise[0])
        ia,dia=divided(qa,dqa,lam,dlam,s.decay[0])
        qg,dqg=divided(s.tm*s.area[1]*b,s.tm*s.area[1]*db,lam,dlam,s.rise[1])
        ig,dig=divided(qg,dqg,lam,dlam,s.decay[1])
        va,dva=divided(s.tm*s.area[0]**2*aa,s.tm*s.area[0]**2*daa,lam,dlam,s.tau[0]/2)
        vg,dvg=divided(s.tm*s.area[1]**2*bb,s.tm*s.area[1]**2*dbb,lam,dlam,s.tau[1]/2)
        m,dm=divided(.5*s.E*v,.5*s.E*dv,lam,dlam,1000.)
        states[:,k]=[xf,xs,qa,ia,qg,ig,va,vg,m]
        derivatives[:,k]=[dxf,dxs,dqa,dia,dqg,dig,dva,dvg,dm]
    return dict(states=states,derivatives=derivatives,rate=cf,rate_derivative=dc,
        lambdas=2j*np.pi*np.arange(len(cf))/T,T=T,dT=dT,dJ=dJ)


def at_time(data, D, dt, origin=0., xp=np):
    """Tangent at fixed physical times, including the period derivative."""
    lam=xp.asarray(data['lambdas']);dlam=-lam*data['dT']/data['T']
    factor=xp.full(len(lam),2.);factor[0]=1.;factor[-1]=1.
    phase=factor*xp.exp(lam*origin)
    state=xp.asarray(data['states']);derivative=xp.asarray(data['derivatives'])
    local=xp.sum(phase[None,:,None]*(derivative+origin*dlam[None,:,None]*state),axis=1).real
    times=origin-xp.arange(1,D+1)*dt
    history_phase=xp.exp(times[:,None]*lam[None,:])*factor[None,:]
    history=(history_phase@xp.asarray(data['rate_derivative'])+
             times[:,None]*(history_phase@(dlam[:,None]*xp.asarray(data['rate'])))).real
    result=xp.concatenate([local.ravel(),history.ravel()])
    return result if xp is np else result.get()


def reconstruct(s, r, T, J, tangent, D, dt):
    return at_time(harmonic_tangent(s,r,T,J,tangent),D,dt)
