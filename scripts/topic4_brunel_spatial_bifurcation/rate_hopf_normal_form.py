"""Hopf cubic solvability for the unchanged 935-group spatial rate DDE.

Eliminate only linear filters; retain all spatial groups and physical delays.
The rate eigenvector is normalized to max(abs(q))=0.001/ms. The amplitude
equation is zdot=(lambda_J*dJ+cubic*abs(z)**2)*z. This normalization fixes
the coefficient's scale; its real sign determines local criticality.
"""
from rate_field import *
from scipy.sparse.linalg import eigs,spsolve

PERIODIC_OUT=RATE_OUT/'periodic_completion'


def local_derivatives(s,r,J):
    import torch
    torch.set_num_threads(1)
    z=torch.tensor(np.array(s.moments(r,J)),dtype=torch.float64,requires_grad=True)
    tm,ref,theta=[torch.tensor(x) for x in (s.tm,s.ref,s.theta)]
    xx=torch.tensor(s.nodes);ww=torch.tensor(s.weights)
    mu,ve,vi=z;var=ve+vi;sig=var.sqrt()
    shift=1.0325*((var/(ve/s.tau[0]+vi/s.tau[1]))/tm).sqrt()
    lo=(s.p['V_reset']-mu)/sig+shift;hi=(theta-mu)/sig+shift
    x=(lo+hi)[:,None]/2+(hi-lo)[:,None]/2*xx
    value=1/(ref+tm*np.sqrt(np.pi)*(hi-lo)/2*(ww*torch.special.erfcx(-x)).sum(1))
    first=torch.autograd.grad(value.sum(),z,create_graph=True)[0]
    second=torch.stack([torch.autograd.grad(first[i].sum(),z,create_graph=True,retain_graph=True)[0] for i in range(3)])
    third=torch.stack([torch.stack([torch.autograd.grad(second[i,j].sum(),z,retain_graph=True)[0] for j in range(3)]) for i in range(3)])
    return [v.detach().numpy() for v in (first,second,third)]


def moment_mode(s,J,lam,v):
    a,b,qa,qb=s.matrices(J,lam)
    mu=s.tm*(s.area[0]*(a@v)/((1+lam*s.rise[0])*(1+lam*s.decay[0]))-
             s.Z*s.area[1]*(b@v)/((1+lam*s.rise[1])*(1+lam*s.decay[1])))-.5*s.E*v/(1+1000*lam)
    ve=s.tm*s.area[0]**2*(qa@v)/(1+lam*s.tau[0]/2)
    vi=s.tm*(s.Z*s.area[1])**2*(qb@v)/(1+lam*s.tau[1]/2)
    return np.array([mu,ve,vi])


def compute(s,core,source=None,parameter_step=1e-6):
    z=np.load(source or RATE_OUT/f'hopf_{core}.npz');r=z['rates'];J=float(z['J']);w=float(z['omega'])
    q=z['vector'];q=q/(abs(q).max()*1000);q*=np.exp(-1j*np.angle(q[np.argmax(abs(q))]))
    gm,B,C=local_derivatives(s,r,J)
    m=s.characteristic(r,J,1j*w);ev,p=eigs(m.conj().T,k=1,sigma=0,tol=1e-12);p=p[:,0]
    h=1e-6;der=(s.characteristic(r,J,1j*w+h)-s.characteristic(r,J,1j*w-h))/(2*h)
    den=np.vdot(p,der@q)
    u=moment_mode(s,J,1j*w,q)
    bil=lambda a,b:np.einsum('ijp,ip,jp->p',B,a,b)
    h11=spsolve(s.characteristic(r,J,0).tocsc(),bil(u,u.conj()))
    h20=spsolve(s.characteristic(r,J,2j*w).tocsc(),bil(u,u))
    n21=.5*np.einsum('ijkp,ip,jp,kp->p',C,u,u,u.conj())
    n21+=bil(u,moment_mode(s,J,0,h11))+.5*bil(u.conj(),moment_mode(s,J,2j*w,h20))
    cubic=np.vdot(p,n21)/den
    hj=parameter_step
    rp,ok,_=s.solve(J+hj,r);assert ok
    rm,ok,_=s.solve(J-hj,r);assert ok
    dJ=(s.characteristic(rp,J+hj,1j*w)-s.characteristic(rm,J-hj,1j*w))/(2*hj)
    crossing=-np.vdot(p,dJ@q)/den
    slope=-cubic.real/crossing.real;dw=cubic.imag+crossing.imag*slope
    checks=[]
    # Independent local Taylor check, including the third order term.
    for eps in [.04,.02,.01]:
        d=u.real;mom=np.array(s.moments(r,J));actual=s.phi(*(mom+eps*d))-s.phi(*(mom-eps*d))-2*eps*np.sum(gm*d,axis=0)
        pred=eps**3/3*np.einsum('ijkp,ip,jp,kp->p',C,d,d,d)
        checks.append(dict(epsilon=eps,relative_cubic_error=float(np.linalg.norm(actual-pred)/np.linalg.norm(pred))))
    row=dict(core=core,J_EE_core=J,omega_per_ms=w,frequency_hz=w*1000/(2*np.pi),
        eigenvector_max_rate_hz=1.,cubic=cubic,lambda_J=crossing,J_shift_per_amplitude_squared=slope,
        omega_shift_per_amplitude_squared=dw,criticality='subcritical' if cubic.real>0 else 'supercritical',
        first_derivative_relative_error=float(np.linalg.norm(gm-np.array(s.gains(s.moments(r,J))))/np.linalg.norm(gm)),
        cubic_taylor_checks=checks,characteristic_residual=float(np.linalg.norm(m@q)/np.linalg.norm(q)),
        normalization='max abs rate eigenvector 1 Hz; z is dimensionless; cubic per ms',
        parameter_step=hj,stationary_continuation_rate_steps_Hz=[float(max(abs(rp-r))*1000),float(max(abs(rm-r))*1000)],
        source=str(source or RATE_OUT/f'hopf_{core}.npz'),
        interpretation='Local Hopf center manifold. Criticality alone does not imply full-system stability when other transverse modes are already unstable.')
    PERIODIC_OUT.mkdir(exist_ok=True)
    np.savez_compressed(PERIODIC_OUT/f'normal_form_{core}.npz',r=r,J=J,w=w,q=q,h11=h11,h20=h20,cubic=cubic,lambda_J=crossing)
    write(PERIODIC_OUT/f'normal_form_{core}.json',row)
    print(row,flush=True);return row


if __name__=='__main__':
    s=RateField();rows=[compute(s,c) for c in 'AB'];write(PERIODIC_OUT/'hopf_criticality.json',dict(rows=rows))
