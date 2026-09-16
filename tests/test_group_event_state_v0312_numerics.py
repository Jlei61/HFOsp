"""v0.3.12 numerical core: exact OU transition, split solver, evidence update.

These verify the filter kernel against an independent reference; they do not
claim anything about a learned human posterior.
"""
import math
import numpy as np
import pytest
import torch

from src.topic5_group_event_state.v0312 import numerics as N


def test_omega_basis_matches_upper_triangle_and_gradient_without_index_put():
    dyn=N.Dynamics(coupled=True)
    with torch.no_grad():dyn.omega.copy_(torch.linspace(-.2,.3,len(dyn.omega)))
    A=dyn.A();expected=torch.zeros(N.LATENT,N.LATENT)
    expected[dyn.omega_i,dyn.omega_j]=dyn.omega.detach()
    expected=expected-expected.T-torch.diag(1/dyn.tau_hours().detach())
    assert torch.equal(A.detach(),expected)
    weight=torch.arange(A.numel(),dtype=A.dtype).reshape_as(A)/A.numel()
    (A*weight).sum().backward()
    expected_grad=weight[dyn.omega_i,dyn.omega_j]-weight[dyn.omega_j,dyn.omega_i]
    assert torch.equal(dyn.omega.grad,expected_grad)


@pytest.fixture(autouse=True)
def _double_precision():
    """FP64 here only; the default dtype must not leak into other modules."""
    prev=torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)



def _reference_transition(A,sigma,dt,n=20000):
    """Independent F and Q: scipy matrix exponential plus Simpson quadrature."""
    from scipy.linalg import expm
    A=np.asarray(A,float);S=np.diag(np.asarray(sigma,float)**2)
    F=expm(A*dt)
    s=np.linspace(0.,dt,n+1)
    vals=np.stack([expm(A*t)@S@expm(A*t).T for t in s])
    w=np.ones(n+1);w[1:-1:2]=4;w[2:-1:2]=2
    Q=(dt/n/3.)*np.einsum('k,kij->ij',w,vals)
    return F,Q


def _dyn(coupled=False,nonlinear=False,seed=0):
    g=torch.Generator().manual_seed(seed)
    d=N.Dynamics(coupled=coupled,nonlinear=nonlinear,generator=g).double()
    with torch.no_grad():
        d.omega.normal_(0.,0.4,generator=g)
        d.log_sigma.copy_(torch.log(torch.full((N.LATENT,),0.1)))
    return d


@pytest.mark.parametrize('dt',[1./60.,7.5/60.,0.5,3.7,26.0])
def test_van_loan_matches_independent_quadrature(dt):
    d=_dyn(coupled=True)
    A=d.A().detach();sigma=d.sigma().detach()
    F,Q=N.ou_transition(A,sigma,dt)
    Fr,Qr=_reference_transition(A.numpy(),sigma.numpy(),dt)
    assert np.abs(F[0].numpy()-Fr).max()/max(np.abs(Fr).max(),1e-12)<1e-6
    assert np.abs(Q[0].numpy()-Qr).max()/max(np.abs(Qr).max(),1e-12)<1e-6


def test_sigma_squared_dt_is_not_a_legal_stand_in():
    """A rotating, decaying A makes the naive diagonal guess visibly wrong."""
    d=_dyn(coupled=True)
    A=d.A().detach();sigma=d.sigma().detach()
    _,Q=N.ou_transition(A,sigma,0.5)
    naive=torch.diag(sigma*sigma)*0.5
    assert (Q[0]-naive).abs().max()>0.05*naive.max()


def test_linear_semigroup_composes_exactly():
    d=_dyn(coupled=True)
    m=torch.randn(1,N.LATENT);P=torch.eye(N.LATENT).expand(1,-1,-1).clone()
    a,Pa=N.propagate_moments(d,m,P,1.0,max_step=1.0)
    b,Pb=N.propagate_moments(d,m,P,1.0,max_step=1./60.)
    assert (a-b).abs().max()<1e-9
    assert (Pa-Pb).abs().max()<1e-9


def _reference_kalman(A,sigma,steps,m0,P0):
    """Textbook prior/posterior recursion with an independent transition."""
    m=np.asarray(m0,float);P=np.asarray(P0,float);out=[]
    for dt,a,R in steps:
        F,Q=_reference_transition(A,sigma,dt,n=4000)
        m=F@m;P=F@P@F.T+Q
        prior=(m.copy(),P.copy())
        if a is not None:
            S=P+R
            K=np.linalg.solve(S.T,P.T).T
            innov=a-m
            m=m+K@innov
            IK=np.eye(len(m))-K
            P=IK@P@IK.T+K@R@K.T
            ll=-0.5*(innov@np.linalg.solve(S,innov)+np.linalg.slogdet(S)[1]+len(m)*math.log(2*math.pi))
        else:ll=None
        out.append((prior,(m.copy(),P.copy()),ll))
    return out


def test_filter_matches_reference_kalman_irregular_dt_strong_weak_and_gap():
    d=_dyn(coupled=True)
    A=d.A().detach().numpy();sigma=d.sigma().detach().numpy()
    rng=np.random.default_rng(20260906)
    eye=np.eye(N.LATENT)
    plan=[(1./60.,1.0),(1./60.,0.01),(4.3/60.,100.0),(0.5,0.01),
          (9.0,None),(1./60.,0.01),(1./60.,1.0),(2./60.,100.0)]
    steps=[]
    for dt,r in plan:
        steps.append((dt,None if r is None else rng.normal(size=N.LATENT),None if r is None else eye*r))
    ref=_reference_kalman(A,sigma,steps,np.zeros(N.LATENT),eye.copy())
    m=torch.zeros(1,N.LATENT);P=torch.eye(N.LATENT).expand(1,-1,-1).clone()
    worst_prior=worst_post=worst_ll=0.
    for (dt,a,R),(rp,rq,rll) in zip(steps,ref):
        m,P=N.propagate_moments(d,m,P,dt)
        worst_prior=max(worst_prior,float((P[0].detach().numpy()-rp[1]).max()/max(abs(rp[1]).max(),1e-12)))
        if a is None:continue
        m,P,ll,_=N.evidence_update(m,P,torch.as_tensor(a).reshape(1,-1),torch.as_tensor(R).reshape(1,N.LATENT,N.LATENT))
        worst_post=max(worst_post,float(abs(P[0].detach().numpy()-rq[1]).max()/max(abs(rq[1]).max(),1e-12)))
        worst_post=max(worst_post,float(abs(m[0].detach().numpy()-rq[0]).max()/max(abs(rq[0]).max(),1e-12)))
        worst_ll=max(worst_ll,abs(float(ll)-rll)/max(abs(rll),1e-12))
    assert worst_prior<1e-6 and worst_post<1e-6 and worst_ll<1e-6


def test_evidence_update_shrinks_inherited_variance():
    """Spec scalar clause: prior 0.98027 with R=0.01 must land near 0.009899.

    The rejected alternative gives every trajectory the same innovation shift
    and leaves the variance at the prior (~0.98), which is what this separates.
    """
    prior=0.98027;R=0.01
    m=torch.zeros(1,1);P=torch.full((1,1,1),prior)
    a=torch.full((1,1),0.37)
    mp,Pp,_,_=N.evidence_update(m,P,a,torch.full((1,1,1),R))
    expected=prior*R/(prior+R)
    assert abs(float(Pp)-expected)<1e-12
    assert abs(float(Pp)-0.0098990)<1e-6
    assert abs(float(mp)-prior/(prior+R)*0.37)<1e-12
    common_innovation=prior
    assert abs(float(Pp)-common_innovation)>0.9


def test_evidence_update_keeps_off_diagonal_coupling():
    P=torch.tensor([[[1.0,0.6],[0.6,1.0]]])
    R=torch.tensor([[[0.05,0.0],[0.0,5.0]]])
    m=torch.zeros(1,2);a=torch.tensor([[1.0,0.0]])
    _,Pp,_,_=N.evidence_update(m,P,a,R)
    assert abs(float(Pp[0,0,1]))>1e-3
    assert float(Pp[0,0,0])<float(P[0,0,0])
    assert float(Pp[0,1,1])<float(P[0,1,1])


def test_missing_packet_variance_settles_to_stationary_not_forced_growth():
    d=_dyn(coupled=False)
    m=torch.zeros(1,N.LATENT);P=torch.eye(N.LATENT).expand(1,-1,-1).clone()*4.0
    trace=[float(P[0].diagonal().mean())]
    for _ in range(6):
        m,P=N.propagate_moments(d,m,P,2.0)
        trace.append(float(P[0].diagonal().mean()))
    assert trace[-1]<trace[0]
    assert trace[-1]>0
    assert abs(trace[-1]-trace[-2])<abs(trace[1]-trace[0])


def _coupled_brownian_paths(d,z0,dt,n_grid,increments):
    """Same underlying Brownian increments consumed at two grid resolutions."""
    h=dt/n_grid
    A=d.A().detach();sigma=d.sigma().detach()
    z=z0.clone()
    fine=increments
    per=len(fine)//(2*n_grid)
    k=0
    for _ in range(n_grid):
        for half in range(2):
            F,_=N.ou_transition(A,sigma,h/2.)
            F=F[0]
            acc=torch.zeros_like(z)
            for j in range(per):
                s=(j+0.5)/per*(h/2.)
                Fs,_=N.ou_transition(A,sigma,h/2.-s)
                acc=acc+fine[k]@ (Fs[0]*sigma).transpose(0,1)
                k+=1
            z=z@F.transpose(0,1)+acc
            if half==0 and d.nonlinear:z=N.midpoint_map(d,z,h)
    return z


def test_split_solver_deterministic_grid_refinement():
    """Halving the step must shrink the splitting error on the same path."""
    d=_dyn(coupled=True,nonlinear=True,seed=3)
    with torch.no_grad():
        d.U.normal_(0.,0.35);d.V.normal_(0.,0.5)
    torch.manual_seed(11)
    dt=1./60.;n_fine=4*24
    zero=[torch.zeros(8,N.LATENT) for _ in range(n_fine)]
    z0=torch.randn(8,N.LATENT)*0.5
    a=_coupled_brownian_paths(d,z0,dt,1,zero)
    b=_coupled_brownian_paths(d,z0,dt,2,zero)
    c=_coupled_brownian_paths(d,z0,dt,4,zero)
    e=_coupled_brownian_paths(d,z0,dt,16,zero)
    first=float((a-e).abs().max());second=float((b-e).abs().max());third=float((c-e).abs().max())
    assert third<second<first
    assert third<0.3*first


def test_split_integration_error_is_below_noise_resampling_spread():
    """A redrawn Brownian path is not an integration error; the test separates them."""
    d=_dyn(coupled=True,nonlinear=True,seed=3)
    with torch.no_grad():
        d.U.normal_(0.,0.35);d.V.normal_(0.,0.5)
    torch.manual_seed(11)
    dt=1./60.;n_fine=4*24
    incr=[torch.randn(8,N.LATENT)*math.sqrt(dt/n_fine) for _ in range(n_fine)]
    other=[torch.randn(8,N.LATENT)*math.sqrt(dt/n_fine) for _ in range(n_fine)]
    z0=torch.randn(8,N.LATENT)*0.5
    b=_coupled_brownian_paths(d,z0,dt,2,incr)
    c=_coupled_brownian_paths(d,z0,dt,4,incr)
    fresh=_coupled_brownian_paths(d,z0,dt,4,other)
    integration=float((b-c).abs().max());resample=float((fresh-c).abs().max())
    assert integration<0.2*resample


def test_nonlinear_jacobian_matches_autograd():
    d=_dyn(coupled=False,nonlinear=True,seed=5)
    with torch.no_grad():
        d.U.normal_(0.,0.4);d.V.normal_(0.,0.7);d.b.normal_(0.,0.2)
    z=torch.randn(3,N.LATENT,requires_grad=True)
    h=1./60.
    J=N.midpoint_jacobian(d,z.detach(),h)
    auto=torch.stack([torch.autograd.functional.jacobian(lambda x:N.midpoint_map(d,x.reshape(1,-1),h).reshape(-1),z[i].detach())
                      for i in range(3)])
    assert float((J-auto).abs().max())<1e-9


def test_cholesky_reports_numerical_failure_instead_of_silent_repair():
    bad=torch.tensor([[[1.,10.],[10.,1.]]])
    with pytest.raises(FloatingPointError):
        N.cholesky_psd(bad,'deliberately singular')


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA parity requires a GPU')
def test_gpu_stationary_covariance_stays_on_gpu_and_preserves_gradients(monkeypatch):
    d=_dyn(coupled=True).cuda()
    seen=[];solve=torch.linalg.solve
    def record(a,b):
        seen.append(a.device.type)
        return solve(a,b)
    monkeypatch.setattr(torch.linalg,'solve',record)
    P=N.stationary_covariance(d.A(),d.sigma())
    weight=torch.arange(P.numel(),device=P.device,dtype=P.dtype).reshape_as(P)/P.numel()
    grads=torch.autograd.grad((P*weight).sum(),(d.log_tau,d.omega,d.log_sigma))
    assert seen==['cuda'] and P.device.type=='cuda'
    assert all(torch.isfinite(g).all() for g in grads)
    residual=d.A()@P+P@d.A().T+torch.diag(d.sigma()**2)
    assert float(residual.abs().max())<1e-8
