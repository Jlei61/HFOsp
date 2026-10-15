import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import src
src.__path__.append(str(ROOT.parent/'topic4-dual-core-z-bifurcation'/'src'))
import numpy as np
from scipy import sparse
from src.topic4_patient_zm_meanfield import homogeneous_one_cell_model
from src.topic4_dual_core_spatial_z_delay import CoarseDelayOperators,simulate_delayed_ou_trajectory
from src.topic4_fig5_local_bifurcation import Family,corrected_delay_matrix,arc_continue,polish_fold


def setup(parameter='z_loss'):
    model=homogeneous_one_cell_model()
    ops=CoarseDelayOperators(dt_ms=.1,max_delay_steps=2,
        **{'w_'+k+'_history':sparse.hstack([sparse.csr_matrix(getattr(model,'w_'+k))*.4,
                      sparse.csr_matrix(getattr(model,'w_'+k))*.6]).tocsr() for k in ('ee','ei','ie','ii')})
    return Family(model,np.array([[.95],[-.15],[.95**2],[-.95*.15],[.15**2]]),.4,parameter,ops)


def test_residual_jacobian_adapted_input():
    f=setup();x=np.array([.08,.12]);h=1e-6
    observed=np.column_stack([(f.f(x+h*v,1)-f.f(x-h*v,1))/(2*h) for v in np.eye(2)])
    np.testing.assert_allclose(f.jac(x,1),observed,rtol=2e-5,atol=1e-6)


def test_delay_tangent_matches_nonlinear_step_with_m():
    f=setup();m,z,z2,eta,tau,ops=f.at(1.);x=np.array([.08,.12]);n=1
    syn=np.array([m.tau_mem_e_ms*(m.w_ee@x[:n]),m.tau_mem_e_ms*(m.w_ei@x[n:]),
                  m.tau_mem_i_ms*(m.w_ie@x[:n]),m.tau_mem_i_ms*(m.w_ii@x[n:])])
    state=np.r_[x,syn.ravel(),np.repeat(x[0],2),np.repeat(x[1],2),tau*x[:n]]
    def step(q):
        r=simulate_delayed_ou_trajectory(m,ops,q[:2],z_field=z,z_second_moment=z2,
            ou_rate_e=np.zeros((1,1)),eta_m=eta,tau_m_slow_ms=tau,
            initial_synapses=q[2:6].reshape(4,1),initial_history_e=q[6:8].reshape(2,1),
            initial_history_i=q[8:10].reshape(2,1),initial_m=q[10:])
        return np.r_[r['final_rates'],r['final_synapses'].ravel(),r['final_history_e'].ravel(),
                     r['final_history_i'].ravel(),r['final_adaptation_state']]
    h=1e-6
    jac=np.column_stack([(step(state+h*v)-step(state-h*v))/(2*h) for v in np.eye(len(state))])
    np.testing.assert_allclose(corrected_delay_matrix(f,x,1).toarray(),jac,atol=1e-6,rtol=3e-5)


def test_pathway_direction_and_variance_scaling():
    f=setup('e_to_i');m,*_=f.at(1.2)
    np.testing.assert_allclose(m.w_ie,f.base.w_ie*1.2)
    np.testing.assert_allclose(m.v_ie,f.base.v_ie*1.2**2)
    np.testing.assert_array_equal(m.w_ei,f.base.w_ei)


def test_matched_adaptation_keeps_equilibria_changes_dynamics():
    f=setup('tau_m_matched');x=np.array([.08,.12])
    np.testing.assert_allclose(f.f(x,.5),f.f(x,2),atol=1e-13)
    assert np.linalg.norm((corrected_delay_matrix(f,x,.5)-corrected_delay_matrix(f,x,2)).toarray())>0


def test_fold_detector_on_known_normal_form():
    # F(x,p)=x^2+p: fold is at zero. Tests traversal and nondegeneracy.
    class Scalar:
        base=type('Base',(),{'n_cells':1,'count_e':np.ones(1)})()
        def f(self,x,p):return x*x+p
        def jac(self,x,p):return np.diag(2*x)
        def dp(self,x,p):return np.ones_like(x)
        def physical(self,x):return True
    f=Scalar()
    pts,reason=arc_continue(f,(np.array([.2]),-.04),(np.array([.19]),-.0361),(-.1,.1),max_steps=70)
    assert any(a[2]*b[2]<0 for a,b in zip(pts[:-1],pts[1:]))
    # Two variables give a separated second singular value.
    class Two(Scalar):
        def f(self,x,p):return np.array([x[0]**2+p,x[1]])
        def jac(self,x,p):return np.diag([2*x[0],1.])
        def dp(self,x,p):return np.array([1.,0.])
    r,x=polish_fold(Two(),np.array([.01,0.]),-.0001)
    assert r['confirmed'] and abs(r['parameter'])<1e-8
