"""Positive numerical retraction for periodic Newton proposals.

This changes the solver's update coordinates only. It never clips the model
flow or changes the residual; every proposal must pass full admissibility and
decrease the actual unchanged shooting residual.
"""
from common import np
from scipy.optimize import brentq


def phase_bound_projection(raw,normal,target,positive,fixed_zero):
    """Euclidean projection of a solver proposal onto bounds and phase.

    Solve the one-dimensional KKT multiplier exactly. For a feasible source
    x, this projection cannot increase distance from x, unlike restoring
    phase through an arbitrarily selected small signed subspace. This acts
    on Newton proposals only, never on the physical evolution.
    """
    def value(lam,return_vector=False):
        y=raw-lam*normal
        y[positive]=np.maximum(y[positive],0.)
        y[fixed_zero]=0.
        return y if return_vector else float(normal@y-target)
    f0=value(0.)
    if abs(f0)<1e-12:return value(0.,True)
    free=(~positive)&(~fixed_zero)
    derivative=float(normal[free]@normal[free])
    assert derivative>1e-14,'A nonzero unconstrained phase direction is required'
    other=f0/derivative
    if f0>0:lo,hi=0.,other
    else:lo,hi=other,0.
    # Piecewise-linear monotonic KKT equation; the unconstrained derivative
    # supplies a guaranteed bracket (apart from endpoint roundoff).
    for _ in range(8):
        if value(lo)>=0 and value(hi)<=0:break
        lo*=2;hi*=2
    lam=brentq(value,lo,hi,xtol=1e-14,rtol=1e-14)
    y=value(lam,True)
    assert abs(normal@y-target)<1e-9
    return y


def retract(A,x,delta,alpha,project_zero=False,diagnostics=None,project_all=False,restore_phase=True,orthogonal_phase=False,refractory_projection=False):
    raw=x+alpha*delta
    # An admissible Newton step already lies in the physical domain. Preserve
    # it exactly; an unnecessary nonlinear retraction destroys its local
    # Newton cancellation, particularly in small positive history entries.
    if (not restore_phase or abs(A.normal@(raw-A.xref))<1e-9) and A.admissible(raw):
        if diagnostics is not None:diagnostics['unmodified_admissible_Newton_step']=True
        return raw
    shape=(-1,A.c.P);a=x.reshape(shape);d=(alpha*delta).reshape(shape)
    y=(a+d).copy()
    positive=np.zeros_like(a,dtype=bool)
    positive[:11]=True;positive[47:]=True
    # M_I is identically zero and is never a degree of freedom.
    positive[4,~A.e.s.E]=False
    if project_all and restore_phase and orthogonal_phase:
        fixed=np.zeros_like(positive);fixed[4,~A.e.s.E]=True
        y=phase_bound_projection(raw,A.normal,float(A.normal@A.xref),positive.ravel(),fixed.ravel())
        if diagnostics is not None:
            diagnostics.update(phase_method='Exact Euclidean projection onto positivity and phase; numerical proposals only',
                original_step_norm=float(np.linalg.norm(alpha*delta)),
                projected_step_norm=float(np.linalg.norm(y-x)),phase_error=float(A.normal@(y-A.xref)))
        assert np.linalg.norm(y-x)<=np.linalg.norm(alpha*delta)*(1+1e-8)+1e-10
        if A.admissible(y):return y
        if refractory_projection:
            from core_a_refractory_projection import project
            return project(A,raw,x,restore_phase=restore_phase,diagnostics=diagnostics)
        return None
    use=positive&(a>0)&(d<0)
    # a/(1-d/a) is positive and has derivative d at zero step.
    # This ordering avoids both a*a underflow and d/a overflow.
    if project_all:
        # Euclidean projection of the numerical Newton proposal onto the
        # nonnegative coordinate bounds. Interior coordinates stay linear.
        # This is never applied to an integrated physical trajectory.
        crossed=positive&(y<0);y[crossed]=0
        if diagnostics is not None:diagnostics['projected_negative_coordinates']=int(crossed.sum())
    else:
        y[use]=(a[use]/(a[use]-d[use]))*a[use]
    boundary=positive&(a==0)&(d<0)
    if project_zero:
        # Bound-constrained solver proposal only. Neither an accepted model
        # trajectory nor its residual/variational flow is clipped. Hookstep
        # prediction must use this actual projected displacement.
        y[boundary]=0
    if diagnostics is not None:
        diagnostics['negative_steps_at_exact_zero']=int(boundary.sum())
        diagnostics['zero_boundary_projection']=bool(project_zero)
    y[4,~A.e.s.E]=0
    y=y.ravel();normal=A.normal
    if not restore_phase:
        pass
    elif np.count_nonzero(normal.reshape(shape)[:47])==0 and np.count_nonzero(normal.reshape(shape)[48:])==0:
        # A rate-section: a common positive scale of participating current
        # rate coordinates restores the exact original section plane.
        participating=normal!=0
        target=float(normal@A.xref);actual=float(normal@y)
        # Reversing the orientation of the same physical rate section must
        # not change the retraction: both target and actual may be negative.
        if target==0 or actual==0 or target/actual<=0:return None
        y[participating]*=target/actual
    else:
        # Input-memory variables may have either sign. Restore the phase in
        # that unconstrained subspace; positivity is left intact.
        free=np.zeros_like(normal);free.reshape(shape)[11:47]=normal.reshape(shape)[11:47]
        den=float(normal@free)
        if den<1e-12:return None
        y-=free*float(normal@(y-A.xref))/den
    if (restore_phase and abs(normal@(y-A.xref))>1e-9) or not A.admissible(y):
        if refractory_projection:
            from core_a_refractory_projection import project
            return project(A,raw,x,restore_phase=restore_phase,diagnostics=diagnostics)
        if diagnostics is not None:
            physical=y.reshape(shape)*A.c.scale/A.c.weight
            diagnostics.update(phase_error=float(normal@(y-A.xref)),
                min_syn_M=float(physical[:5].min()),min_covariance=float(physical[5:11].min()),
                min_history=float(physical[47:].min()))
            maximum=[]
            for mask,ref in [(A.e.s.E,2.),(~A.e.s.E,1.)]:
                h=physical[47:,mask];n=round(ref/A.e.dt)
                sums=np.vstack([np.zeros((1,h.shape[1])),np.cumsum(h,axis=0)])
                maximum.append(float(((sums[n:]-sums[:-n])*A.e.dt).max()))
            diagnostics['max_refractory_occupancy_E_I']=maximum
        return None
    return y
