"""Convex projection of Newton initial-state proposals, never physical flows.

The intersection contains positivity, fixed inhibitory M, the optional phase
plane, and every sliding refractory-occupancy halfspace. Dykstra corrections
are retained for each set; violated history halfspaces are added on demand.
The actual unchanged model residual still decides whether to accept a step.
"""
import numpy as np
from core_a_positive_newton_coordinates import phase_bound_projection


def project(A,raw,current,restore_phase=True,maximum=600,diagnostics=None):
    P=A.c.P;shape=(-1,P);rows=len(raw)//P;depth=rows-47
    positive=np.zeros(shape=(rows,P),dtype=bool)
    positive[:11]=True;positive[47:]=True;positive[4,~A.e.s.E]=False
    fixed=np.zeros_like(positive);fixed[4,~A.e.s.E]=True
    positive=positive.ravel();fixed=fixed.ravel()
    normal=A.normal;target=float(normal@A.xref)
    physical_scale=A.c.scale/A.c.weight
    widths=np.where(A.e.s.E,round(2./A.e.dt),round(1./A.e.dt))
    scale=physical_scale[47:]*A.e.dt
    cuts={};correction=np.zeros_like(raw);y=raw.copy()
    def bounds(v):
        if restore_phase:return phase_bound_projection(v,normal,target,positive,fixed)
        out=v.copy();out[positive]=np.maximum(out[positive],0.);out[fixed]=0.
        return out
    def discover(v):
        h=v.reshape(shape)[47:]*scale
        cs=np.vstack([np.zeros((1,P)),np.cumsum(h,axis=0)])
        maximum_occupancy=0.;added=0
        for width in np.unique(widths):
            groups=np.flatnonzero(widths==width)
            q=cs[width:,groups]-cs[:-width,groups]
            maximum_occupancy=max(maximum_occupancy,float(q.max()))
            starts,gg=np.where(q>1.+1e-10)
            for start,g in zip(starts,groups[gg]):
                key=(int(start),int(g),int(width))
                if key in cuts:continue
                rr=np.arange(start,start+width)
                indices=(47+rr)*P+g;coeff=scale[rr,g].copy()
                cuts[key]=[indices,coeff,float(coeff@coeff),0.];added+=1
        return maximum_occupancy,added
    previous=y.copy();passed=False;maxocc=float('inf')
    for cycle in range(maximum):
        candidate=y+correction;y=bounds(candidate);correction=candidate-y
        discover(y)
        for cut in cuts.values():
            indices,coeff,norm2,multiplier=cut
            candidate=y[indices]+multiplier*coeff
            new_multiplier=max(0.,float(coeff@candidate-1.)/norm2)
            y[indices]=candidate-new_multiplier*coeff;cut[3]=new_multiplier
        maxocc,added=discover(y)
        change=float(np.linalg.norm(y-previous));previous[:]=y
        phase=abs(float(normal@y-target)) if restore_phase else 0.
        if change<1e-10*max(1.,np.linalg.norm(y)) and phase<1e-9 and maxocc<=1+1e-9 and A.admissible(y):
            passed=True;break
    norm=float(np.linalg.norm(y-current));original=float(np.linalg.norm(raw-current))
    passed=bool(passed and norm<=original*(1+1e-8)+1e-10)
    if diagnostics is not None:
        diagnostics['refractory_projection']=dict(status='PASS' if passed else 'NOT_CONVERGED',
            cycles=cycle+1,active_halfspaces=len(cuts),maximum_occupancy=maxocc,
            phase_error=phase,final_cycle_change=change,original_step_norm=original,
            projected_step_norm=norm,
            method='Dykstra projection onto positivity/fixed-M/phase plus all sliding refractory halfspaces, active cuts added by full-history checks. Solver initial state only; physical flow unchanged.')
    return y if passed else None
