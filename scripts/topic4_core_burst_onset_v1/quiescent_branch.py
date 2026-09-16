"""Exact silent-branch check of the native piecewise-linear, deterministic map.

This does not linearize the noisy active population state. It tests whether
the all-silent equilibrium can lose local stability when only core EE changes.
"""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path('/home/honglab/leijiaxin/HFOsp')
sys.path.insert(0,str(ROOT/'scripts/topic4_core_burst_onset_v1'))
from run import native,Drive,OUT

def main():
    sub,groups,loading,det,applied,cores=native.setup(.85,1.,848101)
    p=sub.params;labels=sub.net['labels'];dt=p.dt
    d=Drive(sub,cores[0],848101,0.)
    rates=np.full(len(labels),d.signal);rates[:sub.n_e][cores[0]>=0]=d.matched
    tau=np.where(labels==0,p.tau_m_E,p.tau_m_I)
    incr=tau/p.tau_r_AMPA*np.where(labels==0,p.J_ext_E,p.J_ext_I)
    gate_decay=np.exp(-dt/p.tau_r_AMPA)
    voltage=rates*dt*incr/(1-gate_decay)
    margin=sub.vtheta-voltage
    summaries={}
    for g in ('coreAE','coreBE','surroundE','allI'):
        idx=groups[g];summaries[g]=dict(n=len(idx),voltage_min_mV=float(voltage[idx].min()),voltage_max_mV=float(voltage[idx].max()),
            margin_min_mV=float(margin[idx].min()),margin_median_mV=float(np.median(margin[idx])),above_or_at_threshold=int((margin[idx]<=0).sum()))
    # One exact native no-spike update applied at this candidate equilibrium.
    s=voltage.copy();ie=voltage.copy();v=voltage.copy()
    s=gate_decay*s+rates*dt*incr
    ie=s+(ie-s)*np.exp(-dt/p.tau_d_AMPA)
    v=ie+(v-ie)*np.exp(-dt/tau)
    residual=float(np.max(abs(v-voltage)));assert residual<1e-10
    decays={k:float(np.exp(-dt/value)) for k,value in dict(E_membrane=p.tau_m_E,I_membrane=p.tau_m_I,
        AMPA_rise=p.tau_r_AMPA,AMPA_decay=p.tau_d_AMPA,GABA_rise=p.tau_r_GABA,GABA_decay=p.tau_d_GABA).items()}
    valid=bool(np.all(margin>0));radius=max(decays.values())
    # Independent finite-difference check of each cell type's five-state map.
    # The state order is sE, sI, IE, II, V; threshold margins justify this branch.
    jac_checks=[]
    for cell_type,tmem in [('E',p.tau_m_E),('I',p.tau_m_I)]:
        a,b,c,e,m=np.exp(-dt/np.array([p.tau_r_AMPA,p.tau_r_GABA,p.tau_d_AMPA,p.tau_d_GABA,tmem]))
        def update(x):
            se=a*x[0];si=b*x[1]
            ie=se+(x[2]-se)*c;ii=si+(x[3]-si)*e
            v=ie-ii+(x[4]-ie+ii)*m
            return np.array([se,si,ie,ii,v])
        point=np.array([.03,.01,.04,.02,.05]);eps=1e-5
        numerical=np.column_stack([(update(point+eps*np.eye(5)[j])-update(point-eps*np.eye(5)[j]))/(2*eps) for j in range(5)])
        expected=np.array([[a,0,0,0,0],[0,b,0,0,0],[a*(1-c),0,c,0,0],
            [0,b*(1-e),0,e,0],[(1-m)*a*(1-c),-(1-m)*b*(1-e),(1-m)*c,-(1-m)*e,m]])
        err=float(abs(numerical-expected).max())
        eig_err=float(abs(np.sort(np.linalg.eigvals(numerical))-np.sort([a,b,c,e,m])).max())
        assert err<1e-9 and eig_err<1e-9
        jac_checks.append(dict(cell_type=cell_type,matrix_max_error=err,eigenvalue_max_error=eig_err))
    payload=dict(status='VALID_SILENT_EQUILIBRIUM' if valid else 'SILENT_CANDIDATE_INVALID',depth=1.,topology=2511,
        ee_values=[.5,.7,.85,1.,1.2],dt_ms=dt,region_summaries=summaries,equilibrium_update_residual_mV=residual,
        subthreshold_map_eigenvalues=decays,delay_shift_eigenvalue=0.,
        finite_difference_checks=jac_checks,
        neuronal_spectral_radius=radius if valid else None,leading_growth_per_s=float(np.log(radius)/dt*1000) if valid else None,
        explanation='In the all-silent, below-threshold neighborhood, the spike map has zero derivative; recurrent EE weights do not enter the Jacobian. Delay-ring shifts are nilpotent. The remaining map is triangular with membrane/gating/current decay eigenvalues.',
        scope='Constant external expected input; all-silent native neuronal subsystem only. This does not exclude Hopf of a nonzero-rate, fluctuation-driven population state or other attractors. Core EE is the only varied physical parameter.',
        parameters={k:float(getattr(p,k)) for k in ('dt','tau_m_E','tau_m_I','tau_r_AMPA','tau_d_AMPA','tau_r_GABA','tau_d_GABA','J_ext_E','J_ext_I')},identity=applied['identity'])
    np.savez_compressed(OUT/'quiescent_branch.npz',voltage_eq_mV=voltage,threshold_mV=sub.vtheta,margin_mV=margin,labels=labels,core_index_E=cores[0])
    (OUT/'quiescent_branch.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload))

if __name__=='__main__':main()
