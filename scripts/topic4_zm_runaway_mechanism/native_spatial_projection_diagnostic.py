"""Instantaneous closure diagnostic at common lifted states, no new simulation."""
from native_spatial_refinement import DEST, INITIAL, restriction_and_parent
from common import *
from scipy import sparse


def arrivals(s,history,dt):
    factor=round(.1/dt)
    values=history[(-factor*np.arange(1,s.prep['max_delay_steps']+1))%len(history)].ravel()
    rows=[]
    for name in ['mean_ampa','mean_gaba','variance_ampa','variance_gaba']:
        matrix=sparse.load_npz(s.folder/(name+'.npz'))
        rows.append(matrix@values)
    return np.asarray(rows)


def common_theta_output(s,y,theta):
    mu=y[5]-y[11]*y[7]-y[10]+s.private_mu
    ve=y[8]+s.private_ve;vi=y[11]**2*y[9];rate=np.zeros(s.P)
    for pop,mask in [('E',s.E),('I',~s.E)]:
        w,_=s.resp.tables[pop].evaluate(mu[mask],ve[mask],vi[mask],theta[mask])
        al,ae,ai,ee,ei=w
        me=al*mu[mask]+(1-al)*y[1,mask]+ee*(ve[mask]-y[2,mask])+ei*(vi[mask]-y[3,mask])
        ev=np.maximum(ae*ve[mask]+(1-ae)*y[12,mask],0)
        iv=np.maximum(ai*vi[mask]+(1-ai)*y[13,mask],0)
        rate[mask]=s.spline[pop].evaluate(me,ev,iv,theta[mask])['rate']
    return rate


def main():
    s20=model(20);s40=model(40);Q,parent=restriction_and_parent(s20,s40)
    original=np.load(INITIAL);coarse=np.load(OUT/'runs/endpoint_D0.2190000_dt0.05/trajectory.npz')
    Z=coarse['Z_source'];w=s40.sizes/s40.sizes.sum()
    states=[('initial',original['state'],original['history']),
            ('coarse_12000ms',coarse['final_state'],coarse['final_history'])]
    rows=[]
    for label,state,history in states:
        y=state.copy();y[11]=Z;yf=y[:,parent];hf=history[:,parent]
        a=arrivals(s20,history,.05);af=arrivals(s40,hf,.05)
        aggregated=(Q@af.T).T
        moment_error=np.linalg.norm(aggregated-a,axis=1)/np.maximum(np.linalg.norm(a,axis=1),1e-30)
        assert np.max(moment_error)<1e-11
        f,r=s20.rhs(y,a,dynamic_z=False);ff,rf=s40.rhs(yf,af,dynamic_z=False)
        r_control=common_theta_output(s40,yf,s20.theta[parent])
        control_error=float(np.max(abs(r_control-r[parent])))
        assert control_error<1e-12
        deviation=ff-f[:,parent]
        fr=(Q@ff.T).T;rr=Q@rf
        forcing={}
        # These coordinates are all mean-input currents, in mV / ms.
        for index,name in [(4,'AMPA_rise'),(5,'AMPA_decay'),(6,'GABA_rise'),(7,'GABA_decay')]:
            forcing[name]=dict(weighted_RMS_mV_per_ms=float(np.sqrt(w@(deviation[index]**2))),
                maximum_abs_mV_per_ms=float(np.max(abs(deviation[index]))),
                projected_max_abs_difference=float(np.max(abs(fr[index]-f[index]))))
        mean_theta=r_control # Explicit evaluator identity control, not a new network.
        rows.append(dict(snapshot=label,moment_aggregation_relative=moment_error.tolist(),
            original_global_E_rate_hz=s20.global_rate(r),
            lifted_fine_global_E_rate_hz=s40.global_rate(rf),
            aggregated_rate_RMS_hz=float(1000*np.sqrt(np.average((rr-r)**2,weights=s20.sizes))),
            fine_vs_parent_rate_RMS_hz=float(1000*np.sqrt(w@((rf-r[parent])**2))),
            parent_theta_evaluator_max_error_hz=control_error*1000,
            actual_theta_vs_parent_effect_RMS_hz=float(1000*np.sqrt(w@((rf-mean_theta)**2))),
            input_forcing_dispersion=forcing,
            full_RHS_aggregation_max_error_by_coordinate=np.max(abs(fr-f),axis=1).tolist()))
        log('PROJECTION DIAGNOSTIC',label,rows[-1]['aggregated_rate_RMS_hz'],forcing)
    q=dict(status='COMPLETE',rows=rows,
        theta_within_parent_RMS_mV=float(np.sqrt(w@((s40.theta-s20.theta[parent])**2))),
        theta_within_parent_max_mV=float(np.max(abs(s40.theta-s20.theta[parent]))),
        scope='Pointwise closure/projection diagnostic at two saved states. Does not prove that a specific discrepancy causes the changed attractor or that the bifurcation itself shifted.')
    write(DEST/'projection_diagnostic.json',q)


if __name__=='__main__':main()
