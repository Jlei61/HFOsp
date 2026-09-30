#!/usr/bin/env python3
"""Bounded numerical fallback for the same unresolved K9 stationary equation."""
import time
import numpy as np
from scipy import sparse
from scipy.optimize import least_squares
from stationary_native_diagnostic import Stationary,OUT
from campaign import write,sha


def main():
    path=OUT/'trust_region_contract.json';assert not path.exists()
    write(path,dict(status='NUMERICAL_FALLBACK_ONLY',source_sha256=sha(__file__),
        reason='Newton initialization reaches a near-zero I rate where positivity backtracking stalls. The subsequent I-only Newton also cannot advance. This says nothing about equilibrium existence.',
        fixed='Exactly same staticv3 equations,Z.21/K9,graph,meaninput. No fit or change to transfer,fields or parameters.',
        solver='One bounded scipyTRF least_squares with analytic same-equation sparseJacobian,maximum200functionevaluations. Initialrateminimum1e-5/ms only keeps the optimizer interior; residual equation is unmodified and bounds remain0..refractoryceiling.',
        acceptance='Maximum residual<1e-6Hz to count as a numerical root; optimizer success or small spatial difference alone is insufficient. No stability or bifurcation.'))
    e=Stationary();e.set_fields(.21,9.);old=np.load(OUT/'exit_z0.21_k9_high.npz')['rate_per_ms']
    initial=np.clip(old,1e-5,1/e.s.ref-1e-5);start=time.time();calls=0
    def fun(r):
        nonlocal calls
        f=e.evaluate(r);calls+=1
        if calls%5==0:write(OUT/'trust_region_progress.json',dict(status='SOLVING',function_evaluations=calls,maximum_residual_Hz=float(abs(f).max()*1000),elapsed_s=time.time()-start))
        return f
    def jac(r):
        _,A,u,v,*_=e.evaluate(r,True)
        return A+sparse.csr_matrix(np.outer(u,v))
    sol=least_squares(fun,initial,jac=jac,bounds=(np.zeros(e.s.P),1/e.s.ref-1e-10),method='trf',tr_solver='lsmr',x_scale='jac',ftol=1e-12,xtol=1e-12,gtol=1e-10,max_nfev=200)
    r=sol.x;f=e.evaluate(r);physical,g,G,_=e.moments(r)
    native=np.load(OUT/'exit_z0.21_k9_high.npz')['native_field_Hz'];field=e.s.cell_field(r)
    np.savez_compressed(OUT/'trust_region_arrays.npz',rate_per_ms=r,residual=f,physical=physical,g=g,candidate_field_Hz=field,native_field_Hz=native)
    result=dict(status='NUMERICAL_DIAGNOSTIC_COMPLETE',solver_success=bool(sol.success),solver_message=sol.message,function_evaluations=int(sol.nfev),maximum_residual_Hz=float(abs(f).max()*1000),
        numerical_root=bool(abs(f).max()*1000<1e-6),wholeE_Hz=float(e.s.global_rate(r)),core_Hz=(e.regional@r*1000)[:2].tolist(),G_raw=G,
        weighted_field_MAE_Hz=float(np.average(abs(field-native),weights=e.counts)),elapsed_s=time.time()-start,formal_bifurcation_allowed=False,native_correspondence_certified=False)
    write(OUT/'trust_region_result.json',result);write(OUT/'trust_region_progress.json',result);print(result,flush=True)


if __name__=='__main__':main()
