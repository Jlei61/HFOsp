#!/usr/bin/env python3
"""Read-only sampling-precision diagnostic, not a new scientific pass gate."""
import numpy as np
from campaign import ROOT,read,write,sha
from conditional_density_inputs import OPS


def main():
    out=ROOT/'direct_newton_validation'
    assert read(out/'result.json')['status']=='DIRECT_NEWTON_STEP_REDUCES_RESIDUAL'
    geo=np.load(OPS/'geometry.npz');group=geo['cell_group'];size=geo['group_size'];E=geo['population']==0
    with np.load(ROOT/'all_target_stationary_direct/response.npz') as z:oldsem=z['cell_SEM_Hz']
    with np.load(ROOT/'all_target_dc_direct/newton_proposal.npz') as z:D=z['adaptation_denominator'];jsem=z['matrix_action_MC_SEM_Hz']
    with np.load(out/'response.npz') as z:newsem=z['group_SEM_Hz'];residual=z['source_residual_Hz']
    oldgroup=np.sqrt(np.bincount(group,weights=(oldsem/D)**2,minlength=len(size)))/size
    combined=np.sqrt(oldgroup**2+newsem**2+jsem**2);rows=[]
    for name,mask in [('E',E),('I',~E)]:
        rows.append(dict(population=name,groups=int(mask.sum()),residual_weighted_RMS_Hz=float(np.sqrt(np.average(residual[mask]**2,weights=size[mask]))),
            combined_sampling_SEM_weighted_RMS_Hz=float(np.sqrt(np.average(combined[mask]**2,weights=size[mask]))),
            groups_outside_3combinedSEM_plus_1e_minus7_Hz=int((abs(residual[mask])>3*combined[mask]+1e-7).sum()),
            zero_count_estimator_SEM_groups=int((combined[mask]==0).sum())))
    np.savez_compressed(out/'sampling_precision.npz',initial_group_SEM_Hz=oldgroup,new_group_SEM_Hz=newsem,
        matrix_action_SEM_Hz=jsem,combined_sampling_SEM_Hz=combined,residual_Hz=residual)
    result=dict(status='SAMPLING_PRECISION_DIAGNOSTIC',rows=rows,
        definition='First-order independent sampling errors from initial direct response, fresh validation, and eight-block measured-matrix action. Numerical absolute floor1e-7Hz only prevents interpreting solver roundoff at zero-count groups.',
        limits='Not a posthoc replacement acceptance gate. Does not include finite-record/reset-phase bias, fixedM approximation, inherited source averaging, or native-network error. Zero counts do not prove zero true rate.',
        decision='Fresh residual reduction is real, and remaining aggregate residual is at the estimated numerical sampling scale. Do not keep iterating the same noisy map merely to chase an arbitrary1e-6Hz residual. Retain finite precision when advancing the related branch and dynamic correspondence.',
        root_certified=False,formal_bifurcation_allowed=False,producer_sha256=sha(__file__))
    write(out/'sampling_precision.json',result);print(result,flush=True)


if __name__=='__main__':main()
