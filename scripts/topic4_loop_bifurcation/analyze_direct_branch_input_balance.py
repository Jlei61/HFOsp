#!/usr/bin/env python3
"""Exact mean-current difference between the checked local K candidates."""
import numpy as np
from campaign import ROOT,read,write,sha
from direct_response_system import DirectDC,EG,EK

out=ROOT/'direct_branch_step_review/input_balance'
assert (out/'contract.json').exists() and not (out/'analysis.json').exists()
e=DirectDC()
a=dict(np.load(ROOT/'direct_newton_validation/inputs.npz'))
b=dict(np.load(ROOT/'direct_exit_K_pair/point_00/evaluation_2/inputs.npz'))
ds=b['source_Hz']-a['proposed_source_rate_Hz']
theta=e.geo['threshold_mv'][e.group]
components=[];names=[]
for j,name in enumerate(['excitation_from_CoreA','excitation_from_CoreB','excitation_from_surround']):
    q=np.where(e.groupE&(e.geo['group_region']==j),ds,0.)
    components.append(e.tm*e.area[0]*(e.W[0]@(q/1000)))
    names.append(name)
components += [-e.Z*e.tm*e.area[1]*(e.W[1]@(ds/1000)),
    -.0005*(b['M']-a['M']),
    -1.*e.E*e.Z*(float(b['Graw'])-float(a['Graw']))*(theta-EG),
    -(b['K']-a['K'])*(theta-EK)]
names += ['local_inhibition','M','global_G','K']
components=np.array(components)
expected=(1+b['g'])*(b['physical'][:,0]-theta)-(1+a['g'])*(a['physical'][:,0]-theta)
err=float(abs(components.sum(0)-expected).max())
assert err<2e-10
rows=[]
for j,name in enumerate(['CoreA','CoreB','surround']):
    m=e.E&(e.geo['group_region'][e.group]==j)
    rows.append(dict(region=name,
        delta_drift_at_threshold_mV_equiv={k:float(v[m].mean()) for k,v in zip(names,components)},
        total_mean=float(expected[m].mean()),
        old_mean_margin_mV_equiv=float(((1+a['g'])*(a['physical'][:,0]-theta))[m].mean()),
        new_mean_margin_mV_equiv=float(((1+b['g'])*(b['physical'][:,0]-theta))[m].mean())))
result=dict(status='COMPLETE_EXACT_LOCAL_INPUT_DECOMPOSITION',rows=rows,reconstruction_max_error=err,
    scope='Differences in supplied stationarymean input balance at grouptargetthresholds between two independentlychecked nearselfconsistent candidates. Not native current measurements, causal ablations or a termination prediction.',
    producer_sha256=sha(__file__))
np.savez_compressed(out/'readouts.npz',components=components,names=names,expected_total=expected)
write(out/'analysis.json',result)
print(result,flush=True)
