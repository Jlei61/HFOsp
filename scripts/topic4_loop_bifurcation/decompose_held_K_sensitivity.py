#!/usr/bin/env python3
"""Input-path decomposition of the completed local conditional K tangent."""
import numpy as np
from held_direct_moments import HeldInputs
from campaign import ROOT,write

E=HeldInputs();out=ROOT/'held_exit_K9p35_tangent'
with np.load(out/'tangent.npz') as z:
    dr=z['source_Hz_per_K'];target=z['target_Hz_per_K'];fk=z['local_K_partial_Hz_per_K']
with np.load(ROOT/'held_exit_phase_dc_operator_K9p35/measured_dc.npz') as z:chi=z['gain'][:,:,0]
h=1+E.reference['g'];D=1+.0005*E.E*chi[:,0]/h
C=np.array([chi[:,0]*E.tm*E.area[0]/(1000*h),-chi[:,0]*E.Z*E.tm*E.area[1]/(1000*h),
    chi[:,1]*E.tm*E.area[0]**2/(1000*h*h),chi[:,2]*E.tm*(E.Z*E.area[1])**2/(1000*h*h)])/D
parts={n:c*(w@dr) for n,c,w in zip(['excitatory_mean','local_inhibitory_mean','excitatory_variance','inhibitory_variance'],C,E.W)}
parts['global_G']=E.E*E.Z*chi[:,3]/D*(.1*E.causal*(E.weights@dr));parts['direct_K']=fk
error=float(abs(sum(parts.values())-target).max());assert error<1e-10
region=E.geo['group_region'];masks=[E.groupE]+[E.groupE&(region==j) for j in range(3)]+[~E.groupE]
rows=[]
for name,mask in zip(['allE','coreA','coreB','surround','I'],masks):
    vals={key:float(np.average((E.S@x)[mask],weights=E.sizes[mask])) for key,x in parts.items()}
    rows.append(dict(region=name,contributions_Hz_per_K=vals,total_Hz_per_K=sum(vals.values())))
write(out/'sensitivity_components.json',dict(status='COMPLETE_LINEAR_INPUT_PATH_DECOMPOSITION',rows=rows,
    maximum_sum_error_Hz_per_K=error,
    scope='Decomposition of measured local static tangent, each term includes implicitlocalM factor. Components depend on the solved recurrent tangent and are not independent intervention outcomes or a native trajectory. Dynamicstability and finiteamplitude failures remain unestablished.'))
np.savez_compressed(out/'sensitivity_components.npz',**parts)
print(rows)
