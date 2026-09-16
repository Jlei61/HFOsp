"""Paired native trajectories around reduced critical intervals."""
from common import *
import numpy as np
pairs=[('Cycle fold',1.12181,1.12183),('Recruitment interval',1.17623,1.17632),('PD1',1.34,1.355),('LP1',1.355,1.37),('PD3',1.375,1.38),('PD2',1.385,1.395)]
out=[]
for name,ga,gb in pairs:
 folders=[OUT/'native/per_run'/f'J{g:g}_s848101' for g in [ga,gb]]
 if not all((f/'result.json').exists() for f in folders):continue
 a,b=[np.load(f/'trajectory.npz') for f in folders];pa,pb=[read(f/'applied_physics.json') for f in folders]
 assert pa['identity']['ampa_values_sha256']!=pb['identity']['ampa_values_sha256']
 for key in ('ampa_topology_sha256','gaba_topology_sha256','gaba_values_sha256','vtheta_float64_sha256','core_index_sha256'):assert pa['identity'][key]==pb['identity'][key]
 rows={}
 for gn in ['coreAE','coreBE','allE','allI']:
  j=a['group_names'].tolist().index(gn);n=int(a['group_sizes'][j]);ca=a['spike_counts_2ms'][1000:6000,j];cb=b['spike_counts_2ms'][1000:6000,j]
  assert len(ca)==len(cb)==5000
  rows[gn]=dict(mean_a_hz=float(ca.mean()/n/.002),mean_b_hz=float(cb.mean()/n/.002),fraction_identical_2ms_counts=float(np.mean(ca==cb)))
 out.append(dict(reference_reduced_critical=name,J_pair=[ga,gb],same_topology_threshold_inhibition=True,different_EE_values=True,window_s=[2,12],groups=rows,
  boundary='Independent cold-start native runs with the same seed. Agreement does not rule out other initial-history-dependent native attractors; mismatch does not locate a native bifurcation.'))
write('native_paired_comparison.json',out);print(out,flush=True)
