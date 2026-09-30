"""Disjoint bounded stability subsets; outputs are merged after completion."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time
import numpy as np
from certify_topic4_fig5_D_stability import cached_characteristic
from topic4_fig5_D_physical_model import OUT
from topic4_fig5_z_characteristic_root_newton import root

def main():
    p=argparse.ArgumentParser();p.add_argument('--branch',default='high');p.add_argument('--indices',required=True);p.add_argument('--name',required=True);a=p.parse_args()
    c=cached_characteristic(1.25);data=np.load(OUT/f'q1.25_{a.branch}.npz');rows=[];v=None;guess=10+150j
    for k in map(int,a.indices.split(',')):
        r,D=data['r_hz'][k],float(data['s'][k]);c.at(r,D);bracket=None;prev=None
        for lam in [0.,1.,3.,10.,30.,100.,300.,1000.]:
            sg=float(np.linalg.slogdet(c.matrix(lam).real)[0])
            if prev is not None and sg*prev[1]<0:bracket=[prev[0],lam];break
            prev=(lam,sg)
        row=dict(branch=a.branch,index=k,D=D,mean_e_hz=float(np.average(r[:c.m.n],weights=c.m.count_e)))
        if bracket:row.update(stability='unstable',method='positive real characteristic root bracket',bracket_per_s=bracket)
        elif a.branch=='middle_extension':
            ll,v,er,h=root(c,guess,v);guess=ll
            if er<1e-7 and ll.real>1e-4:row.update(stability='unstable',method='positive complex characteristic root',lambda_per_s=[ll.real,ll.imag],residual=er)
        if 'stability' not in row:
            checks=[c.rhp_count(spacing=5.)]
            if checks[0]['unstable_multiplier_count']==0:checks.append(c.rhp_count(spacing=3.))
            assert len({x['unstable_multiplier_count'] for x in checks})==1
            assert max(x['max_phase_step'] for x in checks)<.4
            row.update(stability='stable' if checks[0]['unstable_multiplier_count']==0 else 'unstable',method='unit-circle winding',counts=checks)
        rows.append(row);(OUT/f'stability_subset_{a.name}.json').write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True)

if __name__=='__main__':main()
