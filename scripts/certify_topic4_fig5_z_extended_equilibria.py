"""Certify an unstable multiplier when a positive-real characteristic root exists."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,numpy as np
from topic4_fig5_z_frozen_v1 import Characteristic
from topic4_fig5_z_branch_dynamics import accelerate
from continue_topic4_fig5_z_periodic_arc import OUT


def main():
    dest=OUT/'equilibrium_stability_certificates.json';rows=json.loads(dest.read_text()) if dest.exists() else []
    done={(r['branch'],r['index']) for r in rows};c=Characteristic();accelerate(c.m)
    grid=[0.,1.,3.,10.,30.,100.,300.,1000.,3000.,10000.]
    for name in ['equilibrium_trusted_forward','equilibrium_low_trusted_continued','equilibrium_high_trusted','equilibrium_high_trusted_continued']:
        file=OUT/f'{name}.npz'
        if not file.exists():continue
        a=np.load(file)
        for k,(r,s) in enumerate(zip(a['r_hz'],a['s'])):
            if (name,k) in done:continue
            c.at(r,float(s));previous=None;bracket=None;signs=[]
            for lam in grid:
                matrix=c.matrix(lam);assert abs(matrix.imag).max()<1e-12
                sign=float(np.linalg.slogdet(matrix.real)[0]);signs.append([lam,sign])
                if previous is not None and sign*previous[1]<0:bracket=[previous[0],lam];break
                previous=(lam,sign)
            rows.append(dict(branch=name,index=k,s=float(s),status='UNSTABLE_REAL_MULTIPLIER_CERTIFIED' if bracket else 'UNCLASSIFIED',positive_growth_bracket_per_s=bracket))
            if k%10==0:
                print(name,k,rows[-1]['status'],flush=True);dest.write_text(json.dumps(rows,indent=2)+'\n')
    dest.write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
