"""Certify each resolved stationary turn on the selected working-point branch."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json
import numpy as np
from topic4_fig5_D_physical_model import OUT
from audit_topic4_fig5_D_folds import audit

def main():
    rows=[]
    for branch in ['middle','middle_extension','high']:
        path=OUT/f'q1.25_{branch}.npz'
        if not path.exists():continue
        a=np.load(path);turns=np.flatnonzero(a['tangent'][:-1,-1]*a['tangent'][1:,-1]<0)
        for number,k in enumerate(turns,1):
            name=f'q1.25_{branch}_SN{number}'
            if not (OUT/f'{name}.json').exists():
                guess=OUT/f'{name}_seed.npz';np.savez_compressed(guess,r_hz=(a['r_hz'][k]+a['r_hz'][k+1])/2,D=(a['s'][k]+a['s'][k+1])/2)
                audit(name,guess,1.25)
            result=json.loads((OUT/f'{name}.json').read_text());result.update(branch=branch,turn_index=int(k));rows.append(result)
            (OUT/'new_stationary_folds.json').write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':main()
