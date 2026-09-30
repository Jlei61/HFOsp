"""Follow a verified complex characteristic root along computed equilibria."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,numpy as np
from topic4_fig5_z_frozen_v1 import Characteristic
from topic4_fig5_z_characteristic_root_newton import root
from continue_topic4_fig5_z_periodic_arc import OUT


def main():
    p=argparse.ArgumentParser();p.add_argument('--direction',type=int,choices=[-1,1],required=True);p.add_argument('--start',type=int,default=100);p.add_argument('--limit',type=int,default=900);a=p.parse_args()
    dest=OUT/f'low_complex_certificates_{a.direction}.json';states=[]
    for name in ['equilibrium_trusted_forward','equilibrium_low_trusted_continued']:
        z=np.load(OUT/f'{name}.npz')
        states.extend([(name,k,r,float(s)) for k,(r,s) in enumerate(zip(z['r_hz'],z['s']))])
    c=Characteristic();lam=57.096619995952445+148.05352637820812j;vector=None;rows=[]
    order=range(a.start,min(len(states),a.start+a.limit)) if a.direction>0 else range(a.start,max(-1,a.start-a.limit),-1)
    for index in order:
        name,k,r,s=states[index];c.at(r,s);lam,vector,err,h=root(c,lam,vector,maxiter=8)
        status='UNSTABLE_COMPLEX_MULTIPLIER_CERTIFIED' if err<1e-7 and lam.real>1e-5 else 'UNCLASSIFIED'
        row=dict(branch=name,index=k,s=s,lambda_per_s=[lam.real,lam.imag],residual=err,status=status);rows.append(row)
        if len(rows)%10==1 or status=='UNCLASSIFIED':print(index,row,flush=True)
        dest.write_text(json.dumps(rows,indent=2)+'\n')
        if err>1e-7:break


if __name__=='__main__':main()
