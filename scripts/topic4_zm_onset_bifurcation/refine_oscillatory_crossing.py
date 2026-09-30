"""Refine a verified imaginary-axis crossing on a fixed equilibrium branch.

The nonlinear Hopf criticality and its periodic branch are NOT inferred from
this linear calculation. Full-spectrum stability is also a separate test.
"""
from temporal_modes import *
import argparse

def main(a):
    s=ZMRate();folder=DEST/'g20/conditional_dynamic_M'/a.branch
    rows=read(folder/'result.json')['rows'];pairs=[]
    for left,right in zip(rows[:-1],rows[1:]):
        if 'lambda_per_ms' in left and 'lambda_per_ms' in right and left['lambda_per_ms'][0]*right['lambda_per_ms'][0]<0:pairs.append((left,right))
    result=[];dest=folder/'crossings';dest.mkdir(exist_ok=True)
    for index,(left,right) in enumerate(pairs):
        ld=np.load(folder/left['point']);rd=np.load(folder/right['point'])
        dl,dr=left['D'],right['D'];rl,rr=ld['r'],rd['r'];vl,vr=ld['vector'],rd['vector']
        ll=complex(*left['lambda_per_ms']);lr=complex(*right['lambda_per_ms']);trace=[]
        for k in range(30):
            D=(dl+dr)/2;r,ok,_=s.solve_D(D,(rl+rr)/2)
            if not ok:raise RuntimeError('Equilibrium solve failed in crossing refinement')
            L=Linearization(s,r,D);found=refine(L,(ll+lr)/2,vl)
            if found is None:raise RuntimeError('Mode refinement failed')
            lam,v,res,it=found;trace.append(dict(D=D,lambda_per_ms=lam,residual=res))
            if abs(lam.real)<1e-10:break
            if ll.real*lam.real>0:dl=D;rl=r;ll=lam;vl=v
            else:dr=D;rr=r;lr=lam;vr=v
        slope=(lr.real-ll.real)/(dr-dl);energy=s.sizes*abs(v)**2;energy/=energy.sum()
        row=dict(D=D,global_E_hz=s.global_rate(r),frequency_hz=lam.imag*1000/(2*np.pi),
            lambda_per_ms=lam,characteristic_residual=res,crossing_slope_per_ms_per_D=slope,
            E_mode_energy_A_B_surround=[float(energy[s.E&(s.geo['group_region']==i)].sum()) for i in range(3)],
            inhibitory_energy=float(energy[~s.E].sum()),
            type='VERIFIED_OSCILLATORY_LINEAR_CROSSING',
            nonlinear_Hopf_criticality='NOT_ESTABLISHED_WITHOUT_SAME_NONLINEAR_RATE_EQUATIONS',
            global_onset_identity='NOT_ESTABLISHED',trace=trace)
        np.savez_compressed(dest/f'crossing{index+1}.npz',r=r,D=D,lam=lam,vector=v,Z=s.Z)
        result.append(row);print({k:v for k,v in row.items() if k!='trace'},flush=True)
    write(dest/'result.json',dict(status='COMPLETE',rows=result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--branch',default='temporal_upper_21Hz');main(p.parse_args())
