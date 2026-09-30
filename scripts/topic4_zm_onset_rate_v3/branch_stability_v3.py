"""Temporal stability of equilibrium branches (frozen Z, dynamic M): argument-principle root counts
at sampled points, refined leading roots, and bisection of Re(lambda)=0 crossings (Hopf candidates).
Every sampled point gets: unstable_roots (or NEEDS_REFINEMENT), det sign at 0, leading root(s)."""
from root_count_v3 import *
import argparse
def leading_roots(s,r,seeds,dynamic_z=False):
    found=[]
    for lam0 in seeds:
        try:f=refine_root(s,r,lam0,dynamic_z=dynamic_z)
        except Exception:f=None
        if f is None:continue
        lam,v,err=f
        if lam.imag<0:lam=lam.conjugate();v=v.conjugate()
        if any(abs(lam-q['lam'])<1e-6 for q in found):continue
        energy=s.sizes*abs(v)**2*s.E;energy/=max(energy.sum(),1e-300)
        found.append(dict(lambda_per_ms=[float(lam.real),float(lam.imag)],frequency_hz=float(lam.imag*1000/(2*np.pi)),residual=err,
            E_energy_A_B_S=[float(energy[s.geo['group_region']==k].sum()) for k in range(3)],lam=lam))
    return found
def main(a):
    resp=ResponseParams(DEST/'response_closure/closure.json');s=DynamicModel(resp=resp,quiet=True)
    out=DEST/'equilibrium_stability';out.mkdir(exist_ok=True);branch=DEST/'equilibria'/a.branch;info=read(branch/'result.json');rows=info['rows']
    idx=sorted(set(list(range(0,len(rows),a.every))+[len(rows)-1]));results=[];prev=None
    for j in idx:
        z=np.load(branch/f'point{j:04d}.npz');r=z['r'];D=float(z['D']);s.set_D(D)
        if abs(s.residual(r)).max()>1e-8:continue
        q=count(s,r,N=a.N,dynamic_z=a.dynamic_z,verbose=False);seeds=[.01+.03j,.015+.13j,.01+.19j,.02+.06j]+([prev] if prev is not None else [])
        roots=leading_roots(s,r,seeds,a.dynamic_z);roots=sorted(roots,key=lambda q:-q['lam'].real)
        if roots:prev=roots[0]['lam']
        row=dict(index=j,D=D,global_E_hz=s.global_rate(r),regional_hz=s.regional_rates(r),unstable_roots=q['unstable_roots'],status=q['status'],det_zero_sign=q['det_zero_sign'],radius=q['radius_per_ms'],
                 leading_roots=[{k:v for k,v in r_.items() if k!='lam'} for r_ in roots[:3]])
        results.append(row);log(a.branch,j,'D %.5f'%D,'g %.2f'%row['global_E_hz'],'unstable',q['unstable_roots'],q['status'],'lead',[(round(r_['lambda_per_ms'][0],4),round(r_['frequency_hz'],1)) for r_ in roots[:2]])
        write(out/f'{a.branch}_sampled.json',dict(branch=a.branch,every=a.every,dynamic_z=a.dynamic_z,rows=results,response=resp.source))
    # crossings: consecutive sampled points where the leading root real part changes sign or the count changes
    cross=[]
    for p_,q_ in zip(results[:-1],results[1:]):
        if p_['unstable_roots'] is not None and q_['unstable_roots'] is not None and p_['unstable_roots']!=q_['unstable_roots']:cross.append([p_['index'],q_['index'],p_['unstable_roots'],q_['unstable_roots']])
    write(out/f'{a.branch}_sampled.json',dict(branch=a.branch,every=a.every,dynamic_z=a.dynamic_z,rows=results,count_changes=cross,response=resp.source));log('count changes',cross)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--branch',default='lower');p.add_argument('--every',type=int,default=25);p.add_argument('--N',type=int,default=128);p.add_argument('--dynamic-z',action='store_true');main(p.parse_args())
