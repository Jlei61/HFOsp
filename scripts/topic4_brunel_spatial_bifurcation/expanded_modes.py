"""Refined unstable eigenmodes of selected high-rate spatial equilibria."""
from common import *
from model import SpatialBrunel
from contour_roots import extract,refine
import argparse

def main(args):
    s=SpatialBrunel(response='calibrated_full');state=np.load(args.state);r=state['rates'];J=float(state['J'])
    dest=OUT/'expanded/modes'/Path(args.state).stem;dest.mkdir(parents=True,exist_ok=True)
    ev,vec,S=extract(s,r,J,n=32,m=36,corners=[.00001+.0001j,.15+.0001j,.15+1.5j,.00001+1.5j])
    roots=[];rows=[];weights=s.geo['group_size'];reg=s.geo['group_region']
    for k,lam in enumerate(ev):
        if not (-.02<lam.real<.2 and 0<lam.imag<1.7):continue
        q=refine(s,r,J,lam,vec[:,k])
        if q is None:continue
        lam,v,err=q
        if not (0<lam.real<.15 and .0001<lam.imag<1.5):continue
        if any(abs(lam-old[0])<1e-6 for old in roots):continue
        energy=weights*abs(v)**2;energy/=energy.sum();roots.append(q)
        row=dict(lambda_per_ms=lam,frequency_hz=lam.imag*1000/(2*np.pi),residual=err,
            regional_energy=[float(energy[s.E&(reg==k)].sum()) for k in range(3)],inhibitory_energy=float(energy[~s.E].sum()))
        rows.append(row);print(J,row,flush=True)
    write(dest/'result.json',dict(J_EE_core=J,roots=rows,singular_values=S,
        status='REFINED_POSITIVE_MODES' if rows else 'NO_REFINED_ROOTS',
        meaning='Individual positive-growth eigenpairs certified numerically in the extrapolated response approximation; not a complete spectrum or native stability validation'))
    np.savez_compressed(dest/'modes.npz',rates=r,J=J,roots=np.array([q[0] for q in roots]),vectors=np.array([q[1] for q in roots]))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--state',required=True);main(p.parse_args())
