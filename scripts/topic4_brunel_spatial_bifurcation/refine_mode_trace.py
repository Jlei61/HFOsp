"""Refine already located modes after measured response correction."""
from common import *
from model import SpatialBrunel
from contour_roots import refine

def main():
    s=SpatialBrunel(response='calibrated_full');dest=OUT/'g20/mode_trace_calibrated_full';dest.mkdir(exist_ok=True);records=[]
    for p in sorted((OUT/'g20/contour_roots').glob('J*.npz')):
        old=np.load(p);J=float(p.stem[1:]);r=old['rates'];roots=[]
        for lam,v in zip(old['roots'],old['vectors']):
            if not (-.015<lam.real<.05 and .001<lam.imag<.12):continue
            q=refine(s,r,J,complex(lam),v)
            if q is None:continue
            l,v,err=q
            if any(abs(l-x[0])<1e-6 for x in roots):continue
            roots.append(q)
        rows=[]
        for l,v,err in roots:
            en=s.geo['group_size']*abs(v)**2;en/=en.sum();reg=s.geo['group_region'];energy=[float(en[s.E&(reg==i)].sum()) for i in range(3)]
            rows.append(dict(lambda_per_ms=l,frequency_hz=l.imag*1000/(2*np.pi),residual=err,regional_energy=energy,core=['A','B','surround'][int(np.argmax(energy))]))
        records.append(dict(J_EE_core=J,roots=rows));print('trace',J,[(x['core'],x['lambda_per_ms']) for x in rows],flush=True)
        np.savez_compressed(dest/p.name,rates=r,roots=np.array([q[0] for q in roots]),vectors=np.array([q[1] for q in roots]))
        write(dest/'result.json',dict(status='RUNNING',rows=records))
    write(dest/'result.json',dict(status='COMPLETE',rows=records))

if __name__=='__main__':main()
