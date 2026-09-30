"""Resolve distinct equilibria where the folded branch crosses one fixed J."""
from common import *
from model import SpatialBrunel
from contour_roots import refine
BASE=OUT/'critical_revision'

def main():
    s=SpatialBrunel(response='calibrated_full');b=np.load(BASE/'branch.npz');J=1.012;dest=BASE/'fixed_J';dest.mkdir(exist_ok=True)
    idx=np.flatnonzero((b['J'][:-1]-J)*(b['J'][1:]-J)<0);rows=[];rates=[]
    for i in idx:
        a=(J-b['J'][i])/(b['J'][i+1]-b['J'][i]);r=(1-a)*b['rates'][i]+a*b['rates'][i+1]
        r,ok,tr=s.solve(J,r);assert ok
        if any(np.linalg.norm(r-old)<1e-7 for old in rates):continue
        rates.append(r);found=[]
        spectra=sorted((BASE/'stability').glob('point*.npz'),key=lambda p:abs(int(p.stem[5:])-i))
        for path in spectra[:2]+[OUT/'g20/mode_trace_calibrated_full/J1.000000.npz']:
            q=np.load(path)
            for lam,v in zip(q['roots'],q['vectors']):
                result=refine(s,r,J,lam,v)
                if result is not None and result[0].real>1e-7:found=[result];break
            if found:break
        row=dict(J_EE_core=J,branch_index=int(i),rates_hz=s.regional_rates(r),residual=tr[-1],
            unstable_certificate=bool(found),positive_root=found[0][0] if found else None,root_residual=found[0][2] if found else None)
        rows.append(row);np.savez_compressed(dest/f'root{len(rows):02d}.npz',rates=r,J=J);print(row,flush=True)
    # Branch order traverses the two folded core responses in a 3 x 3 pattern;
    # assign levels by one-dimensional clustering only for its visualization.
    from scipy.cluster.vq import kmeans2
    regional=np.array([q['rates_hz'][:2] for q in rows]);levels=[]
    for k in range(2):
        centers,labels=kmeans2(regional[:,k,None],np.quantile(regional[:,k],[1/6,.5,5/6])[:,None],minit='matrix',iter=40)
        ordering=np.argsort(centers[:,0]);remap=np.argsort(ordering);levels.append(remap[labels])
    for i,q in enumerate(rows):q.update(A_level=int(levels[0][i]),B_level=int(levels[1][i]))
    write(dest/'result.json',dict(J_EE_core=J,distinct_roots=len(rows),rows=rows,
        occupied_level_pairs=len(set(zip(*levels))),meaning='Different stationary solutions at the SAME J; level names describe branch positions, not three native-SNN activity classes.'))

if __name__=='__main__':main()
