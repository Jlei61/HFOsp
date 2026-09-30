"""Extract unstable spatial modes without equating a static Jacobian to time."""
from temporal_modes import *
from scipy.sparse.linalg import splu
import argparse,time

def main(a):
    s=ZMRate(a.grid);d=np.load(a.state);r=d['r'];D=float(d['D']);L=Linearization(s,r,D)
    dest=DEST/f'g{a.grid}'/'contour_modes'/a.label;dest.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(930917);V=rng.normal(size=(s.P,a.rank))+1j*rng.normal(size=(s.P,a.rank))
    A0=np.zeros_like(V);A1=np.zeros_like(V);nodes,weights=np.polynomial.legendre.leggauss(a.nodes)
    corners=np.array([1e-6+1e-5j,.12+1e-5j,.12+.65j,1e-6+.65j]);started=time.time()
    for k in range(4):
        left=corners[k];right=corners[(k+1)%4]
        for n,(x,w) in enumerate(zip(nodes,weights)):
            z=(left+right)/2+(right-left)/2*x;dz=(right-left)/2*w/(2j*np.pi)
            X=splu(L.matrix(z).tocsc()).solve(V);A0+=dz*X;A1+=dz*z*X
        print('edge',k,'time',round(time.time()-started,1),flush=True)
    U,S,Vh=np.linalg.svd(A0,full_matrices=False);rank=int(np.sum(S>S[0]*1e-7))
    B=(U[:,:rank].conj().T@A1@Vh[:rank].conj().T)/S[:rank][None,:]
    ev,W=np.linalg.eig(B);vec=U[:,:rank]@W;roots=[];vectors=[]
    for k,lam in enumerate(ev):
        if not (-.003<lam.real<.13 and -.002<lam.imag<.66):continue
        q=refine(L,lam,vec[:,k])
        if q is None:continue
        l,v,res,it=q
        if l.imag<0:l=l.conjugate();v=v.conjugate()
        if not (0<l.real<.12 and 0<l.imag<.65):continue
        if any(abs(l-complex(*rr['lambda_per_ms']))<1e-6 for rr in roots):continue
        energy=s.sizes*abs(v)**2;energy/=energy.sum()
        row=dict(lambda_per_ms=[l.real,l.imag],frequency_hz=l.imag*1000/(2*np.pi),residual=res,
            E_mode_energy_A_B_surround=[float(energy[s.E&(s.geo['group_region']==i)].sum()) for i in range(3)],
            inhibitory_energy=float(energy[~s.E].sum()))
        roots.append(row);vectors.append(v);print('root',row,flush=True)
    np.savez_compressed(dest/'modes.npz',r=r,D=D,vectors=np.array(vectors),roots=np.array([complex(*q['lambda_per_ms']) for q in roots]))
    write(dest/'result.json',dict(status='COMPLETE',D=D,global_E_hz=s.global_rate(r),roots=roots,
        singular_values=S,retained_rank=rank,nodes_per_edge=a.nodes,spectrum_complete=False,
        stability='UNSTABLE' if roots else 'UNRESOLVED',wall_s=time.time()-started))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--state',required=True);p.add_argument('--label',required=True)
    p.add_argument('--grid',type=int,default=20);p.add_argument('--nodes',type=int,default=32);p.add_argument('--rank',type=int,default=20)
    main(p.parse_args())
