"""Extract and independently refine all candidate roots in a sampled RHP box."""
from rate_stationary_root_count import *
from rate_field_spectrum import refine


def extract(s,r,J,n=48,m=24,right=.12,imag=.5,left=1e-6):
    c=Characteristic(s,r,J);corners=np.array([left-1j*imag,right-1j*imag,right+1j*imag,left+1j*imag])
    nodes,weights=np.polynomial.legendre.leggauss(n);rng=np.random.default_rng(948361)
    V=rng.normal(size=(s.P,m))+1j*rng.normal(size=(s.P,m));A0=np.zeros_like(V);A1=np.zeros_like(V)
    for i in range(4):
        a,b=corners[i],corners[(i+1)%4]
        for x,w in zip(nodes,weights):
            z=(a+b)/2+(b-a)*x/2;dz=(b-a)*w/(4j*np.pi)
            X=splu(c.matrix(z).tocsc()).solve(V);A0+=dz*X;A1+=dz*z*X
        print('RATE CONTOUR EDGE',i,n,J,flush=True)
    U,S,Vh=np.linalg.svd(A0,full_matrices=False);rank=int(np.sum(S>S[0]*1e-7))
    B=(U[:,:rank].conj().T@A1@Vh[:rank].conj().T)/S[:rank][None,:]
    eig,v=np.linalg.eig(B);return eig,U[:,:rank]@v,S,corners


def main(a):
    s=RateField();branch=np.load(RATE_OUT/'equilibrium_branch.npz');r=branch['rates'][a.index];J=float(branch['J'][a.index])
    ev,vec,S,corners=extract(s,r,J,a.nodes,a.probes,a.right,a.imag)
    roots=[]
    for k,lam in enumerate(ev):
        if not(-.03<lam.real<a.right+.05 and abs(lam.imag)<a.imag+.1):continue
        q=refine(s,r,J,lam,vec[:,k])
        if q is None:continue
        l,v,err=q
        if l.real<=1e-7 or abs(l.imag)>a.imag or l.real>a.right:continue
        if any(abs(l-old[0])<1e-6 for old in roots):continue
        roots.append(q);print('REFINED SPATIAL RATE ROOT',l,'residual',err,flush=True)
    roots.sort(key=lambda q:(-q[0].real,-q[0].imag));rows=[]
    for l,v,e in roots:
        mass=s.geo['group_size']*abs(v)**2*s.E;mass/=mass.sum()
        rows.append(dict(lambda_per_ms=l,frequency_hz=abs(l.imag)*1000/(2*np.pi),residual=e,
            E_rate_mode_energy_by_region=[float(mass[s.geo['group_region']==k].sum()) for k in range(3)]))
    dest=DEST/f'branch{a.index:04d}_modes_n{a.nodes}_m{a.probes}'
    write(dest.with_suffix('.json'),dict(branch_index=a.index,J_EE_core=J,roots=rows,
        extracted_positive_roots=len(roots),singular_values=S,contour_corners=corners,nodes_per_edge=a.nodes,
        candidate_extraction_method='Beyn moments of the full normalized rate-DDE matrix',
        refinement_method='Bordered Newton on the original, unnormalized RateField.characteristic matrix',
        inventory_complete=False,scope='Independent spatial eigenpairs; compare contour meshes and full determinant count before claiming root coverage.'))
    np.savez_compressed(dest.with_suffix('.npz'),rates=r,J=J,roots=np.array([q[0] for q in roots]),vectors=np.array([q[1] for q in roots]))
    print('RATE CONTOUR ROOTS SAVED',dest,len(rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);p.add_argument('--nodes',type=int,default=48)
    p.add_argument('--probes',type=int,default=24);p.add_argument('--right',type=float,default=.12);p.add_argument('--imag',type=float,default=.5)
    main(p.parse_args())
