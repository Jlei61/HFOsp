"""Right-preconditioned GMRES, with the true residual as stopping criterion."""
import numpy as np
import torch


def solve(A,b,P,restart=100,cycles=10,rtol=1e-7):
    x=torch.zeros_like(b);bn=float(torch.linalg.vector_norm(b));history=[]
    n=len(b)
    for outer in range(cycles):
        r=b-A(x);beta=float(torch.linalg.vector_norm(r));history.append(beta)
        print('GPU KRYLOV',outer,beta,'target',rtol*bn,flush=True)
        if beta<=rtol*bn:return x,history
        V=torch.empty((n,restart+1),dtype=b.dtype,device=b.device)
        H=np.zeros((restart+1,restart));V[:,0]=r/beta
        best=None
        for j in range(restart):
            w=A(P(V[:,j]));basis=V[:,:j+1]
            # Twice-orthogonalized Arnoldi, evaluated as dense vector products.
            h=basis.T@w;w-=basis@h
            correction=basis.T@w;w-=basis@correction;h+=correction
            norm=float(torch.linalg.vector_norm(w));H[:j+1,j]=h.cpu().numpy();H[j+1,j]=norm
            if norm>1e-20:V[:,j+1]=w/norm
            rhs=np.zeros(j+2);rhs[0]=beta
            coef=np.linalg.lstsq(H[:j+2,:j+1],rhs,rcond=None)[0]
            estimate=float(np.linalg.norm(H[:j+2,:j+1]@coef-rhs))
            if (j+1)%25==0:print('GPU ARNOLDI',outer,j+1,estimate,flush=True)
            best=coef
            if estimate<rtol*bn or norm<1e-20:break
        x+=P(V[:,:len(best)]@torch.as_tensor(best,device=b.device))
        del V
    history.append(float(torch.linalg.vector_norm(b-A(x))))
    return x,history
