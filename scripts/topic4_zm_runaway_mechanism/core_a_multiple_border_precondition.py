"""Exact inverse of an approximate cyclic matching/phase/time operator.

Only Newton's numerical right coordinates change. The full physical derivative
is still applied to every returned vector, including all spatial feedback.
The approximation keeps the identity cyclic matching, intrinsic M decay, all
four actual endpoint time slopes, and all four phase planes.
"""
import numpy as np


class CyclicBorderInverse:
    def __init__(self, slopes, normals, groups, h_over_tau, time_weight=.02):
        self.slopes=np.asarray(slopes);self.normals=np.asarray(normals)
        self.P=groups;self.a=np.exp(-np.asarray(h_over_tau));self.wt=time_weight
        self.N=self.slopes.shape[1]
        assert self.slopes.shape==self.normals.shape==(4,self.N)
        assert self.N%groups==0 and self.N//groups>=5
        assert self.a.shape==(4,) and np.all((self.a>0)&(self.a<1))
        self.den=-np.expm1(-sum(h_over_tau))
        columns=[]
        for j in range(4):
            rhs=np.zeros_like(self.slopes);rhs[j]=self.slopes[j]
            columns.append(self.state_inverse(rhs))
        self.columns=np.asarray(columns)
        self.schur=np.array([[self.normals[j]@column[j] for column in self.columns] for j in range(4)])
        self.condition=float(np.linalg.cond(self.schur))
        assert np.isfinite(self.condition) and self.condition<1e10,('Numerical preconditioner singular; not a physical bifurcation',self.condition)

    def state_inverse(self,rhs):
        u=np.roll(rhs,1,axis=0).copy()
        b=rhs.reshape(4,-1,self.P)[:,4];m=u.reshape(4,-1,self.P)[:,4];a=self.a
        m[0]=(b[3]+a[3]*b[2]+a[3]*a[2]*b[1]+a[3]*a[2]*a[1]*b[0])/self.den
        for j in range(3):m[j+1]=a[j]*m[j]+b[j]
        return u

    def __call__(self,v):
        u=self.state_inverse(v[:-4].reshape(4,self.N))
        rhs=v[-4:]-np.einsum('ij,ij->i',self.normals,u)
        dh=np.linalg.solve(self.schur,rhs)
        for q,col in zip(dh,self.columns):u+=q*col
        return np.r_[u.ravel(),self.wt*dh]

    def approximate_operator(self,v):
        u=v[:-4].reshape(4,self.N);dh=v[-4:]/self.wt
        f=np.roll(u,-1,axis=0).copy()
        f.reshape(4,-1,self.P)[:,4]-=self.a[:,None]*u.reshape(4,-1,self.P)[:,4]
        f-=self.slopes*dh[:,None]
        return np.r_[f.ravel(),np.einsum('ij,ij->i',self.normals,u)]

    def verify(self,seed=92602):
        rng=np.random.default_rng(seed);rows=[]
        for j in range(2):
            v=rng.normal(size=4*self.N+4);q=self.approximate_operator(self(v))
            error=float(np.linalg.norm(q-v)/np.linalg.norm(v))
            assert error<1e-9,error
            rows.append(dict(direction=j,approximate_operator_inverse_error=error))
        return dict(status='PASS',schur_condition=self.condition,checks=rows,
                    scope='Exact inverse of the stated approximate numerical operator; the original full variational equation still controls every Newton update. Not a physical eigenvalue or stability test.')
