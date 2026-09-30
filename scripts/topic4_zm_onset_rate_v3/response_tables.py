"""Workpoint-dependent response weights (alpha, a_E, a_I, eta_E, eta_I) as tensor cubic B-splines on the
coarse assay grid (u=asinh x, sigma_E, sigma_I), sigma axes mirrored (even). Outside the grid: clamped (zero gradient).
alpha (0) clipped to [0,1], a_E, a_I (1,2) to [-1,1] on evaluation; eta_E, eta_I unclipped. Same evaluation in numpy and CUDA.
"""
from common_v3 import *
from transfer_spline import cox_de_boor,VR
from scipy.interpolate import make_interp_spline
NPAR=5
class ResponseTable:
    def __init__(self,x,sE,sI,values,k=3):
        """values: (NPAR, nx, nE, nI). Both sigma axes are mirrored (even extension) so that the tables are
        smooth in the variances at sigma=0 (finite d/dv)."""
        self.x=np.asarray(x,float);self.u=np.arcsinh(self.x);self.k=k;self.values=np.asarray(values,float)
        sE=np.asarray(sE,float);sI=np.asarray(sI,float);assert self.values.shape==(NPAR,len(x),len(sE),len(sI))
        def mirror(grid,arr,axis):
            if grid[0]==0:g=np.r_[-grid[:0:-1],grid];a=np.concatenate([np.flip(arr,axis).take(range(arr.shape[axis]-1),axis=axis),arr],axis=axis)
            else:g=np.r_[-grid[::-1],grid];a=np.concatenate([np.flip(arr,axis),arr],axis=axis)
            return g,a
        self.sEmax=sE[-1];self.sImax=sI[-1];self.sE,V=mirror(sE,self.values,2);self.sI,V=mirror(sI,V,3);cs=[]
        for v in V:
            c=v.copy()
            for axis,grid in [(0,self.u),(1,self.sE),(2,self.sI)]:
                c=np.moveaxis(c,axis,0);sp=make_interp_spline(grid,c.reshape(len(grid),-1),k=k);c=sp.c.reshape(c.shape);c=np.moveaxis(c,0,axis)
                setattr(self,['tu','tE','tI'][axis],sp.t)
            cs.append(c)
        self.c=np.array(cs)
    def evaluate(self,mu,ve,vi,theta):
        mu=np.asarray(mu,float);ve=np.asarray(ve,float);vi=np.asarray(vi,float);theta=np.asarray(theta,float)
        sc=theta-VR;x=(mu-VR)/sc;u=np.arcsinh(x);sE=np.sqrt(np.maximum(ve,0))/sc;sI=np.sqrt(np.maximum(vi,0))/sc
        uc=np.clip(u,self.u[0],self.u[-1]);sEc=np.clip(sE,0,self.sEmax);sIc=np.clip(sI,0,self.sImax)
        iu,Bu,dBu,_=cox_de_boor(self.tu,self.k,uc);iE,BE,dBE,d2BE=cox_de_boor(self.tE,self.k,sEc);iI,BI,dBI,d2BI=cox_de_boor(self.tI,self.k,sIc)
        k=self.k;idx_u=iu[:,None]-k+np.arange(k+1);idx_E=iE[:,None]-k+np.arange(k+1);idx_I=iI[:,None]-k+np.arange(k+1)
        C=self.c[:,idx_u[:,:,None,None],idx_E[:,None,:,None],idx_I[:,None,None,:]]
        f=np.einsum('pnijk,ni,nj,nk->pn',C,Bu,BE,BI);fu=np.einsum('pnijk,ni,nj,nk->pn',C,dBu,BE,BI);fE=np.einsum('pnijk,ni,nj,nk->pn',C,Bu,dBE,BI);fI=np.einsum('pnijk,ni,nj,nk->pn',C,Bu,BE,dBI)
        fEE=np.einsum('pnijk,ni,nj,nk->pn',C,Bu,d2BE,BI);fII=np.einsum('pnijk,ni,nj,nk->pn',C,Bu,BE,d2BI)
        inu=(u>=self.u[0])&(u<=self.u[-1]);inE=sE<=self.sEmax;inI=sI<=self.sImax
        on=np.ones_like(f,bool);w=f.copy()
        on[0]=(f[0]>0)&(f[0]<1);w[0]=np.clip(f[0],0,1)
        for p in (1,2):on[p]=(f[p]>-1)&(f[p]<1);w[p]=np.clip(f[p],-1,1)
        dmu=fu/np.sqrt(1+x*x)/sc*inu*on
        dvE=np.where(sE<1e-6,fEE/(2*sc*sc),fE/(2*np.maximum(sE,1e-300)*sc*sc))*inE*on
        dvI=np.where(sI<1e-6,fII/(2*sc*sc),fI/(2*np.maximum(sI,1e-300)*sc*sc))*inI*on
        return w,np.stack([dmu,dvE,dvI],axis=1)   # (5,n), (5,3,n)
    def device_block(self):
        head=np.array([len(self.u),len(self.sE),len(self.sI),self.u[0],self.u[-1],0.,self.sEmax,0.,self.sImax],float)
        return np.concatenate([head,self.tu,self.tE,self.tI,self.c.ravel()])

CUDA_RESP=r'''
#define NPAR 5
// block: [0]=nu,[1]=nE,[2]=nI,[3]=umin,[4]=umax,[5]=sEmin,[6]=sEmax,[7]=sImin,[8]=sImax, then tu,tE,tI, then c (NPAR x nu x nE x nI)
__device__ void resp_weights(const double* S,double mu,double ve,double vi,double theta,double* w,double* grad){
 int nu=(int)S[0],nE=(int)S[1],nI=(int)S[2];double umin=S[3],umax=S[4],sEmax=S[6],sImax=S[8];
 const double* tu=S+9;const double* tE=tu+nu+KDEG+1;const double* tI=tE+nE+KDEG+1;const double* c=tI+nI+KDEG+1;
 double sc=theta-11.;double x=(mu-11.)/sc;double u=asinh(x);double sE=sqrt(fmax(ve,0.))/sc,sI=sqrt(fmax(vi,0.))/sc;
 double uc=fmin(fmax(u,umin),umax),sEc=fmin(sE,sEmax),sIc=fmin(sI,sImax);
 int iu,iE,iI;double Bu[4],dBu[4],d2u[4],BE[4],dBE[4],d2E[4],BI[4],dBI[4],d2I[4];
 basis3(tu,nu,uc,&iu,Bu,dBu,d2u);basis3(tE,nE,sEc,&iE,BE,dBE,d2E);basis3(tI,nI,sIc,&iI,BI,dBI,d2I);
 double inu=(u>=umin&&u<=umax)?1.:0.,inE=(sE<=sEmax)?1.:0.,inI=(sI<=sImax)?1.:0.;
 double dudmu=1/sqrt(1+x*x)/sc;
 int stride=nu*nE*nI;
 for(int p=0;p<NPAR;p++){double f=0,fu=0,fE=0,fI=0,fEE=0,fII=0;const double* cp=c+p*stride;
  for(int a=0;a<4;a++)for(int b=0;b<4;b++)for(int d=0;d<4;d++){double cc=cp[((iu+a)*nE+(iE+b))*nI+(iI+d)];
   f+=cc*Bu[a]*BE[b]*BI[d];fu+=cc*dBu[a]*BE[b]*BI[d];fE+=cc*Bu[a]*dBE[b]*BI[d];fI+=cc*Bu[a]*BE[b]*dBI[d];fEE+=cc*Bu[a]*d2E[b]*BI[d];fII+=cc*Bu[a]*BE[b]*d2I[d];}
  double on=1.;if(p==0){on=(f>0&&f<1)?1.:0.;f=fmin(fmax(f,0.),1.);}else if(p<3){on=(f>-1&&f<1)?1.:0.;f=fmin(fmax(f,-1.),1.);}w[p]=f;
  double gE=(sE<1e-6)?fEE/(2*sc*sc):fE/(2*sE*sc*sc);double gI=(sI<1e-6)?fII/(2*sc*sc):fI/(2*sI*sc*sc);
  grad[3*p]=fu*dudmu*inu*on;grad[3*p+1]=gE*inE*on;grad[3*p+2]=gI*inI*on;}
}
'''
