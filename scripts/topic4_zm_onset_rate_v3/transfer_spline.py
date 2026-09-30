"""Smooth tensor-product B-spline representation of the MC transfer table.

g = log(max(r, 1e-4 Hz)) is interpolated (degree k, not-a-knot) on (u=asinh(x), sigma_E, sigma_I).
Both sigma axes are mirrored (sigma -> -sigma with the same data) so the interpolant is
even in sigma and d(phi)/d(variance) stays finite at zero variance. For x beyond ~60 the
interpolant is blended smoothly into the deterministic rate of the native discrete-time LIF
(1/(t_ref + dt*S), S a smoothed version of the ISI step count ceil((tau_m/dt) ln(x/(x-1)))-1). Evaluation (value, gradient, Hessian) is
analytic from the B-spline basis; the same knots/coefficients are used by the CUDA kernel.
"""
from common_v3 import *
from scipy.interpolate import make_interp_spline
R_FLOOR=1e-4   # Hz: table rates below this are floored before the log transform
PARAMS=read(OPERATORS/'g20/prepared.json')['params'];VR=PARAMS['V_reset']

DT=0.1
def _smooth_steps(y,eps=.1):
    """Smooth stand-in for (ceil(y)-1): the engine fires on the step in which V crosses theta, so the
    deterministic ISI is t_ref + dt*(ceil(y)-1) with y = (tau_m/dt) ln(x/(x-1)); the average of the
    staircase is y-0.5, kept >=0 by a smooth max."""
    q=y-.5;return .5*(q+np.sqrt(q*q+eps*eps)),.5*(1+q/np.sqrt(q*q+eps*eps))
def deterministic_rate(x,tm,ref):
    """Rate (Hz) of the native discrete-time LIF without noise: 1/(t_ref + dt*S(y)), S = smoothed staircase."""
    x=np.asarray(x,float);out=np.zeros_like(x);m=x>1.
    y=(tm/DT)*np.log(x[m]/(x[m]-1));S,_=_smooth_steps(y);out[m]=1000/(ref+DT*S)
    return out
def deterministic_rate_dx(x,tm,ref):
    x=np.asarray(x,float);out=np.zeros_like(x);m=x>1.
    y=(tm/DT)*np.log(x[m]/(x[m]-1));S,dS=_smooth_steps(y);r=1/(ref+DT*S)
    dy=-(tm/DT)/(x[m]*(x[m]-1));out[m]=1000*(-r*r*DT*dS*dy)
    return out

def cox_de_boor(t,k,x):
    """Return (i, B[...,k+1], dB, d2B) for query x on knot vector t (clamped). B nonzero for coeffs i-k..i."""
    x=np.asarray(x,float);n=len(t)-k-1
    i=np.clip(np.searchsorted(t,x,side='right')-1,k,n-1)
    B=np.zeros(x.shape+(k+1,));B[...,0]=1.
    dB=np.zeros_like(B);d2B=np.zeros_like(B)
    for d in range(1,k+1):
        Bn=np.zeros_like(B);dn=np.zeros_like(B);d2n=np.zeros_like(B)
        for j in range(d+1):
            # basis index i-d+j
            idx=i-d+j
            if j>0:
                den=t[idx+d]-t[idx];den=np.where(den==0,1,den)
                a=(x-t[idx])/den;Bn[...,j]+=a*B[...,j-1];dn[...,j]+=(B[...,j-1]+a*dB[...,j-1]*1.0)*0+ (d/den)*B[...,j-1] if False else 0
            if j<d:
                den=t[idx+d+1]-t[idx+1];den=np.where(den==0,1,den)
                b=(t[idx+d+1]-x)/den;Bn[...,j]+=b*B[...,j]
        # derivatives via standard formula: B'_{i,d} = d*(B_{i,d-1}/(t_{i+d}-t_i) - B_{i+1,d-1}/(t_{i+d+1}-t_{i+1}))
        for j in range(d+1):
            idx=i-d+j
            if j>0:
                den=t[idx+d]-t[idx];den=np.where(den==0,1,den);dn[...,j]+=d*B[...,j-1]/den;d2n[...,j]+=d*dB[...,j-1]/den
            if j<d:
                den=t[idx+d+1]-t[idx+1];den=np.where(den==0,1,den);dn[...,j]-=d*B[...,j]/den;d2n[...,j]-=d*dB[...,j]/den
        B,dB,d2B=Bn,dn,d2n
    return i,B,dB,d2B

class TransferSpline:
    def __init__(self,table,k=3,blend=(60.,200.)):
        z=np.load(table,allow_pickle=True);self.pop=str(z['pop']);self.tm=PARAMS['tau_m_E'] if self.pop=='E' else PARAMS['tau_m_I']
        self.ref=PARAMS['tau_ref_E'] if self.pop=='E' else PARAMS['tau_ref_I'];self.k=k
        x=z['x'];sE=z['sigma_E'];sI=z['sigma_I'];rate=z['rate_hz'];self.scale=float(z['theta']-z['v_reset'])
        self.u=np.arcsinh(x);g=np.log(np.maximum(rate,R_FLOOR))
        # mirror sigma axes
        self.sE=np.r_[-sE[:0:-1],sE];self.sI=np.r_[-sI[:0:-1],sI]
        g=np.concatenate([g[:,:0:-1,:],g],axis=1);g=np.concatenate([g[:,:,:0:-1],g],axis=2)
        c=g.copy()
        for axis,grid in [(0,self.u),(1,self.sE),(2,self.sI)]:
            c=np.moveaxis(c,axis,0);sp=make_interp_spline(grid,c.reshape(len(grid),-1),k=k);c=sp.c.reshape(c.shape);c=np.moveaxis(c,0,axis)
            setattr(self,['tu','tE','tI'][axis],sp.t)
        self.c=c;self.blend=blend;self.umin,self.umax=self.u[0],self.u[-1]
        self.sEmax,self.sImax=self.sE[-1],self.sI[-1]
    def _basis(self,t,q):
        i,B,dB,d2B=cox_de_boor(t,self.k,q);return i,B,dB,d2B
    def evaluate(self,mu,ve,vi,theta,order=1):
        """phi (per ms) and derivatives wrt mu (mV), ve, vi (mV^2). Arrays over groups."""
        mu=np.asarray(mu,float);ve=np.asarray(ve,float);vi=np.asarray(vi,float);theta=np.asarray(theta,float)
        sc=theta-VR;x=(mu-VR)/sc;u=np.arcsinh(x);sE=np.sqrt(np.maximum(ve,0))/sc;sI=np.sqrt(np.maximum(vi,0))/sc
        uc=np.clip(u,self.umin,self.umax);sEc=np.clip(sE,0,self.sEmax);sIc=np.clip(sI,0,self.sImax)
        iu,Bu,dBu,d2Bu=self._basis(self.tu,uc);iE,BE,dBE,d2BE=self._basis(self.tE,sEc);iI,BI,dBI,d2BI=self._basis(self.tI,sIc)
        k=self.k;n=len(mu);idx_u=iu[:,None]-k+np.arange(k+1);idx_E=iE[:,None]-k+np.arange(k+1);idx_I=iI[:,None]-k+np.arange(k+1)
        C=self.c[idx_u[:,:,None,None],idx_E[:,None,:,None],idx_I[:,None,None,:]]   # n,4,4,4
        def contract(a,b,c):return np.einsum('nijk,ni,nj,nk->n',C,a,b,c)
        g=contract(Bu,BE,BI);gu=contract(dBu,BE,BI);gE=contract(Bu,dBE,BI);gI=contract(Bu,BE,dBI)
        r=np.exp(g);dr=r
        dudx=1/np.sqrt(1+x*x);inside_u=(u>=self.umin)&(u<=self.umax)
        # derivatives in original variables
        dmu=dr*gu*dudx/sc*inside_u
        # d/dv = dg/dsigma /(2 sigma sc^2); at sigma->0 use second derivative limit (even function)
        gEE=contract(Bu,d2BE,BI);gII=contract(Bu,BE,d2BI)
        small=sE<1e-6;dvE=np.where(small,dr*gEE/(2*sc*sc),dr*gE/(2*np.maximum(sE,1e-300)*sc*sc))*(sE<=self.sEmax)
        small=sI<1e-6;dvI=np.where(small,dr*gII/(2*sc*sc),dr*gI/(2*np.maximum(sI,1e-300)*sc*sc))*(sI<=self.sImax)
        # blend with deterministic rate for large x
        x1,x2=self.blend;w=np.clip((x-x1)/(x2-x1),0,1);s=w*w*(3-2*w);dsdx=6*w*(1-w)/(x2-x1)
        rd=deterministic_rate(x,self.tm,self.ref);drd=deterministic_rate_dx(x,self.tm,self.ref)
        out_r=(1-s)*r+s*rd
        out_mu=((1-s)*dmu*sc+s*drd-dsdx*(r-rd))/sc
        out_vE=(1-s)*dvE;out_vI=(1-s)*dvI
        res=dict(rate=out_r/1000,d_mu=out_mu/1000,d_ve=out_vE/1000,d_vi=out_vI/1000,x=x,blend=s)
        if order>=2:
            # Hessian in (mu,ve,vi) from spline second derivatives (used for normal forms); ignore blend region (s>0 flagged)
            guu=contract(d2Bu,BE,BI);guE=contract(dBu,dBE,BI);guI=contract(dBu,BE,dBI);gEI=contract(Bu,dBE,dBI)
            ch=r;sh=r
            d2udx2=-x/(1+x*x)**1.5
            # r = R0 sinh(g): r_ab = sh*g_a*g_b + ch*g_ab
            def sig_d(sg,g1,g2):   # d/dv of (g1/(2 sg sc^2)) where g1=dg/dsigma, g2=d2g/dsigma2
                sgs=np.maximum(sg,1e-6);return (g2/sgs-g1/sgs**2)/(4*sgs*sc**4)
            Hmm=(sh*(gu*dudx)**2+ch*(guu*dudx**2+gu*d2udx2))/sc**2
            HmE=(sh*gu*dudx*gE/(2*np.maximum(sE,1e-6))+ch*guE*dudx/(2*np.maximum(sE,1e-6)))/sc**3
            HmI=(sh*gu*dudx*gI/(2*np.maximum(sI,1e-6))+ch*guI*dudx/(2*np.maximum(sI,1e-6)))/sc**3
            HEE=sh*(gE/(2*np.maximum(sE,1e-6)*sc**2))**2+ch*sig_d(sE,gE,gEE)
            HII=sh*(gI/(2*np.maximum(sI,1e-6)*sc**2))**2+ch*sig_d(sI,gI,gII)
            HEI=(sh*gE*gI+ch*gEI)/(4*np.maximum(sE,1e-6)*np.maximum(sI,1e-6)*sc**4)
            res['hessian']=np.array([[Hmm,HmE,HmI],[HmE,HEE,HEI],[HmI,HEI,HII]])/1000
        return res
    def device_arrays(self):
        return dict(tu=self.tu,tE=self.tE,tI=self.tI,c=self.c,shape=np.array(self.c.shape),umin=self.umin,umax=self.umax,sEmax=self.sEmax,sImax=self.sImax,
                    blend=np.array(self.blend),tm=self.tm,ref=self.ref,k=self.k)

CUDA_DEVICE=r'''
#define KDEG 3
__device__ void basis3(const double* t,int n,double x,int* i0,double* B,double* dB,double* d2B){
 // n = number of coefficients; t has n+KDEG+1 knots; clamped domain
 int lo=KDEG,hi=n-1;
 while(lo<hi){int mid=(lo+hi+1)/2;if(t[mid]<=x)lo=mid;else hi=mid-1;}
 int i=lo;*i0=i-KDEG;
 double Bp[4]={1,0,0,0},dBp[4]={0,0,0,0},d2Bp[4]={0,0,0,0};
 for(int d=1;d<=KDEG;d++){
  double Bn[4]={0,0,0,0},dn[4]={0,0,0,0},d2n[4]={0,0,0,0};
  for(int j=0;j<=d;j++){int idx=i-d+j;
   if(j>0){double den=t[idx+d]-t[idx];if(den==0)den=1;double a=(x-t[idx])/den;Bn[j]+=a*Bp[j-1];dn[j]+=d*Bp[j-1]/den;d2n[j]+=d*dBp[j-1]/den;}
   if(j<d){double den=t[idx+d+1]-t[idx+1];if(den==0)den=1;double b=(t[idx+d+1]-x)/den;Bn[j]+=b*Bp[j];dn[j]-=d*Bp[j]/den;d2n[j]-=d*dBp[j]/den;}
  }
  for(int j=0;j<4;j++){Bp[j]=Bn[j];dBp[j]=dn[j];d2Bp[j]=d2n[j];}
 }
 for(int j=0;j<4;j++){B[j]=Bp[j];dB[j]=dBp[j];d2B[j]=d2Bp[j];}
}
// spline block layout (doubles): [0]=nu,[1]=nE,[2]=nI,[3]=umin,[4]=umax,[5]=sEmax,[6]=sImax,[7]=x1,[8]=x2,[9]=tm,[10]=ref, then tu, tE, tI, c
__device__ void phi_spline(const double* S,double mu,double ve,double vi,double theta,double* rate,double* dmu,double* dve,double* dvi){
 int nu=(int)S[0],nE=(int)S[1],nI=(int)S[2];double umin=S[3],umax=S[4],sEmax=S[5],sImax=S[6],x1=S[7],x2=S[8],tm=S[9],ref=S[10];
 const double* tu=S+11;const double* tE=tu+nu+KDEG+1;const double* tI=tE+nE+KDEG+1;const double* c=tI+nI+KDEG+1;
 double sc=theta-11.;double x=(mu-11.)/sc;double u=asinh(x);double sE=sqrt(fmax(ve,0.))/sc,sI=sqrt(fmax(vi,0.))/sc;
 double uc=fmin(fmax(u,umin),umax),sEc=fmin(sE,sEmax),sIc=fmin(sI,sImax);
 int iu,iE,iI;double Bu[4],dBu[4],d2Bu[4],BE[4],dBE[4],d2BE[4],BI[4],dBI[4],d2BI[4];
 basis3(tu,nu,uc,&iu,Bu,dBu,d2Bu);basis3(tE,nE,sEc,&iE,BE,dBE,d2BE);basis3(tI,nI,sIc,&iI,BI,dBI,d2BI);
 double g=0,gu=0,gE=0,gI=0,gEE=0,gII=0;
 for(int a=0;a<4;a++)for(int b=0;b<4;b++)for(int d=0;d<4;d++){double cc=c[((iu+a)*nE+(iE+b))*nI+(iI+d)];
  g+=cc*Bu[a]*BE[b]*BI[d];gu+=cc*dBu[a]*BE[b]*BI[d];gE+=cc*Bu[a]*dBE[b]*BI[d];gI+=cc*Bu[a]*BE[b]*dBI[d];gEE+=cc*Bu[a]*d2BE[b]*BI[d];gII+=cc*Bu[a]*BE[b]*d2BI[d];}
 double r=exp(g),dr=r;double dudx=1/sqrt(1+x*x);double inu=(u>=umin&&u<=umax)?1.:0.;
 double dm=dr*gu*dudx/sc*inu;
 double dE=(sE<1e-6?dr*gEE/(2*sc*sc):dr*gE/(2*sE*sc*sc))*(sE<=sEmax?1.:0.);
 double dI=(sI<1e-6?dr*gII/(2*sc*sc):dr*gI/(2*sI*sc*sc))*(sI<=sImax?1.:0.);
 double w=fmin(fmax((x-x1)/(x2-x1),0.),1.);double s=w*w*(3-2*w),dsdx=6*w*(1-w)/(x2-x1);
 double rd=0,drd=0;if(x>1.){double y=(tm/0.1)*log(x/(x-1));double qq=y-.5,sq=sqrt(qq*qq+.01);double S=.5*(qq+sq),dS=.5*(1+qq/sq);
  double q=1/(ref+0.1*S);rd=1000*q;double dy=-(tm/0.1)/(x*(x-1));drd=1000*(-q*q*0.1*dS*dy);}
 *rate=((1-s)*r+s*rd)/1000.;*dmu=(((1-s)*dm*sc+s*drd-dsdx*(r-rd))/sc)/1000.;*dve=(1-s)*dE/1000.;*dvi=(1-s)*dI/1000.;
}
'''
def device_block(sp):
    d=sp.device_arrays()
    head=np.array([d['shape'][0],d['shape'][1],d['shape'][2],d['umin'],d['umax'],d['sEmax'],d['sImax'],d['blend'][0],d['blend'][1],d['tm'],d['ref']],float)
    return np.concatenate([head,d['tu'],d['tE'],d['tI'],d['c'].ravel()])

if __name__=='__main__':
    # Self-test on a synthetic smooth table: interpolation exactness at nodes and derivative accuracy.
    import tempfile
    x=np.sinh(np.arange(-3.9,6.,.15));sE=np.array([0,.1,.2,.35,.5,.7,1.,1.4,2.,2.8,4.,5.5]);sI=np.array([0,.1,.2,.35,.5,.7,1.,1.4,2.,2.8,4.,5.5,7.5])
    X,SEv,SIv=np.meshgrid(x,sE,sI,indexing='ij')
    from scipy.special import erfcx
    nodes,weights=np.polynomial.legendre.leggauss(64)
    def siegert(x,se,si):
        sig=np.sqrt(se*se+si*si)+1e-9;lo=(-x)/sig;hi=(1-x)/sig
        q=(lo+hi)[...,None]/2+(hi-lo)[...,None]/2*nodes
        with np.errstate(over='ignore'):integ=(hi-lo)/2*np.sum(weights*erfcx(-q),axis=-1)
        return 1000/(2+20*np.sqrt(np.pi)*integ)
    rate=siegert(X,SEv,SIv)
    f=Path(tempfile.mkdtemp())/'t.npz';np.savez(f,x=x,sigma_E=sE,sigma_I=sI,rate_hz=rate,theta=18.,v_reset=11.,pop='E')
    sp=TransferSpline(f)
    rng=np.random.default_rng(0);n=2000;xq=rng.uniform(-2,30,n);seq=rng.uniform(0.02,5,n);siq=rng.uniform(0,7,n)
    theta=np.where(rng.uniform(size=n)<.5,18.,15.);sc=theta-11;mu=11+sc*xq;ve=(sc*seq)**2;vi=(sc*siq)**2
    ev=sp.evaluate(mu,ve,vi,theta,order=2);truth=siegert(xq,seq,siq)/1000
    rel=abs(ev['rate']-truth)/np.maximum(truth,1e-6);print('interp rel err: median %.2e p95 %.2e max %.2e (rates>0.1Hz)'%tuple(np.percentile(rel[truth>1e-4],[50,95,100])))
    h=1e-4
    fd=(siegert(xq+h/sc,seq,siq)-siegert(xq-h/sc,seq,siq))/(2*h)/1000
    print('d_mu: spline vs truth rel err median %.2e p95 %.2e'%tuple(np.percentile(abs(ev['d_mu']-fd)/np.maximum(abs(fd),1e-9),[50,95])))
    hv=1e-3*np.maximum(ve,1);fdv=(siegert(xq,np.sqrt(ve+hv)/sc,siq)-siegert(xq,np.sqrt(np.maximum(ve-hv,0))/sc,siq))/(2*hv)/1000
    print('d_ve: rel err median %.2e p95 %.2e'%tuple(np.percentile(abs(ev['d_ve']-fdv)/np.maximum(abs(fdv),1e-9),[50,95])))
    # internal consistency: spline derivative vs finite differences of the spline itself (must be ~1e-8)
    e1=sp.evaluate(mu+h,ve,vi,theta)['rate'];e2=sp.evaluate(mu-h,ve,vi,theta)['rate'];print('self FD d_mu max rel dev',np.max(abs((e1-e2)/(2*h)-ev['d_mu'])/np.maximum(abs(ev['d_mu']),1e-9)))
    e1=sp.evaluate(mu,ve+hv,vi,theta)['rate'];e2=sp.evaluate(mu,ve-hv,vi,theta)['rate'];print('self FD d_ve max rel dev',np.max(abs((e1-e2)/(2*hv)-ev['d_ve'])/np.maximum(abs(ev['d_ve']),1e-9)))
    hv=1e-3*np.maximum(vi,1);e1=sp.evaluate(mu,ve,vi+hv,theta)['rate'];e2=sp.evaluate(mu,ve,np.maximum(vi-hv,0),theta)['rate'];print('self FD d_vi max rel dev',np.max(abs((e1-e2)/(2*hv)-ev['d_vi'])/np.maximum(abs(ev['d_vi']),1e-9)))
    # zero-variance derivative finite
    ev0=sp.evaluate(np.array([20.,30.]),np.array([2.,2.]),np.array([0.,0.]),np.array([18.,18.]));print('d_vi at vi=0:',ev0['d_vi'],'finite',np.isfinite(ev0['d_vi']).all())
    # CUDA parity
    import cupy as cp
    blk=cp.asarray(device_block(sp));mod=cp.RawModule(code=CUDA_DEVICE+r'''
extern "C" __global__ void test(const double* S,const double* mu,const double* ve,const double* vi,const double* th,double* out,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;double r,a,b,c;phi_spline(S,mu[i],ve[i],vi[i],th[i],&r,&a,&b,&c);out[4*i]=r;out[4*i+1]=a;out[4*i+2]=b;out[4*i+3]=c;}
''',options=('--fmad=false',),name_expressions=['test']);kern=mod.get_function('test')
    out=cp.zeros((n,4));kern(((n+127)//128,),(128,),(blk,cp.asarray(mu),cp.asarray(ve),cp.asarray(vi),cp.asarray(theta),out,np.int32(n)));o=out.get()
    for j,key in enumerate(['rate','d_mu','d_ve','d_vi']):
        print('CUDA parity',key,np.max(abs(o[:,j]-ev[key])/np.maximum(abs(ev[key]),1e-12)))
