"""Time-domain v3 model: 12 states per group, physical delays, spline transfer, Z/M closures.

States per group g:  0 mu_f (unused, kept for layout), 1 mu_s (slow mean filter, tau_s), 2 vE_f, 3 vI_f
(fast-transient filters, tau_cE, tau_cI), 4 qa, 5 ia (AMPA rise/decay mean), 6 qg, 7 ig (GABA), 8 va,
9 vg (variance filters, tau/2), 10 m (eta_M M, mV), 11 z, 12 vE_v, 13 vI_v (variance mixing filters, tau_vE, tau_vI).
Instantaneous moments: mu = ia - z ig - m + pm(t); vE = va + pv(t); vI = z^2 vg.
Response (structure V3, fitted 2026-09-18):
  tau_s d mu_s/dt = mu - mu_s ;  tau_cc d vc_f/dt = vc - vc_f ;  tau_vc d vc_v/dt = vc - vc_v   (c = E, I)
  mu_eff = a mu + (1-a) mu_s + eta_E (vE - vE_f) + eta_I (vI - vI_f)
  vc_eff = a_c vc + (1-a_c) vc_v   (a_c in [-1,1]; clamped at 0 inside phi)
  r = phi(mu_eff, vE_eff, vI_eff)   (spline transfer)
  a, a_E, a_I, eta_E, eta_I = constants or smooth tables of the instantaneous workpoint (mu, vE, vI; theta)
Synaptic: d qa/dt = (tau_m area_A A[r](t) - qa)/tau_rA ; d ia/dt = (qa - ia)/tau_dA ; GABA analogous;
  d va/dt = (tau_m area_A^2 QA[r] - va)/(tau_A/2) ; d vg/dt = (tau_m area_G^2 QB[r] - vg)/(tau_G/2)
M: d m/dt = (0.5 E r - m)/tau_M.   Z: d z/dt = E (z_inf - z)/tau_Z, z_inf = Phi((theta_Z - ig)/sd), sd^2 = tau_m vg/(2 tau_G).
Heun (dt=0.1 ms default) on GPU; CPU RHS mirrors the kernel for parity tests.
Optional stochastic contrast: Poisson finite-size sampling of each group's emitted rate.
"""
from model_v3 import *
from response_tables import ResponseTable,CUDA_RESP
from scipy.special import ndtr
from scipy.stats import norm

NS=14
class ResponseParams:
    """Fixed poles per population + mixing weights (constants or workpoint tables)."""
    def __init__(self,table=None):
        self.kind='constant';self.source='PLACEHOLDER_CONSTANTS'
        self.poles={'E':dict(tau_f=1.,tau_s=10.,tau_cE=10.,tau_cI=15.,tau_vE=4.,tau_vI=7.),'I':dict(tau_f=1.,tau_s=5.,tau_cE=5.,tau_cI=8.,tau_vE=2.,tau_vI=4.)}
        self.weights_const={'E':dict(alpha=.5,a_E=1.,a_I=1.,eta_E=0.,eta_I=0.),'I':dict(alpha=.5,a_E=1.,a_I=1.,eta_E=0.,eta_I=0.)}
        self.tables={}
        if table is not None:self.load(table)
    def load(self,table):
        d=read(table);self.kind=d['kind'];self.poles={p:dict(tau_f=1.,**{k:v for k,v in d['poles'][p].items() if k!='tau_f'}) for p in d['poles']};self.source=str(table)
        if self.kind=='constant':self.weights_const=d['weights']
        else:
            z=np.load(Path(table).with_suffix('.npz'))
            self.tables={p:ResponseTable(z[f'x_{p}'],z[f'sE_{p}'],z[f'sI_{p}'],z[f'values_{p}']) for p in 'EI'}
    def pole_arrays(self,s):
        return np.array([[self.poles['E' if e else 'I'][k] for e in s.E] for k in ['tau_f','tau_s','tau_cE','tau_cI','tau_vE','tau_vI']])
    def weights(self,s,mu,ve,vi):
        """(5,P) weights (alpha,a_E,a_I,eta_E,eta_I) and (5,3,P) gradients wrt (mu,vE,vI)."""
        if self.kind=='constant':
            w=np.array([[self.weights_const['E' if e else 'I'][k] for e in s.E] for k in ['alpha','a_E','a_I','eta_E','eta_I']]);return w,np.zeros((5,3,s.P))
        w=np.zeros((5,s.P));g=np.zeros((5,3,s.P))
        for p,m in [('E',s.E),('I',~s.E)]:
            ww,gg=self.tables[p].evaluate(mu[m],ve[m],vi[m],s.theta[m]);w[:,m]=ww;g[:,:,m]=gg
        return w,g
    def device_blocks(self):
        if self.kind=='constant':return None
        return {p:self.tables[p].device_block() for p in 'EI'}

class DynamicModel(SpatialRateV3):
    def __init__(self,grid=20,resp=None,**kw):
        super().__init__(grid,**kw);self.resp=resp or ResponseParams();self.poles=self.resp.pole_arrays(self)
    # ---------- states ----------
    def equilibrium_state(self,r):
        a,b,qa,qb=self.matrices();ma=self.tm*self.area[0]*(a@r);mg=self.tm*self.area[1]*(b@r)
        va=self.tm*self.area[0]**2*(qa@r);vg=self.tm*self.area[1]**2*(qb@r)
        mu=ma-self.Z*mg-.5*self.E*r+self.private_mu;vE=va+self.private_ve;vI=self.Z**2*vg
        return np.array([mu,mu,vE,vI,ma,ma,mg,mg,va,vg,.5*self.E*r,self.Z,vE,vI])
    def effective(self,y,pm=None,pv=None):
        muf,mus,vEf,vIf,qa,ia,qg,ig,va,vg,m,z,vEv,vIv=y
        pm=self.private_mu if pm is None else pm;pv=self.private_ve if pv is None else pv
        mu=ia-z*ig-m+pm;vE=va+pv;vI=z*z*vg;w,_=self.resp.weights(self,mu,vE,vI);al,aE,aI,eE,eI=w
        return al*mu+(1-al)*mus+eE*(vE-vEf)+eI*(vI-vIf),np.maximum(aE*vE+(1-aE)*vEv,0),np.maximum(aI*vI+(1-aI)*vIv,0),(mu,vE,vI)
    def output(self,y):
        me,ve,vi,_=self.effective(y);return self.phi(me,ve,vi)['rate']
    def rhs(self,y,arrivals,dynamic_z=True,pm=None,pv=None):
        muf,mus,vEf,vIf,qa,ia,qg,ig,va,vg,m,z,vEv,vIv=y;a,b,aa,bb=arrivals
        me,veff,vieff,(mu,vE,vI)=self.effective(y,pm,pv);r=self.phi(me,veff,vieff)['rate'];tf,ts,tE,tI,tvE,tvI=self.poles
        sd=np.sqrt(np.maximum(self.tm*vg/(2*self.tau[1]),1e-20));zinf=ndtr((THRESHOLD_Z-ig)/sd)
        return np.array([(mu-muf)/tf,(mu-mus)/ts,(vE-vEf)/tE,(vI-vIf)/tI,
            (self.tm*self.area[0]*a-qa)/self.rise[0],(qa-ia)/self.decay[0],
            (self.tm*self.area[1]*b-qg)/self.rise[1],(qg-ig)/self.decay[1],
            (self.tm*self.area[0]**2*aa-va)/(self.tau[0]/2),(self.tm*self.area[1]**2*bb-vg)/(self.tau[1]/2),
            (.5*self.E*r-m)/TAU_M,self.E*(zinf-z)/TAU_Z if dynamic_z else np.zeros(self.P),(vE-vEv)/tvE,(vI-vIv)/tvI]),r
    # ---------- linear response filters at an equilibrium ----------
    def H(self,lam,r):
        """Mean filter H_mu, variance static filters H_vE, H_vI, and fast-transient terms T_E, T_I (added to the mean gain)."""
        mu,ve,vi=self.moments(r);w,_=self.resp.weights(self,mu,ve,vi);al,aE,aI,eE,eI=w;tf,ts,tE,tI,tvE,tvI=self.poles
        return al+(1-al)/(1+lam*ts),aE+(1-aE)/(1+lam*tvE),aI+(1-aI)/(1+lam*tvI),eE*lam*tE/(1+lam*tE),eI*lam*tI/(1+lam*tI)
    def characteristic(self,r,lam,dynamic_z=False):
        """M(lam) with det M = 0 at characteristic roots. M(0) = -Jacobian (frozen Z)."""
        a,b,qa,qb=self.matrices(lam);mu,ve,vi=self.moments(r);g=self.phi(mu,ve,vi);Hm,HvE,HvI,TE,TI=self.H(lam,r)
        ha=1/((1+lam*self.rise[0])*(1+lam*self.decay[0]));hg=1/((1+lam*self.rise[1])*(1+lam*self.decay[1]))
        hva=1/(1+lam*self.tau[0]/2);hvg=1/(1+lam*self.tau[1]/2);hm=1/(1+lam*TAU_M)
        gm=g['d_mu']*Hm;gE=g['d_ve']*HvE+g['d_mu']*TE;gI=g['d_vi']*HvI+g['d_mu']*TI
        K=sparse.diags(gm*self.tm)@(self.area[0]*ha*a-sparse.diags(self.Z*self.area[1]*hg)@b)
        K=K+sparse.diags(gE*self.tm*self.area[0]**2*hva)@qa+sparse.diags(gI*self.Z**2*self.tm*self.area[1]**2*hvg)@qb
        if dynamic_z:
            b0=self.matrices()[1];qb0=self.matrices()[3];ig=self.tm*self.area[1]*(b0@r);vg=self.tm*self.area[1]**2*(qb0@r)
            sd=np.sqrt(np.maximum(self.tm*vg/(2*self.tau[1]),1e-20));u=(THRESHOLD_Z-ig)/sd;pdf=norm.pdf(u)
            dz_dig=-pdf/sd;dz_dvg=-pdf*u/(2*np.maximum(vg,1e-20));hz=self.E/(1+lam*TAU_Z)
            Zop=sparse.diags(hz*dz_dig*self.tm*self.area[1]*hg)@b+sparse.diags(hz*dz_dvg*self.tm*self.area[1]**2*hvg)@qb
            K=K+sparse.diags(-gm*ig+gI*2*self.Z*vg)@Zop
        return (sparse.identity(self.P,format='csc')+sparse.diags(.5*self.E*gm*hm)-K).tocsc()

# ---------------- CUDA integrator ----------------
def cuda_code(s):
    return f'#define P {s.P}\n'+CUDA_DEVICE+CUDA_RESP+r'''
#include <curand_kernel.h>
extern "C" __global__ void delayed(const int* pa,const int* ca,const double* wa,const double* va,
 const int* pb,const int* cb,const double* wb,const double* vb,const double* hist,double* out,int tick,int depth,int delay_factor){
 int g=blockIdx.x,lane=threadIdx.x;double a=0.,b=0.,q=0.,v=0.;
 for(int i=pa[g]+lane;i<pa[g+1];i+=blockDim.x){int d=(ca[i]/P+1)*delay_factor,slot=(tick-d)%depth;if(slot<0)slot+=depth;
  double r=hist[slot*P+ca[i]%P];a+=wa[i]*r;q+=va[i]*r;}
 for(int i=pb[g]+lane;i<pb[g+1];i+=blockDim.x){int d=(cb[i]/P+1)*delay_factor,slot=(tick-d)%depth;if(slot<0)slot+=depth;
  double r=hist[slot*P+cb[i]%P];b+=wb[i]*r;v+=vb[i]*r;}
 __shared__ double buf[4][128];buf[0][lane]=a;buf[1][lane]=b;buf[2][lane]=q;buf[3][lane]=v;__syncthreads();
 for(int k=64;k>0;k/=2){if(lane<k)for(int j=0;j<4;j++)buf[j][lane]+=buf[j][lane+k];__syncthreads();}
 if(lane==0)for(int j=0;j<4;j++)out[j*P+g]=buf[j][0];
}
// pars rows: 0 tm,1 ref,2 theta,3 E,4 areaA,5 areaG,6 pm,7 pv,8 tau_f(unused),9 tau_s,10 tau_cE,11 tau_cI,12 tau_vE,13 tau_vI,14 alpha_c,15 aE_c,16 aI_c,17 etaE_c,18 etaI_c,19 dynZ,20 dynM,21 group_size,22 jext
// consts: 0 rise_A,1 decay_A,2 rise_G,3 decay_G,4 tauA/2,5 tauG/2,6 tauM,7 tauZ,8 thetaZ,9 use_tables
__device__ void weights_at(const double* pars,const double* consts,const double* WE,const double* WI,int g,double mu,double vE,double vI,double* w,double* gr){
 if(consts[9]>.5){resp_weights(pars[3*P+g]>.5?WE:WI,mu,vE,vI,pars[2*P+g],w,gr);}
 else{for(int p=0;p<5;p++)w[p]=pars[(14+p)*P+g];for(int i=0;i<15;i++)gr[i]=0.;}
}
extern "C" __global__ void rhs(const double* y,const double* arr,const double* pars,const double* consts,const double* SE,const double* SI,const double* WE,const double* WI,
 const double* drive,int drive_index,int drive_on,double* out,double* rate_out){
 int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=P)return;
 double tm=pars[g],th=pars[2*P+g],E=pars[3*P+g],aa=pars[4*P+g],ag=pars[5*P+g],pm=pars[6*P+g],pv=pars[7*P+g];
 double tf=pars[8*P+g],ts=pars[9*P+g],tE=pars[10*P+g],tI=pars[11*P+g],tvE=pars[12*P+g],tvI=pars[13*P+g],dynZ=pars[19*P+g],dynM=pars[20*P+g];
 if(drive_on){double nu=drive[drive_index*P+g];double jext=pars[22*P+g];pm=tm*aa*jext*nu;pv=tm*aa*aa*jext*jext*nu;}
 double muf=y[g],mus=y[P+g],vEf=y[2*P+g],vIf=y[3*P+g],qa=y[4*P+g],ia=y[5*P+g],qg=y[6*P+g],ig=y[7*P+g],va=y[8*P+g],vg=y[9*P+g],m=y[10*P+g],z=y[11*P+g],vEv=y[12*P+g],vIv=y[13*P+g];
 double mu=ia-z*ig-m+pm,vE=va+pv,vI=z*z*vg;double w[5],gr[15];weights_at(pars,consts,WE,WI,g,mu,vE,vI,w,gr);
 double mueff=w[0]*mu+(1-w[0])*mus+w[3]*(vE-vEf)+w[4]*(vI-vIf);double vEeff=fmax(w[1]*vE+(1-w[1])*vEv,0.),vIeff=fmax(w[2]*vI+(1-w[2])*vIv,0.);
 double r,d1,d2,d3;phi_spline(E>0.5?SE:SI,mueff,vEeff,vIeff,th,&r,&d1,&d2,&d3);
 double sd=sqrt(fmax(tm*vg/(2*(consts[2]+consts[3])),1e-20));double zi=.5*erfc((ig-consts[8])/(sqrt(2.)*sd));
 out[g]=(mu-muf)/tf;out[P+g]=(mu-mus)/ts;out[2*P+g]=(vE-vEf)/tE;out[3*P+g]=(vI-vIf)/tI;
 out[4*P+g]=(tm*aa*arr[g]-qa)/consts[0];out[5*P+g]=(qa-ia)/consts[1];
 out[6*P+g]=(tm*ag*arr[P+g]-qg)/consts[2];out[7*P+g]=(qg-ig)/consts[3];
 out[8*P+g]=(tm*aa*aa*arr[2*P+g]-va)/consts[4];out[9*P+g]=(tm*ag*ag*arr[3*P+g]-vg)/consts[5];
 out[10*P+g]=dynM*(.5*E*r-m)/consts[6];out[11*P+g]=dynZ*E*(zi-z)/consts[7];out[12*P+g]=(vE-vEv)/tvE;out[13*P+g]=(vI-vIv)/tvI;
 rate_out[g]=r;
}
extern "C" __global__ void predictor(const double* y,const double* f,double* pred,double dt){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<14*P)pred[i]=y[i]+dt*f[i];}
extern "C" __global__ void finish(double* y,const double* f,const double* f2,const double* rate,const double* pars,double* hist,double dt,int tick,int depth,
 int noise,unsigned long long seed){
 int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=P)return;
 for(int j=0;j<14;j++){int i=j*P+g;y[i]+=.5*dt*(f[i]+f2[i]);}
 double r=rate[g];
 if(noise){curandStatePhilox4_32_10_t rng;curand_init(seed,g,(unsigned long long)tick,&rng);double n=pars[21*P+g];
  double lambda=fmax(r,0.)*n*dt;unsigned int c=curand_poisson(&rng,lambda);r=(double)c/(n*dt);}
 hist[(tick%depth)*P+g]=r;
}
'''

def model_device_arrays(s,cp,dynamic_z=True,dynamic_m=True):
    """pars, consts, spline blocks, weight blocks shared by the integrator and the Floquet kernels."""
    wc=s.resp.weights_const if s.resp.kind=='constant' else {'E':dict(alpha=0,a_E=0,a_I=0,eta_E=0,eta_I=0),'I':dict(alpha=0,a_E=0,a_I=0,eta_E=0,eta_I=0)}
    wrows=[np.array([wc['E' if e else 'I'][k] for e in s.E]) for k in ['alpha','a_E','a_I','eta_E','eta_I']]
    pars=cp.asarray(np.array([s.tm,s.ref,s.theta,s.E.astype(float),np.full(s.P,s.area[0]),np.full(s.P,s.area[1]),s.private_mu,s.private_ve,
        *s.poles,*wrows,np.full(s.P,float(dynamic_z)),np.full(s.P,float(dynamic_m)),s.sizes,s.jext]))
    consts=cp.asarray(np.array([s.rise[0],s.decay[0],s.rise[1],s.decay[1],s.tau[0]/2,s.tau[1]/2,TAU_M,TAU_Z,THRESHOLD_Z,float(s.resp.kind!='constant')]))
    SE=cp.asarray(device_block(s.spline['E']));SI=cp.asarray(device_block(s.spline['I']))
    blocks=s.resp.device_blocks();WE=cp.asarray(blocks['E']) if blocks else cp.zeros(1);WI=cp.asarray(blocks['I']) if blocks else cp.zeros(1)
    return pars,consts,SE,SI,WE,WI

def shared_fraction_operators(s):
    """Private-variance operators for the stochastic contrast: the part of each cell's Poisson input
    variance that is COMMON to the cells of its group (shared presynaptic cells) is generated by the
    sampled group counts; only the remainder (1 - f_gh) stays as private diffusion variance.
    f_gh = n_ih/N_h with n_ih = A_gh^2/Q_gh (connections from group h to a cell of g) and N_h the
    size of group h (delay-summed operators). External-input private variance is untouched."""
    out={}
    for kind in ['ampa','gaba']:
        a=sparse.load_npz(s.folder/f'mean_{kind}.npz').tocsr();q=sparse.load_npz(s.folder/f'variance_{kind}.npz').tocsr()
        rows=np.repeat(np.arange(s.P),np.diff(a.indptr));cols=a.indices%s.P;key=rows.astype(np.int64)*s.P+cols
        uniq,inv=np.unique(key,return_inverse=True);A=np.bincount(inv,weights=a.data);Q=np.bincount(inv,weights=q.data)
        n_ih=np.where(Q>0,A*A/np.maximum(Q,1e-300),0.);Nh=s.sizes[uniq%s.P];f=np.clip(n_ih/np.maximum(Nh,1),0,1)
        q2=q.copy();q2.data=q.data*(1-f[inv]);out[kind]=(q2,f,inv)
    return out

class Integrator:
    def __init__(self,s,dt=.1,initial=None,history=None,dynamic_z=True,dynamic_m=True,drive=None,noise=False,seed=1,device=0,shared_split=None):
        """shared_split: None (full diffusion variance; deterministic skeleton) or the dict from
        shared_fraction_operators(s) (stochastic contrast with Poisson group sampling and reduced private variance)."""
        import cupy as cp
        self.cp=cp;cp.cuda.Device(device).use();self.s=s;self.dt=dt;self.tick=0;self.noise=noise;self.seed=seed
        self.factor=round(.1/dt);assert abs(self.factor*dt-.1)<1e-12
        self.depth=s.prep['max_delay_steps']*self.factor+1
        y0=np.zeros((NS,s.P)) if initial is None else np.asarray(initial,float)
        if initial is None:y0[11]=s.Z
        self.y=cp.asarray(y0);r0=s.output(y0)
        self.history=cp.asarray(np.broadcast_to(r0,(self.depth,s.P)).copy() if history is None else history)
        self.arr=cp.zeros((4,s.P));self.f=cp.zeros_like(self.y);self.f2=cp.zeros_like(self.y);self.pred=cp.zeros_like(self.y);self.rate=cp.zeros(s.P);self.rate2=cp.zeros(s.P)
        self.ops=[]
        for kind in ['ampa','gaba']:
            a=sparse.load_npz(s.folder/f'mean_{kind}.npz').tocsr();q=sparse.load_npz(s.folder/f'variance_{kind}.npz').tocsr()
            if shared_split is not None:q=shared_split[kind][0]
            assert np.array_equal(a.indices,q.indices) and np.array_equal(a.indptr,q.indptr)
            self.ops.extend([cp.asarray(a.indptr,dtype=cp.int32),cp.asarray(a.indices,dtype=cp.int32),cp.asarray(a.data),cp.asarray(q.data)])
        self.pars,self.consts,self.SE,self.SI,self.WE,self.WI=model_device_arrays(s,cp,dynamic_z,dynamic_m)
        self.drive=cp.asarray(drive) if drive is not None else cp.zeros((1,s.P));self.drive_on=drive is not None;self.n_drive=self.drive.shape[0]
        self.module=cp.RawModule(code=cuda_code(s),options=('--fmad=false',),name_expressions=['delayed','rhs','predictor','finish'])
        self.k={n:self.module.get_function(n) for n in ['delayed','rhs','predictor','finish']}
    def arrivals(self,tick):
        self.k['delayed']((self.s.P,),(128,),(*self.ops,self.history,self.arr,np.int32(tick),np.int32(self.depth),np.int32(self.factor)))
    def drive_index(self,tick):return min(int(tick*self.dt/1.),self.n_drive-1)
    def eval_rhs(self,y,tick,out,rate):
        self.k['rhs'](((self.s.P+127)//128,),(128,),(y,self.arr,self.pars,self.consts,self.SE,self.SI,self.WE,self.WI,self.drive,np.int32(self.drive_index(tick)),np.int32(self.drive_on),out,rate))
    def step(self):
        n=(self.s.P+127)//128
        self.arrivals(self.tick);self.eval_rhs(self.y,self.tick,self.f,self.rate)
        self.k['predictor'](((NS*self.s.P+127)//128,),(128,),(self.y,self.f,self.pred,self.dt))
        self.arrivals(self.tick+1);self.eval_rhs(self.pred,self.tick+1,self.f2,self.rate2)
        self.tick+=1;self.rate+=self.rate2;self.rate*=.5
        self.k['finish']((n,),(128,),(self.y,self.f,self.f2,self.rate,self.pars,self.history,self.dt,np.int32(self.tick),np.int32(self.depth),np.int32(self.noise),np.uint64(self.seed)))
        return self.history[self.tick%self.depth]

if __name__=='__main__':
    s=DynamicModel(quiet=True);s.set_D(.01);r,ok,_=s.solve();assert ok
    y=s.equilibrium_state(r);arr=np.array([a@r for a in s.matrices()])
    f,rr=s.rhs(y,arr);print('equilibrium RHS residual max',abs(f[:11]).max(),'rate residual',abs(rr-r).max())
    C=s.characteristic(r,0.);J=s.jacobian(r);print('M(0)+J max',abs((C+J).data).max() if (C+J).nnz else 0)
    import cupy as cp
    y2=y.copy();y2[5]*=1.2;y2[7]*=.9;y2[0]*=1.05
    e=Integrator(s,initial=y2);e.arrivals(0);e.eval_rhs(e.y,0,e.f,e.rate);cpu,rc=s.rhs(y2,e.arr.get())
    print('GPU rhs parity',abs(cpu-e.f.get()).max(),'rate parity',abs(rc-e.rate.get()).max())
    # table-kind parity test with a synthetic weight table
    x=np.array([-3.,-1.,0.,.5,1.,2.,5.,15.]);sE=np.array([.15,.5,1.,2.,4.]);sI=np.array([0.,.5,1.5,3.,6.])
    X,SEg,SIg=np.meshgrid(x,sE,sI,indexing='ij');vals=np.stack([1/(1+np.exp(-(X-1))),np.tanh(SEg-1),np.tanh(SIg-2),.05*np.tanh(X),.02*np.tanh(X-1)])
    rp=ResponseParams();rp.kind='tables';rp.tables={'E':ResponseTable(x,sE,sI,vals),'I':ResponseTable(x,sE,sI,vals*.9)};s.resp=rp
    e=Integrator(s,initial=y2);e.arrivals(0);e.eval_rhs(e.y,0,e.f,e.rate);cpu,rc=s.rhs(y2,e.arr.get())
    print('GPU rhs parity (tables)',abs(cpu-e.f.get()).max(),'rate parity',abs(rc-e.rate.get()).max())
    mu,ve,vi=s.moments(r);w,g=rp.weights(s,mu,ve,vi);h=1e-4;w2,_=rp.weights(s,mu+h,ve,vi);print('weight grad FD check (mu)',np.max(abs((w2-w)/h-g[:,0])))
