"""Full time-dependent tangent of the autonomous spatial density map.

The reference remains the original FP64 map. Delayed synaptic currents and M
are differentiated along with the conditional voltage/refractory density.
Private Poisson statistics and the spatial Z parameter stay fixed.
"""
from autonomous_density import *


CODE=r'''
extern "C" __global__ void voltage_tangent(
 const double* R,const double* P,double* Q,double* flux,const double* edges,const double* centers,const double* widths,
 const double* nodes,const double* ratio,const double* decay,const double* drive,const double* delta_drive,const int* refs,
 int K,int nv,int width){
 int row=blockIdx.x,g=row/K,s=row%K,t=threadIdx.x,nr=refs[g];long long base=(long long)row*width;
 for(int j=t;j<width;j+=blockDim.x)Q[base+j]=0.;if(t==0)flux[row]=0.;__syncthreads();
 const double* e=edges+(long long)g*(nv+1);const double* c=centers+(long long)g*nv;const double* w=widths+(long long)g*nv;
 double dec=decay[g],current=nodes[s]*ratio[g]+drive[g],theta=e[nv],dd=(1-dec)*delta_drive[g];
 for(int j=t;j<=nv;j+=blockDim.x){
  double pm=P[base+j],rm=R[base+j];if(pm==0.&&rm==0.)continue;
  double v=j<nv?c[j]:11.,dest=dec*v+(1-dec)*current;
  if(j==nv){
   if(dest>=theta)atomicAdd(flux+row,pm);
   else if(dest<=c[0])atomicAdd(Q+base,pm);
   else if(dest>=c[nv-1])atomicAdd(Q+base+nv-1,pm);
   else{int lo=0,hi=nv-1;while(hi-lo>1){int m=(lo+hi)/2;if(c[m]<=dest)lo=m;else hi=m;}
    double h=c[lo+1]-c[lo],f=(dest-c[lo])/h;
    atomicAdd(Q+base+lo,pm*(1-f)-rm*dd/h);atomicAdd(Q+base+lo+1,pm*f+rm*dd/h);}continue;
  }
  double slope=0.,rslope=0.,f=pm/w[j],rf=rm/w[j];
  if(j>0&&j<nv-1){
   double rdl=(rf-R[base+j-1]/w[j-1])/(v-c[j-1]),rdr=(R[base+j+1]/w[j+1]-rf)/(c[j+1]-v);
   if(rdl*rdr>0){
    if(fabs(rdl)<=fabs(rdr)){rslope=rdl;slope=(f-P[base+j-1]/w[j-1])/(v-c[j-1]);}
    else{rslope=rdr;slope=(P[base+j+1]/w[j+1]-f)/(c[j+1]-v);}
   }
  }
  double left=dest-.5*dec*w[j],right=dest+.5*dec*w[j];
  if(left>=theta){atomicAdd(flux+row,pm);continue;}
  int lo=0,hi=nv;while(hi-lo>1){int m=(lo+hi)/2;if(e[m]<=left)lo=m;else hi=m;}
  double prev=-.5*w[j],dprev=0.,used=0.;int at=lo;
  while(at<nv&&e[at]<right){
   double raw=(e[at+1]-dest)/dec,b=fmax(prev,fmin(.5*w[j],raw));
   double db=(raw>-.5*w[j]&&raw<.5*w[j])?-dd/dec:0.;
   double value=pm*(b-prev)/w[j]+.5*slope*(b*b-prev*prev)+(rf+rslope*b)*db-(rf+rslope*prev)*dprev;
   atomicAdd(Q+base+at,value);used+=value;prev=b;dprev=db;if(b>=.5*w[j])break;at++;
  }
  if(right>=theta)atomicAdd(flux+row,pm-used);else atomicAdd(Q+base+min(at,nv-1),pm-used);
 }
 __syncthreads();if(t==0){Q[base+nv+nr-1]+=flux[row];for(int j=1;j<nr;j++)Q[base+nv+j-1]+=P[base+nv+j];}
}
extern "C" __global__ void observe_tangent(
 const double* flux,const double* mass,const unsigned char* pop,double* M,double* history,double* activity,
 int K,int P,int D,int step){
 int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=P)return;
 double r=0.;for(int k=0;k<K;k++)r+=mass[k]*flux[g*K+k];
 activity[g]=r;history[(long long)(step%D)*P+g]=r;
 if(!pop[g])M[g]=(1.-.1/1000.)*M[g]+r;
}
'''

STATE_NAMES=('F','qa','ia','qg','ig','M','history')


class NetworkTangent:
    def __init__(self,reference):
        self.reference=reference;m=reference
        for name in STATE_NAMES:setattr(self,name,cp.zeros_like(getattr(m,name)))
        self.Q=cp.empty_like(self.F);self.flux=cp.empty_like(m.flux)
        self.qe,self.ie,self.drive,self.activity,self.zero_nu=[cp.zeros(m.P) for _ in range(5)]
        mod=cp.RawModule(code=CODE,options=('--fmad=false',),name_expressions=['voltage_tangent','observe_tangent'])
        self.voltage=mod.get_function('voltage_tangent');self.observe=mod.get_function('observe_tangent')

    def advance(self):
        m=self.reference;i32=np.int32;step=m.step_index
        m.recurrent((m.P,),(128,),(*m.operators,self.history,self.zero_nu,m.tm,m.dr,
            self.qa,self.ia,self.qg,self.ig,self.qe,self.ie,m.Z,self.M,self.drive,
            i32(m.P),i32(m.D),i32(step),*m.synpars))
        # Reuse the original reference noise product without changing its map.
        original_mix=m.mix_noise;reference_moved=original_mix()
        m.mix_noise=lambda:reference_moved
        try:m.advance_step()
        finally:m.mix_noise=original_mix
        moved=cp.ascontiguousarray(cp.matmul(m.transition,self.F))
        self.voltage((m.P*m.K,),(128,),(reference_moved,moved,self.Q,self.flux,
            m.de,m.dc,m.dw,m.nodes,m.dr,m.decay,m.drive,self.drive,m.drefs,
            i32(m.K),i32(m.nv),i32(m.width)))
        self.observe(((m.P+127)//128,),(128,),(self.flux,m.mass,m.pop,self.M,self.history,self.activity,
            i32(m.K),i32(m.P),i32(m.D),i32(step)))
        self.F,self.Q=self.Q,self.F
        return self.activity

    def canonical(self,name):
        if name=='history':
            m=self.reference;return self.history[(m.step_index-np.arange(m.D))%m.D]
        return getattr(self,name)

    def project_noise_marginal(self):
        m=self.reference
        self.F-=self.F.sum(2)[:,:,None]*m.F/m.F.sum(2)[:,:,None]


def copy_reference(source,target):
    for name in ('F','history','qa','ia','qg','ig','qe','ie','Z','M'):
        getattr(target,name)[:]=getattr(source,name)
    target.step_index=source.step_index


def audit(args):
    source=Path(args.source);cfg=read(source/'config.json')
    folder=OUT/'network_tangent_audits'/args.label;folder.mkdir(parents=True,exist_ok=False)
    def model():return AutonomousDensity(cfg['D'],cfg['degree'],cfg['voltage_dv'],args.device,basis_mode=cfg.get('basis_mode','legacy'))
    reference=model();reference.restore(source)
    for _ in range(round(args.warmup/DT)):reference.advance_step()
    plus=model();minus=model();copy_reference(reference,plus);copy_reference(reference,minus)
    tangent=NetworkTangent(reference);rng=np.random.default_rng(3317)
    for name in ('qa','ia','qg','ig','M'):
        direction=getattr(reference,name)*cp.asarray(rng.normal(size=reference.P))*.01
        if name=='M':direction*=reference.pop==0
        getattr(tangent,name)[:]=direction
    tangent.history[:]=reference.history*cp.asarray(rng.normal(size=reference.history.shape))*.01
    for name in STATE_NAMES:
        getattr(plus,name)[:]+=args.epsilon*getattr(tangent,name)
        getattr(minus,name)[:]-=args.epsilon*getattr(tangent,name)
    rows=[];started=time.time();last=started
    write(folder/'config.json',dict(source=str(source),D=cfg['D'],epsilon=args.epsilon,warmup_ms=args.warmup,
        duration_ms=args.duration,scope='Complete dynamic-map derivative versus independent symmetric finite differences',
        seed=3317,perturbation='Valid small relative changes to recurrent currents, E-M and delay probabilities; initial PDF perturbation zero'))
    for step in range(round(args.duration/DT)):
        tangent.advance();plus.advance_step();minus.advance_step()
        if step in (0,9) or (step+1)%100==0:
            row=dict(time_ms=(step+1)*DT,components={})
            for name in (*STATE_NAMES,'activity'):
                exact=getattr(tangent,name);fd=(getattr(plus,name)-getattr(minus,name))/(2*args.epsilon)
                err=fd-exact;den=float(cp.linalg.norm(exact.ravel()).get())
                row['components'][name]=dict(max_abs_error=float(cp.max(abs(err)).get()),
                    relative_L2=float(cp.linalg.norm(err.ravel()).get())/max(den,1e-30),tangent_norm=den)
            row['noise_marginal_tangent_max']=float(cp.max(abs(tangent.F.sum(2))).get())
            rows.append(row)
        if time.time()-last>20:
            write(folder/'progress.json',dict(status='RUNNING',pid=os.getpid(),time_ms=(step+1)*DT,wall_s=time.time()-started))
            print('network tangent audit',args.label,(step+1)*DT,flush=True);last=time.time()
    maximum=max(v['relative_L2'] for row in rows for v in row['components'].values() if v['tangent_norm']>1e-10)
    result=dict(status='COMPLETE',rows=rows,maximum_relative_L2=maximum,
                pass_finite_difference=bool(maximum<.005),wall_s=time.time()-started,
                interpretation='One perturbation and epsilon; require epsilon refinement and an active burst window before Floquet use.')
    write(folder/'result.json',result);print('tangent audit result',maximum,result['pass_finite_difference'],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--label',required=True)
    ap.add_argument('--epsilon',type=float,default=1e-5);ap.add_argument('--warmup',type=float,default=100.)
    ap.add_argument('--duration',type=float,default=50.);ap.add_argument('--device',type=int,default=0);audit(ap.parse_args())
