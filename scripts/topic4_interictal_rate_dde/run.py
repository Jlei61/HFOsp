"""Spatial rate DDE simulation with frozen observed inputs or autonomous input."""
from common import *
from model import RateSystem
import argparse
import cupy as cp
from scipy.sparse import load_npz
from population_field_gpu import CODE as SOURCE_CODE
from interictal_common import smooth

def table_code(pop,tab):
    text=''
    for name,value in [('t',tab['theta']),('n',tab['nu']),('x',tab['input_knots']),('c',np.ravel(tab['coefficients']))]:
        a=np.asarray(value).ravel();text+=f'__device__ __constant__ double {pop}_{name}[{len(a)}]={{'+','.join(format(float(x),'.17g') for x in a)+'};\n'
    nt=len(tab['theta']);nn=len(tab['nu']);nd=len(tab['input_knots'])-1
    text+=f'''__device__ double phi_{pop}(double d,double t,double n){{
      int a=0,b=0,j=0;while(a<{nt-2}&&t>={pop}_t[a+1])a++;while(b<{nn-2}&&n>={pop}_n[b+1])b++;
      double ft={f'fmin(1.,fmax(0.,(t-{pop}_t[a])/({pop}_t[a+1]-{pop}_t[a])))' if nt>1 else '0.'};
      double fn=fmin(1.,fmax(0.,(n-{pop}_n[b])/({pop}_n[b+1]-{pop}_n[b]))),x=asinh(d/8.);
      while(j<{nd-1}&&x>={pop}_x[j+1])j++;double f=fmin(1.,fmax(0.,(x-{pop}_x[j])/({pop}_x[j+1]-{pop}_x[j]))),v=0.;
      for(int i=0;i<{2 if nt>1 else 1};i++)for(int k=0;k<2;k++){{
       double y=0.;int off=(((a+i)*{nn}+b+k)*{nd}+j)*6;for(int h=0;h<6;h++)y=y*f+{pop}_c[off+h];
       v+=y*{('(i?ft:1-ft)' if nt>1 else '1.')}*(k?fn:1-fn);
      }}return 1./({tab['ref_ms']:.17g}+exp(fmin(700.,v)));
    }}\n'''
    return text

RATE=r'''
extern "C" __global__ void advance(double* r,double* history,double* Z,double* M,const double* drive,const double* ie,const double* ig,
 const double* theta,const double* tau,const double* tau_c,const double* private_factor,const unsigned char* pop,const double* nu,int P,int D,int step,int dynamic_z){
 int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=P)return;
 double d=drive[g]+ie[g]-private_factor[g]*nu[g];double f=pop[g]?phi_I(d,theta[g],nu[g]):phi_E(d,theta[g],nu[g]);
 double tr=tau_c[g]>0?.1+(tau[g]-.1)/(1+(d/tau_c[g])*(d/tau_c[g])):tau[g];
 double a=exp(-.1/tr);r[g]=a*r[g]+(1-a)*f;history[(long long)(step%D)*P+g]=r[g]*.1;
 if(!pop[g]){M[g]+=.1*(r[g]-M[g]/1000.);if(dynamic_z)Z[g]+=.1/5000.*((ig[g]<95.19851312666987?1.:0.)-Z[g]);}
}
'''

def main(args):
    cp.cuda.Device(0).use();cal=read(BASE/'dynamic_calibration/result.json');taus={r['population']:r['tau_rate_ms'] for r in cal['rows']}
    s=RateSystem(args.grid,taus['E'],taus['I'],adaptive_tau=args.adaptive_tau);g=s.geo;p=s.p;P=s.P;D=s.prep['max_delay_steps']+1
    dest=BASE/'runs'/args.label;dest.mkdir(parents=True,exist_ok=True);assert not (dest/'result.json').exists()
    if args.frozen_z_ms is not None:
        check=np.load(SOURCE/f'replay/runs/eta0.0005_s{args.seed}/checkpoints/t{args.frozen_z_ms}ms.npz')
        native=check['slow__z'];s.Z=np.r_[np.bincount(g['cell_group'][:NE],weights=native,minlength=P)/np.maximum(np.bincount(g['cell_group'][:NE],minlength=P),1.)]
        s.Z[~s.E]=1.
    steps=round(args.duration_ms/DT)
    inp=np.load(FIELD/f'runs/g40_degree6_dv0.125_s{args.seed}_8000ms_convergence_bound/external_input.npz');global_nu=inp['global_nu_per_ms'][:steps].copy()
    if args.constant_input:global_nu[:]=s.nu0;local=np.zeros((steps//10,P))
    else:
        sys.path.insert(0,str(ROOT/'src'));from topic4_spatial_ou_drive import SpatialOUDrive,SpatialOUConfig
        config=dict(s.prep['spatial_ou']);config.pop('role');config.pop('seed_offset');config['seed']=args.seed+500000
        drive=SpatialOUDrive(g['original_positions'][:NE],20.,DT,SpatialOUConfig(**config));groups=g['cell_group'][:NE]
        local=np.array([np.bincount(groups,weights=drive.step(k*DT),minlength=P)/g['group_size'] for k in range(0,steps,10)])
    dlocal=cp.asarray(local);r=cp.zeros(P);history=cp.zeros((D,P));states=[cp.zeros(P) for _ in range(7)]
    qa,ia,qg,ig,qe,ie,drive=states;Z=cp.asarray(s.Z);M=cp.zeros(P)
    ops=[]
    for kind in ['ampa','gaba']:
        w=load_npz(s.folder/f'delay_{kind}.npz').tocsr()
        if kind=='ampa':
            row=np.repeat(np.arange(P),np.diff(w.indptr));col=w.indices%P
            mask=s.E[row]&s.E[col]&(g['group_region'][row]<2)&(g['group_region'][row]==g['group_region'][col]);w.data[mask]*=args.core_ee
        ops.extend([cp.asarray(w.indptr,dtype=np.int64),cp.asarray(w.indices,dtype=np.int32),cp.asarray(w.data)])
    recurrent=cp.RawKernel(SOURCE_CODE.split('extern "C" __global__ void moments')[0],'recurrent',options=('--fmad=false',))
    advance=cp.RawKernel(table_code('E',s.tables[0])+table_code('I',s.tables[1])+RATE,'advance',options=('--fmad=false',))
    tm=cp.asarray(s.tm);ratio=cp.asarray(np.where(s.E,1.,p['tau_m_I']*p['J_ext_I']/(p['tau_m_E']*p['J_ext_E'])))
    theta=cp.asarray(s.theta);tau=cp.asarray(s.tr);tau_c=cp.asarray(s.tau_c);pop=cp.asarray(s.pop,dtype=np.uint8);private=cp.asarray(s.tm*s.area[0]*s.jext)
    synpars=tuple(np.float64(x) for x in [np.exp(-DT/s.rise[0]),np.exp(-DT/s.decay[0]),np.exp(-DT/s.rise[1]),np.exp(-DT/s.decay[1]),*s.rise,p['tau_m_E']/p['tau_r_AMPA']*p['J_ext_E']])
    write(dest/'contract.json',dict(model='Continuous spatial rate DDE, exponentially integrated at native0.1ms',grid=args.grid,populations=P,
        continuous_states=8*P+(int(s.E.sum()) if args.frozen_z_ms is None else 0),particle_count=0,J_EE_core=args.core_ee,seed=args.seed,duration_ms=args.duration_ms,
        tau_rate_ms=taus,frozen_z_ms=args.frozen_z_ms,constant_input=args.constant_input,
        state_dependent_tau=args.adaptive_tau,
        slow_scope='Original Z/M rules during correspondence; Z clamped to an interictal field for autonomous fast-system analysis when specified',
        no_native_future_activity_forcing=True,transfer_source=str(BASE/'transfer'),
        continuation_api=['model.RateSystem.equilibrium_residual','model.RateSystem.rhs','model.RateSystem.jvp','model.RateSystem.characteristic']))
    started=time.time();activity=[];slow=[];acc=cp.zeros(P)
    for step in range(steps):
        nu=dlocal[step//10]+float(global_nu[step]);i32=np.int32
        recurrent((P,),(128,),(*ops,history,nu,tm,ratio,qa,ia,qg,ig,qe,ie,Z,M,drive,i32(P),i32(D),i32(step),*synpars))
        advance(((P+127)//128,),(128,),(r,history,Z,M,drive,ie,ig,theta,tau,tau_c,private,pop,nu,i32(P),i32(D),i32(step),i32(args.frozen_z_ms is None)))
        acc+=r*DT
        if (step+1)%10==0:activity.append(acc.copy());acc.fill(0)
        if (step+1)%500==0:
            slow.append(np.stack([Z.get(),M.get()]));assert bool(cp.isfinite(r).all())
            write(dest/'progress.json',dict(status='RUNNING',simulated_ms=(step+1)*DT,seconds=time.time()-started,pid=os.getpid()))
            if (step+1)%10000==0:print(args.label,(step+1)*DT,time.time()-started,flush=True)
    activity=cp.stack(activity).get();counts=activity*g['group_size'];e=s.E
    cell=g['group_cell'];ratio=args.grid//20;cell20=cell%args.grid//ratio+20*(cell//args.grid//ratio)
    field=np.array([np.bincount(cell20[e],weights=x[e],minlength=400) for x in counts]);regions=np.array([np.bincount(g['group_region'][e],weights=x[e],minlength=3) for x in counts])
    whole=np.column_stack([counts[:,e].sum(1),counts[:,~e].sum(1)]);contact=activity.reshape(-1,2,P).sum(1)@g['contact_rate_weights']
    np.savez_compressed(dest/'trajectory.npz',group_spikes_1ms=activity,field_1ms=field,regions_1ms=regions,spikes_1ms=whole,
        contact_envelope=smooth(contact),contact_raw=contact,slow_Z_M=slow,global_external_rate=global_nu,
        contact_names=np.load(SOURCE/'replay/geometry.npz')['contact_names'])
    result=dict(status='COMPLETE',seconds=time.time()-started,grid=args.grid,rate_groups=P,particle_count=0,seed=args.seed,
        mean_E_rate_hz=float(whole[:,0].sum()/NE/(args.duration_ms/1000)),mean_I_rate_hz=float(whole[:,1].sum()/NI/(args.duration_ms/1000)),
        spatial_correspondence='PENDING_ANALYSIS',strict_bifurcation_accepted=False)
    write(dest/'result.json',result);write(dest/'progress.json',result);print(result,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--grid',type=int,default=20);ap.add_argument('--seed',type=int,default=9108401)
    ap.add_argument('--label',required=True);ap.add_argument('--core-ee',type=float,default=1.);ap.add_argument('--duration-ms',type=int,default=8000)
    ap.add_argument('--frozen-z-ms',type=int);ap.add_argument('--constant-input',action='store_true');ap.add_argument('--adaptive-tau',action='store_true');main(ap.parse_args())
