"""Bounded mixed-precision trajectory pilot, never a branch-stability solver.

Only the conditional-noise matrix product is FP32. Voltage, recurrent currents,
delay history and M stay FP64. The stationary noise marginal is constrained to
its known invariant vector after the product, preventing cumulative mass drift.
Critical points and final orbit correction still use the FP64 reference map.
"""
from autonomous_density import *


class MixedDensity(AutonomousDensity):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.transition32=self.transition.astype(cp.float32)
        values,vectors=np.linalg.eig(cp.asnumpy(self.transition));at=np.argmin(abs(values-1.))
        marginal=vectors[:,at].real;marginal/=cp.asnumpy(self.mass)@marginal
        assert abs(values[at]-1.)<1e-10 and marginal.min()>0
        self.fixed_marginal=cp.asarray(marginal)
        assert cp.cuda.cublas.getMathMode(cp.cuda.device.get_cublas_handle())==cp.cuda.cublas.CUBLAS_DEFAULT_MATH

    def mix_noise(self):
        moved=cp.matmul(self.transition32,self.F.astype(cp.float32)).astype(cp.float64)
        moved*=self.fixed_marginal[None,:,None]/cp.sum(moved,axis=2)[:,:,None]
        return cp.ascontiguousarray(moved)


def run(args):
    source=Path(args.source);cfg=read(source/'config.json')
    folder=OUT/'mixed_precision_pilots'/args.label;folder.mkdir(parents=True,exist_ok=False)
    modified='FP32 noise-basis matrix product, invariant marginal constrained'
    retained='Voltage/refractory transport, Z/M, delayed recurrent currents and state storage'
    if args.pdf_fp32 or args.conservative_pdf_fp32:
        from fast_search_density import FastSearchDensity,ConservativeFastSearchDensity
        fast_class=ConservativeFastSearchDensity if args.conservative_pdf_fp32 else FastSearchDensity
        modified='FP32 PDF/noise product and voltage/refractory transport, invariant marginal constrained'
        retained='Z/M, delayed recurrent currents/history and weighted firing readout'
        if args.conservative_pdf_fp32:modified+='; exact modal mass also restored after voltage transport'
    else:fast_class=MixedDensity
    write(folder/'contract.json',dict(source=str(source),duration_ms=args.duration,
        modified_operation=modified,retained_FP64=retained,
        criteria=dict(global_rate_relative_RMS=.001,global_rate_max_absolute_hz=.1,
                      spatial_weighted_relative_RMS=.001,mean_M_absolute_difference=.002),
        inference='Short local trajectory accuracy/speed only; no long-time, bifurcation or Floquet acceptance'))
    ref=AutonomousDensity(cfg['D'],cfg['degree'],cfg['voltage_dv'],args.device,basis_mode=cfg.get('basis_mode','legacy'));ref.restore(source)
    fast=fast_class(cfg['D'],cfg['degree'],cfg['voltage_dv'],args.device,basis_mode=cfg.get('basis_mode','legacy'));fast.restore(source)
    rates=[[],[]];fields=[[],[]];timing=[0.,0.];blocks=[cp.zeros(ref.P),cp.zeros(ref.P)]
    models=[ref,fast];started=time.time();last=started
    for step in range(round(args.duration/DT)):
        for j,m in enumerate(models):
            a=cp.cuda.Event();b=cp.cuda.Event();a.record();activity=m.advance_step();b.record();b.synchronize()
            timing[j]+=cp.cuda.get_elapsed_time(a,b);blocks[j]+=activity
            if (step+1)%10==0:
                rate=blocks[j]*1000.;rates[j].append(cp.asnumpy(cp.r_[m.e_weights@rate,m.region_weights@rate]))
                fields[j].append(cp.asnumpy(cp.bincount(m.observable_cell,weights=rate*m.e_sizes,minlength=1600)))
                blocks[j].fill(0.)
        if time.time()-last>20:
            write(folder/'status.json',dict(status='RUNNING',pid=os.getpid(),completed_ms=(step+1)*DT))
            print('mixed pilot',args.label,(step+1)*DT,flush=True);last=time.time()
    rates=np.asarray(rates);fields=np.asarray(fields)
    count=np.bincount(ref.geo['group_cell'],weights=np.where(ref.geo['population']==0,ref.geo['group_size'],0),minlength=1600)
    fields/=np.maximum(count,1.)[None,None,:]
    dr=rates[1,:,0]-rates[0,:,0];df=fields[1]-fields[0];weights=count/32000.
    rerr=np.sqrt(np.mean(dr*dr)/max(np.mean(rates[0,:,0]**2),1e-30))
    ferr=np.sqrt(np.mean((df*df)@weights)/max(np.mean((fields[0]*fields[0])@weights),1e-30))
    mdiff=float(abs((ref.e_weights@(fast.M-ref.M)).get()))
    qa=dict(global_rate_relative_RMS=float(rerr),global_rate_max_absolute_hz=float(np.max(abs(dr))),
        spatial_weighted_relative_RMS=float(ferr),mean_M_absolute_difference=mdiff,
        elapsed_GPU_ms=timing,speed_ratio=timing[0]/timing[1],
        reference_diagnostics=ref.diagnostics(),mixed_diagnostics=fast.diagnostics(),
        pass_short_trajectory=bool(rerr<.001 and np.max(abs(dr))<.1 and ferr<.001 and mdiff<.002))
    np.savez_compressed(folder/'trajectories.npz',rate_1ms=rates,field_1ms=fields,count_e=count)
    write(folder/'status.json',dict(status='SHORT_TRAJECTORY_PILOT_COMPLETE',qa=qa,wall_s=time.time()-started,
        accepted_for_final_bifurcation='NO; FP64 correction/stability remains required'))
    print(qa,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--duration',type=float,default=200.);ap.add_argument('--label',required=True)
    ap.add_argument('--device',type=int,default=1)
    g=ap.add_mutually_exclusive_group();g.add_argument('--pdf-fp32',action='store_true');g.add_argument('--conservative-pdf-fp32',action='store_true');run(ap.parse_args())
