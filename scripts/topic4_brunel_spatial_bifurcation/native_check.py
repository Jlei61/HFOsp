"""Native 40,000-LIF network at the same frozen interictal spectral conditions.

Only within-core EE is varied. Z=1 is prescribed for this autonomous fast
subsystem check; original M feedback, all delays, thresholds, private Poisson
input and individual synaptic currents remain. OU fluctuations are zero,
matching the spectral operating point rather than the original OU trial.
"""
from common import *
sys.path.insert(0,str(ROOT/'scripts/topic4_interictal_surrogate'))
from native_physics_graph import NativePhysicsGraph,NATIVE,FIELD,cp,DT
import argparse

def main(args):
    for J in args.values:
        dest=OUT/args.label/f'J{J:.4f}_s{args.seed}';dest.mkdir(parents=True,exist_ok=True)
        if (dest/'result.json').exists():continue
        e=NativePhysicsGraph(FIELD/'operators/g40_theta0.25',1,args.device)
        # Remove just the Z evolution statement; retain individual M updates.
        statement='Zi[i]+=.1/5000.*((recurrent_g[i]<95.19851312666987?1.:0.)-Zi[i]);'
        assert statement in NATIVE;e.cells=cp.RawKernel(NATIVE.replace(statement,''),'native_particles',options=('--fmad=false',))
        pop=e.geo['population'][e.geo['cell_group']][e.native_order]
        region=e.geo['group_region'][e.geo['cell_group']][e.native_order]
        count=e.geo['group_size'];group_region=e.geo['group_region'];group_E=e.geo['population']==0
        ptra,col,w=e.operators[:3];rr=cp.asarray(np.repeat(np.arange(e.N,dtype=np.int32),np.diff(ptra.get())));cc=col%e.N
        region_gpu=cp.asarray(region);is_e=cp.asarray(pop==0)
        mask=is_e[rr]&is_e[cc]&(region_gpu[rr]<2)&(region_gpu[rr]==region_gpu[cc])
        e.operators[2]=w*cp.where(mask,J,1.)
        warmup_steps=round(args.warmup_ms/DT)
        if warmup_steps:
            e.operators[2]=w*cp.where(mask,args.warmup_J,1.)
            warm_rng=cp.random.RandomState(args.seed+199)
            for start in range(0,warmup_steps,100):
                n=min(100,warmup_steps-start);ext=warm_rng.poisson(float(e.prep['nu_ext_per_ms'])*DT,size=(n,e.N)).astype(cp.float64)
                for j,k in enumerate(range(start,start+n)):e.step(ext,j,k,None)
            e.operators[2]=w*cp.where(mask,J,1.)
        rng=cp.random.RandomState(args.seed);nu=float(e.prep['nu_ext_per_ms']);steps=round(args.duration/DT)
        accum=cp.zeros(e.P);samples=[];ras_t=[];ras_i=[]
        exact_raw=[];exact_lfp=[]
        if args.exact_readout:
            from interictal_common import weights
            ww,ll=weights();rate_w=cp.asarray(np.r_[ww,np.zeros((8000,15))][e.native_order])
            lfp_w=cp.asarray(np.r_[ll,np.zeros((8000,15))][e.native_order]);neuron_counts=cp.zeros(e.N)
        draw=np.random.default_rng(771);selected=[]
        for region_id in (0,1,2):
            ids=np.flatnonzero((pop==0)&(region==region_id));selected.extend(draw.choice(ids,min(250,len(ids)),False))
        selected=np.array(selected);dsel=cp.asarray(selected);started=time.time()
        write(dest/'contract.json',dict(J_EE_core=J,seed=args.seed,duration_ms=args.duration,neurons=e.N,
            topology=6101,private_input_per_ms=nu,common_OU='zero fluctuation',Z='fixed 1',M='original dynamic',
            initial_state='original reset, zero currents' if not warmup_steps else f'carried from J={args.warmup_J:g} after {args.warmup_ms:g} ms',
            warmup_J=args.warmup_J if warmup_steps else None,warmup_ms=args.warmup_ms,
            spatial_readout='original 15 firing-envelope contacts',scope='Native stochastic check of the same spectral parameter conditions'))
        for start in range(0,steps,100):
            n=min(100,steps-start);ext=rng.poisson(nu*DT,size=(n,e.N)).astype(cp.float64);sp=[]
            for j,k in enumerate(range(start,start+n)):
                accum+=e.step(ext,j,k+warmup_steps,None);sp.append(e.flags[dsel].copy())
                if args.exact_readout:
                    neuron_counts+=e.flags
                    if (k+1)%5==0:exact_lfp.append((cp.abs(e.ie+e.direct_ia)+cp.abs(e.native_Z*e.direct_ig))@lfp_w)
                    if (k+1)%10==0:exact_raw.append(neuron_counts@rate_w*1000);neuron_counts.fill(0)
                if (k+1)%10==0:samples.append(accum.copy());accum.fill(0)
            rr0,cc0=np.nonzero(cp.stack(sp).get());ras_t.extend((rr0+start+1)*DT);ras_i.extend(cc0)
            if start%10000==0:print('native',J,start*DT,'elapsed',round(time.time()-started,1),flush=True)
        activity=cp.stack(samples).get();rates=[]
        for k in (0,1,2):
            m=group_E&(group_region==k);rates.append(np.average(activity[:,m],axis=1,weights=count[m])*1000)
        m=~group_E;rates.append(np.average(activity[:,m],axis=1,weights=count[m])*1000)
        contact=activity@e.geo['contact_rate_weights']*1000
        field=np.zeros((len(activity),1600))
        for g in np.flatnonzero(group_E):field[:,e.geo['group_cell'][g]]+=activity[:,g]*count[g]
        totals=np.bincount(e.geo['group_cell'][group_E],weights=count[group_E],minlength=1600)
        field=field/np.maximum(totals,1)*1000
        np.savez_compressed(dest/'trajectory.npz',time_ms=np.arange(len(activity))+1,regional_rates_hz=np.array(rates).T,
            field_E_hz=field.astype('float32'),contact_rate_hz=contact.astype('float32'),raster_time_ms=np.array(ras_t),raster_id=np.array(ras_i),raster_region=region[selected])
        if args.exact_readout:
            names=np.load(ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/geometry.npz')['contact_names']
            np.savez_compressed(dest/'exact_readout.npz',contact_rate_hz=cp.stack(exact_raw).get(),lfp_raw=cp.stack(exact_lfp).get(),contact_names=names)
            write(dest/'exact_readout.json',dict(status='COMPLETE',observation='Exact original neuron weights; recorded during the same native trajectory',
                lfp_sampling_ms=.5,contact_sampling_ms=1.,J_EE_core=J))
        write(dest/'result.json',dict(status='COMPLETE',J_EE_core=J,seconds=time.time()-started,mean_rates_hz=np.mean(np.array(rates)[:,500:],axis=1)))
        del e;cp.get_default_memory_pool().free_all_blocks()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--values',type=float,nargs='+',required=True);p.add_argument('--seed',type=int,default=199771)
    p.add_argument('--device',type=int,default=0);p.add_argument('--duration',type=int,default=5000)
    p.add_argument('--label',default='native_private_only');p.add_argument('--warmup-ms',type=float,default=0.)
    p.add_argument('--warmup-J',type=float,default=2.);p.add_argument('--exact-readout',action='store_true');main(p.parse_args())
