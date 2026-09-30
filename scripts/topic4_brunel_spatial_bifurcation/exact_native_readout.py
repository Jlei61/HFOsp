"""Passive exact-neuron contact and current observers on paired native replays."""
from common import *
sys.path.insert(0,str(ROOT/'scripts/topic4_interictal_surrogate'))
from native_physics_graph import NativePhysicsGraph,NATIVE,FIELD,cp,DT
from interictal_common import weights
import argparse

def main(args):
    base=OUT/'expanded/native'
    w,lw=weights()
    native_geo=np.load(ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/geometry.npz')
    for J in args.values:
        dest=base/f'J{J:.4f}_s199771'
        if (dest/'exact_readout.json').exists():continue
        contract=read(dest/'contract.json');reference=np.load(dest/'trajectory.npz')
        e=NativePhysicsGraph(FIELD/'operators/g40_theta0.25',1,args.device)
        statement='Zi[i]+=.1/5000.*((recurrent_g[i]<95.19851312666987?1.:0.)-Zi[i]);'
        e.cells=cp.RawKernel(NATIVE.replace(statement,''),'native_particles',options=('--fmad=false',))
        region=e.geo['group_region'][e.geo['cell_group']][e.native_order]
        pop=e.geo['population'][e.geo['cell_group']][e.native_order]
        pa,ca,wa=e.operators[:3];rr=cp.asarray(np.repeat(np.arange(e.N,dtype=np.int32),np.diff(pa.get())));cc=ca%e.N
        rg=cp.asarray(region);ise=cp.asarray(pop==0)
        mask=ise[rr]&ise[cc]&(rg[rr]<2)&(rg[rr]==rg[cc]);e.operators[2]=wa*cp.where(mask,J,1.)
        rate_w=cp.asarray(np.r_[w,np.zeros((8000,15))][e.native_order])
        lfp_w=cp.asarray(np.r_[lw,np.zeros((8000,15))][e.native_order])
        assert np.allclose(rate_w.sum(0).get(),1) and np.allclose(lfp_w.sum(0).get(),1)
        rng=cp.random.RandomState(contract['seed']);nu=contract['private_input_per_ms'];steps=round(contract['duration_ms']/DT)
        accumulator=cp.zeros(e.N);groups=cp.zeros(e.P);raw=[];currents=[];regional=[];started=time.time()
        gr=e.geo['group_region'];ge=e.geo['population']==0;size=e.geo['group_size']
        aggregation=np.zeros((e.P,4))
        for k in range(4):
            take=ge&(gr==k) if k<3 else ~ge
            aggregation[take,k]=size[take]/size[take].sum()
        for start in range(0,steps,100):
            n=min(100,steps-start);ext=rng.poisson(nu*DT,size=(n,e.N)).astype(cp.float64)
            for j,k in enumerate(range(start,start+n)):
                groups+=e.step(ext,j,k,None);accumulator+=e.flags
                if (k+1)%5==0:
                    currents.append((cp.abs(e.ie+e.direct_ia)+cp.abs(e.native_Z*e.direct_ig))@lfp_w)
                if (k+1)%10==0:
                    raw.append(accumulator@rate_w*1000);accumulator.fill(0)
                    regional.append(groups.copy());groups.fill(0)
            if start%20000==0:print('exact',J,start*DT,round(time.time()-started,1),flush=True)
        regional=cp.stack(regional).get()@aggregation*1000
        error=float(abs(regional-reference['regional_rates_hz']).max());assert error<1e-9,error
        raw=cp.stack(raw).get();current=cp.stack(currents).get()
        np.savez_compressed(dest/'exact_readout.npz',contact_rate_hz=raw,lfp_raw=current,contact_names=native_geo['contact_names'])
        write(dest/'exact_readout.json',dict(status='COMPLETE',same_trajectory_regional_rate_max_error=error,
            contact_rate_source='Exact original neuron identities and original Gaussian contact weights; Hz at 1 ms',
            current_source='Original Eq9-11 spatial weights applied to abs(total AMPA)+abs(Z*GABA) of individual E neurons',
            lfp_sampling_ms=.5,lfp_units='model current proxy, arbitrary amplitude; not physical SEEG microvolts',
            seconds=time.time()-started,J_EE_core=J))
        print('EXACT COMPLETE',J,error,flush=True);del e;cp.get_default_memory_pool().free_all_blocks()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0);p.add_argument('--values',type=float,nargs='+',required=True);main(p.parse_args())
