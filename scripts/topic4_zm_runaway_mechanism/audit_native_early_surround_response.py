"""Independent binning and actual GPU local-equation parity for early inputs."""
from common import OUT,np,read,write,log,model
from refractory_rate_cuda import LocalGPUResponse
import argparse,gc

DEST=OUT/'native_early_surround_inputs'


def main(device):
    assert read(DEST/'input_response_result.json')['status']=='FIXED_RESPONSE_DIAGNOSTIC_COMPLETE'
    g=np.load(DEST/'membership.npz');selected=g['selected_groups'];G=len(selected)
    sources=[np.load(f) for f in sorted((DEST/'inputs').glob('*.npz'))]
    counts=np.concatenate([z['spikes'][:,selected] for z in sources]);mom=np.concatenate([z['moments'] for z in sources])
    names=sources[0]['moment_names'].tolist();zpath=mom[:,names.index('z')];mu=mom[:,names.index('net')]
    target=np.load(DEST/'fixed_readout.npz');raw=np.load(DEST/'projected_inputs.npz');s=model(40)
    native=counts.reshape(60,500,G).sum(1)[10:]
    assert np.array_equal(native,target['native_counts']) and np.array_equal(target['bin_start_ms'],np.arange(500,3000,50))
    # Original per-cell current snapshots are float32; bound their quantization
    # explicitly when comparing new float64 observed group moments.
    worst_ratio=0.;moment_checks=0
    ref=OUT.parent/'fig5_current_network_z_state_v1/replay/runs/eta0.0005_s9108401/fields'
    for path in sorted((DEST/'inputs').glob('*.npz')):
        with np.load(path) as observed,np.load(ref/path.name) as original:
            ie=original['ie'];ii=original['ii'];zz=original['z'];mm=original['m']*.0005
            target_moments=observed['moments'];start_step=int(observed['start_step'])
            for k,step in enumerate(original['zm_step']):
                index=int(step-start_step)
                for j,group in enumerate(selected):
                    if g['population'][group]!=0:continue
                    mask=g['cell_group'][:32000]==group
                    ea=ie[k,mask].astype(float);ib=ii[k,mask].astype(float)
                    da=.5*np.abs(np.spacing(ie[k,mask])).astype(float);db=.5*np.abs(np.spacing(ii[k,mask])).astype(float)
                    zs=zz[k,mask];ms=mm[k,mask]
                    values=[('ampa',ea,da),('gaba',ib,db),('net',ea-zs*ib-ms,da+zs*db),
                        ('ampa2',ea*ea,2*abs(ea)*da+da*da),('gaba2',ib*ib,2*abs(ib)*db+db*db)]
                    for name,value,bound in values:
                        difference=abs(value.mean()-target_moments[index,names.index(name),j])
                        limit=bound.mean()+1e-10*max(abs(value.mean()),1.)
                        assert difference<=limit,(path.name,int(group),int(step),name,difference,limit)
                        worst_ratio=max(worst_ratio,float(difference/limit));moment_checks+=1
    for label in ['native_mean_private_variance','projected_mean_private_variance','native_mean_native_marginals']:
        expected=target[label].reshape(60,500,G).sum(1)[10:]*.1/1000*s.sizes[selected]
        assert np.max(abs(expected-target['counts_'+label]))<1e-9
    projected=raw['filtered_moments'];raw_variance=raw['raw_private_variance_forcing'];rows=[]
    for label,net in [('native_mean_private_variance',mu),
                      ('projected_mean_private_variance',projected[:,0]-zpath*projected[:,1]-mom[:,names.index('mcurrent')])]:
        physical=np.concatenate([net[:,None],raw_variance],axis=1)
        e=LocalGPUResponse(g['population'][selected],g['threshold_mv'][selected],.1,len(physical)+1,physical,device=device)
        cp=e.cp;zp=cp.asarray(zpath)
        loadz=cp.RawKernel(r'''
extern "C" __global__ void load_z(double* z,const double* path,const int* clock,int G){
 int g=blockIdx.x*blockDim.x+threadIdx.x;if(g<G)z[g]=path[(long long)clock[0]*G+g];
}''','load_z',options=('--fmad=false',))
        def step():
            loadz(((G+127)//128,),(128,),(e.Z,zp,e.clock,np.int32(G)));e.step()
        step();cp.cuda.get_current_stream().synchronize()
        e.state.fill(0);e.history.fill(0);e.rate.fill(0);e.clock.fill(0);e.Z.fill(1)
        cp.cuda.get_current_stream().synchronize();stream=cp.cuda.Stream(non_blocking=True)
        with stream:
            stream.begin_capture()
            for _ in range(100):step()
            graph=stream.end_capture()
        for _ in range(len(physical)//100):graph.launch(stream)
        stream.synchronize();observed=e.history.get()[1:]*1000;expected=target[label]
        error=float(np.max(abs(observed-expected)));relative=float(np.linalg.norm(observed-expected)/max(np.linalg.norm(expected),1.))
        assert error<1e-4 and relative<1e-8,(label,error,relative)
        assert np.array_equal(e.Z.get(),zpath[-1]) and int(e.clock.get()[0])==30000
        rows.append(dict(label=label,dt_ms=.1,steps=30000,maximum_rate_difference_hz=error,relative_L2=relative))
        log('EARLY INPUT CUDA PARITY',rows[-1])
        del e,zp,graph;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    write(DEST/'independent_response_audit.json',dict(status='PASS',native_fixed_bins_exact=True,rate_bins_exact=True,
        original_raw_current_moment_checks=moment_checks,maximum_quantization_bound_ratio=worst_ratio,
        current_GPU_local_response_parity=rows,
        scope='Numerical readout identity for the actual input waveforms and prescribed nativeZ, plus independent fixedbin reconstruction. No native/model scientific acceptance or bifurcation claim.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1);main(p.parse_args().device)
