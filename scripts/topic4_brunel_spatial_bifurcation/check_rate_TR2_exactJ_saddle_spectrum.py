"""Paired full-history spectrum at the exact-J target of the closest TR2 torus."""
from complete_rate_positive_stability import *
from audit_current_rate_model import full_rhs_check
import subprocess


FOLDER=DEST/'TR2_saddle_directions/exactJ_spectrum'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',type=int,default=0)
    parser.add_argument('--min-free-gib',type=float,default=8.)
    parser.add_argument('--after-pids',type=int,nargs='*',default=[])
    args=parser.parse_args();assert args.min_free_gib>=8.
    FOLDER.mkdir(exist_ok=True)
    def status(stage,**kw):
        write(FOLDER/'worker.json',dict(status=stage,pid=os.getpid(),timestamp=time.time(),**kw))
        print(stage,kw,flush=True)
    source=read(PERIODIC_OUT/'TR2_same_parameter_saddle_approach.json')['rows'][-1]
    target=Path(source['target']);torus=np.load(source['torus']);z=np.load(target)
    assert float(z['J'])==float(torus['J'])==source['J_EE_core']
    assert z['r'].shape==(128,935)
    s=RateField();check=full_rhs_check(s,target,factor=4)
    assert check['between_nodes_integrated_error_Hz']<1e-8
    assert check['full_RHS_rate_error_Hz_per_ms']<1e-8
    assert check['minimum_interpolated_group_rate_Hz']>0
    N=len(z['r']);M=4*N;k=np.arange(N//2+1)
    lam=2j*np.pi*k[:,None]/float(z['T']);rf=np.fft.rfft(z['r'],axis=0)/N
    targetf=rf/s.filter_response(lam)
    minima=[]
    for tau in [s.tf,s.ts]:
        padded=np.zeros((M//2+1,s.P),complex)
        padded[:len(k)]=M*targetf/(1+lam*tau)
        padded[len(k)-1]*=.5
        minima.append(float(np.fft.irfft(padded,n=M,axis=0).min()*1000))
    assert min(minima)>0
    write(FOLDER/'physical_check.json',dict(source=source['target'],J_EE_core=float(z['J']),
        torus_source=source['torus'],independent_CPU_full_RHS=check,minimum_fast_slow_filter_Hz=minima,
        model='Unchanged 400-cell / 935-population rate DDE'))
    dependencies={pid:Path(f'/proc/{pid}/cmdline').read_bytes() for pid in args.after_pids
                  if Path(f'/proc/{pid}/cmdline').exists()}
    while dependencies:
        for pid,identity in list(dependencies.items()):
            file=Path(f'/proc/{pid}/cmdline')
            try:live=file.read_bytes()==identity
            except FileNotFoundError:live=False
            if not live:dependencies.pop(pid)
        if dependencies:
            status('WAITING_DEPENDENCIES',dependencies=list(dependencies));time.sleep(30)
    spectra=[];sources=[]
    for dt in [.1,.05]:
        name=f'TR2_exactJ_a0038_saddle_k8_20260920_dt{dt:g}'
        output=PERIODIC_OUT/'poincare_floquet'/(name+'.json')
        if output.exists():q=read(output)
        else:
            release(args.device)
            while True:
                free=float(subprocess.check_output(['nvidia-smi','-i',str(args.device),
                    '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
                if free>=args.min_free_gib*1024:break
                status('WAITING_GPU_RESOURCE',free_mib=free,required_free_gib=args.min_free_gib,dt_ms=dt)
                time.sleep(30)
            status('FULL_HISTORY_SPECTRUM',dt_ms=dt,orbit=str(target))
            q=compute_return(target,dt,8,args.device,24,stream_harmonics=True,
                             output_label='TR2_exactJ_a0038_saddle_k8_20260920')
        assert Path(q['orbit']).resolve()==target.resolve()
        assert q['J_EE_core']==float(z['J'])
        spectra.append(q);sources.append(str(output))
    verdict=paired_modes(*spectra)
    write(FOLDER/'result.json',dict(status=verdict['status'],sources=sources,orbit=str(target),
        J_EE_core=float(z['J']),classification=verdict,physical_source=str(FOLDER/'physical_check.json'),
        scope='Paired full-history spectrum at this exact-J saddle only. Does not prove torus stability or a global invariant-manifold connection.'))
    release(args.device)
    status('SPECTRUM_FINISHED',scientific_status=verdict['status'])


if __name__=='__main__':
    try:main()
    except Exception as exc:
        FOLDER.mkdir(exist_ok=True)
        write(FOLDER/'worker.json',dict(status='COMPUTATION_FAILED',pid=os.getpid(),error=repr(exc)))
        raise
