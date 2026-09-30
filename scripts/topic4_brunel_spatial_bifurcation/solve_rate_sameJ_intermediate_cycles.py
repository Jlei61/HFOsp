"""Resolve omitted H1 return cycles at the exact b/c parameter, J=.942."""
from rate_periodic_accuracy import *
from run_rate_sameJ_basin_bridge import DEST
import gc


def main(device):
    import cupy as cp
    s=RateField();rows=[];target=.942
    sources=[('H1_middle','arcA_0069_N64',128),('H1_return','arcAreturnStrong_0047_N512',512)]
    for name,stem,N in sources:
        source=PERIODIC_OUT/'orbits'/f'{stem}.npz'
        path=PERIODIC_OUT/'orbits'/f'connection_J0942_{name}_N{N}.npz'
        if not path.exists():
            z=np.load(source);o=Periodic(s,N,device);o.low_memory=True;o.stream_harmonics=True
            o.harmonic_chunk_size=32;o.derivative_chunk_size=32;o.host_krylov=True
            o.normalize_linear_rhs=True
            r,T,J,e,h=o.solve(resample(z['r'],N,axis=0),float(z['T']),target,tol=1e-10,maxiter=14)
            assert J==target
            path=save_orbit(s,r,T,J,e,h,path.stem);del o;gc.collect()
            import cupy as cp
            cp.fft.config.get_plan_cache().clear();cp.get_default_memory_pool().free_all_blocks()
            if e>=1e-10:
                rows.append(dict(label=name,status='CORRECTION_UNRESOLVED',source=str(source),orbit=str(path),residual_Hz=e))
                write(DEST/'sameJ_intermediate_cycles.json',dict(status='PARTIAL',rows=rows));continue
        actual,check=prepare(path,device,max_N=4096,check_filter_states=True,stream_harmonics=True,harmonic_chunk_size=32,host_krylov=True)
        meta=read(Path(actual).with_suffix('.json'))
        rows.append(dict(label=name,status=check['status'],source=str(source),orbit=str(actual),
            J_EE_core=meta['J_EE_core'],T_ms=meta['T_ms'],mean_rates_hz=meta['mean_rates_hz'],
            resolution=check,stability='NOT_YET_CHECKED_ON_THIS_EXACT_PROFILE'))
        write(DEST/'sameJ_intermediate_cycles.json',dict(status='RUNNING',rows=rows))
        print('INTERMEDIATE',name,meta['J_EE_core'],meta['T_ms'],meta['mean_rates_hz'],check['status'],flush=True)
        gc.collect();cp.fft.config.get_plan_cache().clear();cp.get_default_memory_pool().free_all_blocks()
    write(DEST/'sameJ_intermediate_cycles.json',dict(status='CORRECTIONS_FINISHED',rows=rows,
        scope='Exact same-J correction of previously traced H1 return branches. These are intermediate mean-rate solutions, not proven connections to the alternating-burst family or basin separators.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1);main(p.parse_args().device)
