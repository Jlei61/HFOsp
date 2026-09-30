"""Bounded Floquet screen for connections missed by branch geometry alone.

Negative multipliers of unstable H1 cycles motivate checking additional
period-doubling branches. A negative multiplier is not itself a PD crossing;
complex-pair rotation can also change the signs of real multipliers.
"""
from rate_floquet import *
from rate_periodic_accuracy import prepare
from rate_stability_coverage import assess
import gc,subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0)
    a=p.parse_args();rows=[];status=PERIODIC_OUT/'H1_secondary_mode_screen.json'
    sites=[('arcAreturnStrong',i) for i in [115,135,155]]+[
           ('arcAglobalConnection',i) for i in [20,60,100]]
    for segment,index in sites:
        label=f'H1_secondary_screen_{segment}_{index:04d}'
        source=PERIODIC_OUT/'orbits'/f'{segment}_{index:04d}_N512.npz'
        output=PERIODIC_OUT/'floquet'/f'{label}_dt0.1.json'
        write(status,dict(status='RUNNING',pid=os.getpid(),current_source=str(source),rows=rows))
        while True:
            free=float(subprocess.check_output(['nvidia-smi','-i',str(a.device),
                '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
            if free>6*1024:break
            time.sleep(20)
        actual,accuracy=prepare(source,a.device)
        assert accuracy['status']=='RESOLUTION_CHECKED',accuracy
        q=read(output) if output.exists() else compute(actual,.1,6,a.device,True,20,output_label=label)
        raw=np.asarray(q['multipliers']);v=raw[:,0]+1j*raw[:,1] if raw.ndim==2 else raw.astype(complex)
        residual=np.asarray(q['residuals'])/np.maximum(1,abs(v))
        real=(abs(v.imag)<1e-6)&(residual<1e-6)
        rows.append(dict(segment=segment,index=index,source=str(source),floquet_source=str(output),
            J_EE_core=q['J_EE_core'],T_ms=q['T_ms'],assessment=assess(q),
            reliable_negative_real_multipliers=v[real&(v.real<0)],
            filtered_coverage_reached=q['smallest_returned_transformed_modulus']<q['filter_coverage_threshold'],
            phase_defect=q['phase_tangent_relative_defect'],accuracy=accuracy))
        print('SECONDARY SCREEN',rows[-1],flush=True)
        gc.collect();cp=__import__('cupy');cp.get_default_memory_pool().free_all_blocks()
    write(status,dict(status='SCREEN_COMPLETE_CROSSINGS_NOT_CLASSIFIED',pid=os.getpid(),rows=rows,
        scope='Six branch-ordered Floquet samples. Negative multipliers motivate tracking; only a refined -1 crossing and independent mesh/mode validation establish a new PD. No complete interval or connection claim.'))


if __name__=='__main__':main()
