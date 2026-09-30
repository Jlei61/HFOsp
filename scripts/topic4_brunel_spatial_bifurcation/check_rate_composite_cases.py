"""Resolve and check the exact periodic profiles used in the all-rate figure."""
from rate_periodic_accuracy import *
from plot_rate_periodic_composite import CASES
import gc


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0)
    a=p.parse_args();rows=[]
    for letter,name,J,title in CASES:
        if name is None:continue
        original=PERIODIC_OUT/'orbits'/(name+'.npz')
        # Prefer a previously refined profile of this exact case, never a
        # neighboring parameter or a different periodic family.
        candidates=[original,*original.parent.glob(name+'_accuracy_N*.npz')]
        candidate=max(candidates,key=lambda f:len(np.load(f)['r']))
        assert abs(float(np.load(candidate)['J'])-J)<1e-12
        actual,check=prepare(candidate,a.device)
        assert check['status']=='RESOLUTION_CHECKED',check
        assert check['minimum_rate_Hz']>=0,check
        rows.append(dict(case=letter,original_orbit=str(original),orbit=str(actual),
                         J_EE_core=J,resolution=check))
        write(PERIODIC_OUT/'composite_case_resolution.json',dict(
            status='COMPLETE' if len(rows)==len(CASES)-1 else 'IN_PROGRESS',rows=rows,
            scope='Same-J waveform resolution only; this check does not establish Floquet stability.'))
        gc.collect()
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()


if __name__=='__main__':main()
