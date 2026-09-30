"""Bounded follow-up of near-unit modes on the existing H1 return branch."""
from rate_stability_coverage import *


def main():
    p=argparse.ArgumentParser();p.add_argument('indices',type=int,nargs='+')
    p.add_argument('--device',type=int,default=0);a=p.parse_args();rows=[]
    dest=PERIODIC_OUT/'H1_secondary_crossing_followup.json'
    for index in a.indices:
        path=PERIODIC_OUT/f'orbits/arcAreturnStrong_{index:04d}_N512.npz'
        assert path.exists()
        while True:
            free=float(subprocess.check_output(['nvidia-smi','-i',str(a.device),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
            if free>8*1024:break
            write(dest,dict(status='WAITING_RESOURCE',pid=os.getpid(),index=index,rows=rows));time.sleep(30)
        write(dest,dict(status='RUNNING',pid=os.getpid(),index=index,rows=rows))
        actual,accuracy=prepare(path,a.device)
        assert accuracy['status']=='RESOLUTION_CHECKED',accuracy
        label=f'H1_followup_arcAreturnStrong_{index:04d}'
        source=PERIODIC_OUT/'floquet'/f'{label}_dt0.05.json'
        q=read(source) if source.exists() else compute(actual,.05,8,a.device,True,24,output_label=label)
        row=dict(index=index,orbit=str(actual),J_EE_core=q['J_EE_core'],T_ms=q['T_ms'],
            source=str(source),assessment=assess(q),accuracy=accuracy,multipliers=q['multipliers'],
            phase_defect=q['phase_tangent_relative_defect'])
        rows.append(row);print('FOLLOWUP',row,flush=True)
        import cupy as cp
        gc.collect();cp.get_default_memory_pool().free_all_blocks()
    write(dest,dict(status='SCREEN_FINISHED_CROSSINGS_UNCLASSIFIED',pid=os.getpid(),rows=rows,
        scope='Discrete Floquet samples in continuation order; no bifurcation label without an independently refined crossing.'))


if __name__=='__main__':main()
