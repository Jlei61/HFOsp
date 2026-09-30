"""Separate exogenous inputs from native outputs used only for validation."""
from common import *
import argparse
import time


def main(a):
    folder=OUT/'reference';folder.mkdir(exist_ok=True)
    source=SOURCE/f'replay/runs/eta0.0005_s{a.seed}'
    inputs=[];globals_=[];fields=[];spikes=[];regions=[];zt=[];zs=[];ms=[]
    for path in sorted((source/'fields').glob('*.npz')):
        with np.load(path) as z:
            inputs.append(z['drive_mean']);globals_.append(z['glob'])
            zt.append(z['zm_step']*.1);zs.append(z['z'].mean(1));ms.append(z['m'].mean(1))
    geo=dict(np.load(GRID/'geometry.npz'))
    for path in sorted((source/'chunks').glob('*.npz')):
        with np.load(path) as z:
            fields.append(z['field_1ms']);spikes.append(z['spikes_1ms']);regions.append(z['regions_1ms'][:,:3])
    inputs=np.concatenate(inputs);glob=np.concatenate(globals_);field=np.concatenate(fields)
    np.savez_compressed(folder/f'input_s{a.seed}.npz',E_rate_per_ms=inputs,I_rate_per_ms=glob,dt_ms=1.)
    np.savez_compressed(folder/f'outputs_s{a.seed}.npz',E_rate_1ms_hz=field/geo['count_e']*1000,
        global_rates_1ms_hz=np.concatenate(spikes)/np.array([32000,8000])*1000,
        region_spikes_1ms=np.concatenate(regions),Z_time_ms=np.concatenate(zt),Z_E=np.concatenate(zs),M_E=np.concatenate(ms),
        cell_count_e=geo['count_e'],centers_mm=geo['centers_mm'])
    write(folder/f'lineage_s{a.seed}.json',dict(source=str(source),seed=a.seed,duration_ms=len(inputs),
        input_file_contents='Only exogenous conditional Poisson intensity, at the recorded1ms clock',
        output_file_role='Validation only; never read by the candidate simulator',
        input_precision='Recorded field mean intensity uses float32 and1ms sampling; not claimed bitwise native private-Poisson replay',
        split='Development' if a.seed==9108401 else 'Second-reference validation; keep separate from fitted pulse inputs'))
    print('prepared',a.seed,len(inputs),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=9108401);main(ap.parse_args())
