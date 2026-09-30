"""Measured resource brackets at operational entry, never a bifurcation label."""
from common import *
import argparse


def main(stochastic):
    s=model();dest=OUT/('closure_stochastic_sensitivity' if stochastic else 'closure_network_sensitivity')
    rows=[]
    for lab in ['frozen','units','units_history']:
        folder=BASE/'runs/A4_stoch_seed9108401' if stochastic and lab=='frozen' else dest/lab
        raw=np.load(folder/'trajectory.npz');entry=read(folder/'result.json')['high_onset_ms']
        ts=(np.arange(len(raw['Z']))+1)*10.
        if entry is None:rows.append(dict(label=lab,entry_ms=None));continue
        i=int(np.searchsorted(ts,entry));indices=sorted(set([max(0,i-1),min(i,len(ts)-1)]));coords=[]
        for j in indices:
            z=raw['Z'][j].astype(float)
            r=dict(time_ms=float(ts[j]),D=float(raw['D'][j]),global_Z=float(1-raw['D'][j]))
            for reg,name in enumerate(['core_A','core_B','surround']):
                mask=s.E&(s.geo['group_region']==reg)
                r['Z_'+name]=float(np.average(z[mask],weights=s.sizes[mask]))
            r['M_current_E_mean_mV']=float(raw['M_current'][j].astype(float)[s.E]@s.mean_weights)
            coords.append(r)
        assert coords[0]['time_ms']<=entry<=coords[-1]['time_ms']
        rows.append(dict(label=lab,entry_ms=entry,adjacent_measured_resource_samples=coords))
    write(dest/'entry_resource_coordinates.json',dict(status='DESCRIPTIVE_COMPLETE',rows=rows,
        region_mapping='group_region 0=Core A, 1=Core B, 2=Surround, as operator producer and resource_path_comparison.py',
        scope='Adjacent measured samples at >=200Hz for200ms operational entry. Different histories and spatial fields; no interpolation, universal threshold or bifurcation claim.'))
    log('ENTRY COORDINATES',rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stochastic',action='store_true');main(p.parse_args().stochastic)
