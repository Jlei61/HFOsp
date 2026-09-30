#!/usr/bin/env python3
"""Check the negligible s/G label correction against every executed input bit."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
from campaign import read,write,sha
from observe_source_aggregation import OUT as SOURCE
from conditional_density_inputs import OPS
from audit_target_root_response import density_condition

def main():
    out=SOURCE/'local_response_factorial'
    d=read(out/'result.json');inputs=dict(np.load(out/'inputs.npz'))
    obs=dict(np.load(SOURCE/'cell_statistics.npz'));G=float(30*obs['global_R_and_s'][:,1].mean())
    params=read(OPS/'prepared.json')['params'];new=[]
    for r in d['rows']:
        E=r['population']=='E';Z=r['held_Z'];K=r['held_K'];M=r['held_native_M']
        g=K+(Z*G if E else 0.);h=1+g;mean=np.array(r['raw_IE_II_mean']);variance=np.array(r['raw_IE_II_variance'])
        effective=variance*np.array([1,Z**2])/h**2
        mu=(mean[0]-Z*mean[1]-.0005*M-30*K-(17.662847938268442*Z*G if E else 0))/h
        _,unit=density_condition([0.,1.,1.],g,r['threshold_mV'],r['population'],params)
        p,_=density_condition(np.r_[mu,effective/unit],g,r['threshold_mV'],r['population'],params);new.append(p)
    new=np.array(new);assert np.array_equal(new,inputs['pars'])
    assert sha(out/'executed_producer.py')==read(out/'contract.json')['producer_sha256']
    write(out/'raw_G_units_qa.json',dict(status='PASS_ALL_EXECUTED_PARAMETERS_BITWISE_UNCHANGED',
        issue='The initial local assay read global_R_and_s[:,1] as G, though it is s. PhysicalGraw=30*s. At this late conditional state both are negligible; reconstruction with correct units reproduces every input parameter bit exactly.',
        stored_s_mean=float(inputs['held_global_G']),correct_raw_G_mean=G,conditions=len(new),
        all_parameter_bits_equal=True,new_MC_required=False,
        original_executed_producer='executed_producer.py',corrected_producer='assay_observed_input_moments.py',
        verification_producer_sha256=sha(__file__)))
    print('All',len(new),'executed parameter vectors unchanged after exact G=30*s correction')

if __name__=='__main__':main()
