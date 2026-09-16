#!/usr/bin/env python3
"""Resume the frozen log-M worker with original ordered CPU scatter only."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import argparse,time
import run_topic4_fig5_log_m_scan as original
def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['worker']);ap.add_argument('--name',required=True)
    ap.add_argument('--producer-script',required=True);args=ap.parse_args()
    assert args.name.startswith('eta1_tau') and args.producer_script.endswith('run_topic4_fig5_log_m_scan.py')
    qa=original.base.read(original.ROOT/'results/topic4_sef_hfo/autonomous_recovery_exploration_20260914/quiet_backend_benchmark.json')
    assert qa['status']=='PASS' and qa['entire_checkpoint_recursive_bitwise']
    folder=original.OUT/'runs'/args.name
    # No change to job, equations, observer, endpoint or initial-state restoration.
    original.wrap_simulator=lambda fn,device_index:fn
    try:
        original.worker(args.name)
    except Exception as e:
        original.base.write(folder/'failure.json',dict(error=repr(e),time=time.time(),backend='original_ordered_CPU_scatter'));raise
if __name__=='__main__':main()
