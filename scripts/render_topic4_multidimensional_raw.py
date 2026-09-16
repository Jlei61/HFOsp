#!/usr/bin/env python3
"""Reuse the audited native movie comparison producer for the multidimensional pilot."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import audit_topic4_xy_raw_propagation_video as producer
from scripts.run_topic4_multidimensional_pilot import OUT,read

if __name__=='__main__':
    if read(ROOT/'config/topic4_joint_xy_kernel_v3.json')['observation']!=read(ROOT/'config/topic4_joint_xy_kernel_v4.json')['observation']:
        raise RuntimeError('raw renderer observation contract differs')
    producer.SEARCH=OUT
    producer.main()
