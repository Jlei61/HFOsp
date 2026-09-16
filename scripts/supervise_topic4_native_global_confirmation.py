#!/usr/bin/env python3
"""Reuse the bounded scheduler for one reviewed native-Z paired noise."""
import subprocess
import run_topic4_native_global_confirmation as physical
import supervise_topic4_adaptation_capacity as controller

if __name__=='__main__':
    assert (physical.OUT/'dispatch_authorization.json').exists()
    original_popen=subprocess.Popen
    def popen(cmd,*args,**kwargs):
        old=physical.ROOT/'scripts/run_topic4_adaptation_capacity.py'
        new=physical.ROOT/'scripts/run_topic4_native_global_confirmation.py'
        return original_popen([str(new) if str(v)==str(old) else v for v in cmd],*args,**kwargs)
    controller.run=physical
    controller.OUT,controller.ROOT=physical.OUT,physical.ROOT
    controller.read,controller.write=physical.carrier.base.read,physical.carrier.base.write
    controller.subprocess.Popen=popen
    try:controller.main()
    finally:controller.subprocess.Popen=original_popen
