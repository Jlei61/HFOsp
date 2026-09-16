#!/usr/bin/env python3
"""Same physiological condition, independent convolution/update realizations.

Numerical QA only; these are not new parameter-search candidates.
"""
import argparse,json
import numpy as np
from scipy.signal import convolve
import run_topic4_liou_original_reference as ref

class DirectProjection(ref.Projection):
    def __call__(self,output):
        glob=self.p.w_global_i*output.mean()
        return 100.*convolve(output,self.ke,mode='same',method='direct'),self.p.w_local_i*convolve(output,self.ki,mode='same',method='direct')+glob,glob

def vector_update(s,pe,pi,stim,u,p):
    s[:]=ref.numpy_reference_update(s,pe,pi,stim,u,p)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=['direct_numba','direct_numpy'],required=True);args=ap.parse_args()
    original=ref.OUT;ref.OUT=original/'solver_validation';ref.OUT.mkdir(exist_ok=True)
    (ref.OUT/(args.variant+'_protocol.json')).write_text(json.dumps({'type':'Numerical QA; unchanged Exp2A equations, initial state and stimulus','variant':args.variant,'convolution':'Direct normalized Gaussian convolution; no FFT','update':'Vectorized source transcription' if args.variant=='direct_numpy' else 'Same scalar Numba update as production'},indent=2)+'\n')
    ref.Projection=DirectProjection
    if args.variant=='direct_numpy':ref.update=vector_update
    ref.run(ref.protocol('exp2a'),args.variant)
