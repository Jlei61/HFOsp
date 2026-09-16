#!/usr/bin/env python3
"""Common random draws for full-duration execution of original MATLAB methods.

This repeats an existing Exp4B protocol; it is numerical fidelity validation,
not a new physiological condition or a search candidate.
"""
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/topic4_sef_hfo/liou_original_design_20260915/octave_source_validation'
rng=np.random.RandomState(20260916)
target=OUT/'exp4b_uniforms_f64.bin'
with target.open('wb') as f:
    for _ in range(50000):
        rng.standard_normal(2000)
        rng.rand(2000).astype('<f8').tofile(f)
(OUT/'full_spike_fixture.json').write_text(json.dumps(dict(
    seed=20260916,n=2000,n_steps=50000,dt_ms=1,
    layout='time-major little-endian float64; Gaussian draws consumed before uniforms at every step',
    scope='Numerical QA of existing Exp4B run; original author update method and spatial convolution',
    bytes=target.stat().st_size),indent=2)+'\n')
