"""Parallelize independent frequencies; retain the original summation order."""
import numpy as np
from numba import njit,prange,set_num_threads


@njit(cache=True,parallel=True)
def frequency_matrix(rows,cols,weights,n,delays,omega):
    out=np.zeros((len(omega),n,n),np.complex128);derivative=np.zeros_like(out)
    for k in prange(len(omega)):
        phase=np.exp(-1j*omega[k]*delays)
        for p in range(len(weights)):
            d=cols[p]//n;value=weights[p]*phase[d]
            out[k,rows[p],cols[p]%n]+=value
            derivative[k,rows[p],cols[p]%n]+=value*(1j*omega[k]*delays[d])
    return out,derivative


def install(threads=4):
    import topic4_fig5_z_frozen_v1
    set_num_threads(threads);topic4_fig5_z_frozen_v1.frequency_matrix=frequency_matrix
