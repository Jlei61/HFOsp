"""Observe ARPACK Ritz diagnostics without changing the iteration or acceptance.

Pointer meanings follow ARPACK-NG SRC/dnaupd.f, IPNTR(6:8).
These unconverged estimates are never used as accepted Floquet evidence.
"""
import importlib
import numpy as np


def observed_eigs(A,observer,**kwargs):
    module=importlib.import_module('scipy.sparse.linalg._eigen.arpack.arpack')
    original=module._UnsymmetricArpackParams
    class Observed(original):
        def iterate(self):
            super().iterate()
            if self.tp not in 'fd' or min(self.ipntr[5:8])<=0:return
            values=[]
            for pointer in self.ipntr[5:8]:
                values.append(self.workl[pointer-1:pointer-1+self.ncv].copy())
            if not np.any(values[0]) and not np.any(values[1]):return
            signature=np.concatenate(values).tobytes()
            if signature==getattr(self,'_last_ritz',None):return
            self._last_ritz=signature
            observer(values[0]+1j*values[1],values[2],int(self.iparam[4]),self.converged)
    module._UnsymmetricArpackParams=Observed
    try:return module.eigs(A,**kwargs)
    finally:module._UnsymmetricArpackParams=original


if __name__=='__main__':
    from scipy.sparse.linalg import eigs
    rng=np.random.default_rng(918)
    A=rng.normal(size=(40,40));v0=rng.normal(size=40)
    expected=eigs(A,k=2,ncv=10,tol=1e-9,v0=v0)
    records=[]
    actual=observed_eigs(A,lambda *q:records.append(q),k=2,ncv=10,tol=1e-9,v0=v0)
    assert np.array_equal(expected[0],actual[0]) and np.array_equal(expected[1],actual[1])
    print('PASS: observer leaves eigenpairs bitwise unchanged;',len(records),'diagnostic updates')
