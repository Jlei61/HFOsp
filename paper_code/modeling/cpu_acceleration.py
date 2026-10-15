"""Optional compiled ordered scatter; every model/RNG operation stays native.

Numba compiles only np.add.at's sequential ring-buffer additions (no fastmath,
no parallel reassociation). A complete small-network trajectory parity test is
in tools/check_computation.py. If Numba is missing the original engine runs.
"""
import inspect
import numpy as np
from modeling.src.snn_engine.kick_probe_core_input_v2 import simulate_kick


def accelerated_engine():
    try:
        from numba import njit
    except ImportError:
        return simulate_kick

    @njit(cache=False, fastmath=False)
    def scatter(target, index, values):
        rows, cols = index
        for j in range(len(values)):
            target[rows[j], cols[j]] += values[j]

    source=inspect.getsource(simulate_kick)
    if source.count('np.add.at(')!=3:
        raise RuntimeError('Engine scatter sites changed; review acceleration before use')
    namespace=dict(simulate_kick.__globals__)
    namespace['_ordered_scatter']=scatter
    exec(compile(source.replace('np.add.at(', '_ordered_scatter('), __file__, 'exec'),namespace)
    return namespace['simulate_kick']
