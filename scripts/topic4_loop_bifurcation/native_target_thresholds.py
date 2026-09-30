"""Recover exact original per-cell thresholds with the native identity check."""
import numpy as np
from campaign import read, sha
from conditional_density_inputs import OPS
import run_topic4_loop_zk_conditional as native


def load():
    namespace = native.base.old.setup.__globals__
    path = namespace['OUT'] / 'substrate.npz'
    z = dict(np.load(path))
    theta = z['vtheta'].copy()
    expected = read(OPS / 'prepared.json')['graph_identity']['vtheta_sha256']
    assert namespace['array_sha256'](np.asarray(theta, np.float32)) == expected
    geo = dict(np.load(OPS / 'geometry.npz'))
    assert np.array_equal(z['positions_e'], geo['original_positions'][:32000])
    assert theta.shape == (40000,) and np.isfinite(theta).all()
    return theta, dict(path=str(path), file_sha256=sha(path), native_vtheta_float32_sha256=expected)
