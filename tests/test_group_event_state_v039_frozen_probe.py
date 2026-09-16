import importlib.util
from pathlib import Path
import numpy as np


def test_cross_task_probe_preserves_opposite_spatial_modulations():
    spec=importlib.util.spec_from_file_location('probe',Path(__file__).resolve().parents[1]/'scripts/probe_group_event_state_v039_instrument.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    x=np.tile(np.linspace(-1,1,30),4)[:,None]
    y=np.c_[.5+.4*x[:,0],.5-.4*x[:,0]]
    assert np.ptp(y.mean(-1))<1e-12
    state=module.ridge(x,y,60,90)
    constant=module.ridge(np.zeros_like(x),y,60,90)
    assert state['held_out_squared_error'].shape==(30,)
    assert constant['held_out_squared_error'].mean()>100*state['held_out_squared_error'].mean()
