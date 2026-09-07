import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from recover_fig5_two_gif_bases import recruitment_start


def test_pre_probe_precedes_first_core_not_delayed_global_recruitment():
    rates=np.zeros((300,4))
    rates[20:25,1]=400  # An earlier returned interictal burst is not onset.
    for col,index in enumerate([190,150,160,200]):rates[index:,col]=400
    onset,starts=recruitment_start(dict(time_ms=np.arange(300)*20+10,rates_hz=rates))
    assert starts==[3800.,3000.,3200.,4000.]
    assert onset==3000.
    assert onset-250+200<onset


def test_terminal_transient_cannot_define_pretransition():
    rates=np.zeros((300,4));rates[-20:]=400
    with pytest.raises(ValueError):
        recruitment_start(dict(time_ms=np.arange(300)*20+10,rates_hz=rates))
