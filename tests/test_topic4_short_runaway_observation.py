import numpy as np
import pytest
from scripts.qualify_topic4_observation_repair import legacy_observation_or_censor
from src.topic4_observation_repaired import observe


@pytest.mark.parametrize('frames',[94,163,180,250])
def test_known_runaway_before_burnin_is_not_an_estimable_fit(frames):
    env=np.zeros((15,frames))
    table,meta=legacy_observation_or_censor(env,2.,{'burnin_ms':500.},frames*2.)
    assert table.shape==(0,15)
    assert meta['status']=='NOT_ESTIMABLE_RUNAWAY_BEFORE_BURNIN'


def test_unexplained_truncation_still_fails():
    with pytest.raises(ValueError,match='without recorded runaway'):
        legacy_observation_or_censor(np.zeros((15,100)),2.,{'burnin_ms':500.},None)
