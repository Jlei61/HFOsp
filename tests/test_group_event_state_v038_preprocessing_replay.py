from types import SimpleNamespace

import numpy as np

from src.topic5_group_event_state.v037.h1_data import _rich_cache_signature, _residualise_grammar


def test_rich_cache_binds_dictionary_and_selected_waveforms_only():
    seq = SimpleNamespace(index={'band_available': [True]}, arrays={
        name: np.zeros((8, 2, 3), dtype=np.float32) for name in ('band_features', 'cross_band_lag', 'waveform')})
    dictionary = SimpleNamespace(community_of_contact=np.array([0, 1]),
        event_repertoire_embedding=np.zeros((4, 2)), event_repertoire_label=np.zeros(4), repertoire_centres=np.zeros((2, 2)))
    args = (seq, np.arange(4), np.ones((4, 2), dtype=bool), np.zeros((4, 2)), dictionary, np.arange(2))
    first = _rich_cache_signature(*args)
    seq.arrays['waveform'][7] = 99
    assert _rich_cache_signature(*args) == first
    seq.arrays['waveform'][1] = 1
    assert _rich_cache_signature(*args) != first
    seq.arrays['waveform'][1] = 0
    dictionary.community_of_contact[:] = [1, 0]
    assert _rich_cache_signature(*args) != first


def test_saved_residual_pca_operator_replays_without_refitting_and_is_fit_only():
    rng = np.random.default_rng(5)
    burden = rng.normal(size=(30, 3)); grammar = rng.normal(size=(30, 8)); fit = np.arange(20)
    compressed, audit = _residualise_grammar(grammar, burden, fit, n_components=4, seed=2)
    op = audit['fitted_operator']; x = np.column_stack((np.ones(30), burden))
    residual = grammar - x @ np.asarray(op['ridge_coefficient'])
    standard = np.clip((residual - op['residual_centre']) / op['residual_scale'], -12, 12).astype(np.float32)
    replay = (standard - np.asarray(op['pca_mean'], dtype=np.float32)) @ np.asarray(op['pca_components'], dtype=np.float32).T
    np.testing.assert_allclose(replay, compressed, rtol=1e-6, atol=1e-6)
    grammar[20:] += 1000
    changed, _ = _residualise_grammar(grammar, burden, fit, n_components=4, seed=2)
    np.testing.assert_array_equal(changed[fit], compressed[fit])
