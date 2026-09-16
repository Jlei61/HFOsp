"""Guard scientific denominators and seizure-level randomization."""
import numpy as np
from scripts.analyze_e1146_preseizure_template_share import association, coverage, overlap


def test_background_ta_preference_is_not_label_specific_enrichment():
    result = association([.8,.8,.8,.8], ['A','A','B','B'])
    assert result['delta_ta_share'] == 0
    assert result['label_permutation_p_greater'] == 1
    assert result['mean_matched_share'] == .5


def test_exact_seizure_allocation_not_event_pseudoreplication():
    result = association([.9,.8,.2,.1], ['A','A','B','B'])
    assert result['n'] == 4
    assert result['n_permutations'] == 6
    assert np.isclose(result['label_permutation_p_greater'], 1/6)
    assert np.isclose(result['circular_rotation_p_greater'], .25)


def test_coverage_never_interpolates_missing_time():
    assert coverage([(0,10),(20,30)], 5,25) == 10
    assert not overlap(0,10,10,20)
    assert overlap(0,10,9,20)


def test_one_label_cannot_establish_specificity():
    result = association([.8,.9,.7], ['A','A','A'])
    assert result['status'] == 'not_estimable'
