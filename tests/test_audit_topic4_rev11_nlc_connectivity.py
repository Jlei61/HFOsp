import numpy as np
from scipy import sparse

from scripts.audit_topic4_rev11_nlc_connectivity import audit_network


def test_audit_distinguishes_parameter_suffix_from_biological_direction():
    positions = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
    ])
    ampa = sparse.csc_matrix(np.array([
        [0.0, 1.0], [1.0, 0.0],
        [1.0, 1.0], [1.0, 1.0],
    ]))
    gaba = sparse.csc_matrix(np.array([
        [1.0, 0.0], [0.0, 1.0],
        [0.0, 1.0], [1.0, 0.0],
    ]))
    payload = {
        "net": {
            "pos": positions, "NE": 2, "NI": 2,
            "ampa_by_delay": [ampa], "gaba_by_delay": [gaba],
        },
        "config": {"theta_EE_deg": 0.0, "C_EE": 1, "C_IE": 2,
                   "C_EI": 1, "C_II": 1},
    }
    result = audit_network(payload, sample_stride=1)
    assert result["pathways"]["E_to_E"]["n_edges"] == 2
    assert result["pathways"]["E_to_I"]["n_edges"] == 4
    assert "Params suffix IE" in result["pathways"]["E_to_I"]["matrix_contract"]
    assert "Params suffix EI" in result["pathways"]["I_to_E"]["matrix_contract"]
    assert not result["scientific_interpretation"]["EE_long_axis_is_population_label"]
