"""Fixed empirical Z strata inside existing spatial/threshold groups.

Membership depends only on the reference resource field, never on output
activity or the tested D. Each node is the exact empirical mean of Z(D) in its
stratum, retaining the original per-neuron resource mean at every D.
"""
import numpy as np


def resource_strata(geometry, reference_z, levels):
    assert isinstance(levels, int) and levels >= 1
    parent = np.asarray(geometry['cell_group'])
    assert parent.shape == reference_z.shape
    population = geometry['population']
    assignment = np.empty(len(parent), dtype=np.int32)
    parents = []
    for g in range(len(population)):
        ids = np.flatnonzero(parent == g)
        assert len(ids) > 0
        # Stable tie handling; quantiles are fixed before any tested output.
        ids = ids[np.argsort(reference_z[ids], kind='stable')]
        n = min(levels, len(ids)) if population[g] == 0 else 1
        for subset in np.array_split(ids, n):
            assignment[subset] = len(parents)
            parents.append(g)
    size = np.bincount(assignment)
    parents = np.asarray(parents, dtype=np.int32)
    assert np.array_equal(parents[assignment], parent)
    assert np.array_equal(np.bincount(parents, weights=size).astype(int), geometry['group_size'])
    return dict(cell_stratum=assignment, parent_group=parents, stratum_size=size)


def quantized_field(z, strata):
    assignment = strata['cell_stratum']
    means = np.bincount(assignment, weights=z) / strata['stratum_size']
    represented = means[assignment]
    assert represented.min() >= 0 and represented.max() <= 1
    assert abs(represented.mean()-z.mean()) < 1e-14
    return represented, means
