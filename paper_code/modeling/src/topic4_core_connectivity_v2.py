"""Core membership and E/I pathway weight scaling."""
import numpy as np

WEIGHT_FACTORS = ('EE_same_core_scale', 'EE_core_to_out_scale', 'EE_out_to_out_scale', 'EI_same_core_scale', 'IE_same_core_scale', 'II_same_core_scale')

def core_index_for(positions, centers, radii):
    """-1 outside every disk; otherwise the nearest containing core (0=A, 1=B)."""
    pos = np.asarray(positions, float)
    centers = np.asarray(centers, float)
    radii = np.asarray(radii, float)
    d = np.linalg.norm(pos[:, None] - centers[None], axis=2)
    inside = d <= radii[None]
    nearest = np.argmin(np.where(inside, d, np.inf), axis=1)
    return np.where(inside.any(1), nearest, -1)

def _block_masks(rows, cols, n_e, e_core, i_core, pathway):
    target_e = rows < n_e
    tc = np.where(target_e, e_core[np.minimum(rows, n_e - 1)], i_core[np.maximum(rows - n_e, 0)])
    sc = e_core[cols] if pathway == 'ampa' else i_core[cols]
    same = (tc >= 0) & (sc == tc)
    if pathway == 'ampa':
        return {'EE_same_core_scale': target_e & same, 'EE_core_to_out_scale': target_e & (tc < 0) & (sc >= 0), 'EE_out_to_out_scale': target_e & (tc < 0) & (sc < 0), 'EI_same_core_scale': ~target_e & same}
    return {'IE_same_core_scale': target_e & same, 'II_same_core_scale': ~target_e & same}

def scale_pathway_blocks(ampa_bins, gaba_bins, n_e, e_core, i_core, factors):
    """C4: multiply weights of the six named blocks; edges, delays and every other block untouched."""
    factors = {k: float(factors.get(k, 1.0)) for k in WEIGHT_FACTORS}
    if any((not np.isfinite(v) or v <= 0 for v in factors.values())):
        raise ValueError('weight factors must be finite and positive')
    e_core = np.asarray(e_core, int)
    i_core = np.asarray(i_core, int)
    audit = dict(factors=factors, blocks={}, exact_noop=all((v == 1.0 for v in factors.values())))
    outputs = []
    for pathway, bins in (('ampa', ampa_bins), ('gaba', gaba_bins)):
        names = [k for k in WEIGHT_FACTORS if k in _block_masks(np.zeros(0, int), np.zeros(0, int), n_e, e_core, i_core, pathway)]
        counts = {k: 0 for k in names}
        before = {k: 0.0 for k in names}
        after = {k: 0.0 for k in names}
        touched = any((factors[k] != 1.0 for k in names))
        new_bins = []
        for m in bins:
            if m.format != 'csc':
                raise ValueError('engine contract requires CSC delay bins')
            if m.nnz == 0:
                new_bins.append(m)
                continue
            coo = m.tocoo(copy=False)
            rows = np.asarray(coo.row, np.int64)
            cols = np.asarray(coo.col, np.int64)
            masks = _block_masks(rows, cols, n_e, e_core, i_core, pathway)
            new = m.copy() if touched else m
            data = new.data if touched else m.data
            for k in names:
                sel = masks[k]
                counts[k] += int(sel.sum())
                before[k] += float(m.data[sel].sum())
                if touched and factors[k] != 1.0:
                    data[sel] *= factors[k]
                after[k] += float(data[sel].sum())
            new_bins.append(new)
        for k in names:
            audit['blocks'][k] = dict(n_edges=counts[k], weight_before=before[k], weight_after=after[k], factor=factors[k])
            if counts[k] and (not np.isclose(after[k], before[k] * factors[k], rtol=1e-09, atol=1e-09)):
                raise RuntimeError(f'block dose audit failed for {k}')
        outputs.append(new_bins)
    return (outputs[0], outputs[1], audit)
