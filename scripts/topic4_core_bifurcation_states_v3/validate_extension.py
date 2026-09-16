"""Independent full-history Floquet checks for the added periodic points."""
from extend import OUT, SOURCE
import floquet
import json
import numpy as np

floquet.OUT = OUT

def main():
    rows = []
    for p in sorted((OUT / 'periodic').glob('g*_N2048.npz')):
        g = float(np.load(p)['g'])
        f = OUT / 'floquet' / f'g{g:.8f}_dt0.1.json'
        row = json.loads(f.read_text()) if f.exists() else floquet.compute(p, .1)
        mm = np.array([complex(*x) for x in row['multipliers']])
        neutral = int(np.argmin(abs(mm-1)))
        phase_error = float(abs(mm[neutral]-1))
        transverse = float(max(abs(np.delete(mm, neutral))))
        row.update(phase_error=phase_error, maximum_transverse_modulus=transverse)
        if phase_error >= .005 or transverse >= 1:
            raise RuntimeError(f'Unverified stable periodic point J={g}: {row}')
        rows.append(row)
    # Refine a new endpoint independently of the v2 convergence test.
    p = OUT / 'periodic/g1.17500000_N2048.npz'
    if p.exists():
        for dt in (.05, .025):
            f = OUT / 'floquet' / f'g1.17500000_dt{dt:g}.json'
            if not f.exists():
                floquet.compute(p, dt)
    (OUT / 'extension_stability.json').write_text(json.dumps(rows, indent=2)+'\n')

if __name__ == '__main__':
    main()
