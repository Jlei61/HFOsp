"""Full delay-history Floquet validation of right-branch periodic solutions."""
from common import *
import floquet, argparse, numpy as np
floquet.OUT=OUT

def main():
    ap=argparse.ArgumentParser();ap.add_argument('paths',nargs='+');ap.add_argument('--dt',type=float,default=.1);a=ap.parse_args()
    for path in a.paths:
        path=Path(path).resolve();z=np.load(path);g=float(z['g'])
        if path.is_relative_to(OUT/'periodic'):
            relative=path.parent.relative_to(OUT/'periodic');target=OUT/relative
        else:
            target=path.parent/'stability'/path.stem
        target.mkdir(parents=True,exist_ok=True);floquet.OUT=target
        dest=target/'floquet'/f'g{g:.8f}_dt{a.dt:g}.json'
        row=read(dest) if dest.exists() else floquet.compute(path,a.dt)
        mm=np.array([complex(*v) for v in row['multipliers']]);i=np.argmin(abs(mm-1))
        row.update(phase_error=float(abs(mm[i]-1)),maximum_transverse_modulus=float(max(abs(np.delete(mm,i)))))
        row['classification']='stable' if row['phase_error']<.005 and row['maximum_transverse_modulus']<1 else ('unstable' if row['phase_error']<.005 else 'refinement_needed')
        dest.write_text(json.dumps(row,indent=2)+'\n');print('STABILITY',json.dumps(row),flush=True)

if __name__=='__main__':main()
