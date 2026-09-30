"""Promote only completed paired-step return-map evidence at the same orbit."""
from rate_floquet_poincare import *


def main():
    updated=[]
    targets={}
    for target in (PERIODIC_OUT/'stability_coverage').glob('*.json'):
        old=read(target)
        if old.get('analyzed_orbit'):
            targets.setdefault(Path(old['analyzed_orbit']).resolve(),[]).append(target)
    for f in (PERIODIC_OUT/'poincare_floquet').glob('*_step_check.json'):
        q=read(f)
        if q['status'] not in ['NUMERICALLY_STABLE','UNSTABLE']:continue
        # A survey site retains its original filename after waveform refinement.
        # Match the actual analyzed waveform, never infer identity from J/T or
        # from stripping an accuracy suffix.
        for target in targets.get(Path(q['orbit']).resolve(),[]):
            old=read(target)
            assert Path(old['analyzed_orbit']).resolve()==Path(q['orbit']).resolve()
            if old.get('poincare_step_check')==str(f):continue
            old.setdefault('original_monodromy_classification',
                {k:old.get(k) for k in ['status','margin','reliable_outside_count','floquet_source']})
            old.update(status=q['status'],margin=q['margin'],reliable_outside_count=q['reliable_outside_count'],
                poincare_step_check=str(f),classification_method='Full-state/delay Poincare spectrum, paired integration steps; numerical safety margin')
            write(target,old);updated.append(dict(orbit=q['orbit'],status=q['status'],margin=q['margin']))
    print('UPDATED RETURN-MAP CHECKS',updated,flush=True)


if __name__=='__main__':main()
