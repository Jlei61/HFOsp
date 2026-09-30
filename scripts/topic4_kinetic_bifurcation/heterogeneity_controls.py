"""Bounded controls separating finite population size and grouping effects.

The selected spatial communication, physical D and dynamic adaptation law are
unchanged. The two complementary N16 runs isolate individual Z from individual
threshold/M. A complete replay of the previous microscopic N1 trajectory must
be bitwise identical before these new controls are launched.
"""
from compare_density_spatial import *
import subprocess


def run(args):
    folder=OUT/'heterogeneity_controls';folder.mkdir(parents=True,exist_ok=True)
    base=OUT/'particle_controls/selected_g40';prefix='D0.225000_Nscale'
    cases=[(1,7,['--individual-theta','--individual-z','--individual-M']),
           (16,2,['--individual-z']),(16,5,['--individual-theta','--individual-M'])]
    for n,flags,options in cases:
        dest=base/f'{prefix}{n}_seed1901_4000ms_individual_flags{flags}'
        if not (dest/'status.json').exists() or json.load(open(dest/'status.json'))['status']!='COMPLETE':
            command=[sys.executable,str(ROOT/'scripts/topic4_kinetic_bifurcation/particle_control.py'),
                '--D','.225','--scale',str(n),'--seed','1901','--duration','4000','--device',str(args.device),*options]
            subprocess.run(command,check=True)
        if flags==7:
            source=base/f'{prefix}1_seed1901_4000ms_microscopic'
            with np.load(source/'trajectory.npz') as a,np.load(dest/'trajectory.npz') as b:
                checks={k:bool(np.array_equal(a[k],b[k])) for k in a.files}
            (folder/'flag_refactor_replay.json').write_text(json.dumps(dict(pass_bitwise=all(checks.values()),arrays=checks,source=str(source),new=str(dest)),indent=2)+'\n')
            assert all(checks.values()),checks
    density=extract(OUT/'qualification/selected_g40/D0.225000_degree6_dv0.125_4000ms',.225)
    rows=[]
    names=[f'{prefix}1_seed1901_4000ms_microscopic',f'{prefix}4_seed1901_4000ms_microscopic',
        f'{prefix}16_seed1901_4000ms_microscopic',f'{prefix}16_seed1901_4000ms',
        f'{prefix}16_seed1901_4000ms_individual_flags2',f'{prefix}16_seed1901_4000ms_individual_flags5']
    for name in names:
        p=base/name;stats=summarize(p,1000,4000);spatial=extract(p,.225)
        rows.append(dict(name=name,config=json.load(open(p/'config.json')),statistics=stats,
            core_B_minus_A_crossing_ms=spatial['core_B_minus_A_crossing_ms'],spatial_vs_density=compare(density,spatial)))
    report=dict(status='COMPLETE_BOUNDED_DIAGNOSTIC',D=.225,rows=rows,
        statistical_unit='One independent input realization per group-size/heterogeneity condition; events and spatial bins are nested descriptors',
        interpretation='Factorial causal diagnostic at one seed and a finite 1--4s window; not asymptotic dynamical or propagation equivalence acceptance')
    (folder/'result.json').write_text(json.dumps(safe(report),indent=2)+'\n')


if __name__=='__main__':
    import argparse,sys
    ap=argparse.ArgumentParser();ap.add_argument('--device',type=int,default=0);run(ap.parse_args())
