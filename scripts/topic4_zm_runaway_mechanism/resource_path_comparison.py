"""Disambiguate regional Z values and the different spatial parameter paths."""
from native_path import *
import csv


def main():
    s=model();rows=[];regions=s.geo['group_region']
    def append(family,point,Z,meaning):
        q=dict(family=family,point=point,global_Z=float(Z[s.E]@s.mean_weights))
        q['D']=1-q['global_Z']
        for i,name in enumerate(['core_A_Z','core_B_Z','surround_Z']):
            mask=s.E&(regions==i);q[name]=float(np.average(Z[mask],weights=s.sizes[mask]))
        q['evidence_scope']=meaning;rows.append(q)
    affine=attach_rate_entry_path(s)
    turn=read(OUT/'periodic/rate_turn_center_G16385_M65536/point0000.json')
    s.set_D(turn['D']);append('rate endpoint affine slice','period-turn candidate',s.Z,
        'Conditional cycle-fold candidate only; upper transverse refinement and critical certification incomplete')
    fine=attach_fine_rate_entry_path(s)
    meanings=['regular self-limited','irregular self-limited','local persistent','local persistent']
    for t,Z,label in zip(fine['times_ms'],fine['fields'],meanings):
        append('actual rate fine Z slice',f'{t} ms field',Z,label+' in same-history 8-s held-Z control; not a located bifurcation')
    native=attach_native_path(s)
    for t,Z in zip(native['times_ms'],native['fields']):
        append('native checkpoint Z slice',f'{t} ms field',Z,'Native SNN spatial Z source; not a certified rate/SNN common bifurcation threshold')
    write(OUT/'resource_path_comparison.json',dict(status='COMPLETE',rows=rows,
        weighting='Original E cell counts, globally and within each region',
        warning='Global Z and core Z are different observables. Do not identify a global native Z~0.77 boundary with the affine rate candidate at global Z~0.855, even though core Z~0.74 there.',
        Z_M_contract='Conditional results hold the full spatial Z field and keep M dynamic'))
    with (OUT/'resource_path_comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    log('RESOURCE PATHS',rows)


if __name__=='__main__':main()
