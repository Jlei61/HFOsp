"""Collect refined native-Z periodic roots without inventing branch edges.

In particular, period is not globally monotone on this branch. Sorting all
independent roots by period would create false connections in a diagram.
"""
from common import *


def main():
    folder=OUT/'periodic';rows=[];edges=[]
    labels=['native_Z219_stable_side_G8505_M65536',
            'native_long_period_bridge_G8505_M65536',
            'native_Z78_refinement_G8193_M65536',
            'native_Z78_second_G8505_M65536',
            'native_unstable_period_extension_G8505_M65536']
    for label in labels:
        source=folder/label/'result.json'
        data=read(source)
        accepted=[q for q in data['rows'] if q['residual']<2e-8]
        for q in accepted:
            row=dict(q,source_summary=str(source),global_Z=1-q['D'],
                     accepted_stability='NOT_ESTABLISHED',stability_evidence=[])
            for f in (OUT/'floquet').glob('*acceptance.json'):
                cert=read(f)
                if cert.get('orbit') and Path(cert['orbit']).resolve()==Path(q['path']).resolve():
                    if cert.get('status') in ['STABLE_WITH_STEP_REFINEMENT','UNSTABLE_WITH_STEP_REFINEMENT']:
                        row['accepted_stability']=cert['status'];row['stability_evidence'].append(str(f))
            rows.append(row)
        # This job follows an explicit sequence starting from the refined T264.286
        # root. Other files are independently refined roots, not accepted bridges.
        if label=='native_unstable_period_extension_G8505_M65536':
            previous=folder/'native_Z78_second_G8505_M65536/point0000.npz'
            for q in accepted:
                edges.append(dict(source=str(previous),target=q['path'],
                    method=q['method'],stability='NOT_ESTABLISHED'))
                previous=Path(q['path'])
    out=dict(status='PARTIAL_BRANCH_EVIDENCE',rows=rows,continuation_edges=edges,
        Z_path='native checkpoint spatial fields, piecewise linear',Z='held',M='dynamic',
        onset_bifurcation='NOT_ESTABLISHED',
        limits=['Independent refined roots are not automatically connected.',
                'Coarse alias-sensitive turns excluded.',
                'Strong nonlinear growth is not a passed Floquet certificate.',
                'This is a conditional rate-model slice, not a new native SNN run.'])
    write(folder/'native_refined_branch_evidence.json',out)
    log('NATIVE REFINED BRANCH',len(rows),'roots',len(edges),'edges')


if __name__=='__main__':main()
