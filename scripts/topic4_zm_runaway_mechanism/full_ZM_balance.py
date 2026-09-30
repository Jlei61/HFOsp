"""Read the input balance of already certified autonomous equilibrium roots.

This explains the endpoint with M active; it does not identify the onset
bifurcation, or infer a transient decrease of the actual inhibitory current.
"""
from common import *


def main():
    s=model();folder=OUT/'full_ZM_equilibria';rows=[]
    for name in ['low','high']:
        path=folder/f'{name}.npz';z=np.load(path);y=z['state'];r=z['r']
        s.set_Z(z['Z']);expected=s.equilibrium_state(r)
        assert np.max(abs(y-expected))<1e-9
        assert np.max(abs(s.output(y)-r))<1e-11
        terms=dict(recurrent_AMPA_mv=y[5],raw_GABA_mv=y[7],
                   effective_GABA_mv=y[11]*y[7],M_feedback_mv=y[10],
                   external_mean_mv=s.private_mu,
                   total_mean_mv=y[5]-y[11]*y[7]-y[10]+s.private_mu)
        regions=[]
        for label,mask in [('all_E',s.E),('core_A',s.E&(s.geo['group_region']==0)),
                           ('core_B',s.E&(s.geo['group_region']==1)),
                           ('surround',s.E&(s.geo['group_region']==2))]:
            w=s.sizes[mask];w=w/w.sum()
            regions.append(dict(region=label,Z=float(y[11,mask]@w),
                rate_hz=float(r[mask]@w*1000),
                **{k:float(v[mask]@w) for k,v in terms.items()}))
        rows.append(dict(root=name,source=str(path),regions=regions,
            M_balance_max_error_mv=float(np.max(abs(y[10]-.5*s.E*r)))))
    out=dict(status='COMPLETE',rows=rows,Z='dynamic',M='dynamic',
        stability_evidence=str(folder/'stability.json'),
        scope='Input balance at previously solved full-system equilibria only; '
              'no transient current-budget, onset type, or SNN correspondence inferred.')
    write(folder/'input_balance.json',out);log('FULL ZM EQUILIBRIUM BALANCE',out)


if __name__=='__main__':main()
