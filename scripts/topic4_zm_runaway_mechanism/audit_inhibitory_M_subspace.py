"""Identify exact stable auxiliary modes outside the physical M_I=0 manifold.

No eigen solve or modification of any running monodromy calculation.
"""
from common import *
from dynamics_v3 import TAU_M


def main():
    s=model();I=~s.E
    path=OUT/'periodic/native_T2644_G16875_M262144/point0000.npz';z=np.load(path)
    s.Z=z['Z'].copy();r=z['r'];T=float(z['T'])
    matrices=s.matrices();rng=np.random.default_rng(920015);rows=[]
    for n in [0,len(r)//4,len(r)//2,3*len(r)//4]:
        y=s.equilibrium_state(r[n]);arrivals=np.array([a@r[n] for a in matrices])
        base=s.rhs(y,arrivals,dynamic_z=False)[0]
        assert np.array_equal(y[10,I],np.zeros(I.sum()))
        assert np.array_equal(base[10,I],np.zeros(I.sum()))
        delta=rng.normal(size=I.sum());errors=[]
        for h in [1e-3,5e-4]:
            plus=y.copy();minus=y.copy();plus[10,I]+=h*delta;minus[10,I]-=h*delta
            fd=(s.rhs(plus,arrivals,dynamic_z=False)[0][10,I]-s.rhs(minus,arrivals,dynamic_z=False)[0][10,I])/(2*h)
            errors.append(float(np.max(abs(fd+delta/TAU_M))))
        assert max(errors)<1e-14
        rows.append(dict(source_rate_index=n,invariant_M_I_zero=True,finite_difference_errors=errors))
    q=dict(status='INHIBITORY_M_INVARIANT_SUBSPACE_AUDIT_PASS',source=str(path),
        groups=s.P,inhibitory_groups=int(I.sum()),excitatory_groups=int(s.E.sum()),
        equation='m_I_dot=-m_I/tau_M, independent of all other states and delay history; physical initial m_I=0.',
        tau_M_ms=float(TAU_M),period_ms=T,
        auxiliary_multiplier=float(np.exp(-T/TAU_M)),multiplicity_at_least=int(I.sum()),
        proof='Ordering physical variables first and m_I last gives a block-upper-triangular monodromy with lower block exp(-T/tau_M) times identity. The physical subspace m_I=0 is invariant. Extra stable eigenvalues do not change physical stability but can cluster in a generic full-space Arnoldi solve.',
        numerical_checks=rows,
        qualification='Matching a Ritz cluster is not proof that every computed Ritz vector belongs to this auxiliary block. No physical multiplier or criticality is inferred; no running job or acceptance gate changed.',
        model_changed=False,onset_type='NOT_ESTABLISHED')
    write(OUT/'inhibitory_M_subspace_audit.json',q);log('INHIBITORY M SUBSPACE',q)


if __name__=='__main__':main()
