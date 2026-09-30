"""Audit the Z-I covariance discarded within the existing spatial groups.

Algebraic native snapshots only. Small global mean error cannot validate the
dynamic closure or exclude sensitivity to a localized near-threshold error.
"""
from common import *


def main():
    s=model();idx=s.members;ne=len(idx);assert ne==32000
    counts=np.bincount(idx,minlength=s.P)
    assert np.array_equal(counts[s.E],s.sizes[s.E]) and counts[~s.E].sum()==0
    folder=ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/runs/eta0.0005_s9108401/checkpoints'
    rows=[]
    def mean(x):return np.bincount(idx,weights=x,minlength=s.P)/np.maximum(counts,1)
    for t in [8000,9000,9420,9870,10370,12500]:
        path=folder/f't{t}ms.npz';z=np.load(path)
        Z=z['slow__z'][:ne];I=z['I_I'][:ne]
        actual=mean(Z*I);product=mean(Z)*mean(I);delta=product-actual
        assert abs(actual[s.E]@s.mean_weights-(Z*I).mean())<1e-10
        assert abs(mean(Z)[s.E]@s.mean_weights-Z.mean())<1e-12
        relative=float(delta[s.E]@s.mean_weights/(Z*I).mean())
        rows.append(dict(source=str(path),time_ms=t,global_Z=float(Z.mean()),
            true_mean_ZI_mV_equivalent=float((Z*I).mean()),
            factorized_mean_error_mV_equivalent=float(delta[s.E]@s.mean_weights),
            factorized_global_relative_error=relative,
            E_weighted_mean_absolute_group_error_mV_equivalent=float(abs(delta[s.E])@s.mean_weights),
            largest_absolute_group_error_mV_equivalent=float(abs(delta[s.E]).max())))
    q=dict(status='DESCRIPTIVE_COMPLETE',rows=rows,
        observable='group_mean(Z)*group_mean(I_I) minus group_mean(Z*I_I), weighted by original E-cell counts',
        model='Existing20x20/935groups; grouping and raw native currents unchanged',
        source_times='Original six available checkpoints, not independent replicates or time-window means',
        limits=['Isolates only the product-of-means approximation at actual native states.',
                'Does not audit Gaussian Z-target closure, dynamic transfer response or recurrent current prediction.',
                'Small global mean error does not rule out sensitivity of a local critical mode.'])
    write(OUT/'native_Z_current_projection_audit.json',q);log('NATIVE Z CURRENT PROJECTION',q)


if __name__=='__main__':main()
