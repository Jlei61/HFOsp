"""Piecewise linear spatial Z path through actual native SNN checkpoint fields.

D is the original-cell-count-weighted mean depletion, not a time coordinate.
The optional endpoint extensions are explicitly distinct from the observed segment.
"""
from common import *


def attach_native_path(s):
    s.path_D_bounds=None
    z=np.load(BASE/'stage_b/z_fields.npz')
    times=[8000,9000,9420,9870,10370]
    observed=np.array([z[f'native_{t}'] for t in times])
    Ds=1-observed[:,s.E]@s.mean_weights
    assert np.all(np.diff(Ds)>0)
    high=np.ones(s.P);high[s.E]=0.
    allD=np.r_[0.,Ds,1.];allZ=np.vstack([np.ones(s.P),observed,high])
    s.path_D_knots=allD.copy();s.path_Z_fields=allZ.copy()
    def set_D(D):
        if not 0<=D<=1:raise ValueError(D)
        j=min(np.searchsorted(allD,D,side='right')-1,len(allD)-2)
        f=(D-allD[j])/(allD[j+1]-allD[j])
        s.set_Z((1-f)*allZ[j]+f*allZ[j+1],source='native-checkpoint piecewise linear spatial Z path')
        assert abs(s.D-D)<2e-14
    s.set_D=set_D
    return dict(times_ms=times,D=Ds,fields=observed,observed_D_interval=[float(Ds[0]),float(Ds[-1])],
                interpolation='linear in per-group Z; original E counts determine D; M remains dynamic')


def attach_rate_entry_path(s):
    """Local autonomous-rate path spanning the observed loss of self-termination.

    Keep to 7.7--7.8 s: the full Z path is not monotonic in its global mean.
    """
    s.path_D_bounds=None
    ck=BASE/'runs/A4_det_meandrive/checkpoints'
    times=[7700,7800];Z=np.array([np.load(ck/f't{t}ms.npz')['Z'] for t in times])
    Ds=1-Z[:,s.E]@s.mean_weights
    s.path_D_knots=Ds.copy();s.path_Z_fields=Z.copy()
    def set_D(D):
        f=(D-Ds[0])/(Ds[1]-Ds[0]);field=(1-f)*Z[0]+f*Z[1]
        if field.min()<0 or field.max()>1+1e-12:raise ValueError('Spatial Z extrapolation leaves the physical domain')
        s.set_Z(np.minimum(field,1.),source='local autonomous-rate 7700--7800 ms spatial Z interpolation')
    s.set_D=set_D
    return dict(times_ms=times,D=Ds,fields=Z,observed_D_interval=Ds.tolist())


def attach_fine_rate_entry_path(s):
    """Observed 10-ms Z fields around actual loss of the regular event sequence.

    This is a separate conditional parameter slice. It must not be spliced into
    the older straight 7700--7800-ms bifurcation branch.
    """
    source=BASE/'runs/A4_det_meandrive/trajectory.npz'
    recorded=np.load(source);times=[7750,7760,7770,7780]
    Z=np.array([recorded['Z'][t//10-1].astype(float) for t in times])
    Ds=1-Z[:,s.E]@s.mean_weights
    assert np.all(np.diff(Ds)>0)
    s.path_D_knots=Ds.copy();s.path_Z_fields=Z.copy()
    def set_D(D):
        # Bounded to the observed monotonic segment; no invented endpoint field.
        if not Ds[0]-1e-12<=D<=Ds[-1]+1e-12:
            raise ValueError('Outside the observed fine Z-path segment')
        j=min(max(np.searchsorted(Ds,D,side='right')-1,0),len(Ds)-2)
        f=(D-Ds[j])/(Ds[j+1]-Ds[j]);field=(1-f)*Z[j]+f*Z[j+1]
        s.set_Z(field,source='actual-rate 7750,7760,7770,7780-ms piecewise spatial Z interpolation')
        assert abs(s.D-D)<2e-14
    s.set_D=set_D
    s.path_D_bounds=(float(Ds[0]),float(Ds[-1]))
    return dict(source=str(source),times_ms=times,D=Ds,fields=Z,
                observed_D_interval=Ds.tolist(),
                interpolation='Piecewise linear between actual 10-ms Z fields; M remains dynamic')


def path_Z_derivative(s,D,h=1e-7):
    """Exact segment slope; right derivative at a knot, left at the last knot.

    Interior knots are nonsmooth parameterization joins. Their one-sided
    Newton column does not qualify such a join as a smooth bifurcation.
    """
    if hasattr(s,'path_D_knots'):
        knots=s.path_D_knots;fields=s.path_Z_fields
        j=min(max(np.searchsorted(knots,D,side='right')-1,0),len(knots)-2)
        return (fields[j+1]-fields[j])/(knots[j+1]-knots[j])
    bounds=getattr(s,'path_D_bounds',None)
    lo=D-h if bounds is None else max(bounds[0],D-h)
    hi=D+h if bounds is None else min(bounds[1],D+h)
    assert hi>lo
    s.set_D(hi);zp=s.Z.copy();s.set_D(lo);zm=s.Z.copy();s.set_D(D)
    return (zp-zm)/(2*h if bounds is None else hi-lo)
