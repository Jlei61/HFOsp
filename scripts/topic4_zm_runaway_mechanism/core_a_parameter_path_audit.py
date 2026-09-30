"""Audit the mean-Z first-crossing coordinate against the native field path.

Native resources can recover between events. Their regional mean is then
not a one-to-one path parameter. Preserve fixed-field results, but do not
silently connect roots across field jumps in a first-upcrossing envelope.
"""
from common import OUT,np,model,read,write,log
from core_a_equilibrium_branch import Family


class NativeTimeFamily(Family):
    """Continuous piecewise-linear native Core-A field, time indexes the path.

    Time here is a continuation parameter indexing a frozen field. Each
    physical orbit still holds every Z fixed and evolves every E M.
    D_A is an observable coordinate, not presumed globally monotone.
    """
    def field_at_time(self,tm):
        assert self.time[0]<=tm<=self.time[-1]
        j=min(np.searchsorted(self.time,tm,side='right')-1,len(self.time)-2)
        a=(tm-self.time[j])/(self.time[j+1]-self.time[j])
        z=self.background.copy();z[self.A]=(1-a)*self.fields[j,self.A]+a*self.fields[j+1,self.A]
        return z,float(1-np.average(z[self.A],weights=self.s.sizes[self.A]))


def main():
    s=model(40);f=NativeTimeFamily(s);w=s.sizes[f.A]/s.sizes[f.A].sum()
    dest=OUT/'core_a_bifurcation_type_20260924/reference_stability_gap'
    start=int(np.flatnonzero(f.time==9000)[0]);stop=int(np.flatnonzero(f.time==10370)[0])
    # Record maxima followed by a genuine recovery and later recrossing.
    running=float(f.D[start]);rows=[];i=start
    while i<stop-1:
        if f.D[i]>=running-1e-14:running=max(running,float(f.D[i]))
        if f.D[i]>=running-1e-14 and f.D[i+1]<f.D[i]-1e-12:
            target=float(f.D[i]);candidates=np.flatnonzero((f.D[i+1:stop]>=target))
            if not len(candidates):break
            k=i+1+int(candidates[0]);a=(target-f.D[k-1])/(f.D[k]-f.D[k-1])
            recross=(1-a)*f.fields[k-1,f.A]+a*f.fields[k,f.A]
            old=f.fields[i,f.A];delta=recross-old
            rows.append(dict(D_A=target,departure_time_ms=float(f.time[i]),
                reentry_time_ms=float((1-a)*f.time[k-1]+a*f.time[k]),
                largest_mean_recovery=float(target-np.min(f.D[i:k+1])),
                zero_mean_field_jump_max_abs=float(abs(delta).max()),
                zero_mean_field_jump_weighted_rms=float(np.sqrt(delta*delta@w)),
                mean_jump=float(delta@w)))
            i=k;running=max(running,float(f.D[k]));continue
        i+=1
    exact,D=f.field_at_time(9000)
    assert np.max(abs(exact-f.background))<2e-12
    # Verify continuity at every stored native path knot in the relevant
    # interval, using the two neighboring interpolation segments.
    maximum=0.
    for k in range(start+1,stop):
        left=(0.*f.fields[k-1,f.A]+1.*f.fields[k,f.A])
        right=(1.*f.fields[k,f.A]+0.*f.fields[k+1,f.A])
        maximum=max(maximum,float(abs(left-right).max()))
    assert maximum==0.
    result=dict(status='FIRST_CROSSING_MEAN_COORDINATE_HAS_FIELD_JUMPS' if rows else 'NO_JUMP_DETECTED',
        source='transient_native_Z_path_20260923/native_Z_path.npz',rows=rows,
        reference_D_A=D,exact_reference_field_restored=True,continuous_time_path_knot_error=maximum,
        decision='Keep all already computed fixed-field trajectories/roots as point results. Parameterize subsequent continuous continuation by native path time with only A field changing and outside9s held; report and plotD_A as a display coordinate. A turn or jump in displayD alone is not an SN. This changes only the field-path parameterization, not network equations or resources during a physical orbit.',
        limitations='The recorded native path is piecewise linear, continuous but with derivative corners at sample times. Critical derivative/nondegeneracy checks must identify their native time segment; a corner alone is not a dynamical bifurcation.',model_promoted=False)
    write(dest/'parameter_path_audit.json',result);log('CORE A PATH AUDIT',result)


if __name__=='__main__':main()
