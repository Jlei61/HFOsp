"""Check primitive periods and near-returns of continued full spatial cycles."""
from plot_rate_periodic_completion import *
from compare_rate_torus_periodic_targets import distances


def main():
    s=RateField();fs=families();weights=s.geo['group_size']/s.geo['group_size'].sum();allrows=[]
    for family,pattern in [('B','arcBsecond_*_N128.json'),('Bleading','arcBleadingExtended_*_N1024.json')]:
        latest=sorted((PERIODIC_OUT/'orbits').glob(pattern))[-1];meta=read(latest);z=np.load(meta['path'])
        N=256 if family=='Bleading' else 128;r=resample(z['r']*1000,N,axis=0)
        scale=np.sqrt(np.mean(np.sum((r-r.mean(0))**2*weights,axis=-1)))
        half=np.sqrt(np.mean(np.sum((r-np.roll(r,N//2,axis=0))**2*weights,axis=-1)))/scale
        candidates=[]
        for name,rr in fs.items():
            for i,q in enumerate(rr):
                if q['path']==meta['path'] or abs(q['J_EE_core']-meta['J_EE_core'])>.001:continue
                if abs(q['T_ms']-meta['T_ms'])>5:continue
                if name==family and i>len(rr)-12:continue
                zz=np.load(q['path']);c=resample(zz['r']*1000,N,axis=0)
                d,shift=distances(r[:,None,:],c,weights)
                candidates.append(dict(family=name,index=i,path=q['path'],J_EE_core=q['J_EE_core'],T_ms=q['T_ms'],
                    rate_profile_RMS_difference_Hz=float(d[0]),relative_waveform_difference=float(d[0]/scale),phase_shift_cycles=float(shift[0])))
        candidates.sort(key=lambda q:q['relative_waveform_difference'])
        row=dict(family=family,source=meta['path'],J_EE_core=meta['J_EE_core'],T_ms=meta['T_ms'],
            half_period_relative_difference=float(half),nearest_candidates=candidates[:12])
        allrows.append(row);print('PERIODIC RETURN AUDIT',family,'half-period error',half,'nearest',candidates[:3],flush=True)
    write(PERIODIC_OUT/'periodic_extension_recurrence_audit.json',dict(rows=allrows,
        scope='Full 935-group waveform comparison after one common phase shift. A near-return at different J/T is not an exact branch connection; common-J BVP correction and branch-tangent checks are needed. Nonzero half-period error excludes a T/2 primitive period at these endpoints.'))


if __name__=='__main__':main()
