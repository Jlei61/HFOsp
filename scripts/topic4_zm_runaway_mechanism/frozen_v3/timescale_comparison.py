"""Stage B3: compare the critical-mode growth/recovery times of the conditional objects with the actual
Z/M evolution time along the skeleton's entry trajectory.
- skeleton: D(t) from A4_det_meandrive; dD/dt over 1-s windows; time to traverse the D interval between the
  entry D and the nearest conditional critical point (LPC D)
- equilibria: leading root real parts along the sampled branches -> e-fold times 1/Re(lambda)
- cycles: Floquet leading nontrivial multiplier -> growth/decay time T/ln|mu|
"""
from common_v3 import *
import glob
def main():
    z=np.load(DEST/'runs/A4_det_meandrive/trajectory.npz');D=z['D'];t=np.arange(len(D))*10.;res=read(DEST/'runs/A4_det_meandrive/result.json');entry=res['high_onset_ms']
    dDdt=np.gradient(D,t/1000.);rows=[]
    for a in range(0,int(t[-1]),1000):
        sl=(t>=a)&(t<a+1000);rows.append(dict(window_s=[a/1000,(a+1000)/1000],D_start=float(D[sl][0]),dD_dt_per_s=float(dDdt[sl].mean())))
    Dentry=float(D[min(len(D)-1,int(entry/10))])
    lpc=[read(f) for f in glob.glob(str(DEST/'periodic/LPC*.json'))];lpcD=[q['D'] for q in lpc]
    eq=[];fl=[]
    for f in glob.glob(str(DEST/'equilibrium_stability/*_sampled.json')):
        for r in read(f)['rows']:
            for q in r['leading_roots'][:1]:eq.append(dict(branch=Path(f).stem,D=r['D'],rate_hz=r['global_E_hz'],re=q['lambda_per_ms'][0],efold_ms=(1/q['lambda_per_ms'][0] if q['lambda_per_ms'][0]!=0 else None),freq_hz=q['frequency_hz']))
    for f in glob.glob(str(DEST/'periodic/floquet/*.json')):
        q=read(f);mu=q['max_other_modulus'];T=q.get('period_ms') or None
        orb=np.load(q['orbit']);T=float(orb['T']);fl.append(dict(orbit=Path(q['orbit']).stem,D=float(orb['D']),T_ms=T,leading_multiplier=mu,growth_time_ms=(T/np.log(mu) if mu and mu>1 else None),decay_time_ms=(-T/np.log(mu) if mu and mu<1 else None),phase_defect=q['phase_defect'],stability=q['stability']))
    out=dict(skeleton=dict(entry_ms=entry,D_at_entry=Dentry,dD_dt_windows=rows,mean_dD_dt_6_to_8s=float(np.mean([r['dD_dt_per_s'] for r in rows if 6<=r['window_s'][0]<8]))),
             LPC_D=lpcD,D_gap_entry_to_LPC=[d-Dentry for d in lpcD],time_to_reach_LPC_at_current_rate_s=[(d-Dentry)/max(np.mean([r['dD_dt_per_s'] for r in rows if 6<=r['window_s'][0]<8]),1e-9) for d in lpcD],
             equilibrium_leading_roots=eq,cycle_floquet=fl)
    write(DEST/'stage_b/timescales.json',out);print(json.dumps(clean(out),indent=1)[:3000])
if __name__=='__main__':main()
