from common import *
import csv
rows=[]
eq=read(V2/'fold.json')
rows.append(dict(label='Low-rate equilibrium fold',JEE_core=eq['g'],T_ms='',A_mean_hz=eq['r_hz'][0],B_mean_hz=eq['r_hz'][1],criterion='Equilibrium real characteristic root zero',nearby_native_ids='5 / 6',source=str(V2/'fold.json'),status='NUMERICALLY_REFINED'))
for q in read(OUT/'joint_critical_gallery.json'):
    z=read(Path(q['source']).with_suffix('.json'))
    rows.append(dict(label=q['label'],JEE_core=q['g'],T_ms=q['T_ms'],A_mean_hz=z['mean_hz'][0],B_mean_hz=z['mean_hz'][1],criterion='Nontrivial Floquet multiplier -1' if 'PD' in q['label'] else 'Nontrivial Floquet multiplier +1',nearby_native_ids=q['nearby_native_ids'],source=q['source'],status='NUMERICALLY_REFINED'))
h=read(OUT/'homoclinic_audit.json')
rows.append(dict(label='HC limit estimate',JEE_core=h['fits'][-1]['J_infinite_period'],T_ms='infinite-period limit',A_mean_hz=h['saddle_hz'][0],B_mean_hz=h['saddle_hz'][1],criterion='Finite-period convergence to the same saddle; no infinite-time BVP',nearby_native_ids='19 / 20 (cycle-fold neighborhood)',source=str(OUT/'homoclinic_audit.json'),status='NUMERICAL_LIMIT_EVIDENCE'))
with (OUT/'critical_points.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
write('critical_points.json',rows);print('CRITICAL_TABLE',len(rows),flush=True)
