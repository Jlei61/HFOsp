"""Branch coverage and critical-point tables (stage C1/C2 deliverables) from the stored results.

Equilibrium branches: start/end (D, rate), D range, number of folds, stability sampling coverage
(fraction of sampled points resolved, classification runs), gaps. Periodic branches: same plus
Floquet coverage and cycle folds. Critical points: SN (static conditions), LPC, Hopf (numerical
criticality), with their verification fields. Writes branch_coverage.json and branch_coverage.md.
"""
from common_v3 import *
import glob
def eq_branch(label):
    p=DEST/'equilibria'/label/'result.json'
    if not p.exists():return None
    info=read(p);rows=info['rows'];D=[r['D'] for r in rows];g=[r['global_E_hz'] for r in rows]
    st=DEST/'equilibrium_stability'/f'{label}_sampled.json';samp=read(st)['rows'] if st.exists() else []
    resolved=[q for q in samp if q['unstable_roots'] is not None]
    runs=[];prev=None
    for q in resolved:
        c='stable' if q['unstable_roots']==0 else f"unstable({q['unstable_roots']})"
        if c!=prev:runs.append(dict(from_index=q['index'],D=q['D'],rate=q['global_E_hz'],classification=c));prev=c
    return dict(label=label,status=info['status'],n_points=len(rows),start=dict(D=D[0],rate_hz=g[0]),end=dict(D=D[-1],rate_hz=g[-1]),D_min=min(D),D_max=max(D),rate_min_hz=min(g),rate_max_hz=max(g),
        n_folds=len(info.get('folds',[])),folds=[dict(D=f['D'],rate_hz=f['global_E_hz'],type=f['static_type'],mode_energy_A_B_S=f['mode_energy_A_B_surround'],transversality=f['transversality']) for f in info.get('folds',[])],
        stability_sampled=len(samp),stability_resolved=len(resolved),classification_runs=runs,
        endpoint=('physical D boundary' if info['status']=='PHYSICAL_DOMAIN_REACHED' else ('continued in another segment' if any((DEST/'equilibria'/f'{label}_cont').exists() for _ in [0]) else info['status'])))
def cycle_branch(label):
    p=DEST/'periodic'/label/'continuation.json'
    if not p.exists():return None
    info=read(p);rows=info['rows'];fl={}
    for f in glob.glob(str(DEST/'periodic/floquet'/f'{label}_*.json')):
        q=read(f);fl[q['orbit']]=q
    D=[r['D'] for r in rows];m=[r['global_mean_hz'] for r in rows]
    return dict(label=label,status=info['status'],n_orbits=len(rows),start=dict(D=D[0],mean_hz=m[0],T_ms=rows[0]['T_ms']),end=dict(D=D[-1],mean_hz=m[-1],T_ms=rows[-1]['T_ms']),D_min=min(D),D_max=max(D),
        brackets=info['brackets'],floquet_orbits=len(fl),floquet=[dict(D=q.get('D'),stability=q['stability'],max_other=q['max_other_modulus'],phase=q['phase_multiplier'],dt=q['dt_ms']) for q in fl.values()])
def main():
    eq=[b for b in (eq_branch(l) for l in ['lower','lower_cont','upper','upper_cont','g40_lower','g40_upper']) if b]
    cy=[b for b in (cycle_branch(l) for l in ['cycleUp','cycleDown']) if b]
    lpc=[read(f) for f in glob.glob(str(DEST/'periodic/LPC*.json'))];hopf=[read(f) for f in glob.glob(str(DEST/'periodic/hopf/*.json'))]
    out=dict(equilibrium_branches=eq,periodic_branches=cy,cycle_folds=lpc,hopf=hopf,note='completeness is limited to the branches relevant to the interictal -> broad-activity path; not all attractors of the high-dimensional delay system')
    write(DEST/'branch_coverage.json',out)
    md=['# 分支覆盖表（自动生成，`branch_coverage_v3.py`）','','## 平衡支','','| 分支 | 状态 | 点数 | 起点 (D, Hz) | 终点 (D, Hz) | D 范围 | 折点数 | 稳定性采样(已判/采样) | 分类段 | 终点性质 |','|---|---|---|---|---|---|---|---|---|---|']
    for b in eq:
        runs='; '.join('%s@D=%.3f'%(r['classification'],r['D']) for r in b['classification_runs'])
        md.append(f"| {b['label']} | {b['status']} | {b['n_points']} | ({b['start']['D']:.4f}, {b['start']['rate_hz']:.2f}) | ({b['end']['D']:.4f}, {b['end']['rate_hz']:.2f}) | [{b['D_min']:.4f}, {b['D_max']:.4f}] | {b['n_folds']} | {b['stability_resolved']}/{b['stability_sampled']} | {runs} | {b['endpoint']} |")
    md+=['','## 周期族','','| 分支 | 状态 | 轨道数 | 起点 (D, 均值 Hz, T ms) | 终点 | D 范围 | 折点括号 | Floquet 轨道数 |','|---|---|---|---|---|---|---|---|']
    for b in cy:md.append(f"| {b['label']} | {b['status']} | {b['n_orbits']} | ({b['start']['D']:.4f}, {b['start']['mean_hz']:.2f}, {b['start']['T_ms']:.1f}) | ({b['end']['D']:.4f}, {b['end']['mean_hz']:.2f}, {b['end']['T_ms']:.1f}) | [{b['D_min']:.4f}, {b['D_max']:.4f}] | {b['brackets']} | {b['floquet_orbits']} |")
    md+=['','## 临界点','','| 类型 | D | 率 (Hz) | 判据 | 备注 |','|---|---|---|---|---|']
    for b in eq:
        for f in b['folds']:md.append(f"| SN ({b['label']}) | {f['D']:.6f} | {f['rate_hz']:.2f} | {f['type']} | 模态能量 A/B/S = {[round(x,2) for x in f['mode_energy_A_B_S']]}, transversality {f['transversality']:.2e} |")
    for q in lpc:md.append(f"| LPC | {q['D']:.9f} | {q.get('global_mean_hz','?')} | {q['check']} | T = {q['T_ms']:.3f} ms, d²D/dc² = {q['d2D_dcoordinate2']:.2e} |")
    for q in hopf:md.append(f"| Hopf | {q['D']:.6f} | {q['global_E_hz']:.2f} | Re λ 穿越，dRe/dD = {q['dRe_dD']:.2e} | f = {q['frequency_hz']:.1f} Hz，数值判型 {q.get('criticality_numerical')} |")
    (DEST/'branch_coverage.md').write_text('\n'.join(md)+'\n');print('\n'.join(md))
if __name__=='__main__':main()
