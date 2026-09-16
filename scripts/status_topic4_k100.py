#!/usr/bin/env python3
"""Compact live status table for the k100 matrix batch (read-only)."""
import json,os,time,glob
from pathlib import Path
R=Path('/data/hfosp/topic4_sef_hfo/fig5_interictal_recurrence_k100_matrix_20260916')
def main():
    rows=[]
    for d in sorted(R.glob('runs/*/')):
        if d.is_symlink():continue
        name=d.name;chunks=sorted(p for p in (d/'chunks').glob('*.npz') if '.tmp.' not in p.name) if (d/'chunks').exists() else []
        own=[p for p in chunks if os.stat(p).st_nlink==1]  # exclude hard-linked prefix of branches
        t_end=max([int(p.stem.split('_')[-1]) for p in chunks],default=0)*1e-4
        wall=None
        if len(own)>=2:
            ts=sorted(os.path.getmtime(p) for p in own);per=[(ts[i+1]-ts[i]) for i in range(len(ts)-1)]
            sim=(int(own[-1].stem.split('_')[-1])-int(own[0].stem.split('_')[-1]))*1e-4/len(per)
            wall=sorted(per)[len(per)//2]/sim
        live=json.loads((d/'live_status.json').read_text()) if (d/'live_status.json').exists() else {}
        res=(d/'result.json').exists();prog=json.loads((d/'progress.json').read_text()) if (d/'progress.json').exists() else {}
        rows.append((name,t_end,'DONE' if res else prog.get('status','?'),live.get('classification','-'),len(live.get('entries',[])),len(live.get('exits',[])),live.get('preentry_brief','-'),live.get('n_events','-'),live.get('Z'),live.get('gK'),wall))
    print(f"{time.strftime('%H:%M:%S')}  load {os.getloadavg()[0]:.0f}")
    for r in rows:
        z='-' if r[8] is None else f'{r[8]:.3f}';gk='-' if r[9] is None else f'{r[9]:.3f}';w='-' if r[10] is None else f'{r[10]:.0f}'
        print(f"{r[0]:48s} t={r[1]:6.1f}s {r[2]:22s} {r[3]:36s} hi={r[4]} ex={r[5]} brief={r[6]} ev={r[7]} Z={z} gK={gk} wall/s={w}")
if __name__=='__main__':main()
