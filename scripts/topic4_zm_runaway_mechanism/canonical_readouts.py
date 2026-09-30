"""Apply the original Fig.5 frozen-Z definitions to both rate and SNN fields.

All complete events must be bounded by >=20-ms quiet on both sides. The main
category uses the final 4 s and four 1-s subwindows, exactly as the native audit.
This is additional analysis; existing trajectories and exploratory labels remain.
"""
import sys,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/topic4_fig5_z_state'))
import readouts as R
OUT=ROOT/'results/topic4_sef_hfo/fig5_zm_runaway_mechanism_20260918'


def assess(field,counts,label,t0=0.):
    assert len(field)>=4000 and np.isfinite(field).all()
    n=len(field)//10*10;cell=field[:n].reshape(-1,10,field.shape[1]).mean(1)
    r=cell@(counts/counts.sum());seps,events=R.find_events(r);high=R.high_rate_entry(r)
    end=len(r);start=end-400
    tail=R.window_stats(r,seps,events,cell,counts,start,end,high)
    sub=[R.window_stats(r,seps,events,cell,counts,start+i*100,start+(i+1)*100,high) for i in range(4)]
    cat,reasons,sensitivity=R.classify(tail,sub,high,r[start:])
    ev=[dict(e,start_ms=t0+e['start_bin']*10,end_ms=t0+e['end_bin']*10) for e in events if e['qualifies']]
    return dict(label=label,category=cat,tail=tail,subwindows=sub,events=ev,reasons=reasons,
                sensitivity=sensitivity,high_rate=high,start_ms=t0,duration_ms=n,
                temporal_unit='10-ms non-overlapping bins',claim='Finite-time category, not asymptotic attractor classification')


def main():
    rows=[]
    for p in sorted((OUT/'runs').glob('*/trajectory.npz')):
        if not (p.parent/'result.json').exists():continue
        z=np.load(p);q=assess(z['field_E_hz'],z['cell_counts'],p.parent.name)
        q['source']=str(p);q['D']=json.loads((p.parent/'result.json').read_text())['D_initial']
        rows.append(q)
    native_root=ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1'
    for parent in ['z9420_h8000_W1','z9420_h8000_W2']:
        folders=list(native_root.glob(f'**/runs/{parent}'))
        if not folders:continue
        original=folders[0];geometry=np.load(original.parent.parent/'geometry.npz');counts=geometry['cell_e_counts']
        d=R.load_chunks(original,keys=('field_1ms',))
        field=d['field_1ms']/np.maximum(counts,1)*1000
        rows.append(dict(assess(field,counts,parent,d['start_step']*.1),source=str(original)))
        ext=OUT/'native_validation/runs'/f'{parent}_plus20000ms'
        if (ext/'chunks').exists():
            e=R.load_chunks(ext,keys=('field_1ms',))
            if e['field_1ms'] is not None:
                assert d['end_step']==e['start_step'], 'Native parent/extension discontinuity'
                field=np.concatenate([field,e['field_1ms']/np.maximum(counts,1)*1000])
                q=assess(field,counts,parent+'_extended',d['start_step']*.1)
                q.update(source=[str(original),str(ext)],extension_complete=(ext/'result.json').exists())
                rows.append(q)
    payload=dict(readout_source=str(Path(R.__file__)),
        readout_sha256=hashlib.sha256(Path(R.__file__).read_bytes()).hexdigest(),rows=rows)
    dest=OUT/'canonical_readouts.json';tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(payload,indent=2));tmp.replace(dest)
    for q in rows:
        t=q['tail'];print(q['label'],q['category'],'mean',round(t['mean_rate_hz'],3),
            'events',t['n_events'],'quiet',t['quiet_runs_ge20ms'],'duration',q['duration_ms'],flush=True)


if __name__=='__main__':main()
