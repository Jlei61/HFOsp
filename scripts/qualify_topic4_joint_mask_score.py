"""Offline qualification of the joint participation-mask term on real patient FIT masks (design §6).

Checks the reward direction with counterexamples built from the patient's own events; this is a
statistic check, not a propagation acceptance and it needs no simulation.
"""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
from src import topic4_initial_state_runtime as rt
from src import topic4_joint_participation_mask as jm
from scripts.run_topic4_core_connectivity_search import OUT,PARENT

def main():
    parent=rt.read(PARENT);ev=rt.load_evaluator(parent);patient=np.asarray(ev.fit);labels=np.asarray(ev.fit_labels);names=np.asarray(rt.load_observation_contract(parent)['contact_names'])
    masks=np.isfinite(patient);blocks=np.asarray(ev.blocks)[ev.index['FIT']];ref=jm.MaskReference(masks)
    cal=jm.calibrate_scale(masks,blocks,ref,sample_count=16,n_samples=128,seed=20260910);ref.a_mask=cal['a_mask']
    scl=np.char.startswith(names,'SCL');icl=np.char.startswith(names,'ICL');rng=np.random.default_rng(20260910);out=[]
    for n in [16,40,100]:
        rows=[]
        for rep in range(20):
            exact=masks[rng.choice(len(masks),n,replace=False)]
            split=np.concatenate([exact&scl[None],exact&icl[None]])[rng.permutation(2*n)][:n]      # rods in separate events, same n
            shuffled=exact.copy()
            for j in range(masks.shape[1]):shuffled[:,j]=rng.permutation(shuffled[:,j])          # marginals kept, joint destroyed
            allon=np.ones_like(exact)
            scl_off=exact.copy();scl_off[:,scl]=False                                            # whole upper rod missing (the observed model failure)
            shifted=exact.copy();shifted[:,scl]&=rng.random((n,scl.sum()))<.3                    # SCL frequency shift
            collapsed=np.repeat(exact[[np.argmax(exact.sum(1))]],n,axis=0)                       # dispersion shrink
            ta_only=masks[labels==1][rng.choice(int((labels==1).sum()),n,replace=False)]         # mode-proportion shift
            row={k:jm.off_diagonal_mask_distance(v,ref) for k,v in dict(exact=exact,split_rods=split,marginal_shuffle=shuffled,all_on=allon,scl_off=scl_off,scl_shift=shifted,collapsed=collapsed,ta_only=ta_only).items()}
            rows.append(row)
        med={k:float(np.median([r[k] for r in rows])) for k in rows[0]}
        order=[k for k in med if k!='exact' and med[k]>med['exact']]
        out.append(dict(n=n,median_D_mask_off=med,scaled_by_a_mask={k:v/ref.a_mask for k,v in med.items()},all_counterexamples_worse_than_exact=len(order)==len(med)-1,
                        fraction_exact_negative=float(np.mean([r['exact']<0 for r in rows]))))
    record=dict(status='QUALIFIED' if all(o['all_counterexamples_worse_than_exact'] for o in out) else 'FAILED',a_mask=ref.a_mask,calibration=dict(statistic=cal['statistic'],n_samples=cal['n_samples'],
        seed=cal['seed'],q05=cal['q05'],q95=cal['q95'],blocks_per_draw=[len(d['blocks']) for d in cal['draws']]),unique_patient_patterns=int(len(ref.patterns)),patient_events=ref.n_events,
        kernel='mean_h exp(-Hamming/h), h in {1,3,6}',results=out,
        interpretation='Exact patient joint masks score lowest (D_mask_off fluctuates around 0, negative allowed); splitting rods across events, marginal-preserving shuffles, all-on, whole-rod loss, SCL frequency shift, single-pattern collapse and mode-proportion shift all score higher. This validates the reward direction of the statistic only; it is not a propagation acceptance.')
    (OUT/'analysis').mkdir(parents=True,exist_ok=True);rt.write(OUT/'analysis/mask_score_validation.json',record);print(json.dumps(record['results'][0]['scaled_by_a_mask'],indent=1),record['status'])

if __name__=='__main__':main()
