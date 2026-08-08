"""Re-run the fit's best candidate on networks that had no say in choosing it.

This was written into a scratch directory and run from there, so the number it
produced had no committed producer. That is fixed here, along with a caution the
first version got wrong: the change from the fit value to this one is NOT a
clean winner's-curse estimate. It also changes the event count, and a
total-variation distance between histograms is biased upward at small samples,
so selection and sample size move the number together and cannot be separated
by this comparison alone.

The comparison floor must be matched on event count too. The patient's own two
halves sit at 0.031 when both sides use every event, but at 80 events a side --
what a six-network confirmation yields -- the same patient-versus-patient
distance is about 0.25.
"""
import json, os, sys, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join("src","snn_engine"))
from scripts.run_topic4_core_field_stage3_fit import _evaluate, STAGE2, OUT
from scripts.run_topic4_core_field_stage3_profile_round1 import (axial_map, distance,
                                                                 patient_events, signed_monotonicity)
from src.topic4_core_field_profile import split_by_block
from src.topic4_core_field_runner import atomic_write_json, provenance
cfg=json.load(open(f"{STAGE2}/config/stage_config.json"))
ck=json.load(open(f"{OUT}/fit/checkpoint_K3_r0.json"))
AX=axial_map(); v,b=patient_events(AX)
tr,te=split_by_block(b,0.3,20260808); P_tr,P_te=v[tr],v[te]
hist=ck["history"]; best=min(hist,key=lambda r:r["distance"])
FIT_SEEDS=set(int(s) for r in hist for s in r["seeds"])
CONFIRM=[s for s in range(501,540) if s not in FIT_SEEDS][:6]
print(f"拟合用过的网络 {sorted(FIT_SEEDS)}")
print(f"确认用的独立网络 {CONFIRM}\n最优候选建图值 {best['distance']:.3f}（{best['n_events']} 事件）")
res=Pool(6, maxtasksperchild=1).map(_evaluate,[(best["theta"],s,cfg,os.path.join(STAGE2,"network_cache")) for s in CONFIRM])
vals=[x for r in res if "error" not in r for e in r["events"]
      if (x:=signed_monotonicity(e.get("ranks"),AX)) is not None]
out=dict(best_theta=best["theta"], fit_value=best["distance"], fit_n_events=best["n_events"],
         confirm_seeds=CONFIRM, confirm_n_events=len(vals),
         confirm_distance_train=distance(vals,P_tr) if len(vals)>=10 else None,
         confirm_distance_heldout=distance(vals,P_te) if len(vals)>=10 else None,
         n_errors=sum(1 for r in res if "error" in r),
         reference=ck["reference"], provenance=provenance())
atomic_write_json(out, f"{OUT}/fit/confirmation_K3_r0.json")
print(f"\n独立网络确认：{len(vals)} 个事件")
print(f"  对病人训练段 {out['confirm_distance_train']}")
print(f"  对病人留出段 {out['confirm_distance_heldout']}")
print(f"  winner's curse = 确认 - 建图 = {(out['confirm_distance_train'] or 0)-best['distance']:+.3f}")
