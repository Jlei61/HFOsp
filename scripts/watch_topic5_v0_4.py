#!/usr/bin/env python3
"""Progress watcher for the v0.4 run."""
import json, subprocess, time, sys
from pathlib import Path
ROOT = Path("/home/honglab/leijiaxin/HFOsp/.worktrees/topic5-structured-rnn-fig6")
OUT = ROOT / "results/topic5_interictal_ictal_shared_axis_rnn_v0_4"
EXPECTED = 204
def gpu():
    try:
        q = subprocess.run(["nvidia-smi","--query-gpu=memory.used,utilization.gpu",
                            "--format=csv,noheader,nounits"],capture_output=True,text=True).stdout.strip()
        m,u = q.split(", "); return float(m), float(u)
    except Exception: return None, None
start = time.time(); first_done = None
while True:
    done = list((OUT/"per_subject").glob("*/*/seed_*/DONE.json")) if (OUT/"per_subject").is_dir() else []
    failed = list((OUT/"per_subject").glob("*/*/seed_*/FAILED.json")) if (OUT/"per_subject").is_dir() else []
    n = len(done)
    if n and first_done is None: first_done = time.time()
    rate = n/((time.time()-first_done)/3600) if first_done and time.time()>first_done else 0.0
    mem, util = gpu()
    running = int(subprocess.run("ps -eo cmd --no-headers | grep -c '[r]un_topic5_shared_axis_rnn_unit_v0_4'",
                                 shell=True,capture_output=True,text=True).stdout.strip() or 0)
    payload = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "stage": "formal_training", "completed": n, "expected": EXPECTED,
               "failed": len(failed), "running_workers": running,
               "units_per_hour": round(rate,1),
               "eta_hours": round((EXPECTED-n)/rate,2) if rate>0 else None,
               "gpu_memory_mb": mem, "gpu_util_pct": util,
               "elapsed_hours": round((time.time()-start)/3600,2)}
    (OUT/"PROGRESS.json").write_text(json.dumps(payload, indent=2)+"\n")
    if n >= EXPECTED or (OUT/"TRAIN_COMPLETE.json").exists(): break
    time.sleep(300)
