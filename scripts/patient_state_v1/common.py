from pathlib import Path
import json
import numpy as np

ROOT = Path('/home/honglab/leijiaxin/HFOsp')
OUT = ROOT/'results/topic5_patient_state_inference/e1146_drift_hypothesis_v1'
RUN = OUT/'overnight_20260909'

def write_json(path, obj):
    def clean(x):
        if isinstance(x, dict): return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x, (tuple,list,np.ndarray)): return [clean(v) for v in x]
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.bool_,)): return bool(x)
        if isinstance(x, (float,np.floating)): return float(x) if np.isfinite(x) else None
        return x
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(clean(obj),ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
