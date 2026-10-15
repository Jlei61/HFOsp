"""New-run output routing and array-safe reports for computational notebooks."""
from datetime import datetime,timezone
from pathlib import Path
import json
import os
import numpy as np
from paper_paths import new_output,OUTPUT_ROOT


def safe(value):
    if isinstance(value,dict):return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):return [safe(x) for x in value]
    if isinstance(value,(np.integer,np.bool_)):return value.item()
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    if isinstance(value,Path):return str(value)
    return value


def write_json(path,value):
    Path(path).write_text(json.dumps(safe(value),indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def new_session(name):
    return new_output('computational/'+name+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f'))


def publish_session(name,path,report):
    write_json(path/'run_report.json',report)
    pointer=new_output('computational')/('latest_'+name+'.json')
    write_json(pointer,dict(output=str(path),report=str(path/'run_report.json')))


def previous_session(name):
    pointer=OUTPUT_ROOT/'computational'/('latest_'+name+'.json')
    if not pointer.exists():raise FileNotFoundError(f'First execute the preceding {name} notebook; missing {pointer}')
    return Path(json.loads(pointer.read_text())['output'])
