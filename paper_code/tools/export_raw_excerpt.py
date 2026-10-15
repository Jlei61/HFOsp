"""Export unreferenced native EDF samples; no event detection or filtering.

python tools/export_raw_excerpt.py record.edf --config config/raw_example.json \
    --start-sec 0 --duration-sec 200 --output ../paper_inputs/computational/example.npz
Only contacts required by bipolar pairs are read when --config is provided.
"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from data_processing.raw_to_events import read_samples
from workflows.common import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input');p.add_argument('--output',required=True)
    p.add_argument('--config');p.add_argument('--start-sec',type=float,default=0);p.add_argument('--duration-sec',type=float,default=200)
    a=p.parse_args();channels=None
    if a.config:
        config=json.loads(Path(a.config).read_text());pairs=config['preprocessing'].get('bipolar_pairs')
        if config['preprocessing']['reference']=='bipolar' and pairs:channels=sorted({n for pair in pairs.values() for n in pair})
    target=Path(a.output)
    if target.exists():raise FileExistsError(target)
    x,fs,names=read_samples(a.input,start_sec=a.start_sec,duration_sec=a.duration_sec,channels=channels)
    target.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target,samples_V=x,sfreq=fs,channel_names=names)
    write_json(target.with_suffix('.json'),dict(source=str(Path(a.input).resolve()),source_start_sec=a.start_sec,
        duration_sec=x.shape[1]/fs,sfreq=fs,channel_names=names,unit='V',processing='none'))
    print(target)

if __name__=='__main__':main()
