"""Resume a completed, finite arc batch into a distinct output directory."""
from common import *
from arcs import continue_arc
import time,argparse
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--parent',required=True);ap.add_argument('--name',required=True);ap.add_argument('--after',type=int,default=70);ap.add_argument('--steps',type=int,default=100);ap.add_argument('--ds',type=float,default=.4);ap.add_argument('--dsmax',type=float,default=1.2);a=ap.parse_args()
    while True:
        path=OUT/'arcs'/a.parent/'progress.json'
        rows=read(path) if path.exists() else []
        if len(rows)>=a.after:break
        time.sleep(5)
    continue_arc(rows[-2]['source'],rows[-1]['source'],a.name,a.steps,a.ds,2048,.01,a.dsmax)
