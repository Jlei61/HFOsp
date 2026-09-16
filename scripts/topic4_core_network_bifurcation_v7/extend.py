from common import *
import argparse
from arcs import continue_arc
from flip_local import refine,doubled
p=argparse.ArgumentParser();p.add_argument('mode',choices=['low','recruited','flip2','double4']);p.add_argument('--N',type=int,default=4096);a=p.parse_args()
if a.mode in ['low','recruited']:
    parent='surround_unstable' if a.mode=='low' else 'surround_recruited_back'
    rows=read(V5/'arcs'/parent/'progress.json')
    continue_arc(rows[-2]['source'],rows[-1]['source'],a.mode+'_global',180,.1,2048,.01,.5)
elif a.mode=='flip2':
    parent=V6/'periodic/surround_period2'
    refine(parent/'amp0.0002_N4096.npz',parent/'amp0.0004_N4096.npz','surround_2T_flip',a.N,pair=True)
else:
    doubled(OUT/'flips/surround_2T_flip_N4096.npz','surround_period4',[.00001,.00002,.00004,.00008])
