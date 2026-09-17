from common import *
from folds import refine
import argparse
p=argparse.ArgumentParser();p.add_argument('family');p.add_argument('--N',type=int,default=2048);a=p.parse_args()
rows=read(OUT/'arcs'/(a.family+'_fast')/'progress.json')
for idx,(left,right) in enumerate(zip(rows[:-1],rows[1:])):
 if left['tangent_g']*right['tangent_g']<0:
  name=f'{a.family}_global_fold_{idx:03d}'
  if not (OUT/'folds'/f'{name}_N{a.N}.json').exists():refine(left['source'],right['source'],name,a.N)
