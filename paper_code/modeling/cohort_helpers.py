import re

import numpy as np

GROUPS = ('rank_pattern','local_and_interrod_timing','participation')

DISPLAY = ('mean_rank_error','within_shaft_order_error','participation_error')

def shaft(name): return re.sub(r'\d+$','',str(name))

def shafts(names):
    keys=list(dict.fromkeys(shaft(n) for n in names))
    return {k:np.array([i for i,n in enumerate(names) if shaft(n)==k],int) for k in keys}
