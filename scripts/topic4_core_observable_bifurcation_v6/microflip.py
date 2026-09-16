"""Resolve the previously hidden flip just after the narrow return fold."""
from common import *
from flip import refine,doubled
import argparse
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--N',type=int,default=2048);ap.add_argument('--double',action='store_true');ap.add_argument('--amplitudes',type=float,nargs='+',default=[.003,.006,.012]);a=ap.parse_args()
    if a.double:doubled(OUT/'flips/surround_micro_flip_N2048.npz','surround_period2',a.amplitudes)
    else:
        rows=read(V4/'arcs/recruitment_turn/progress.json')
        refine(V5/'folds/surround_second_fold_N2048.npz',rows[6]['source'],'surround_micro_flip',a.N,pair=True)
