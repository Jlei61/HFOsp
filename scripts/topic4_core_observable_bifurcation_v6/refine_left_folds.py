"""Double-grid validation of the two extremely close low-surround folds."""
from common import *
from folds import refine
import argparse
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--which',type=int,choices=[1,2],required=True);a=ap.parse_args()
    old=read(V5/'folds'/('surround_first_fold_N2048.json' if a.which==1 else 'surround_second_fold_N2048.json'))
    refine(old['left_source'],old['right_source'],'surround_first_fold' if a.which==1 else 'surround_second_fold',4096)
