from common import *
from transitions import run
import argparse
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--name',required=True);ap.add_argument('--g',type=float,required=True);ap.add_argument('--dt',type=float,default=.025);a=ap.parse_args();run(a.name,a.source,a.g,a.dt)
