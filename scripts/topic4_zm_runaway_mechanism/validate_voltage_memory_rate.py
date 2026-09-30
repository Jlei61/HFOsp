"""Reuse unchanged stimulus/score protocol with autonomous voltage memory."""
import validate_reset_memory_rate as protocol
from voltage_memory_rate import DEST, load_models, read, write
from voltage_memory_numerics import evaluate
import argparse


def main(command):
    protocol.DEST=DEST; protocol.VDIR=DEST/'validation'
    protocol.load_models=load_models; protocol.evaluate=evaluate
    if command=='predict':
        protocol.predict()
        path=protocol.VDIR/'predictions_locked.json'; d=read(path)
        d.pop('own_reset_trace',None); d['own_voltage_memory']=True
        d['physical_memory']='Input-weighted refractory clamping and threshold-reset charge, driven by own predicted flux.'
        write(path,d)
    else:
        protocol.score()
        path=protocol.VDIR/'result.json'; d=read(path)
        d['scope']='Fixed original training targets with one input-weighted voltage-memory coordinate. Reused waveforms only; no spatial/onset acceptance. Independent new waveforms and matched DC/AC required on pass.'
        write(path,d)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['predict','score']); main(p.parse_args().command)
