#!/usr/bin/env python3
"""Dense observation of an exact completed R7/R8 revised-Z source trajectory."""
import argparse
import time
from pathlib import Path
import record_topic4_resource_candidate_native_fields as adapter

physical = adapter.physical
observer = adapter.observer


def select_source(root):
    root = root.resolve()
    protocol = physical.carrier.base.read(root / 'protocol.json')
    assert protocol['recovery_producer_sha256'] == physical.carrier.base.sha(physical.__file__)
    assert root.parent == physical.PARENT.resolve()
    physical.OUT = root
    observer.SOURCE = root
    observer.destination = lambda name: physical.PARENT / 'native_field_candidates_recurrence' / name


def prepare(name):
    result = observer.SOURCE / 'runs' / name / 'result.json'
    if not result.exists():
        raise RuntimeError('Source must finish before selecting complete transition/return windows')
    protocol = adapter.prepare(name)
    sha = physical.carrier.base.sha(__file__)
    if 'source_selection_adapter_sha256' in protocol:
        assert protocol['source_selection_adapter_sha256'] == sha
    else:
        protocol.update(source_selection_adapter=str(Path(__file__).resolve()),
            source_selection_adapter_sha256=sha,
            source_selection='Exact completed named source from R7/R8; actual ResourceRecoverySlow retained. Dense contact windows are based on its recorded entry and return times, not on a different candidate.')
        physical.carrier.base.write(observer.destination(name) / 'protocol.json', protocol)
    return protocol


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'worker', 'verify'])
    parser.add_argument('--name', required=True)
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--producer-script')
    args = parser.parse_args()
    if args.producer_script:
        assert Path(args.producer_script).resolve() == Path(physical.carrier.__file__).resolve()
    select_source(args.source_root)
    observer.prepare = prepare
    try:
        if args.mode == 'prepare':
            prepare(args.name)
        elif args.mode == 'worker':
            prepare(args.name)
            observer.worker(args.name)
        else:
            adapter.verify(args.name)
    except Exception as exc:
        physical.carrier.base.write(observer.destination(args.name) / 'observation_failure.json',
                                    dict(error=repr(exc), time=time.time()))
        raise
