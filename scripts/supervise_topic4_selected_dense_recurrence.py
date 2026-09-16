#!/usr/bin/env python3
"""Reuse the verified dense replay pipeline for another reviewed R7 candidate."""
import argparse
import time
import supervise_topic4_dense_recurrence as pipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    assert args.name == 'resource_rho0.25_k200_tau10_s9108401'
    pipeline.NAME = args.name
    pipeline.OUT = pipeline.BASE / 'native_field_candidates_recurrence' / args.name
    try:
        pipeline.main()
    except Exception as exc:
        pipeline.write(pipeline.OUT / 'recorder_status.json',
                       dict(status='FAILED_REVIEW', error=repr(exc), updated_at=time.time()))
        raise
