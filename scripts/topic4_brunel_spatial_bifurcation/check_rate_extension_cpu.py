"""Independent full-RHS checks at frozen locations on a continued segment."""
from audit_current_rate_model import full_rhs_check, RateField, PER, read, write
from audit_rate_filter_states import filter_state_minima
from pathlib import Path
import argparse
import os
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('label')
    parser.add_argument('--indices', type=int, nargs='+', required=True)
    parser.add_argument('--factor', type=int, default=2)
    parser.add_argument('--output', help='Separate evidence basename for an additional frozen selection')
    args = parser.parse_args()
    assert args.factor >= 2
    metadata = [read(path) for path in sorted((PER / 'orbits').glob(args.label + '_*_N*.json'))
                if '_accuracy_' not in path.stem]
    selected = [(index, Path(metadata[index]['path'])) for index in args.indices]
    destination = PER / ((args.output or args.label + '_independent_CPU_checks') + '.json')
    if destination.exists():
        previous = read(destination)
        assert previous.get('selected_orbits') == [str(path) for _, path in selected]
        rows = previous['rows']
    else:
        rows = []
    model = RateField()

    def record(status):
        write(destination, dict(status=status, pid=os.getpid(), rows=rows,
            selected_orbits=[str(path) for _, path in selected],
            scope='Independent CPU physical-delay and nine-state equation checks at selected frozen continuation points. No whole-segment acceptance, critical-point naming, Floquet or global-connection inference.'))

    for index, path in selected:
        if any(row['path'] == str(path) for row in rows):
            continue
        record('RUNNING')
        row = full_rhs_check(model, path, factor=args.factor)
        profile = np.load(path)
        row['index'] = index
        row['filter_state_check'] = filter_state_minima(model, profile['r'], float(profile['T']))
        row['status'] = ('POINT_CHECK_PASS' if
            row['between_nodes_integrated_error_Hz'] < .001 and
            row['full_RHS_rate_error_Hz_per_ms'] < .001 and
            row['full_RHS_linear_state_max_abs'] < 1e-8 and
            row['filter_state_check']['positive'] else 'POINT_CHECK_UNRESOLVED')
        rows.append(row)
        record('RUNNING')
    record('SELECTED_POINTS_PASS' if all(row['status'] == 'POINT_CHECK_PASS' for row in rows)
           else 'SELECTED_POINTS_UNRESOLVED')


if __name__ == '__main__':
    main()
