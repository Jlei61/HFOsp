"""Explain frozen-observer exclusions without changing its event definition."""
from rate_periodic import PERIODIC_OUT, read, write
import numpy as np


def main():
    source = PERIODIC_OUT / 'periodic_contact_observations.json'
    data = read(source)
    rows = []
    for row in data['rows']:
        lo, hi = row['window_ms']
        events = [event for event in row['observation']['events']
                  if lo <= np.mean(event['window_ms']) < hi]
        anchors = np.array([np.mean(event['window_ms']) for event in events])
        widths = [np.diff(event['window_ms'])[0] for event in events]
        result = {key: row[key] for key in [
            'case', 'T_ms', 'qualified_events', 'all_detected_events',
            'maximum_unique_contacts_in_observer_window', 'required_unique_contacts',
            'exclusion_counts']}
        assert len(events) == row['all_detected_events']
        result['observer_anchor_intervals_ms'] = np.diff(anchors)
        result['observer_window_widths_ms'] = sorted(set(widths))
        result['events_per_orbit'] = len(events) / 8
        period = row['T_ms']
        if row['maximum_unique_contacts_in_observer_window'] < row['required_unique_contacts']:
            result.update(status='BELOW_FROZEN_GROUP_CONTACT_REQUIREMENT',
                meaning='No qualified group; this does not imply zero field or contact activity.')
        elif (period is not None and len(events) == 8 and
              0 < row['qualified_events'] < 8 and widths and
              period < min(widths) and
              set(row['exclusion_counts']) == {'overlapping_window'}):
            result.update(status='PERIODIC_WINDOW_OVERLAP_BIN_PHASE_SENSITIVE',
                continuous_anchor_overlap_ms=min(widths) - period,
                meaning='One event per exact orbit. Continuous equally spaced windows overlap every cycle; bin-quantized anchors sometimes make windows touch instead. Selected fractions cannot be interpreted as irregular event generation.')
        else:
            result.update(status='FROZEN_OBSERVER_APPLIED',
                meaning='Repeated-orbit descriptors, not independent event samples or native-SNN equivalence.')
        rows.append(result)
    output = dict(source=str(source), observer=data['observer'], rows=rows,
        equations_changed=False, observer_changed=False,
        scope='Interpretation of existing exact-orbit readouts. Keeps the frozen observer results; does not recalibrate thresholds or infer absent propagation from excluded events.')
    write(PERIODIC_OUT / 'periodic_observer_exclusion_audit.json', output)
    for row in rows:
        print(row['case'], row['status'], row['qualified_events'], '/', row['all_detected_events'])


if __name__ == '__main__':
    main()
