# FCXR-LC2 closed-loop exploration status

- **Status**: `COMPLETE_BOUNDED_NEGATIVE`
- **Canonical verdict**: `H_BOUNDED_HIGH_POSITIVE_ONSET_OFFSET_CONTROL_NEGATIVE`
- **Stage reached**: E4 frozen H/X forks; E5 dynamic Z/H/X was not unlocked.

## Plain-language result

R1 sensor recharacterization completed, and the full 90-cell upper-bound H screen completed. The screen yielded 52 developmental survivors and 38 saturated-tonic cells, but this high-initial-condition screen is not a basin test. In the canonical matched forks, both finalists self-escalated from low H on the healthy D=0 substrate. The susceptible low/high starts therefore did not establish a selective bistable window. The tested H loop therefore establishes a finite bounded high state, but Z has no selective onset control. Both accepted frozen X-load levels reduce its amplitude, yet neither returns it to the interictal workpoint, so X has no offset state-transition authority at those loads.

## Claim boundary

The formal claim is: bounded high-state generation positive; susceptibility-selective onset and X-controlled offset negative for the tested architecture and loads. This is a bounded negative for the two locked finalists and the tested post-X H architecture. It is not a global impossibility result for H, not a reversal of LC1 X termination authority, and not a dynamic lifecycle result. M, K, A and ELR were never introduced.

## Canonical artifacts

- `r1_resegmentation_summary.json` / `r1_sensor_pareto.csv`
- `h_loop_screen.json` (90/90)
- `frozen_fork_map.json` (2 finalists x 6 matched arms)
- `candidate_verdict.json`
- `figures/` and Chinese `figures/README.md`
