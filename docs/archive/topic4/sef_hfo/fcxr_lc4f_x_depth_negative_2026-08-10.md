# FCXR-LC4f X-depth closure — bounded-negative

LC4f transferred the cleanest archived X-depth candidate to the accepted no-kick D/Z entry. The
activity threshold stayed at the frozen LC1 Q99.9 value 76.6386; only `K_y` changed from 5 to 3,
because a late-bout fork had reached X=0.3776 at that value, just below the archived late-bout
reference 0.380. That number is state- and field-conditional, not a universal termination boundary.
The LC4 M actuator was disabled exactly.

The fresh 40k trajectory entered at 11 s after 29 returning events, but did not offset by the 22 s
record end. Population-mean X reached only 0.488 and ended at 0.501; final D/H were 0.540/2.124.
Thus the late-bout depth calibration did not transfer to the natural-entry closed loop. X1 is
`X_DEPTH_OFFSET_NEGATIVE`, so the 70 s lifecycle and exact-D stages were not authorised.

The simulation itself completed and wrote the durable NPZ. Initial JSON serialization failed only
afterward because a false `numpy.bool_` clause was not cast to Python `bool`. The gate prefix already
contained the negative verdict; rate/X/D/H summaries and the 29 pre-onset returning events were
recovered from the stored 10 ms/1 ms traces without rerunning the simulation. The serialization
contract now has a regression test.

Safe interpretation: the locked K-only deepening does not autonomously offset the natural-entry
trajectory within the preregistered 1--5 s target window. The run does not identify an asymptotic X
floor and does not establish that population-wide coverage is the unique blocker: closed-loop dose
self-limitation, the moving D/H state, the X spatial field, and the channel through which X acts remain
coupled. The current X implementation is closed as the lifecycle exit path; activity-dependent relay
depression remains available only for short, explicitly matched diagnostics.
