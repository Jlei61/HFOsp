"""Group-Event State v0.3.2 measurement / evaluation package (Agent 2).

Everything here answers one paired question on one frozen partition::

    H  vs  H + S_correct  vs  H + S_shifted  vs  H + S_mean

The package never trains the residual state itself; it consumes the frozen
state registry written by the model agent and fits identical readouts on top of
identical explicit history for every arm.  Contract:
``docs/archive/topic5/group_event_state_v0_3_2_measurement_contract_2026-09-02.md``.
"""
