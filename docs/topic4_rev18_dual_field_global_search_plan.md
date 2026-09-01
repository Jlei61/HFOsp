# Topic 4 rev18 execution plan

1. Freeze the 52-field manifest at a clean commit and audit exact-anchor parity,
   unique dual mappings, unit directions and absence of observation coordinates.
2. Launch 52 fields by three fit networks with 12 one-thread workers, at least
   40 GiB available-memory reserve and a 600 s controller interval.
3. Aggregate all 156 runs only after the Cartesian product is complete. Compute
   patient-training `J14` and per-network natural KMeans from the same causal
   event units.
4. Nominate the six lowest frozen robust-loss fields. Do not read patient
   held-out, source topology, interventions, figures or ictal data.
5. Run nominees plus the exact anchor on fresh selection networks 2401--2403.
   Freeze one candidate only if its robust loss improves the anchor, natural
   KMeans remains patient-aligned in every network, both modes remain supported,
   and no network shows a large complete-distribution regression.
6. Confirm the frozen candidate without reranking on networks 2411--2413.
7. Only after confirmation open complete held-out event distribution, weakest
   mode, source-topology permutation null and same-checkpoint crossed hotspot
   intervention. Only the intervention can freeze Node and open pathway/ZM work.

The fit screen stops on missing/invalid artifacts, provenance drift, nonzero
EE/E-to-I/ZM coefficients, or resource-floor violation. Scientific
non-improvement produces a valid negative screen and a new field-design
iteration; it is not repaired by relaxing the objective after seeing results.
