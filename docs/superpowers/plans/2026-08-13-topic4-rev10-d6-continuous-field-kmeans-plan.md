# Topic 4 rev10-D6 execution plan

1. Freeze 49 fields before reading seeds 1331-1332: one warm baseline plus 12
   whole-sheet sin/cos directions, two amplitudes, and both signs.
2. Unit-test field construction, exact edge no-op, provenance, and the
   recruitment-aware KMeans score.
3. Commit code/config, then freeze the manifest from that clean commit.
4. Launch through `systemd-run --user -> nohup`; measure one sentinel RSS,
   limit each worker to one numerical thread and 24 GiB, and choose parallelism
   from current memory headroom with a maximum of nine workers.
5. Aggregate only returned events and audit candidates with equal network
   weight. Report KMeans purity, signed patient matrix, A/B recruitment,
   OOD, occupancy, and runaway status.
6. If a nonbaseline direction is descriptively better, freeze a small
   combination/refinement library before seeds 1341-1343. Otherwise stop the
   local field-sensitivity branch and reconsider the mechanism family.
7. Do not render a replacement Fig.4 from fit seeds. Only a later untouched
   confirmation may generate the direct-readout and KMeans acceptance pair.
