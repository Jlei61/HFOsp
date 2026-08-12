# Topic 4 rev10-D5.4: fresh KMeans selection

## Question

D5.3 selected `sigma=0.075/ms, tau=40ms` from 12 continuous spatial-OU cells by a canonical masked-rank KMeans score. Its balanced direction purity was `0.833`, but only two of three fit networks had at least three formal events per direction and its bootstrap interval was wide. D5.4 asks whether that result repeats on untouched selection networks before any final Fig.4 confirmation.

## Frozen arms

Run three arms on seeds `1311-1313`: exact off, the selected local OU, and a matched permutation that preserves the exact OU trajectory and per-update value multiset while destroying neural-sheet adjacency. The Node field, common detector, direction classifier, topology and all edge coefficients remain frozen. No patient held-out data are read.

The local candidate proceeds when at least two networks provide three formal events per direction, balanced canonical KMeans purity exceeds the old D5.2 anchor `0.674`, and the supervised model-patient matrix has positive diagonal and negative crossed cells. Patient matched `q05=0.884` is reported, not imposed as a new blocker. The local-permuted difference is a locality diagnostic and does not retrospectively redefine selection success.

Passing D5.4 authorizes a separate manifest on six untouched confirmation networks. Failing D5.4 closes OU dose/time tuning without opening more cores, `beta`, topology or an optimizer comparison.
