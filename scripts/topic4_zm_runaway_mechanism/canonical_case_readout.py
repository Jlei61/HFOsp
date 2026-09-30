"""Original Fig.5 readouts in a separate import context from the rate engine."""
from canonical_readouts import assess
from pathlib import Path
import sys
import json
import numpy as np

folder=Path(sys.argv[1])
z=np.load(folder if folder.suffix=='.npz' else folder/'trajectory.npz')
print(json.dumps(assess(z['field_E_hz'],z['cell_counts'],folder.stem if folder.suffix=='.npz' else folder.name)))
