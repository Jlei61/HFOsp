"""Analysis adapter for the actual finite-count model's conditional drift.

ONLY count innovations are suppressed. The private variance split is retained;
this is NOT the separate full-Q mean-field model and not an ensemble mean.
Both objects stay explicitly named and their results must not be combined.
"""
from current_rate_characteristic import CurrentRateCharacteristic, np
from shared_variance_network_sensitivity import split
from scipy import sparse


class CurrentConditionalDrift(CurrentRateCharacteristic):
    def __init__(self, grid=40):
        super().__init__(grid)
        private,qa=split(self,.05)
        self.raw=list(self.raw)
        self.private_operators=private;self.private_split_qa=qa
        for k,kind in enumerate(['ampa','gaba']):
            oldrow,oldcol,_=self.raw[k+2];basekeys=oldrow*self.P+oldcol
            q=private[kind].tocoo();source=q.col%self.P;delay=q.col//self.P
            keys=q.row*self.P+source;indices=np.searchsorted(basekeys,keys)
            assert np.array_equal(basekeys[indices],keys)
            matrix=sparse.coo_matrix((q.data,(indices,delay)),shape=(len(basekeys),len(self.delays))).tocsr()
            self.raw[k+2]=(oldrow,oldcol,matrix)
        self.response_kind='conditioned39 private-Q conditional drift; count innovations suppressed only'
