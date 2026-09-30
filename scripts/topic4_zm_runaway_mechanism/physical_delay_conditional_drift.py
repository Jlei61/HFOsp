"""Conditional drift of the rate model with physical-delay variance repaired.

Full-Q and legacy-private adapters remain separate. M is dynamic; Z is a
prescribed full field for conditional analysis. No native acceptance implied.
"""
from current_rate_characteristic import CurrentRateCharacteristic,np
from physical_delay_variance_split import physical_split
from scipy import sparse


class PhysicalDelayConditionalDrift(CurrentRateCharacteristic):
    def __init__(self,grid=40):
        super().__init__(grid)
        private,qa=physical_split(self);self.raw=list(self.raw)
        self.private_operators=private;self.private_split_qa=qa
        for k,kind in enumerate(['ampa','gaba']):
            row,col,_=self.raw[k+2];basekeys=row*self.P+col
            q=private[kind].tocoo();keys=q.row*self.P+q.col%self.P
            indices=np.searchsorted(basekeys,keys);assert np.array_equal(basekeys[indices],keys)
            matrix=sparse.coo_matrix((q.data,(indices,q.col//self.P)),shape=(len(row),len(self.delays))).tocsr()
            self.raw[k+2]=(row,col,matrix)
        self.response_kind='conditioned39 conditional drift; private-Q uses physical0.1ms delays'
