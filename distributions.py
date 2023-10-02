import torch
from torch_scatter import scatter_log_softmax, scatter_max


class ScatterCategorical:
    def __init__(self, logits, indices):
        assert logits.dtype == torch.float, "`logits` should be torch.float data type"
        assert indices.dtype == torch.long, "`indices` should be torch.long data type"
        assert (
            logits.shape == indices.shape
        ), f"The shape of `logits` and `indices` should be the same, got logits={logits.shape}, indices={indices.shape}"

        self.indices, mapping = torch.sort(indices, stable=True)
        self.logits = logits[mapping]
        self.size = max(self.indices.tolist(), default=-1) + 1

        count = torch.bincount(self.indices)
        self._offsets = torch.cumsum(count, 0) - count

    def log_prob(self, value):
        assert self.size == len(
            value
        ), f"size doesn't match, size {self.size}, got {value.shape}"
        log_probs = scatter_log_softmax(self.logits, self.indices)
        return log_probs[value + self._offsets]

    @torch.no_grad()
    def sample(self):
        unif = torch.rand_like(self.logits)
        gumbel = -(-unif.log()).log()
        _, samples = scatter_max(self.logits + gumbel, self.indices)
        return samples - self._offsets

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size})"
