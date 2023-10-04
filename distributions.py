import torch
from torch_scatter import scatter_log_softmax, scatter_max
from mol_env import idx_to_Action, Action_to_idx


class ActionCategorical:
    def __init__(self, states, logits, indices):
        assert logits.dtype == torch.float, "`logits` should be torch.float data type"
        assert indices.dtype == torch.long, "`indices` should be torch.long data type"
        assert (
            logits.shape == indices.shape
        ), f"The shape of `logits` and `indices` should be the same, got logits={logits.shape}, indices={indices.shape}"
        self.states = states
        self.indices, mapping = torch.sort(indices, stable=True)
        self.logits = logits[mapping]
        self.size = max(self.indices.tolist(), default=-1) + 1

        count = torch.bincount(self.indices)
        self._offsets = torch.cumsum(count, 0) - count

    def log_prob(self, actions):
        assert self.size == len(
            actions
        ), f"size doesn't match, size {self.size}, got {len(actions)}"
        action_indices = torch.LongTensor(
            [Action_to_idx(s, a) for s, a in zip(self.states, actions)]
        )
        log_probs = scatter_log_softmax(self.logits, self.indices)
        return log_probs[action_indices + self._offsets]

    @torch.no_grad()
    def sample(self):
        unif = torch.rand_like(self.logits)
        gumbel = -(-unif.log()).log()
        _, samples = scatter_max(self.logits + gumbel, self.indices)
        indices = samples - self._offsets
        return [idx_to_Action(s, i.item()) for s, i in zip(self.states, indices)]

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size})"
