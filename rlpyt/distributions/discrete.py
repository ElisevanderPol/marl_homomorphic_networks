
import torch

from rlpyt.utils.tensor import to_onehot, from_onehot

import itertools


class DiscreteMixin:

    def __init__(self, dim, n_agents=3, n_actions=5, dtype=torch.long, onehot_dtype=torch.float):
        self._dim = dim
        self.dtype = dtype
        self.onehot_dtype = onehot_dtype
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.act_prod = list(itertools.product(*[[i for i in
                                                  range(self.n_actions)]
                                                                   for _ in
                                                                   range(self.n_agents)]))

    @property
    def dim(self):
        return self._dim

    def to_onehot(self, indexes, dtype=None):
        return to_onehot(indexes, self._dim, dtype=dtype or self.onehot_dtype)

    def from_onehot(self, onehot, dtype=None):
        return from_onehot(onehot, dtpye=dtype or self.dtype)
