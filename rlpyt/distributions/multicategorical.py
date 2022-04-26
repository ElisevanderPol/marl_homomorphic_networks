
import torch

from rlpyt.distributions.base import Distribution
from rlpyt.distributions.discrete import DiscreteMixin
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.tensor import valid_mean, select_at_indexes

EPS = 1e-8

DistInfo = namedarraytuple("DistInfo", ["prob"])


class MultiCategorical(DiscreteMixin, Distribution):

    def kl(self, old_dist_info, new_dist_info):
        p = old_dist_info.prob
        q = new_dist_info.prob
        return torch.sum(p * (torch.log(p + EPS) - torch.log(q + EPS)), dim=-1)

    def mean_kl(self, old_dist_info, new_dist_info, valid=None):
        return valid_mean(self.kl(old_dist_info, new_dist_info), valid)

    def sample(self, dist_info):
        ### Rewritten to make batch * n_agents
        p = dist_info.prob
        if len(p.shape) > 2:
            bs = p.shape[0]
        else:
            bs = 1
        sample = torch.multinomial(p.view(bs*self.n_agents, self.n_actions),
                                   num_samples=1)
        if bs == 1:
            sample = sample.view(self.n_agents)
        else:
            sample = sample.view(bs, self.n_agents)

        return sample.type(self.dtype)

    def entropy(self, dist_info, product=False):
        if product:
            p = torch.prod(dist_info.prob, dim=1)
        else:
            p = dist_info.prob
        return -torch.sum(p * torch.log(p + EPS), dim=-1)

    def log_likelihood(self, indexes, dist_info):
        selected_likelihood = select_at_indexes(indexes, dist_info.prob)
        return torch.log(selected_likelihood + EPS)

    def likelihood_ratio(self, indexes, old_dist_info, new_dist_info):
        num = select_at_indexes(indexes, new_dist_info.prob)
        den = select_at_indexes(indexes, old_dist_info.prob)
        return (num + EPS) / (den + EPS)

    def multiply(self, dist_info):
        """
        """
        return DistInfo(prob=torch.prod(dist_info.prob, dim=1))
