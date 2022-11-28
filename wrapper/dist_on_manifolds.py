import math
from numbers import Number

import torch
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import _standard_normal, broadcast_all

from wrapper.manifolds import Euclidean, Sphere

class ManifoldNormal(ExponentialFamily):
    r"""
    Creates a normal (also called Gaussian) distribution parameterized by
    :attr:`loc` and :attr:`scale`.

    Example::

        >>> m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # normally distributed with loc=0 and scale=1
        tensor([ 0.1046])exp

    Args:
        loc (float or Tensor): mean of the distribution (often referred to as mu)
        scale (float or Tensor): standard deviation of the distribution
            (often referred to as sigma)
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.loc

    @property
    def stddev(self):
        return self.scale

    @property
    def variance(self):
        return self.stddev.pow(2)

    def __init__(self, manifold, loc, scale, validate_args=None):
        self.manifold = manifold
        self.loc, self.scale = broadcast_all(loc, scale)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(ManifoldNormal, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(ManifoldNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(ManifoldNormal, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        raise NotImplementedError


    def rsample(self, obs, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        self.manifold = Sphere(shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        #pos = self.manifold.exp(self.loc[:, :], eps[:, :] * self.scale[:, :])
        a = self.loc + eps * self.scale
        #pos = self.manifold.exp(obs, a[:, 3])
        #pos = pos - obs
        #gripper = a[:, -1].reshape(-1,1)
        return a # torch.cat((pos, gripper), axis=1)
        # pos = self.manifold.exp(self.loc[:, :3], eps[:, :3] * self.scale[:, :3])
        # gripper = self.loc[:, 3:] + eps[:, 3:] * self.scale[:, 3:]
        # return torch.cat((pos, gripper), axis=1)

    def log_prob(self, obs, value):
        self.manifold = Sphere(value.shape[0])
        if self._validate_args:
            self._validate_sample(value)
        # compute the variance
        var = (self.scale ** 2)
        log_scale = math.log(self.scale) if isinstance(self.scale, Number) else self.scale.log()
        b = obs + value[:, :3].clone()
        pos = self.manifold.log(obs, b).clone()
        gripper = value[:,3].reshape(-1,1).clone()
        a = torch.cat((pos, gripper, value[:, 4:]), axis=1).clone()
        return -((a - self.loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))
        #return -((value-self.loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))
        # pos = -(self.manifold.log(self.loc[:, :5], value[:, :5]) ** 2) / (2 * var[:, :5]) - log_scale[:, :5] - math.log(math.sqrt(2 * math.pi))
        #gripper = -((value[:, 3:] - self.loc[:, 3:]) ** 2) / (2 * var[:, 3:]) - log_scale[:, 3:] - math.log(math.sqrt(2 * math.pi))
        #gripper = -(self.manifold.log(self.loc[:, 3:], value[:, 3:]) ** 2) / (2 * var[:, 3:]) - log_scale[:, 3:] - math.log(math.sqrt(2 * math.pi))
        #return torch.cat((pos, gripper), axis=1)
        # return pos
        #return -(self.manifold.log(self.loc, value) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        #return 0.5 * (1 + torch.erf(self.manifold.log(self.loc, value) * self.scale.reciprocal() / math.sqrt(2)))
        return 0.5 * (1 + torch.erf((value - self.loc) * self.scale.reciprocal() / math.sqrt(2)))

    def icdf(self, value):
        raise NotImplementedError

    def entropy(self):
        return 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.scale)

    @property
    def _natural_params(self):
        raise NotImplementedError

    def _log_normalizer(self, x, y):
        raise NotImplementedError
