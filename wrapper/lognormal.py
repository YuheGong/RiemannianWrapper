import math
from numbers import Number

import torch
from torch.distributions import constraints, Normal
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import _standard_normal, broadcast_all
from numbers import Real
import matplotlib.colors as pltc
from wrapper.manifolds import Euclidean, Sphere
from geomstats.distributions.lognormal import LogNormal
from .von_mises_fisher import VonMisesFisher
#from tensorflow_probability.python.distributions.von_mises_fisher import VonMisesFisher
import numpy as np



class CombineNormal(ExponentialFamily):
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

        self.index = {}
        self.index['position'] = [0,3]
        self.index['gripper'] = [3,4]
        self.index['orientation'] = [4,8]

        self.manifold_dist = {}

        self.loc = loc
        self.scale = scale


        for i in manifold:
            if i == 'position':
                self.manifold_dist[i] = LogNormal(
                    manifold[i], loc[:, self.index[i][0]:self.index[i][1]],
                    scale[:, self.index[i][0]:self.index[i][1]], validate_args=validate_args)
            elif i == 'gripper':
                self.manifold_dist[i] = Normal(loc[:, self.index[i][0]:self.index[i][1]],
                    scale[:, self.index[i][0]:self.index[i][1]], validate_args=validate_args)
            elif i == 'orientation':
                self.manifold_dist[i] = VonMisesFisher(
                    loc=loc[:, self.index[i][0]:self.index[i][1]],#.reshape(-1),
                    scale=scale[:, self.index[i][0]].reshape(-1,1))
                a = self.manifold_dist[i].rsample()

    def sample(self, sample_shape=torch.Size()):
        raise NotImplementedError


    def rsample(self, obs, sample_shape=torch.Size()):

        s_pos_gri = torch.cat([self.manifold_dist['position'].rsample(), self.manifold_dist['gripper'].rsample(),], dim=1)
                               #torch.tensor(self.manifold_dist['orientation'].sample()).to(device='cuda')], dim=1)
        return s_pos_gri#torch.cat([ s_pos_gri, s_ori],dim=1)

    #def log_prob(self, obs, value):
    #    if self._validate_args:
    #        self._validate_sample(value)
        # compute the variance
    #    var = (self.scale ** 2)
    #    log_scale = self.scale.log()
    #    return -((value - self.loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))

    def log_prob(self, obs, value):
        pos = self.manifold_dist['position'].log_prob(value[:,self.index['position'][0]:self.index['position'][1]])
        gripper = self.manifold_dist['gripper'].log_prob(value[:,self.index['gripper'][0]:self.index['gripper'][1]])
        # orientation = self.manifold_dist['orientation'].log_prob(value[:, self.index['orientation'][0]:self.index['orientationr'][1]])
        return torch.cat([pos, gripper], dim=1)


    def cdf(self, value):
        raise NotImplementedError
        #if self._validate_args:
        #    self._validate_sample(value)
        #return 0.5 * (1 + torch.erf((value - self.loc) * self.scale.reciprocal() / math.sqrt(2)))

    def icdf(self, value):
        raise NotImplementedError

    def entropy(self):
        return torch.cat([self.manifold_dist['position'].entropy(), self.manifold_dist['gripper'].entropy(), ],
                              dim=1)

    @property
    def _natural_params(self):
        raise NotImplementedError

    def _log_normalizer(self, x, y):
        raise NotImplementedError



class LogNormal(ExponentialFamily):
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

        #self.dist = LogNormal(manifold, loc[0].cpu().detach().numpy(), torch.diag(scale[0], 0).cpu().detach().numpy())

        self.loc, self.scale = broadcast_all(loc, scale)

        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(LogNormal, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(ManifoldNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(LogNormal, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        raise NotImplementedError


    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)

        pos_eps = _standard_normal(self.loc.shape[1], dtype=self.loc.dtype, device=self.loc.device)
        # while torch.square(pos_eps).mean() > 0.1:
        #    pos_eps = _standard_normal(self.pos_index_high - self.pos_index_low, dtype=self.loc.dtype, device=self.loc.device)
        pos_sample = self.manifold.exp(self.loc, pos_eps * self.scale)

        return pos_sample

    #def log_prob(self, obs, value):
    #    if self._validate_args:
    #        self._validate_sample(value)
        # compute the variance
    #    var = (self.scale ** 2)
    #    log_scale = self.scale.log()
    #    return -((value - self.loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # compute the variance
        log_scale = math.log(self.scale) if isinstance(self.scale, Real) else self.scale.log()

        batch_size = value.shape[0]
        pos = torch.cat([self.manifold.log(self.loc[i,:], value[i,:]) for i in range(batch_size)]).reshape(-1,self.loc.shape[1])\
            .double().to(device="cuda").reshape(self.loc.shape[0], self.loc.shape[1])
        #pos_exp = torch.cat([pos * self.scale * pos] ).to(device="cuda").reshape(-1, 3)

        # plot
        """
        base_point = self.pos_loc[0, :]
        data = value_pos[0,:]
        u = self.manifold.log(base_point, data)
        geodesic_tensor = torch.cat([self.manifold.exp(base_point, u * t) for t in torch.linspace(0., 1., 20)]).reshape(-1,3)
        y1 = base_point.cpu().detach().numpy()
        y2 = data.cpu().detach().numpy()
        geodesic = geodesic_tensor.cpu().detach().numpy()
        u = u.cpu().detach().numpy()

        
        #fig_maps = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(500, 500))
        mlab.clf()
        plot_sphere()
        mlab.points3d(y1[0], y1[1], y1[2], color=(0., 0., 0.), scale_factor=0.1)
        mlab.points3d(0, 0, 1, color=(0., 0., 0.), scale_factor=0.1)
        #mlab.points3d(y2[0], y2[1], y2[2], color=pltc.to_rgb('crimson'), scale_factor=0.1)
        #plot_sphere_tangent_plane(y1, l_vert=1.3, opacity=0.3)
        plot_vector_on_tangent_plane(y1, u, pltc.to_rgb('red'))
        #mlab.plot3d(geodesic[:, 0], geodesic[:, 1], geodesic[:, 2], color=pltc.to_rgb('crimson'),
        #            line_width=2, tube_radius=None)

        mlab.view(110, 60)
        mlab.show()
        #fig_maps
        """

        #pos_exp = torch.cat([pos[i, :] * pos[i, :] for i in range(batch_size)]).to(device="cuda").reshape(-1, 3)
        #pos_log = - pos_exp/ 2 - math.log(math.sqrt(2 * math.pi)) - log_scale[:,self.pos_index_low:self.pos_index_high]

        var = (self.scale ** 2)
        pos_log = -((pos * self.scale * pos)) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))
        #expand_pos_log = torch.expand_copy(pos_log.reshape((-1,1)), (-1, 3))
        #return torch.cat([pos_log,gripper_log], dim=1)
        return pos_log


    def cdf(self, value):
        raise NotImplementedError
        #if self._validate_args:
        #    self._validate_sample(value)
        #return 0.5 * (1 + torch.erf((value - self.loc) * self.scale.reciprocal() / math.sqrt(2)))

    def icdf(self, value):
        raise NotImplementedError

    def entropy(self):
        return 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.scale)

    @property
    def _natural_params(self):
        raise NotImplementedError

    def _log_normalizer(self, x, y):
        raise NotImplementedError
