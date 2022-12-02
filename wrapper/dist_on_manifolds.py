import math
from numbers import Number

import torch
from torch.distributions import constraints, Normal
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import _standard_normal, broadcast_all
from numbers import Real
from mayavi import mlab
import matplotlib.colors as pltc
from wrapper.draw import plot_sphere, plot_sphere_tangent_plane, plot_gaussian_mesh_on_tangent_plane, plot_vector_on_tangent_plane, draw_arrow_mayavi
from wrapper.manifolds import Euclidean, Sphere
from geomstats.distributions.lognormal import LogNormal

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

        #self.dist = LogNormal(manifold, loc[0].cpu().detach().numpy(), torch.diag(scale[0], 0).cpu().detach().numpy())

        self.loc, self.scale = broadcast_all(loc, scale)

        self.pos_index_low = 0
        self.pos_index_high = 3
        self.gripper_index_low = 3
        self.gripper_index_high = 4
        self.pos_loc = self.loc[:, self.pos_index_low:self.pos_index_high]
        self.pos_scale = self.scale[:, self.pos_index_low:self.pos_index_high]
        self.gripper_loc = self.loc[:, self.gripper_index_low:self.gripper_index_high]
        self.gripper_scale = self.scale[:, self.gripper_index_low:self.gripper_index_high]


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


    def rsample(self,obs, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)

        pos_eps = _standard_normal(self.pos_index_high-self.pos_index_low, dtype=self.loc.dtype, device=self.loc.device)
        while torch.square(pos_eps).mean() > 0.1:
            pos_eps = _standard_normal(self.pos_index_high - self.pos_index_low, dtype=self.loc.dtype,
                                       device=self.loc.device)
        pos_sample = self.manifold.exp(self.pos_loc, pos_eps * self.pos_scale)

        gripper_eps = _standard_normal(self.gripper_index_high-self.gripper_index_low, dtype=self.loc.dtype, device=self.loc.device)
        gripper_sample = self.gripper_loc + gripper_eps * self.gripper_scale

        return torch.cat([pos_sample, gripper_sample],dim=1)

    #def log_prob(self, obs, value):
    #    if self._validate_args:
    #        self._validate_sample(value)
        # compute the variance
    #    var = (self.scale ** 2)
    #    log_scale = self.scale.log()
    #    return -((value - self.loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))

    def log_prob(self, obs, value):
        if self._validate_args:
            self._validate_sample(value)
        # compute the variance
        log_scale = math.log(self.gripper_scale) if isinstance(self.gripper_scale, Real) else self.gripper_scale.log()

        value_pos = value[:, self.pos_index_low:self.pos_index_high]
        value_gripper = value[:, self.gripper_index_low:self.gripper_index_high]

        pos_dim = self.pos_index_high-self.pos_index_low
        batch_size = value.shape[0]
        pos = torch.cat([self.manifold.log(self.pos_loc[i, :], value_pos[i,:]) for i in range(batch_size)]).reshape(-1,pos_dim)\
            .double().to(device="cuda").reshape(self.pos_loc.shape[0], self.pos_loc.shape[1])
        pos_exp = torch.cat([pos * self.pos_scale * pos] ).to(device="cuda").reshape(-1, 3)

        # plot
        base_point = self.pos_loc[0, :]
        data = value_pos[0,:]
        u = self.manifold.log(base_point, data)
        geodesic_tensor = torch.cat([self.manifold.exp(base_point, u * t) for t in torch.linspace(0., 1., 20)]).reshape(-1,3)
        y1 = base_point.cpu().detach().numpy()
        y2 = data.cpu().detach().numpy()
        geodesic = geodesic_tensor.cpu().detach().numpy()
        u = u.cpu().detach().numpy()
        """
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
        pos_log = - pos_exp/ 2 - math.log(math.sqrt(2 * math.pi)) - log_scale[:,self.pos_index_low:self.pos_index_high]

        var = (self.scale[:,:3] ** 2)
        pos_log = -((pos * self.pos_scale * pos)) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))
        #expand_pos_log = torch.expand_copy(pos_log.reshape((-1,1)), (-1, 3))

        var = (self.gripper_scale ** 2)
        gripper_log = - ((value_gripper - self.gripper_loc) ** 2) / (2 * var) \
                       - log_scale\
                    - math.log(math.sqrt(2 * math.pi))

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
