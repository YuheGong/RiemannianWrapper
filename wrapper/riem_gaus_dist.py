from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import torch as th
from gym import spaces
from torch import nn
from torch.distributions import Bernoulli, Categorical, Normal
from wrapper.dist_on_manifolds import ManifoldNormal

from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.distributions import DiagGaussianDistribution, Distribution
from wrapper.manifolds import Sphere, Euclidean
from scipy.stats import rv_continuous
import numpy as np

def sum_independent_dims(tensor: th.Tensor) -> th.Tensor:
    """
    Continuous actions are usually considered to be independent,
    so we can sum components of the ``log_prob`` or the entropy.

    :param tensor: shape: (n_batch, n_actions) or (n_batch,)
    :return: shape: (n_batch,)
    """
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor

'''
class RGD_distribution(rv_continuous):
    #Riemannian Gaussian distribution


    def __init__(self, mean, cov):
        self.loc = mean
        self.cov = cov
        super(RGD_distribution, self).__init__()

    def _pdf(self, x):
        return np.exp(-x ** 2 / 2.) / np.sqrt(2.0 * np.pi)

    def rsample(self, random_state=None):
        """
        Return a sample from PDF - Probability Distribution Function.
        calling - rv_continuous class.
        """
        return self._rvs(size=1, random_state=random_state)

'''


class RiemannianGaussianDistribution(DiagGaussianDistribution):
    """
    Riemannian Gaussian distribution.

    :param action_dim:  Dimension of the action space.
    """

    def __init__(self, manifold, action_dim: int):
        super(RiemannianGaussianDistribution, self).__init__(action_dim)
        self.manifold = manifold#(action_dim-1)
        self.distribution = None
        self.action_dim = action_dim
        self.mean_actions = None
        self.log_std = None

    def log_prob(self, obs, actions: th.Tensor) -> th.Tensor:
        """
        Get the log probabilities of actions according to the distribution.
        Note that you must first call the ``proba_distribution()`` method.

        :param actions:
        :return:
        """
        log_prob = self.distribution.log_prob(obs, actions)
        return sum_independent_dims(log_prob)
    def get_actions(self, obs, deterministic: bool = False) -> th.Tensor:
        """
        Return actions according to the probability distribution.

        :param deterministic:
        :return:
        """
        if deterministic:
            return self.mode()
        return self.sample(obs)

    def sample(self, obs) -> th.Tensor:
        # Reparametrization trick to pass gradients
        return self.distribution.rsample(obs)

    def proba_distribution(self, mean_actions: th.Tensor, log_std: th.Tensor) -> "DiagGaussianDistribution":
        """
        Create the distribution given its parameters (mean, std)

        :param mean_actions:
        :param log_std:
        :return:
        """
        action_std = th.ones_like(mean_actions) * log_std.exp()
        self.distribution = ManifoldNormal(self.manifold, mean_actions, action_std)
        #self.distribution = Normal(mean_actions, action_std)
        return self

    '''
    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0) -> Tuple[nn.Module, nn.Parameter]:
        """
        Create the layers and parameter that represent the distribution:
        one output will be the mean of the Gaussian, the other parameter will be the
        standard deviation (log std in fact to allow negative values)

        :param latent_dim: Dimension of the last layer of the policy (before the action layer)
        :param log_std_init: Initial value for the log standard deviation
        :return:
        """
        mean_actions = nn.Linear(latent_dim, self.action_dim)
        # TODO: allow action dependent std
        log_std = nn.Parameter(th.ones(self.action_dim) * log_std_init, requires_grad=True)
        return mean_actions, log_std
    
    def get_actions(self, deterministic: bool = False) -> th.Tensor:
    
        """
        Return actions according to the probability distribution.

        :param deterministic:
        :return:
        """
        
        if deterministic:
            a_exp = self.mode()
            a_mani = self.manifolds.exp(self.mean_actions[:,:3].clone(), a_exp[:,:3].clone())
            a_exp[:, :3] = (a_mani).clone()
            a_exp[:, 3:] = self.mean_actions[:, 3:].clone() + a_exp[:, 3:].clone()
            return a_exp
        a_exp = self.sample()
        a_mani = self.manifolds.exp(self.mean_actions[:,:3].clone(), a_exp[:, :3].clone())
        a_exp[:, :3] = (a_mani).clone()
        a_exp[:, 3:] = self.mean_actions[:,3:].clone() + a_exp[:, 3:].clone()
        return a_exp
        
        if deterministic:
            a_exp = self.mode()
            a_mani = self.manifolds.exp(self.mean_actions[:,:3].clone(), a_exp[:, :3].clone())
            a_exp[:,:3] = a_mani #- self.mean_actions[:,:3]
            return a_exp
        a_exp = self.sample()
        a_mani = self.manifolds.exp(self.mean_actions[:,:3].clone(), a_exp[:, :3].clone())
        a_exp[:,:3] = a_mani # - self.mean_actions[:,:3]
        return a_exp


    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        """
        Get the log probabilities of actions according to the distribution.
        Note that you must first call the ``proba_distribution()`` method.

        :param actions:
        :return:
        """
        
        #actions[:, :3] = self.manifolds.log(self.mean_actions[:,:3], actions[:, :3].clone())
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)
        
        # actions = self.manifolds.log(self.mean_actions, actions)
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)
        '''



