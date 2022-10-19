from cs285.policies.MLP_policy import MLPPolicy
import torch
import numpy as np
from cs285.infrastructure import sac_utils
from cs285.infrastructure import pytorch_util as ptu
from cs285.infrastructure import sac_utils as sacu
from torch import nn, distributions
from torch import optim
import itertools


class MLPPolicySAC(MLPPolicy):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=3e-4,
                 training=True,
                 log_std_bounds=[-20, 2],
                 action_range=[-1, 1],
                 init_temperature=1.0,
                 **kwargs
                 ):
        super(MLPPolicySAC, self).__init__(ac_dim, ob_dim, n_layers, size, discrete, learning_rate, training, **kwargs)
        self.log_std_bounds = log_std_bounds
        self.action_range = action_range
        self.init_temperature = init_temperature
        self.learning_rate = learning_rate

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(ptu.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.target_entropy = -ac_dim

    @property
    def alpha(self):
        # TODO: Formulate entropy term
        # Mohsen: I guess it's better to return the entropy scale (alpha) here.
        return torch.exp(self.log_alpha)

    def get_action(self, obs: np.ndarray, sample=True, return_logprob=False):
        # TODO: return sample from distribution if sampling
        # if not sampling return the mean of the distribution
        observation = ptu.from_numpy(obs)

        action_range = (self.action_range[1] - self.action_range[0]) / 2
        action_bias = (self.action_range[0] + self.action_range[1]) / 2

        if len(observation.shape) > 1:
            observation = observation
        else:
            observation = observation[None]

        action_dist = self.forward(observation)
        if sample:
            action = action_dist.sample()
        else:
            action = action_dist.mean

        log_prob = action_dist.log_prob(action) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdims=True).detach().cpu().numpy()

        action = action * action_range + action_bias
        if return_logprob:
            return ptu.to_numpy(action), log_prob
        else:
            return ptu.to_numpy(action)

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        if isinstance(observation, np.ndarray):
            observation = ptu.from_numpy(observation)

        # TODO: Implement pass through network, computing logprobs and apply correction for Tanh squashing
        if len(observation.shape) > 1:
            observation = observation
        else:
            observation = observation[None]

        action_mean = self.mean_net(observation)
        scale_tril = torch.exp(self.logstd)
        batch_dim = action_mean.shape[0]
        log_std = scale_tril.repeat(batch_dim, 1)
        log_std = torch.clamp(
            log_std, min=self.log_std_bounds[0], max=self.log_std_bounds[1])

        action_distribution = sacu.SquashedNormal(
            loc=action_mean,
            scale=log_std,
        )

        # HINT:
        # You will need to clip log values
        # You will need SquashedNormal from sac_utils file
        return action_distribution

    def update(self, obs, critic):
        # TODO Update actor network and entropy regularizer
        # return losses and alpha value
        self.optimizer.zero_grad()

        eps = 1e-6
        action_dist = self.forward(obs)
        actions = action_dist.rsample()

        q1, q2 = critic.forward(obs, actions)
        min_q = torch.min(q1, q2)

        log_prob = action_dist.log_prob(actions) - torch.log(1 - actions.pow(2) + eps)
        log_prob = log_prob.sum(dim=1, keepdims=True)
        actor_loss = ((self.alpha * log_prob) - min_q).mean()
        actor_loss.backward()
        self.optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        action_dist = self.forward(obs)
        actions = action_dist.rsample()
        log_prob = action_dist.log_prob(actions) - torch.log(1 - actions.pow(2) + eps)
        log_prob = log_prob.sum(dim=1, keepdims=True)
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy)).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        return actor_loss.item(), alpha_loss.item(), self.alpha.item()
