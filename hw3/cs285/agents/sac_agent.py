import copy
from collections import OrderedDict

import torch
from cs285.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from cs285.policies.MLP_policy import MLPPolicyAC

from cs285.infrastructure.sac_utils import soft_update_params
from .base_agent import BaseAgent
import gym
from cs285.policies.sac_policy import MLPPolicySAC
from cs285.critics.sac_critic import SACCritic
import cs285.infrastructure.pytorch_util as ptu


class SACAgent(BaseAgent):
    def __init__(self, env: gym.Env, agent_params):
        super(SACAgent, self).__init__()

        self.env = env
        self.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.critic_tau = 0.005
        self.learning_rate = self.agent_params['learning_rate']

        self.actor = MLPPolicySAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
            action_range=self.action_range,
            init_temperature=self.agent_params['init_temperature']
        )
        self.actor_update_frequency = self.agent_params['actor_update_frequency']
        self.critic_target_update_frequency = self.agent_params['critic_target_update_frequency']

        self.critic = SACCritic(self.agent_params)
        self.critic_target = copy.deepcopy(self.critic).to(ptu.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.training_step = 0
        self.replay_buffer = ReplayBuffer(max_size=100000)

    def update_critic(self, ob_no, ac_na, next_ob_no, re_n, terminal_n):
        # TODO: 
        # 1. Compute the target Q value.
        with torch.no_grad():
            next_action, next_log_prob = self.actor.get_action(next_ob_no, sample=True, return_logprob=True)
            q1_target_tp1, q2_target_tp1 = self.critic_target.forward(next_ob_no, next_action)
            target_q_value = torch.min(q1_target_tp1,
                                       q2_target_tp1).detach().cpu().numpy() - self.actor.alpha.detach().cpu().numpy() * next_log_prob
            target_q_value = target_q_value.squeeze()
        # HINT: You need to use the entropy term (alpha)
        # 2. Get current Q estimates and calculate critic loss
        y_i = (re_n + (1 - terminal_n) * self.gamma * target_q_value).reshape(-1, 1)

        y_i = ptu.from_numpy(y_i)
        q1, q2 = self.critic.forward(ob_no, ac_na)
        critic_loss = self.critic.loss(q1, y_i) + self.critic.loss(q2, y_i)
        # 3. Optimize the critic
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        return critic_loss.item()

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        loss = OrderedDict()
        # TODO 
        # 1. Implement the following pseudocode:
        # for agent_params['num_critic_updates_per_agent_update'] steps,
        #     update the critic
        for i in range(self.agent_params['num_critic_updates_per_agent_update']):
            loss['Critic_Loss'] = self.update_critic(ob_no, ac_na, next_ob_no, re_n, terminal_n)

        # 2. Softly update the target every critic_target_update_frequency (HINT: look at sac_utils)
        if self.training_step % self.critic_target_update_frequency == 0:
            soft_update_params(self.critic, self.critic_target, self.critic_tau)

        # 3. Implement following pseudocode:
        # If you need to update actor
        # for agent_params['num_actor_updates_per_agent_update'] steps,
        #     update the actor
        # 4. gather losses for logging
        if self.training_step % self.actor_update_frequency == 0:
            for i in range(self.agent_params['num_actor_updates_per_agent_update']):
                loss['Actor_Loss'], loss['Alpha_Loss'], loss['Temperature'] = self.actor.update(ob_no, self.critic)

        self.training_step += 1

        return loss

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_random_data(batch_size)
