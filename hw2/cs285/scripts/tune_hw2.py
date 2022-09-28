import copy
import os
import time

import numpy as np
import pandas as pd
from cs285.infrastructure.rl_trainer import RL_Trainer
from cs285.agents.pg_agent import PGAgent


class PG_Trainer(object):

    def __init__(self, params):
        #####################
        ## SET AGENT PARAMS
        #####################

        computation_graph_args = {
            'n_layers': params['n_layers'],
            'size': params['size'],
            'learning_rate': params['learning_rate'],
        }

        estimate_advantage_args = {
            'gamma': params['discount'],
            'standardize_advantages': not (params['dont_standardize_advantages']),
            'reward_to_go': params['reward_to_go'],
            'nn_baseline': params['nn_baseline'],
            'gae_lambda': params['gae_lambda'],
        }

        train_args = {
            'num_agent_train_steps_per_iter': params['num_agent_train_steps_per_iter'],
        }

        agent_params = {**computation_graph_args, **estimate_advantage_args, **train_args}

        self.params = params
        self.params['agent_class'] = PGAgent
        self.params['agent_params'] = agent_params
        self.params['batch_size_initial'] = self.params['batch_size']

        ################
        ## RL TRAINER
        ################

        self.rl_trainer = RL_Trainer(self.params)

    def run_training_loop(self):
        self.rl_trainer.run_training_loop(
            self.params['n_iter'],
            collect_policy=self.rl_trainer.agent.actor,
            eval_policy=self.rl_trainer.agent.actor,
        )


def tune_policy(params):
    np.random.seed(int(time.time() % 1e4))
    args = {
        'gae_lambda': None if params['gae_lambda'] is None else np.random.uniform(0.9, 1.0),

        'num_agent_train_steps_per_iter': 1,
        'discount': 0.9,

        'batch_size': int(np.random.choice([1024, 2048, 4096, 8192])) if params['batch_size'] is None else
        params['batch_size'],
        'eval_batch_size': 400,

        'n_layers': 2,
        'size': 64,
        'learning_rate': np.random.uniform(1e-3, 5e-1) if params['learning_rate'] is None else params['learning_rate'],

        'video_log_freq': -1,
        'scalar_log_freq': 1,
        'no_gpu': False,
        'which_gpu': 0,
        'max_replay_buffer_size': 1000000,
        'save_params': False,
        'seed': int(time.time() % 1e4),
    }

    args['train_batch_size'] = args['batch_size']

    np.random.seed(args['seed'])

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    logdir_prefix = 'q2_pg_'  # keep for autograder

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../tune_exps')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path, exist_ok=True)

    logdir = logdir_prefix + params['exp_name'] + '_' + params['env_name'] + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params.update(copy.deepcopy(args))
    params['logdir'] = logdir
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    ###################
    ### RUN TRAINING
    ###################

    print(params)

    trainer = PG_Trainer(params)
    trainer.run_training_loop()

    import json
    del params['agent_class']
    json.dump(params, fp=open(os.path.join(logdir, 'hyper-params.json'), 'w'))
    print('hyper-params saved!')

    progress = pd.DataFrame.from_dict(trainer.rl_trainer.history)
    progress.to_csv(os.path.join(logdir, 'progress.csv'), sep=',', index=False)

    print('progress saved!')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str)
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--ep_len', type=int, default=None)

    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--nn_baseline', action='store_true')
    parser.add_argument('--dont_standardize_advantages', '-dsa', action='store_true')

    parser.add_argument('--gae_lambda', type=float, default=None)

    parser.add_argument('--batch_size', '-b', type=int, default=None)  # steps collected per train iteration

    parser.add_argument('--learning_rate', '-lr', type=float, default=None)

    parser.add_argument('--action_noise_std', type=float, default=0)

    parser_args = parser.parse_args()
    params = vars(parser_args)

    for _ in range(50000):
        tune_policy(copy.deepcopy(params))


if __name__ == "__main__":
    main()
