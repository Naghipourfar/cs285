import copy
import os
import time
import numpy as np
import pandas as pd

from cs285.infrastructure.rl_trainer import RL_Trainer
from cs285.agents.bc_agent import BCAgent
from cs285.policies.loaded_gaussian_policy import LoadedGaussianPolicy
from cs285.infrastructure.utils import MJ_ENV_KWARGS, MJ_ENV_NAMES


class BC_Trainer(object):

    def __init__(self, params):
        #######################
        ## AGENT PARAMS
        #######################

        agent_params = {
            'n_layers': params['n_layers'],
            'size': params['size'],
            'learning_rate': params['learning_rate'],
            'max_replay_buffer_size': params['max_replay_buffer_size'],
        }

        self.params = params
        self.params['agent_class'] = BCAgent  ## HW1: you will modify this
        self.params['agent_params'] = agent_params

        self.params["env_kwargs"] = MJ_ENV_KWARGS[self.params['env_name']]

        ################
        ## RL TRAINER
        ################

        self.rl_trainer = RL_Trainer(self.params)  ## HW1: you will modify this

        #######################
        ## LOAD EXPERT POLICY
        #######################

        print('Loading expert policy from...', self.params['expert_policy_file'])
        self.loaded_expert_policy = LoadedGaussianPolicy(self.params['expert_policy_file'])
        print('Done restoring expert policy...')

    def run_training_loop(self):
        self.rl_trainer.run_training_loop(
            n_iter=self.params['n_iter'],
            initial_expertdata=self.params['expert_data'],
            collect_policy=self.rl_trainer.agent.actor,
            eval_policy=self.rl_trainer.agent.actor,
            relabel_with_expert=self.params['do_dagger'],
            expert_policy=self.loaded_expert_policy,
        )


def tune_policy(params):
    np.random.seed(int(time.time()))
    args = {
        'ep_len': 1024,

        'num_agent_train_steps_per_iter': int(np.random.choice([32, 64, 128, 256, 512, 1024, 2048])),
        'n_iter': 1 if not params['do_dagger'] else int(np.random.choice([3, 5, 10, 15, 20, 25, 50])),

        'batch_size': int(np.random.choice([128, 256, 512, 1024, 2048, 4096])),
        'eval_batch_size': 10000,
        'train_batch_size': int(np.random.choice([16, 32, 64, 128, 256])),

        'n_layers': int(np.random.randint(1, 5)),
        'size': int(np.random.choice([32, 64, 128, 256])),
        'learning_rate': np.random.uniform(1e-5, 5e-3),

        'video_log_freq': -1,
        'scalar_log_freq': 1,
        'no_gpu': False,
        'which_gpu': 0,
        'max_replay_buffer_size': 1000000,
        'save_params': False,
        'seed': int(np.random.randint(1, 2022)),
    }

    np.random.seed(args['seed'])

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    if params['do_dagger']:
        # Use this prefix when submitting. The auto-grader uses this prefix.
        logdir_prefix = 'q2_'
        assert args['n_iter'] > 1, (
            'DAGGER needs more than 1 iteration (n_iter>1) of training, to iteratively query the expert and train (after 1st warmstarting from behavior cloning).')
    else:
        # Use this prefix when submitting. The auto-grader uses this prefix.
        logdir_prefix = 'q1_'
        assert args['n_iter'] == 1, \
            ('Vanilla behavior cloning collects expert data just once (n_iter=1)')

    ## directory for logging
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../tune_exps/')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = logdir_prefix + params['exp_name'] + '_' + params['env_name'] + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)

    params.update(copy.deepcopy(args))
    params['logdir'] = logdir

    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    ###################
    ### RUN TRAINING
    ###################

    print('\n\nRunning Experiment with the following params:\n\t')
    print(params)

    trainer = BC_Trainer(params)
    trainer.run_training_loop()

    del params['agent_class']

    import json
    json.dump(params, fp=open(os.path.join(logdir, 'hyper-params.json'), 'w'))
    print('hyper-params saved!')

    progress = pd.DataFrame.from_dict(trainer.rl_trainer.history)
    progress.to_csv(os.path.join(logdir, 'progress.csv'), sep=',', index=False)

    print('progress saved!')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert_policy_file', '-epf', type=str,
                        required=True)  # relative to where you're running this script from
    parser.add_argument('--expert_data', '-ed', type=str,
                        required=True)  # relative to where you're running this script from
    parser.add_argument('--env_name', '-env', type=str, help=f'choices: {", ".join(MJ_ENV_NAMES)}', required=True)
    parser.add_argument('--exp_name', '-exp', type=str, default='pick an experiment name', required=True)
    parser.add_argument('--do_dagger', action='store_true')

    parser_args = parser.parse_args()
    params = vars(parser_args)

    for _ in range(50000):
        tune_policy(copy.deepcopy(params))


if __name__ == "__main__":
    main()
