#python -m cs285.scripts.run_hw5_iql --env_name PointmassEasy-v0 --exp_name "q5_iql_easy_supervised_lam1.0_tau$1" --use_rnd --num_exploration_steps=20000 --awac_lambda=0.1 --iql_expectile=$1 --exploit_rew_shift=0 --exploit_rew_scale=1 --seed 2
#python -m cs285.scripts.run_hw5_iql --env_name PointmassEasy-v0 --exp_name "q5_iql_easy_unsupervised_lam2.0_tau$1" --use_rnd --unsupervised_exploration --num_exploration_steps=20000 --awac_lambda=2.0 --iql_expectile=$1 --exploit_rew_shift=0 --exploit_rew_scale=1 --seed 2
#
#python -m cs285.scripts.run_hw5_iql --env_name PointmassMedium-v0 --exp_name "q5_iql_medium_supervised_lam50_tau$1" --use_rnd --num_exploration_steps=20000 --awac_lambda=50.0 --iql_expectile=$1 --exploit_rew_shift=0 --exploit_rew_scale=1 --seed 2
python -m cs285.scripts.run_hw5_iql --env_name PointmassMedium-v0 --exp_name "q5_iql_medium_unsupervised_lam2_tau$1" --use_rnd --unsupervised_exploration --num_exploration_steps=20000 --awac_lambda=2.0 --iql_expectile=$1 --exploit_rew_shift=0 --exploit_rew_scale=1 --seed 2

