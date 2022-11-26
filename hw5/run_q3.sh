#python -m cs285.scripts.run_hw5_expl --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=20000 --cql_alpha=0.0 --exp_name q3_medium_dqn
#python -m cs285.scripts.run_hw5_expl --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=20000 --cql_alpha=1.0 --exp_name q3_medium_cql

python -m cs285.scripts.run_hw5_expl --env_name PointmassHard-v0 --use_rnd --num_exploration_steps=20000 --cql_alpha=0.0 --exp_name q3_hard_dqn --seed 2 --exploit_rew_shift 1 --exploit_rew_scale 100
python -m cs285.scripts.run_hw5_expl --env_name PointmassHard-v0 --use_rnd --num_exploration_steps=20000 --cql_alpha=1.0 --exp_name q3_hard_cql --seed 2 --exploit_rew_shift 1 --exploit_rew_scale 100