from src.plot import plot_test_avg

results_dir = "qppo-slurm/results"
plots_dir = "final-plots"


#Test 0 - ppo default:
"""
plot_dir = plots_dir + "/Test_0"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = ["ppo_default_lr2.5e-4"]
seeds = [10, 20, 30, 40, 50]
stepsize = 4*128
max_steps = 300000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, stepsize, max_steps)
"""



#Hyperparameter Test 1: find optimal learning rate for ppo:

plot_dir = plots_dir + "/Hyperparameter-Test-1_PPO-LR"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = ["ppo_default_lr2.0e-4", "ppo_default_lr2.5e-4", "ppo_default_lr3.0e-4"]
seeds = [10, 20, 30, 40, 50]
stepsize = 4*128
max_steps = 300000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, stepsize, max_steps)

