from src.plot import plot_test_avg

results_dir = "qppo-slurm/results"
plots_dir = "final-plots"


# Test 0 - ppo default:
"""
plot_dir = plots_dir + "/Test_0"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = ["ppo_default_lr2.5e-4"]
seeds = [10, 20, 30, 40, 50]
stepsize = 4*128
max_steps = 300000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, stepsize, max_steps)
"""


# Hyperparameter Test 1: find optimal learning rate for ppo with 4-4-NN:
"""
plot_dir = plots_dir + "/Hyperparameter-Test-1_PPO-LR"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "ppo_default_lr2.0e-4",
    # "ppo_default_lr2.5e-4",
    # "ppo_default_lr3.0e-4",
    # "ppo_default_lr3.5e-4",
    "ppo_default_lr4.0e-4",
    # "ppo_default_lr4.5e-4",
    "ppo_default_lr6.0e-4",
    # "ppo_default_lr8.0e-4",
    "ppo_default_lr10.0e-4",
    "ppo_default_lr15.0e-4",
    "ppo_default_lr25.0e-4",
    "ppo_default_lr50.0e-4",
    "ppo_default_lr100.0e-4",
    "ppo_default_lr250.0e-4",
]
seeds = [10, 20, 30, 40, 50]
stepsize = 4 * 128 * 4
max_steps = 300000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, stepsize, max_steps)
"""

# Hyperparameter Test 2: find optimal learning rate and clip_coef for qppo:


plot_dir = plots_dir + "/Hyperparameter-Test-2_QPPO-LR"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "qppo_Jerbi-no-reuploading-no-input-scaleing_lr-sceduling_qlr0.1e-3",
    "qppo_Jerbi-no-reuploading-no-input-scaleing_lr-sceduling_qlr0.5e-3",
]
seeds = [10, 20, 30]
stepsize = 4 * 128 * 1
max_steps = 500000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, stepsize, max_steps)
