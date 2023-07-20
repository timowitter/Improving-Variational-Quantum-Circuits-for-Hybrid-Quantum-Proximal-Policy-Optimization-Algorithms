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



# Hyperparameter Test 1: find optimal learning rate for ppo with 4-4-NN:

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

# Hyperparameter Test 2: find optimal learning rate and clip_coef for qppo: (failed due to error in sceduling)


plot_dir = plots_dir + "/Hyperparameter-Test-2_QPPO-LR"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "qppo_Jerbi-no-reuploading-no-input-scaleing_lr-sceduling_qlr0.1e-3",
    "qppo_Jerbi-no-reuploading-no-input-scaleing_lr-sceduling_qlr0.5e-3",
    "qppo_Jerbi-no-reuploading-no-input-scaleing_lr-sceduling_qlr2.5e-3",
]
seeds = [10, 20, 30]
stepsize = 4 * 128 * 8
max_steps = 500000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, stepsize, max_steps)


# Hyperparameter Test 2b: find optimal learning rate and clip_coef for qppo: sceduled output scaleing and no clip_grad_norm for circuit parameters (failed due to error in sceduling)
plot_dir = plots_dir + "/Hyperparameter-Test-2b_QPPO-LR"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "qppo_Jerbi-no-reuploading-no-input-scaleing_lrANDoutpScl-sceduling_no-clip-grad-norm_qlr0.1e-3",
    "qppo_Jerbi-no-reuploading-no-input-scaleing_lrANDoutpScl-sceduling_no-clip-grad-norm_qlr0.5e-3",
    "qppo_Jerbi-no-reuploading-no-input-scaleing_lrANDoutpScl-sceduling_no-clip-grad-norm_qlr2.5e-3",
]
seeds = [10, 20]
stepsize = 4 * 128 * 8
max_steps = 500000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, stepsize, max_steps)


# Hyperparameter Test 2c: 2b in default Frozen Lake (failed due to error in sceduling)
plot_dir = plots_dir + "/Hyperparameter-Test-2c_QPPO-LR"
gym_id = "FrozenLake-v1"
exp_names = [
    "qppo_Jerbi-no-reuploading-no-input-scaleing_default-FrozenLake_lrANDoutpScl-sceduling_no-clip-grad-norm_qlr0.1e-3",
    "qppo_Jerbi-no-reuploading-no-input-scaleing_default-FrozenLake_lrANDoutpScl-sceduling_no-clip-grad-norm_qlr0.5e-3",
    "qppo_Jerbi-no-reuploading-no-input-scaleing_default-FrozenLake_lrANDoutpScl-sceduling_no-clip-grad-norm_qlr2.5e-3",
]
seeds = [10, 20]
stepsize = 4 * 128 * 8
max_steps = 500000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, stepsize, max_steps)



# Hyperparameter Test 2d: find optimal learning rate and clip_coef for qppo: fixed sceduling
plot_dir = plots_dir + "/Hyperparameter-Test-2d_QPPO-LR"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "qppo-Jerbi_1enc_layer_no_input_scaleing-lrANDoutpScl_sceduling-qlr0.1e-3",
    "qppo-Jerbi_1enc_layer_no_input_scaleing-lrANDoutpScl_sceduling-qlr0.5e-3",
    "qppo-Jerbi_1enc_layer_no_input_scaleing-lrANDoutpScl_sceduling-qlr2.5e-3",
]
seeds = [10, 20]
stepsize = 4 * 128 * 8
max_steps = 500000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, stepsize, max_steps)



# Hyperparameter Test 2e: find optimal learning rate and clip_coef for qppo: fixed sceduling
plot_dir = plots_dir + "/Hyperparameter-Test-2e_QPPO-LR-Plot2"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    # "qppo-Jerbi_1enc_layer_no_input_scaleing-lr_sceduling_AND_trainable_outpScl-qlr0.1e-3",
    # "qppo-Jerbi_1enc_layer_no_input_scaleing-lr_sceduling_AND_trainable_outpScl-qlr0.5e-3",
    # "qppo-Jerbi_1enc_layer_no_input_scaleing-lr_sceduling_AND_trainable_outpScl-qlr2.5e-3",
    # "qppo-Jerbi_1enc_layer_no_input_scaleing-lrANDoutpScl_sceduling-qlr0.1e-3",
    # "qppo-Jerbi_1enc_layer_no_input_scaleing-lrANDoutpScl_sceduling-qlr0.5e-3",
    # "qppo-Jerbi_1enc_layer_no_input_scaleing-lrANDoutpScl_sceduling-qlr2.5e-3",
    "qppo-Jerbi_1enc_layer_no_input_scaleing-lrANDoutpSclx1.0_sceduling-qlr0.1e-3",
    "qppo-Jerbi_1enc_layer_no_input_scaleing-lrANDoutpSclx1.0_sceduling-qlr0.5e-3",
    "qppo-Jerbi_1enc_layer_no_input_scaleing-lrANDoutpSclx1.0_sceduling-qlr2.5e-3",
    "qppo-Jerbi_1enc_layer_no_input_scaleing-lrANDoutpSclx3.0_sceduling-qlr0.1e-3",
    "qppo-Jerbi_1enc_layer_no_input_scaleing-lrANDoutpSclx3.0_sceduling-qlr0.5e-3",
    "qppo-Jerbi_1enc_layer_no_input_scaleing-lrANDoutpSclx3.0_sceduling-qlr2.5e-3",
    "qppo-Jerbi_1enc_layer_no_input_scaleing-lrANDoutpSclx5.0_sceduling-qlr0.25e-3",
    "qppo-Jerbi_1enc_layer_no_input_scaleing-lrANDoutpSclx5.0_sceduling-qlr0.5e-3",
    "qppo-Jerbi_1enc_layer_no_input_scaleing-lrANDoutpSclx5.0_sceduling-qlr1.0e-3",
    "random-baseline",
    "qppo-Jerbi_2enc_layer_no_input_scaleing-lrANDoutpSclx3.0_sceduling-qlr0.1e-3",
    "qppo-Jerbi_2enc_layer_no_input_scaleing-lrANDoutpSclx3.0_sceduling-qlr0.5e-3",
    "qppo-Jerbi_2enc_layer_no_input_scaleing-lrANDoutpSclx3.0_sceduling-qlr2.5e-3",
]
seeds = [10, 20]
stepsize = 4 * 128 * 8
max_steps = 500000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, stepsize, max_steps)
"""


# Hyperparameter Test 3a: find optimal learning rate for qppo
plot_dir = plots_dir + "/Hyperparameter-Test-3a_QPPO-LR"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "qppo-simple-qlr0.1e-3",
    # "qppo-simple-qlr0.25e-3",
    "qppo-simple-qlr0.5e-3",
    # "qppo-simple-qlr1.0e-3",
    "qppo-simple-qlr2.5e-3",
    "qppo-Hgog-qlr0.1e-3",
    # "qppo-Hgog-qlr0.25e-3",
    "qppo-Hgog-qlr0.5e-3",
    # "qppo-Hgog-qlr1.0e-3",
    "qppo-Hgog-qlr2.5e-3",
    "random-baseline",
]
seeds = [10, 20, 30]
stepsize = 4 * 128 * 8
max_steps = 1000000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, stepsize, max_steps)

"""
# Hyperparameter Test 3a-outscale:
plot_dir = plots_dir + "/Hyperparameter-Test-3a-outscale_QPPO-LR"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "qppo-simple-outscale-qlr0.1e-3",
    # "qppo-simple-outscale-qlr0.25e-3",
    "qppo-simple-outscale-qlr0.5e-3",
    # "qppo-simple-outscale-qlr1.0e-3",
    "qppo-simple-outscale-qlr2.5e-3",
    "random-baseline",
]
seeds = [10, 20, 30]
stepsize = 4 * 128 * 8
max_steps = 1000000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, stepsize, max_steps)
"""
