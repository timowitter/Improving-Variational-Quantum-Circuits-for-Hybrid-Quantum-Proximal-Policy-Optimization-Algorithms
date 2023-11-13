from src.plot import plot_test_avg_final
from src.plot_grads import plot_gradient_avg, plot_insider_info
from src.plot_old import plot_training_results

results_dir = "qppo-slurm/results"
plots_dir = "plots"
test_plots_dir = "test-plots"


# Test 0 - ppo default:
"""
plot_dir = plots_dir + "/Test_0"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = ["ppo_default_lr2.5e-4"]
seeds = [10, 20, 30, 40, 50]
batchsize = 4*128
max_steps = 300000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)



# Hyperparameter Test 1: find optimal learning rate for ppo with 4-4-NN:

plot_dir = plots_dir + "/Hyperparameter-Test-1_PPO-LR_plot2"
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
    # "ppo_default_lr15.0e-4",
    # "ppo_default_lr25.0e-4",
    "ppo_default_lr50.0e-4",
    "ppo_default_lr100.0e-4",
    # "ppo_default_lr250.0e-4",
]
seeds = [10, 20, 30, 40, 50]
batchsize = 4 * 128 * 4
max_steps = 300000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)

# Hyperparameter Test 2: find optimal learning rate and clip_coef for qppo: (failed due to error in scheduling)


plot_dir = plots_dir + "/Hyperparameter-Test-2_QPPO-LR"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "qppo_Jerbi-no-reuploading-no-input-scaleing_lr-sceduling_qlr0.1e-3",
    "qppo_Jerbi-no-reuploading-no-input-scaleing_lr-sceduling_qlr0.5e-3",
    "qppo_Jerbi-no-reuploading-no-input-scaleing_lr-sceduling_qlr2.5e-3",
]
seeds = [10, 20, 30]
batchsize = 4 * 128 * 8
max_steps = 500000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)


# Hyperparameter Test 2b: find optimal learning rate and clip_coef for qppo: scheduled output scaleing and no clip_grad_norm for circuit parameters (failed due to error in scheduling)
plot_dir = plots_dir + "/Hyperparameter-Test-2b_QPPO-LR"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "qppo_Jerbi-no-reuploading-no-input-scaleing_lrANDoutpScl-sceduling_no-clip-grad-norm_qlr0.1e-3",
    "qppo_Jerbi-no-reuploading-no-input-scaleing_lrANDoutpScl-sceduling_no-clip-grad-norm_qlr0.5e-3",
    "qppo_Jerbi-no-reuploading-no-input-scaleing_lrANDoutpScl-sceduling_no-clip-grad-norm_qlr2.5e-3",
]
seeds = [10, 20]
batchsize = 4 * 128 * 8
max_steps = 500000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)


# Hyperparameter Test 2c: 2b in default Frozen Lake (failed due to error in scheduling)
plot_dir = plots_dir + "/Hyperparameter-Test-2c_QPPO-LR"
gym_id = "FrozenLake-v1"
exp_names = [
    "qppo_Jerbi-no-reuploading-no-input-scaleing_default-FrozenLake_lrANDoutpScl-sceduling_no-clip-grad-norm_qlr0.1e-3",
    "qppo_Jerbi-no-reuploading-no-input-scaleing_default-FrozenLake_lrANDoutpScl-sceduling_no-clip-grad-norm_qlr0.5e-3",
    "qppo_Jerbi-no-reuploading-no-input-scaleing_default-FrozenLake_lrANDoutpScl-sceduling_no-clip-grad-norm_qlr2.5e-3",
]
seeds = [10, 20]
batchsize = 4 * 128 * 8
max_steps = 500000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)



# Hyperparameter Test 2d: find optimal learning rate and clip_coef for qppo: fixed scheduling
plot_dir = plots_dir + "/Hyperparameter-Test-2d_QPPO-LR"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "qppo-Jerbi_1enc_layer_no_input_scaleing-lrANDoutpScl_sceduling-qlr0.1e-3",
    "qppo-Jerbi_1enc_layer_no_input_scaleing-lrANDoutpScl_sceduling-qlr0.5e-3",
    "qppo-Jerbi_1enc_layer_no_input_scaleing-lrANDoutpScl_sceduling-qlr2.5e-3",
]
seeds = [10, 20]
batchsize = 4 * 128 * 8
max_steps = 500000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)



# Hyperparameter Test 2e: find optimal learning rate and clip_coef for qppo: fixed scheduling
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
batchsize = 4 * 128 * 8
max_steps = 500000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)



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
batchsize = 4 * 128 * 8
max_steps = 1000000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)


# Hyperparameter Test 3a-outscale:
plot_dir = plots_dir + "/Hyperparameter-Test-3a-outscale_QPPO-LR"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "qppo-simple-outscale-qlr0.1e-3",
    "qppo-simple-outscale-qlr0.5e-3",
    "qppo-simple-outscale-qlr2.5e-3",
    "random-baseline",
]
seeds = [10, 20, 30]
batchsize = 4 * 128 * 8
max_steps = 1000000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)


# Hyperparameter Test 3b: find minimum number of layers needed for simple circuit to have a good learning performance (without output scaleing)
plot_dir = plots_dir + "/Hyperparameter-Test-3b_QPPO-num-varlayers"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "qppo-simple-qlr0.5e-3",  # (24-params)
    "qppo-simple-qlr0.5e-3-4varlayers-(48-params)",
    "qppo-simple-qlr0.5e-3-6varlayers-(72-params)",
    "qppo-simple-qlr0.5e-3-8varlayers-(96-params)",
    "qppo-simple-qlr0.5e-3-10varlayers-(120-params)",
    "random-baseline",
]
seeds = [10, 20, 30]
batchsize = 4 * 128 * 8
max_steps = 500000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)


plot_dir = plots_dir + "/Hyperparameter-Test-3b_QPPO-with-outscale"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "qppo-simple-qlr0.5e-3-6varlayers-(72-params)",
    "qppo-simple-qlr0.5e-3-6varlayers-outscale-(76-params)",
    "qppo-simple-qlr0.5e-3-6varlayers-outscale-expLRscedule-(76-params)",
    "random-baseline",
]
seeds = [10, 20, 30]
batchsize = 4 * 128 * 8
max_steps = 500000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)



# Hyperparameter Test 3c: test reuploading
plot_dir = plots_dir + "/Hyperparameter-Test-3c_QPPO-reuploading"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "qppo-simple-qlr0.5e-3-6varlayers-(72-params)",
    "qppo-simple_reuploading-qlr0.5e-3-6varlayers-(72-params)",
    "qppo-Hgog-qlr0.5e-3-6varlayers-(72-params)",
    "qppo-Hgog_reuploading-qlr0.5e-3-6varlayers-(72-params)",
    "qppo-Jerbi_reuploading_no_input-scaleing-qlr0.5e-3-9varlayers-(72-params)",
    "random-baseline",
]
seeds = [10, 20, 30]
batchsize = 4 * 128 * 8
max_steps = 500000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)


plot_dir = plots_dir + "/Hyperparameter-Test-3c_QPPO-reuploading-more-steps"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "qppo-simple-qlr0.5e-3-6varlayers-(72-params)",
    "random-baseline",
]
seeds = [10, 20, 30]
batchsize = 4 * 128 * 8
max_steps = 1000000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)

# Hyperparameter Test 3d: test reuploading more layers
plot_dir = plots_dir + "/Hyperparameter-Test-3d_QPPO-reuploading-more-layers"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "qppo-simple_reuploading-qlr0.5e-3-6varlayers-(72-params)",
    "qppo-simple_reuploading-qlr0.5e-3-10varlayers-(120-params)",
    "qppo-simple_reuploading-qlr0.5e-3-15varlayers-(180-params)",
    "random-baseline",
]
seeds = [10, 20, 30]
batchsize = 4 * 128 * 8
max_steps = 200000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)


# Hyperparameter Test 4a: test prob circuit output vs logprob output with and without output scaleing
plot_dir = plots_dir + "/Hyperparameter-Test-4a_QPPO-prob-vs-logprob-output-vs-output-scaleing"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "qppo-simple_reuploading-qlr0.5e-3-6varlayers-(72-params)",
    "qppo-simple_reuploading-qlr0.5e-3-6varlayers-nologoutput-(72-params)",
    "qppo-simple_reuploading-qlr0.5e-3-6varlayers-outscale_scheduling_4.0-(72-params)",
    "random-baseline",
]
seeds = [10, 20, 30]
batchsize = 4 * 128 * 8
max_steps = 200000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)



# Hyperparameter Test 4b: test clipcoef and clipgradnorm
plot_dir = plots_dir + "/Hyperparameter-Test-4b_QPPO-clipcoef-clipgradnorm"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "qppo-simple_reuploading-qlr0.5e-3-6varlayers-(72-params)",
    "qppo-simple_reuploading-qlr0.5e-3-6varlayers-nologoutput-(72-params)",
    "qppo-simple_reuploading-qlr0.5e-3-6varlayers-outscale_scheduling_4.0-(72-params)",
    "qppo-simple_reuploading-qlr0.5e-3-6varlayers-nologoutput-clipped-(72-params)",
    "qppo-simple_reuploading-qlr0.5e-3-6varlayers-outscale_scheduling_2.0-(72-params)",
    "qppo-simple_reuploading-qlr1.0e-3-6varlayers-nologoutput-clipcoef0.02-(72-params)",
    "qppo-simple_reuploading-qlr0.5e-3-6varlayers-nologoutput-clipcircuitgradnorm-(72-params)",
    "random-baseline",
]
seeds = [10, 20, 30]
batchsize = 4 * 128 * 8
max_steps = 200000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)


# Hyperparameter Test 4c: test different parameter initialisation methods and record gradient mean and var

plot_dir = plots_dir + "/Hyperparameter-Test-4c_QPPO-param-init"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "qppo-simple_reuploading-qlr0.5e-3-6varlayers-nologoutput-random-init-recordgrads-(72-params)",
    "qppo-simple_reuploading-qlr0.5e-3-6varlayers-nologoutput-clippedinit-recordgrads-(72-params)",
    "qppo-simple_reuploading-qlr0.5e-3-6varlayers-nologoutput-gaussinit-recordgrads-(72-params)",
    "random-baseline",
]
seeds = [10, 20, 30]
batchsize = 4 * 128 * 8
max_steps = 350000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)

exp_names = [
    "qppo-simple_reuploading-qlr0.5e-3-6varlayers-nologoutput-random-init-recordgrads-(72-params)",
    "qppo-simple_reuploading-qlr0.5e-3-6varlayers-nologoutput-clippedinit-recordgrads-(72-params)",
    "qppo-simple_reuploading-qlr0.5e-3-6varlayers-nologoutput-gaussinit-recordgrads-(72-params)",
]
plot_gradient_avg(results_dir, plot_dir, gym_id, exp_names, seeds, max_steps)


# Hyperparameter Test 4d: test different parameter initialisation methods and record gradient mean and var

plot_dir = plots_dir + "/Hyperparameter-Test-4d_QPPO-param-init"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "qppo-simple_reuploading-qlr0.5e-3-6varlayers-nologoutput-clippedinit-recordgrads-(72-params)",
    "qppo-simple_reuploading-qlr0.5e-3-6varlayers-nologoutput-allsmall-recordgrads-(72-params)",
    "qppo-simple_reuploading-qlr0.5e-3-6varlayers-nologoutput-allmid-recordgrads-(72-params)",
    "qppo-simple_reuploading-qlr0.5e-3-6varlayers-nologoutput-allbig-recordgrads-(72-params)",
    "qppo-simple_reuploading-qlr0.5e-3-6varlayers-nologoutput-allverysmall-recordgrads-(72-params)",
    # "qppo-simple_reuploading-qlr0.5e-3-6varlayers-nologoutput-alltoosmall-recordgrads-(72-params)",
    "random-baseline",
]
seeds = [10, 20, 30]
batchsize = 4 * 128 * 8
max_steps = 500000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)

exp_names = [
    "qppo-simple_reuploading-qlr0.5e-3-6varlayers-nologoutput-clippedinit-recordgrads-(72-params)",
    "qppo-simple_reuploading-qlr0.5e-3-6varlayers-nologoutput-allsmall-recordgrads-(72-params)",
    "qppo-simple_reuploading-qlr0.5e-3-6varlayers-nologoutput-allmid-recordgrads-(72-params)",
    "qppo-simple_reuploading-qlr0.5e-3-6varlayers-nologoutput-allbig-recordgrads-(72-params)",
    "qppo-simple_reuploading-qlr0.5e-3-6varlayers-nologoutput-allverysmall-recordgrads-(72-params)",
    # "qppo-simple_reuploading-qlr0.5e-3-6varlayers-nologoutput-alltoosmall-recordgrads-(72-params)",
]
plot_gradient_avg(results_dir, plot_dir, gym_id, exp_names, seeds, max_steps)


# Hyperparameter Test 4e: test different (higher) learning rates with normal and low clipfracs

plot_dir = plots_dir + "/Hyperparameter-Test-4e_QPPO-lr+clipfrac"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "qppo-simple_reuploading-qlr0.5e-3-6varlayers-nologoutput-allsmall-recordgrads-(72-params)",
    "qppo-simple_reuploading-qlr0.5e-3-lr0.5e-4-clipcoef0.20-6varlayers-nologoutput-allsmall-(72-params)",
    "qppo-simple_reuploading-qlr0.5e-3-lr0.5e-4-clipcoef0.02-6varlayers-nologoutput-allsmall-(72-params)",
    "qppo-simple_reuploading-qlr2.5e-3-lr2.5e-4-clipcoef0.20-6varlayers-nologoutput-allsmall-(72-params)",
    "qppo-simple_reuploading-qlr2.5e-3-lr2.5e-4-clipcoef0.02-6varlayers-nologoutput-allsmall-(72-params)",
    "qppo-simple_reuploading-qlr10.e-3-lr10.e-4-clipcoef0.20-6varlayers-nologoutput-allsmall-(72-params)",
    "qppo-simple_reuploading-qlr10.e-3-lr10.e-4-clipcoef0.02-6varlayers-nologoutput-allsmall-(72-params)",
    #   "qppo-simple_reuploading-qlr50.e-3-lr50.e-4-clipcoef0.20-6varlayers-nologoutput-allsmall-(72-params)",
    "qppo-simple_reuploading-qlr50.e-3-lr50.e-4-clipcoef0.02-6varlayers-nologoutput-allsmall-(72-params)",
    "random-baseline",
]
seeds = [10, 20]
batchsize = 4 * 128 * 8
max_steps = 200000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)

exp_names = [
    "qppo-simple_reuploading-qlr0.5e-3-6varlayers-nologoutput-allsmall-recordgrads-(72-params)",
    "qppo-simple_reuploading-qlr0.5e-3-6varlayers-nologoutput-allsmall-recordgrads-(72-params)",
    "qppo-simple_reuploading-qlr0.5e-3-lr0.5e-4-clipcoef0.20-6varlayers-nologoutput-allsmall-(72-params)",
    "qppo-simple_reuploading-qlr0.5e-3-lr0.5e-4-clipcoef0.02-6varlayers-nologoutput-allsmall-(72-params)",
    "qppo-simple_reuploading-qlr2.5e-3-lr2.5e-4-clipcoef0.20-6varlayers-nologoutput-allsmall-(72-params)",
    "qppo-simple_reuploading-qlr2.5e-3-lr2.5e-4-clipcoef0.02-6varlayers-nologoutput-allsmall-(72-params)",
    "qppo-simple_reuploading-qlr10.e-3-lr10.e-4-clipcoef0.20-6varlayers-nologoutput-allsmall-(72-params)",
    "qppo-simple_reuploading-qlr10.e-3-lr10.e-4-clipcoef0.02-6varlayers-nologoutput-allsmall-(72-params)",
    #   "qppo-simple_reuploading-qlr50.e-3-lr50.e-4-clipcoef0.20-6varlayers-nologoutput-allsmall-(72-params)",
    "qppo-simple_reuploading-qlr50.e-3-lr50.e-4-clipcoef0.02-6varlayers-nologoutput-allsmall-(72-params)",
]
plot_gradient_avg(results_dir, plot_dir, gym_id, exp_names, seeds, max_steps)


# Hyperparameter Test 4f: repeat 4e with fixed transform functions

plot_dir = plots_dir + "/Hyperparameter-Test-4f_QPPO-lr+clipfrac"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "qppo-simple_reuploading-qlr0.5e-3-lr0.5e-4-clipcoef0.20-fixed",
    "qppo-simple_reuploading-qlr0.5e-3-lr0.5e-4-clipcoef0.02-fixed",
    "qppo-simple_reuploading-qlr2.5e-3-lr2.5e-4-clipcoef0.20-fixed",
    "qppo-simple_reuploading-qlr2.5e-3-lr2.5e-4-clipcoef0.02-fixed",
    "qppo-simple_reuploading-qlr10.e-3-lr10.e-4-clipcoef0.20-fixed",
    "qppo-simple_reuploading-qlr10.e-3-lr10.e-4-clipcoef0.02-fixed",
    "qppo-simple_reuploading-qlr50.e-3-lr50.e-4-clipcoef0.02-fixed",
    "random-baseline",
]
seeds = [10, 20]
batchsize = 4 * 128 * 8
max_steps = 200000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)

exp_names = [
    "qppo-simple_reuploading-qlr0.5e-3-lr0.5e-4-clipcoef0.20-fixed",
    "qppo-simple_reuploading-qlr0.5e-3-lr0.5e-4-clipcoef0.02-fixed",
    "qppo-simple_reuploading-qlr2.5e-3-lr2.5e-4-clipcoef0.20-fixed",
    "qppo-simple_reuploading-qlr2.5e-3-lr2.5e-4-clipcoef0.02-fixed",
    "qppo-simple_reuploading-qlr10.e-3-lr10.e-4-clipcoef0.20-fixed",
    "qppo-simple_reuploading-qlr10.e-3-lr10.e-4-clipcoef0.02-fixed",
    "qppo-simple_reuploading-qlr50.e-3-lr50.e-4-clipcoef0.02-fixed",
]
plot_gradient_avg(results_dir, plot_dir, gym_id, exp_names, seeds, max_steps)




# Frozen Lake: Hyperparameter Test 1: Learning Rates

plot_dir = plots_dir + "/FL-Hyperparameter-Test-1_lr_qlr"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-simple_reuploading-qlr2.5e-3-lr2.5e-4-clipcoef0.20-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-qlr5.0e-3-lr2.5e-4-clipcoef0.20-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-qlr10.e-3-lr2.5e-4-clipcoef0.20-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-qlr2.5e-3-lr5.0e-4-clipcoef0.20-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-qlr5.0e-3-lr5.0e-4-clipcoef0.20-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-qlr10.e-3-lr5.0e-4-clipcoef0.20-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-qlr2.5e-3-lr10.e-4-clipcoef0.20-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-qlr5.0e-3-lr10.e-4-clipcoef0.20-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-qlr10.e-3-lr10.e-4-clipcoef0.20-6varlayers-(72-params)",
    "random-baseline",
]
seeds = [10, 20, 30]
batchsize = 4 * 128 * 8
max_steps = 150000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)




plot_dir = plots_dir + "/FL-Hyperparameter-Test-1a_qlr"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-simple_reuploading-qlr0.5e-3-lr2.5e-4-clipcoef0.20-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-qlr2.5e-3-lr2.5e-4-clipcoef0.20-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-qlr5.0e-3-lr2.5e-4-clipcoef0.20-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-qlr10.e-3-lr2.5e-4-clipcoef0.20-6varlayers-(72-params)",
    "random-baseline",
]
seeds = [10, 20, 30]
batchsize = 4 * 128 * 8
max_steps = 150000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)


plot_dir = plots_dir + "/FL-Hyperparameter-Test-1b_qlr"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-simple_reuploading-qlr2.5e-3-lr5.0e-4-clipcoef0.20-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-qlr5.0e-3-lr5.0e-4-clipcoef0.20-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-qlr10.e-3-lr5.0e-4-clipcoef0.20-6varlayers-(72-params)",
    "random-baseline",
]
seeds = [10, 20, 30]
batchsize = 4 * 128 * 8
max_steps = 150000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)


plot_dir = plots_dir + "/FL-Hyperparameter-Test-1c_qlr"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-simple_reuploading-qlr2.5e-3-lr10.e-4-clipcoef0.20-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-qlr5.0e-3-lr10.e-4-clipcoef0.20-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-qlr10.e-3-lr10.e-4-clipcoef0.20-6varlayers-(72-params)",
    "random-baseline",
]
seeds = [10, 20, 30]
batchsize = 4 * 128 * 8
max_steps = 150000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)


plot_dir = plots_dir + "/FL-Hyperparameter-Test-1d_lr"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-simple_reuploading-qlr2.5e-3-lr0.5e-4-clipcoef0.20-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-qlr2.5e-3-lr2.5e-4-clipcoef0.02-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-qlr2.5e-3-lr2.5e-4-clipcoef0.20-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-qlr2.5e-3-lr5.0e-4-clipcoef0.20-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-qlr2.5e-3-lr10.e-4-clipcoef0.20-6varlayers-(72-params)",
    "random-baseline",
]
seeds = [10, 20, 30]
batchsize = 4 * 128 * 8
max_steps = 150000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)


plot_dir = plots_dir + "/FL-Hyperparameter-Test-1e_lr"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-simple_reuploading-qlr5.0e-3-lr2.5e-4-clipcoef0.20-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-qlr5.0e-3-lr5.0e-4-clipcoef0.20-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-qlr5.0e-3-lr10.e-4-clipcoef0.20-6varlayers-(72-params)",
    "random-baseline",
]
seeds = [10, 20, 30]
batchsize = 4 * 128 * 8
max_steps = 150000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)


plot_dir = plots_dir + "/FL-Hyperparameter-Test-1f_lr"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-simple_reuploading-qlr10.e-3-lr2.5e-4-clipcoef0.20-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-qlr10.e-3-lr5.0e-4-clipcoef0.20-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-qlr10.e-3-lr10.e-4-clipcoef0.20-6varlayers-(72-params)",
    "random-baseline",
]
seeds = [10, 20, 30]
batchsize = 4 * 128 * 8
max_steps = 150000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)


plot_dir = plots_dir + "/FL-Hyperparameter-Test-1g_sced_qlr"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-simple_reuploading-qlr2.5e-3-lr2.5e-4-clipcoef0.20-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-qlr2.5e-3-lr0.5e-4-clipcoef0.20-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-qlr2.5e-3-lr2.5e-4-clipcoef0.02-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-qlr0.5e-3-lr2.5e-4-clipcoef0.20-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-lin_sced-start-qlr5.0e-3-end-qlr1.0e-5-lr2.5e-4-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-exp_sced-start-qlr10.e-3-end-qlr0.5e-3-lr2.5e-4-6varlayers-(72-params)",
    "random-baseline",
    "ppo_default_lr100.0e-4",
]
seeds = [10, 20, 30]
batchsize = 4 * 128 * 8
max_steps = 150000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)



plot_dir = plots_dir + "/FL-Hyperparameter-Test-1h_sced_qlr"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-simple_reuploading-qlr2.5e-3-lr2.5e-4-clipcoef0.20-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-lin_sced-start-qlr5.0e-3-end-qlr1.0e-5-lr2.5e-4-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-exp_sced-start-qlr10.e-3-end-qlr0.5e-3-lr2.5e-4-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-lin_sced-start-qlr5.0e-3-end10e4-qlr0.5e-3-lr2.5e-4-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-exp_sced-start-qlr10.e-3-end10e4-qlr0.5e-3-lr2.5e-4-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-lin_sced-start-qlr5.0e-3-end10e4-qlr0.5e-3-lr0.5e-4-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-exp_sced-start-qlr10.e-3-end10e4-qlr0.5e-3-lr0.5e-4-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-double_sced-start-qlr10.e-3-mid05e4-qlr2.5e-3-end15e4-qlr0.5e-3-lr2.5e-4-6varlayers-(72-params)",
    "random-baseline",
    "ppo_default_lr100.0e-4",
]
seeds = [10, 20, 30]
batchsize = 4 * 128 * 8
max_steps = 150000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)



plot_dir = plots_dir + "/FL-Hyperparameter-Test-1i_sced_qlr"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-simple_reuploading-qlr2.5e-3-lr2.5e-4-clipcoef0.20-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-exp_sced-start-qlr10.e-3-end-qlr0.5e-3-lr2.5e-4-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-exp_sced-start-qlr10.e-3-end10e4-qlr0.5e-3-lr2.5e-4-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-exp_sced-start-qlr10.e-3-end50e3-qlr0.5e-3-lr2.5e-4-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-exp_sced-start-qlr10.e-3-end25e3-qlr0.5e-3-lr2.5e-4-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-double_sced-start-qlr10.e-3-mid05e4-qlr2.5e-3-end15e4-qlr0.5e-3-lr2.5e-4-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-double_sced-start-qlr10.e-3-mid25e3-qlr2.5e-3-end15e4-qlr0.5e-3-lr0.5e-4-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-double_sced-start-qlr10.e-3-mid25e3-qlr2.5e-3-end15e4-qlr0.5e-3-lr2.5e-4-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-double_sced-start-qlr10.e-3-mid25e3-qlr2.5e-3-end15e4-qlr0.5e-3-lr10.e-4-6varlayers-(72-params)",
    "random-baseline",
    "ppo_default_lr100.0e-4",
]
seeds = [10, 20, 30]
batchsize = 4 * 128 * 8
max_steps = 150000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)



# Frozen Lake: Actor Hyperparameter Test 1: Learning Rates

plot_dir = plots_dir + "/FL-actor-Hyperparameter-Test-1a-qlr"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-simple_reuploading-qlr0.5e-3-lr2.5e-4-clipcoef0.20-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-qlr2.5e-3-lr2.5e-4-clipcoef0.20-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-qlr5.0e-3-lr2.5e-4-clipcoef0.20-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-qlr10.e-3-lr2.5e-4-clipcoef0.20-6varlayers-(72-params)",
    "random-baseline",
]
seeds = [10, 20, 30]
batchsize = 4 * 128 * 8
max_steps = 150000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)


plot_dir = plots_dir + "/FL-actor-Hyperparameter-Test-1b-lr"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-simple_reuploading-qlr2.5e-3-lr0.5e-4-clipcoef0.20-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-qlr2.5e-3-lr2.5e-4-clipcoef0.02-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-qlr2.5e-3-lr2.5e-4-clipcoef0.20-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-qlr2.5e-3-lr5.0e-4-clipcoef0.20-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-qlr2.5e-3-lr10.e-4-clipcoef0.20-6varlayers-(72-params)",
    "random-baseline",
]
seeds = [10, 20, 30]
batchsize = 4 * 128 * 8
max_steps = 150000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)


plot_dir = plots_dir + "/FL-actor-Hyperparameter-Test-1c-lin-sced"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-simple_reuploading-qlr2.5e-3-lr2.5e-4-clipcoef0.20-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-lin_sced-start-qlr5.0e-3-end-qlr1.0e-5-lr2.5e-4-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-lin_sced-start-qlr5.0e-3-end10e4-qlr0.5e-3-lr2.5e-4-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-lin_sced-start-qlr5.0e-3-end10e4-qlr0.5e-3-lr0.5e-4-6varlayers-(72-params)",
    "random-baseline",
    "ppo_default_lr100.0e-4",
]
seeds = [10, 20, 30]
batchsize = 4 * 128 * 8
max_steps = 150000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)


plot_dir = plots_dir + "/FL-actor-Hyperparameter-Test-1c-sq-sced"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-simple_reuploading-exp_sced-start-qlr10.e-3-end-qlr0.5e-3-lr2.5e-4-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-exp_sced-start-qlr10.e-3-end10e4-qlr0.5e-3-lr2.5e-4-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-exp_sced-start-qlr10.e-3-end50e3-qlr0.5e-3-lr2.5e-4-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-exp_sced-start-qlr10.e-3-end25e3-qlr0.5e-3-lr2.5e-4-6varlayers-(72-params)",
    "random-baseline",
]
seeds = [10, 20, 30]
batchsize = 4 * 128 * 8
max_steps = 150000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)


plot_dir = plots_dir + "/FL-actor-Hyperparameter-Test-1c-double-sced"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-simple_reuploading-qlr2.5e-3-lr2.5e-4-clipcoef0.20-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-exp_sced-start-qlr10.e-3-end10e4-qlr0.5e-3-lr2.5e-4-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-exp_sced-start-qlr10.e-3-end25e3-qlr0.5e-3-lr2.5e-4-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-double_sced-start-qlr10.e-3-mid05e4-qlr2.5e-3-end15e4-qlr0.5e-3-lr2.5e-4-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-double_sced-start-qlr10.e-3-mid25e3-qlr2.5e-3-end15e4-qlr0.5e-3-lr2.5e-4-6varlayers-(72-params)",
    "random-baseline",
    "ppo_default_lr100.0e-4",
]
seeds = [10, 20, 30]
batchsize = 4 * 128 * 8
max_steps = 150000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)


plot_dir = plots_dir + "/FL-actor-Hyperparameter-Test-1d-double-sced-lr"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-simple_reuploading-double_sced-start-qlr10.e-3-mid25e3-qlr2.5e-3-end15e4-qlr0.5e-3-lr0.5e-4-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-double_sced-start-qlr10.e-3-mid25e3-qlr2.5e-3-end15e4-qlr0.5e-3-lr2.5e-4-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-double_sced-start-qlr10.e-3-mid25e3-qlr2.5e-3-end15e4-qlr0.5e-3-lr10.e-4-6varlayers-(72-params)",
    "random-baseline",
]
seeds = [10, 20, 30]
batchsize = 4 * 128 * 8
max_steps = 150000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)



plot_dir = plots_dir + "/FL-actor-Hyperparameter-Test-1e-final-results"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-simple_reuploading-qlr2.5e-3-lr2.5e-4-clipcoef0.20-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-qlr2.5e-3-lr2.5e-4-clipcoef0.02-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-exp_sced-start-qlr10.e-3-end10e4-qlr0.5e-3-lr2.5e-4-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-exp_sced-start-qlr10.e-3-end25e3-qlr0.5e-3-lr2.5e-4-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-exp_sced-start-qlr10.e-3-end25e3-qlr0.5e-3-lr10.e-4-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-double_sced-start-qlr10.e-3-mid05e4-qlr2.5e-3-end15e4-qlr0.5e-3-lr2.5e-4-6varlayers-(72-params)",
    "random-baseline",
    "ppo_default_lr100.0e-4",
]
seeds = [10, 20, 30, 40, 50]
batchsize = 4 * 128 * 8
max_steps = 300000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)

plot_dir = plots_dir + "/FL-actor-Hyperparameter-Test-1c-exp-sced"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-simple_reuploading-qlr2.5e-3-lr2.5e-4-clipcoef0.20-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-exp_sced-start-qlr10.e-3-end10e4-qlr0.5e-3-lr2.5e-4-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-double_sced-start-qlr10.e-3-mid05e4-qlr2.5e-3-end15e4-qlr0.5e-3-lr2.5e-4-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-exp_sced_fixed-ht15000-start-qlr10.e-3-end-qlr0.1e-3-lr2.5e-4-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-exp_sced_fixed-ht25000-start-qlr10.e-3-end-qlr0.1e-3-lr2.5e-4-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-exp_sced_fixed-ht40000-start-qlr10.e-3-end-qlr0.1e-3-lr2.5e-4-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-exp_sced_fixed-ht10000-start-qlr10.e-3-end-qlr1.0e-3-lr2.5e-4-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-exp_sced_fixed-ht20000-start-qlr10.e-3-end-qlr1.0e-3-lr2.5e-4-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-exp_sced_fixed-ht35000-start-qlr10.e-3-end-qlr1.0e-3-lr2.5e-4-6varlayers-(72-params)",
    "random-baseline",
    "ppo_default_lr100.0e-4",
]
seeds = [10, 20, 30, 40, 50]
batchsize = 4 * 128 * 8
max_steps = 150000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)



plot_dir = plots_dir + "/FL-actor-Hyperparameter-Test-2-number-of-layers"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-ac-simple_reuploading-exp_sced-ht25000-start-qlr10.e-3-end-qlr0.1e-3-lr2.5e-4-2varlayers-(24-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-ht25000-start-qlr10.e-3-end-qlr0.1e-3-lr2.5e-4-4varlayers-(48-params)",
    "FL-qppo-simple_reuploading-exp_sced_fixed-ht25000-start-qlr10.e-3-end-qlr0.1e-3-lr2.5e-4-6varlayers-(72-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-ht25000-start-qlr10.e-3-end-qlr0.1e-3-lr2.5e-4-8varlayers-(96-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-ht25000-start-qlr10.e-3-end-qlr0.1e-3-lr2.5e-4-10varlayers-(120-params)",
    "random-baseline",
    "ppo_default_lr100.0e-4",
]
seeds = [10, 20, 30]
batchsize = 4 * 128 * 8
max_steps = 150000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)


plot_dir = plots_dir + "/FL-actor-Hyperparameter-Test-2b-number-of-layers-fixed-qlr"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-ac-simple_reuploading-qlr2.5e-3-lr2.5e-4-2varlayers-(24-params)",
    "FL-qppo-ac-simple_reuploading-qlr2.5e-3-lr2.5e-4-4varlayers-(48-params)",
    "FL-qppo-simple_reuploading-qlr2.5e-3-lr2.5e-4-clipcoef0.20-6varlayers-(72-params)",
    "FL-qppo-ac-simple_reuploading-qlr2.5e-3-lr2.5e-4-8varlayers-(96-params)",
    "random-baseline",
    "ppo_default_lr100.0e-4",
]
seeds = [10, 20, 30]
batchsize = 4 * 128 * 8
max_steps = 150000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)


plot_dir = plots_dir + "/FL-actor-Hyperparameter-Test-3-init"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-simple_reuploading-exp_sced_fixed-ht25000-start-qlr10.e-3-end-qlr0.1e-3-lr2.5e-4-6varlayers-(72-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-ht25000-start-qlr10.e-3-end-qlr0.1e-3-lr2.5e-4-clippedrandominit-6varlayers-(72-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-ht25000-start-qlr10.e-3-end-qlr0.1e-3-lr2.5e-4-gaussinit-6varlayers-(72-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-ht25000-start-qlr10.e-3-end-qlr0.1e-3-lr2.5e-4-allsmallinit-6varlayers-(72-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-ht25000-start-qlr10.e-3-end-qlr0.1e-3-lr2.5e-4-allmidinit-6varlayers-(72-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-ht25000-start-qlr10.e-3-end-qlr0.1e-3-lr2.5e-4-allbiginit-6varlayers-(72-params)",
    "random-baseline",
    "ppo_default_lr100.0e-4",
]
seeds = [10, 20, 30]
batchsize = 4 * 128 * 8
max_steps = 150000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)


plot_dir = plots_dir + "/FL-actor-Test-1-output-scaleing"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-simple_reuploading-exp_sced_fixed-ht25000-start-qlr10.e-3-end-qlr0.1e-3-lr2.5e-4-6varlayers-(72-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-1param-1e-3-(73-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-1param-1e-2-(73-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-1param-5e-2-(73-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-4params-1e-3-(76-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-4params-1e-2-(76-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-4params-5e-2-(76-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-hybrid_output-20params-(92-params)",
    "random-baseline",
]
seeds = [10, 20, 30]
batchsize = 4 * 128 * 8
max_steps = 150000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)


plot_dir = plots_dir + "/FL-actor-Test-1a-output-scaleing-1-param"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-simple_reuploading-exp_sced_fixed-ht25000-start-qlr10.e-3-end-qlr0.1e-3-lr2.5e-4-6varlayers-(72-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-1param-1e-4-(73-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-1param-1e-3-(73-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-1param-1e-2-(73-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-1param-5e-2-(73-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-1param-1e-1-(73-params)",
    "random-baseline",
    "ppo_default_lr100.0e-4",
]
seeds = [10, 20, 30]
batchsize = 4 * 128 * 8
max_steps = 150000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)


plot_dir = plots_dir + "/FL-actor-Test-1a-output-scaleing-4-params"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-simple_reuploading-exp_sced_fixed-ht25000-start-qlr10.e-3-end-qlr0.1e-3-lr2.5e-4-6varlayers-(72-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-4params-1e-4-(76-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-4params-1e-3-(76-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-4params-1e-2-(76-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-4params-5e-2-(76-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-4params-1e-1-(76-params)",
    "random-baseline",
    "ppo_default_lr100.0e-4",
]
seeds = [10, 20, 30]
batchsize = 4 * 128 * 8
max_steps = 150000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)


plot_dir = plots_dir + "/FL-actor-Test-1b-output-scaleing"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-1param-1e-3-(73-params)",
    "FL-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-3-(73-params)",
    "FL-qppo-ac-simple_reuploading-qlr1.0e-3-output_scaleing-1param-1e-3-(73-params)",
    "random-baseline",
    "ppo_default_lr100.0e-4",
]
seeds = [10, 20, 30]
batchsize = 4 * 128 * 8
max_steps = 150000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)





plot_dir = plots_dir + "/FL-actor-Test-1c-output-scaleing-in-alternate-Frozen-Lake-Environment-2"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0-alt"
exp_names = [
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-1param-1e-2-altFLenvforBIAStest-(73-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-4params-5e-2-altFLenvforBIAStest-(76-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-4params-1e-2-altFLenvforBIAStest-(76-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-1param-1e-3-altFLenvforBIAStest-(73-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-4params-1e-3-altFLenvforBIAStest-(76-params)",
]
seeds = [10, 20, 30, 40, 50]
batchsize = 4 * 128 * 8
max_steps = 150000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)
"""


"""
plot_dir = plots_dir + "/FL-actor-Test-1c-NN-PPO-in-alternate-Frozen-Lake-Environment"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0-alt"
exp_names = [
    "FL-alt-qppo-ac-NN(4,4)-(lr=1.0e-4)-(108-params)",
    "FL-alt-qppo-ac-NN(4,4)-(lr=1.0e-3)-(108-params)",
    "FL-alt-qppo-ac-NN(4,4)-(lr=1.0e-2)-(108-params)",
]
seeds = [10, 20, 30, 40, 50]
batchsize = 4 * 128 * 8
max_steps = 150000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)
"""
#    "FL-qppo-ac-NN(4,4)-(lr=1.0e-4)-altFLenvforBIAStest-(108-params)",
#    "FL-qppo-ac-NN(4,4)-(lr=1.0e-3)-altFLenvforBIAStest-(108-params)",
#    "FL-qppo-ac-NN(4,4)-(lr=1.0e-2)-altFLenvforBIAStest-(108-params)",
"""
plot_dir = plots_dir + "/FL-actor-Test-1c-output-scaleing-in-alternate-Frozen-Lake-Environment"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0-alt"
exp_names = [
    "FL-alt-qppo-ac-simple_reuploading-exp_sced-output_scaleing-1param-1e-2-(73-params)",
    "FL-alt-qppo-ac-simple_reuploading-exp_sced-output_scaleing-4params-1e-2-(76-params)",
    "FL-alt-qppo-ac-simple_reuploading-exp_sced-output_scaleing-4params-5e-2-(76-params)",
    "FL-alt-qppo-ac-NN(4,4)-(lr=1.0e-2)-(108-params)",
]
seeds = [10, 20, 30, 40, 50]
batchsize = 4 * 128 * 8
max_steps = 150000
plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)



plot_dir = plots_dir + "/FL-actor-Test-2-reuploading"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-1param-1e-3-(73-params)",
    "FL-qppo-ac-simple-exp_sced-output_scaleing-1param-1e-3-(73-params)",
    "random-baseline",
    "ppo_default_lr100.0e-4",
]
seeds = [10, 20, 30]
batchsize = 4 * 128 * 8
max_steps = 150000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)


plot_dir = plots_dir + "/FL-actor-Test-2-alternate-circuit-architecture"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-1param-1e-2-(73-params)",
    "FL-qppo-ac-Hgog_reuploading-exp_sced-output_scaleing-1param-1e-2-(73-params)",
    "FL-qppo-ac-Jerbi-reuploading-no-input-scaleing-exp_sced-output_scaleing-1param-1e-2-9var_8enc_layers(73-params)",
    "random-baseline",
    "ppo_default_lr100.0e-4",
]
seeds = [10, 20, 30, 40, 50]
batchsize = 4 * 128 * 8
max_steps = 50000


plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)





# Cartpole hyperparam tests

plot_dir = plots_dir + "/CP-actor-Hyperparameter-Test-0-PPO-NN(6,6)-lr"
gym_id = "CartPole-v1"
exp_names = [
    "CP-ppo-ac-NN(6,6)-(lr=1.0e-5)-(86-params)",
    "CP-ppo-ac-NN(6,6)-(lr=1.0e-4)-(86-params)",
    "CP-ppo-ac-NN(6,6)-(lr=1.0e-3)-(86-params)",
    "CP-ppo-ac-NN(6,6)-(lr=2.5e-3)-(86-params)",
    "CP-ppo-ac-NN(6,6)-(lr=0.5e-2)-(86-params)",
    "CP-ppo-ac-NN(6,6)-(lr=1.0e-2)-(86-params)",
    "CP-ppo-ac-NN(6,6)-(lr=2.5e-2)-(86-params)",
]
seeds = [10, 20, 30, 40, 50]
batchsize = 4 * 128 * 8
max_steps = 500000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)


plot_dir = plots_dir + "/CP-actor-Hyperparameter-Test-0-PPO-NN(6,6)-lr-plot2"
gym_id = "CartPole-v1"
exp_names = [
    "CP-ppo-ac-NN(6,6)-(lr=1.0e-4)-(86-params)",
    "CP-ppo-ac-NN(6,6)-(lr=1.0e-3)-(86-params)",
    "CP-ppo-ac-NN(6,6)-(lr=2.5e-3)-(86-params)",
    "CP-ppo-ac-NN(6,6)-(lr=0.5e-2)-(86-params)",
    "CP-ppo-ac-NN(6,6)-(lr=1.0e-2)-(86-params)",
]
seeds = [10, 20, 30, 40, 50]
batchsize = 4 * 128 * 8
max_steps = 150000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)


plot_dir = plots_dir + "/CP-actor-Hyperparameter-Test-0b-PPO-lr-NN-size"
gym_id = "CartPole-v1"
exp_names = [
    "CP-ppo-ac-NN(5,5)-(lr=1.0e-3)-(67-params)",
    "CP-ppo-ac-NN(5,5)-(lr=2.5e-3)-(67-params)",
    "CP-ppo-ac-NN(6,6)-(lr=1.0e-3)-(86-params)",
    "CP-ppo-ac-NN(6,6)-(lr=2.5e-3)-(86-params)",
    "CP-ppo-ac-NN(7,7)-(lr=1.0e-3)-(107-params)",
    "CP-ppo-ac-NN(7,7)-(lr=2.5e-3)-(107-params)",
]
seeds = [10, 20, 30, 40, 50]
batchsize = 4 * 128 * 8
max_steps = 150000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)


plot_dir = plots_dir + "/CP-actor-Hyperparameter-Test-00-insider_info"
gym_id = "CartPole-v1"
exp_names = ["CP-ppo-ac-NN(6,6)-(lr=2.5e-3)-(86-params)-record-insider-info"]
seeds = [10, 20, 30, 40, 50]
batchsize = 4 * 128 * 8
max_steps = 150000

# plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, batchsize, max_steps)
plot_insider_info(results_dir, plot_dir, gym_id, exp_names, seeds, max_steps)


plot_dir = plots_dir + "/CP-actor-Hyperparameter-Test-1a-qlr-no_output_scale"
gym_id = "CartPole-v1"
exp_names = [
    "CP-qppo-ac-simple_reuploading-exp_sced-ht15000-start-qlr1e-2-end-qlr1e-4-(72-params)",
    "CP-qppo-ac-simple_reuploading-exp_sced-ht25000-start-qlr1e-2-end-qlr1e-4-(72-params)",
    "CP-qppo-ac-simple_reuploading-exp_sced-ht35000-start-qlr1e-2-end-qlr1e-4-(72-params)",
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-(72-params)",
    "CP-qppo-ac-simple_reuploading-qlr1.0e-2-(72-params)",
    "CP-random-baseline-(0-params)",
    "CP-ppo-ac-NN(6,6)-(lr=1.0e-3)-(86-params)",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 500000
alpha = 0.1

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)


plot_dir = plots_dir + "/CP-actor-Hyperparameter-Test-1b-qlr-no_output_scale2"
gym_id = "CartPole-v1"
exp_names = [
    "CP-qppo-ac-simple_reuploading-exp_sced-ht15000-start-qlr1e-2-end-qlr1e-4-(72-params)",
    "CP-qppo-ac-simple_reuploading-exp_sced-ht25000-start-qlr1e-2-end-qlr1e-4-(72-params)",
    "CP-qppo-ac-simple_reuploading-exp_sced-ht35000-start-qlr1e-2-end-qlr1e-4-(72-params)",
    "CP-qppo-ac-simple_reuploading-exp_sced-ht15000-start-qlr1e-1-end-qlr1e-3-(72-params)",
    "CP-qppo-ac-simple_reuploading-exp_sced-ht25000-start-qlr1e-1-end-qlr1e-3-(72-params)",
    "CP-qppo-ac-simple_reuploading-exp_sced-ht35000-start-qlr1e-1-end-qlr1e-3-(72-params)",
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-(72-params)",
    "CP-random-baseline-(0-params)",
    "CP-ppo-ac-NN(6,6)-(lr=1.0e-3)-(86-params)",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.05

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)


plot_dir = plots_dir + "/CP-actor-Hyperparameter-Test-1c-qlr-no_output_scale3"
gym_id = "CartPole-v1"
exp_names = [
    "CP-qppo-ac-simple_reuploading-qlr0.5e-3-(72-params)",
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-(72-params)",
    "CP-qppo-ac-simple_reuploading-qlr1.0e-2-(72-params)",
    "CP-qppo-ac-simple_reuploading-qlr5.0e-2-(72-params)",
    "CP-random-baseline-(0-params)",
    "CP-ppo-ac-NN(6,6)-(lr=1.0e-3)-(86-params)",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.1

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)


plot_dir = plots_dir + "/CP-actor-Hyperparameter-Test-2a-outscale-lr"
gym_id = "CartPole-v1"
exp_names = [
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-(72-params)",
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-5-(73-params)",
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)",
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-3-(73-params)",
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-2-(73-params)",
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-1-(73-params)",
    "CP-random-baseline-(0-params)",
    "CP-ppo-ac-NN(6,6)-(lr=1.0e-3)-(86-params)",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 500000
alpha = 0.1

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)


plot_dir = plots_dir + "/CP-actor-Hyperparameter-Test-2b-qlr_x_outscale-lr"
gym_id = "CartPole-v1"
exp_names = [
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)",
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-3-(73-params)",
    "CP-qppo-ac-simple_reuploading-qlr5.0e-3-output_scaleing-1param-1e-3-(73-params)",
    "CP-qppo-ac-simple_reuploading-qlr1.0e-2-output_scaleing-1param-1e-3-(73-params)",
    "CP-qppo-ac-simple_reuploading-qlr1.0e-2-output_scaleing-1param-1e-2-(73-params)",
    "CP-qppo-ac-simple_reuploading-qlr5.0e-2-output_scaleing-1param-1e-3-(73-params)",
    "CP-qppo-ac-simple_reuploading-qlr5.0e-2-output_scaleing-1param-1e-2-(73-params)",
    "CP-random-baseline-(0-params)",
    "CP-ppo-ac-NN(6,6)-(lr=1.0e-3)-(86-params)",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 500000
alpha = 0.05

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)


plot_dir = plots_dir + "/CP-actor-Hyperparameter-Test-2c-outscale-lr-fixedlastentanglement"
gym_id = "CartPole-v1"
exp_names = [
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)",
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-3-(73-params)",
    "CP-qppo-ac-simple_reuploading-qlr5.0e-3-output_scaleing-1param-1e-3-(73-params)",
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)-fixedlastentanglement",
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-3-(73-params)-fixedlastentanglement",
    "CP-qppo-ac-simple_reuploading-qlr5.0e-3-output_scaleing-1param-1e-3-(73-params)-fixedlastentanglement",
    "CP-random-baseline-(0-params)",
    "CP-ppo-ac-NN(6,6)-(lr=1.0e-3)-(86-params)",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 500000
alpha = 0.05

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)
# for the tests before this one, circular entanglement was used for the last layer instead of onely entangeling the 1st and 3th Qubit plus the 2nd an 4th Qubit


plot_dir = plots_dir + "/CP-actor-Hyperparameter-Test-3a-random-inits"
gym_id = "CartPole-v1"
exp_names = [
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)-fixedlastentanglement",  # fully radom init
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)-clippedrandominit",
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)-allverysmall",
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)-allsmallinit",
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)-allmidinit",
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)-allbiginit",
    "CP-random-baseline-(0-params)",
    "CP-ppo-ac-NN(6,6)-(lr=1.0e-3)-(86-params)",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.05

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)


plot_dir = plots_dir + "/CP-actor-Hyperparameter-Test-3b-gaussinits"
gym_id = "CartPole-v1"
exp_names = [
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)-fixedlastentanglement",
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)-unclipped_gaussinit",
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)-gaussinit",
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)-rescaled_gaussinit",
    "CP-random-baseline-(0-params)",
    "CP-ppo-ac-NN(6,6)-(lr=1.0e-3)-(86-params)",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.05


plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)

plot_dir = plots_dir + "/CP-actor-Hyperparameter-Test-3c-best-inits"
gym_id = "CartPole-v1"
exp_names = [
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)-fixedlastentanglement",
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)-gaussinit",
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)-allsmallinit",
    "CP-random-baseline-(0-params)",
    "CP-ppo-ac-NN(6,6)-(lr=1.0e-3)-(86-params)",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 500000
alpha = 0.1

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)


plot_dir = plots_dir + "/CP-actor-Hyperparameter-Test-4a-qlr"
gym_id = "CartPole-v1"
exp_names = [
    "CP-qppo-ac-simple_reuploading-qlr0.5e-3-output_scaleing-1param-1e-4-(73-params)-allsmallinit",
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)-allsmallinit",
    "CP-qppo-ac-simple_reuploading-qlr10.e-3-output_scaleing-1param-1e-4-(73-params)-allsmallinit",
    "CP-qppo-ac-simple_reuploading-exp_sced-ht_75000-start-qlr1e-2-end-qlr1e-4-output_scaleing-1param-1e-4-(73-params)-allsmallinit",
    "CP-qppo-ac-simple_reuploading-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-(73-params)-allsmallinit",
    "CP-qppo-ac-simple_reuploading-exp_sced-ht150000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-(73-params)-allsmallinit",
    "CP-random-baseline-(0-params)",
    "CP-ppo-ac-NN(6,6)-(lr=1.0e-3)-(86-params)",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 500000
alpha = 0.05

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)


plot_dir = plots_dir + "/CP-actor-Hyperparameter-Test-4b-qlr"
gym_id = "CartPole-v1"
exp_names = [
    "CP-qppo-ac-simple_reuploading-qlr0.5e-3-output_scaleing-1param-1e-4-(73-params)-allsmallinit",
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)-allsmallinit",
    "CP-qppo-ac-simple_reuploading-exp_sced-ht_50000-start-qlr2.5e-3-end-qlr1e-5-output_scaleing-1param-1e-4-(73-params)-allsmallinit",
    "CP-qppo-ac-simple_reuploading-exp_sced-ht_50000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-(73-params)-allsmallinit",
    "CP-qppo-ac-simple_reuploading-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-(73-params)-allsmallinit",
    "CP-qppo-ac-simple_reuploading-exp_sced-ht150000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-(73-params)-allsmallinit",
    "CP-random-baseline-(0-params)",
    "CP-ppo-ac-NN(6,6)-(lr=1.0e-3)-(86-params)",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.05

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)
"""
# "CP-qppo-ac-simple_reuploading-exp_sced-ht100000-start-qlr1e-2-end-qlr1e-4-output_scaleing-1param-1e-4-(73-params)-allsmallinit", #canceled ~350000 steps
# "CP-qppo-ac-simple_reuploading-exp_sced-ht150000-start-qlr1e-2-end-qlr1e-4-output_scaleing-1param-1e-4-(73-params)-allsmallinit", #canceled ~350000 steps
# "CP-qppo-ac-simple_reuploading-qlr1.5e-3-output_scaleing-1param-1e-4-(73-params)-allsmallinit",                                   #150000 steps

"""
plot_dir = plots_dir + "/CP-actor-Test-1a-input-scaleing"
gym_id = "CartPole-v1"
exp_names = [
    "CP-qppo-ac-simple_reuploading-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-(73-params)-allsmallinit",
    "CP-qppo-ac-simple_reuploading-insider-input-rescale-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(73-params)",
    "CP-qppo-ac-simple_reuploading-insider-input-rescale-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(97-params)-8-layers",
    "CP-qppo-ac-simple_reuploading_with_shared_input_scaleing-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(77-params)",
    "CP-qppo-ac-simple_reuploading_with_input_scaleing-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(81-params)-5-layers",
    "CP-qppo-ac-simple_reuploading_with_input_scaleing-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(97-params)",
    "CP-random-baseline-(0-params)",
    "CP-ppo-ac-NN(6,6)-(lr=1.0e-3)-(86-params)",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.05

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)


plot_dir = plots_dir + "/CP-actor-Hyperparameter-Test-5a-insider-input-rescaleing-qlrs"
gym_id = "CartPole-v1"
exp_names = [
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)-fixedlastentanglement",
    # "CP-qppo-ac-simple_reuploading-qlr0.5e-3-output_scaleing-1param-1e-4-(73-params)-allsmallinit",
    # "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)-allsmallinit",
    # "CP-qppo-ac-simple_reuploading-insider-input-rescale-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(73-params)",
    "CP-qppo-ac-simple_reuploading-insider-input-rescale-qlr0.5e-3-output_scaleing-1param-1e-4-(73-params)",
    "CP-qppo-ac-simple_reuploading-insider-input-rescale-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)",
    "CP-qppo-ac-simple_reuploading-insider-input-rescale-qlr10.e-3-output_scaleing-1param-1e-4-(73-params)",
    "CP-random-baseline-(0-params)",
    "CP-ppo-ac-NN(6,6)-(lr=1.0e-3)-(86-params)",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.05

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)


plot_dir = plots_dir + "/CP-actor-Hyperparameter-Test-5b-shared-input-scaleing-qlrs"
gym_id = "CartPole-v1"
exp_names = [
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)-fixedlastentanglement",
    # "CP-qppo-ac-simple_reuploading-qlr0.5e-3-output_scaleing-1param-1e-4-(73-params)-allsmallinit",
    # "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)-allsmallinit",
    # "CP-qppo-ac-simple_reuploading_with_shared_input_scaleing-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(77-params)",
    "CP-qppo-ac-simple_reuploading_with_shared_input_scaleing-qlr0.5e-3-output_scaleing-1param-1e-4-(77-params)",
    "CP-qppo-ac-simple_reuploading_with_shared_input_scaleing-qlr2.5e-3-output_scaleing-1param-1e-4-(77-params)",
    "CP-qppo-ac-simple_reuploading_with_shared_input_scaleing-qlr10.e-3-output_scaleing-1param-1e-4-(77-params)",
    "CP-random-baseline-(0-params)",
    "CP-ppo-ac-NN(6,6)-(lr=1.0e-3)-(86-params)",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.05

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)


plot_dir = plots_dir + "/CP-actor-Hyperparameter-Test-5c-input-scaleing-qlrs"
gym_id = "CartPole-v1"
exp_names = [
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)-fixedlastentanglement",
    # "CP-qppo-ac-simple_reuploading-qlr0.5e-3-output_scaleing-1param-1e-4-(73-params)-allsmallinit",
    # "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)-allsmallinit",
    # "CP-qppo-ac-simple_reuploading_with_input_scaleing-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(97-params)",
    "CP-qppo-ac-simple_reuploading_with_input_scaleing-qlr0.5e-3-output_scaleing-1param-1e-4-(97-params)",
    "CP-qppo-ac-simple_reuploading_with_input_scaleing-qlr2.5e-3-output_scaleing-1param-1e-4-(97-params)",
    "CP-qppo-ac-simple_reuploading_with_input_scaleing-qlr10.e-3-output_scaleing-1param-1e-4-(97-params)",
    "CP-random-baseline-(0-params)",
    "CP-ppo-ac-NN(6,6)-(lr=1.0e-3)-(86-params)",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.05

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)


plot_dir = plots_dir + "/CP-actor-Hyperparameter-Test-5a-insider-input-rescaleing-qlrs"
gym_id = "CartPole-v1"
exp_names = [
    "CP-qppo-ac-simple_reuploading-qlr0.5e-3-output_scaleing-1param-1e-4-(73-params)-allsmallinit",
    "CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)-allsmallinit",
    "CP-qppo-ac-simple_reuploading-insider-input-rescale2-qlr0.5e-3-output_scaleing-1param-1e-4-allsmallinit-(73-params)",
    "CP-qppo-ac-simple_reuploading-insider-input-rescale2-qlr2.5e-3-output_scaleing-1param-1e-4-allsmallinit-(73-params)",
    "CP-qppo-ac-simple_reuploading-insider-input-rescale2-qlr10.e-3-output_scaleing-1param-1e-4-allsmallinit-(73-params)",
    "CP-random-baseline-(0-params)",
    "CP-ppo-ac-NN(6,6)-(lr=1.0e-3)-(86-params)",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.05

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)


######useless tests


plot_dir = plots_dir + "/CP-actor-Test-1a-input-scaleing"
gym_id = "CartPole-v1"
exp_names = [
    "CP-qppo-ac-simple_reuploading-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-(73-params)-allsmallinit",
    "CP-qppo-ac-simple_reuploading-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(97-params)-8-layers",
    "CP-qppo-ac-simple_reuploading_with_shared_input_scaleing-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(77-params)",
    "CP-qppo-ac-simple_reuploading_with_input_scaleing-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(81-params)-5-layers",
    "CP-qppo-ac-simple_reuploading_with_input_scaleing-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(97-params)",
    "CP-random-baseline-(0-params)",
    "CP-ppo-ac-NN(6,6)-(lr=1.0e-3)-(86-params)",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.05

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)

""""""
plot_dir = plots_dir + "/CP-actor-Test-1b-input-rescaleings"
gym_id = "CartPole-v1"
exp_names = [
    "CP-qppo-ac-simple_reuploading-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-(73-params)-allsmallinit", # tanh(x)
    "CP-qppo-ac-simple_reuploading-tanh-x2-rescale-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(73-params)",  # tanh(2x)
    #"CP-qppo-ac-simple_reuploading-insider-input-rescale-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(73-params)",  # 10% clip
    # "CP-qppo-ac-simple_reuploading-insider-input-rescale-qlr0.5e-3-output_scaleing-1param-1e-4-(73-params)",
    #"CP-qppo-ac-simple_reuploading-insider-input-rescale-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)",  # 10%clip
    # "CP-qppo-ac-simple_reuploading-insider-input-rescale-qlr10.e-3-output_scaleing-1param-1e-4-(73-params)",
    # "CP-qppo-ac-simple_reuploading-insider-input-rescale2-qlr0.5e-3-output_scaleing-1param-1e-4-allsmallinit-(73-params)",
    #"CP-qppo-ac-simple_reuploading-insider-input-rescale2-qlr2.5e-3-output_scaleing-1param-1e-4-allsmallinit-(73-params)",  # 25%clip
    # "CP-qppo-ac-simple_reuploading-insider-input-rescale2-qlr10.e-3-output_scaleing-1param-1e-4-allsmallinit-(73-params)",
    "CP-qppo-ac-simple_reuploading-arctan-x1-rescale-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(73-params)",  # 2*arctan(x)
    "CP-qppo-ac-simple_reuploading-arctan-rescale-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(73-params)",  # 2*arctan(2x)
    
    
    "CP-random-baseline-(0-params)",
    "CP-ppo-ac-NN(6,6)-(lr=1.0e-3)-(86-params)",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.05

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)

""""""
plot_dir = plots_dir + "/CP-actor-Test-1c-input-scaleing"
gym_id = "CartPole-v1"
exp_names = [
    "CP-qppo-ac-simple_reuploading-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-(73-params)-allsmallinit",
    "CP-qppo-ac-simple_reuploading_with_input_scaleing-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(65-params)-4-layers",
    "CP-qppo-ac-simple_reuploading_with_input_scaleing-exp_sced-ht_80000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(81-params)-5-layers",
    "CP-qppo-ac-simple_reuploading_with_input_scaleing-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(81-params)-5-layers",
    "CP-qppo-ac-simple_reuploading_with_input_scaleing-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(97-params)",
    "CP-qppo-ac-simple_reuploading_with_input_scaleing-exp_sced-ht100000-start-qlr1.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(97-params)",
    "CP-random-baseline-(0-params)",
    "CP-ppo-ac-NN(6,6)-(lr=1.0e-3)-(86-params)",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.05

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)


plot_dir = plots_dir + "/CP-actor-Test-1d-input-scaleing-500000-Steps"
gym_id = "CartPole-v1"
exp_names = [
    "CP-qppo-ac-simple_reuploading-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-(73-params)-allsmallinit",
    "CP-qppo-ac-simple_reuploading_with_shared_input_scaleing-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(77-params)",
    # "CP-qppo-ac-simple_reuploading_with_input_scaleing-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(65-params)-4-layers",
    # "CP-qppo-ac-simple_reuploading_with_input_scaleing-exp_sced-ht_80000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(81-params)-5-layers",
    "CP-qppo-ac-simple_reuploading_with_input_scaleing-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(81-params)-5-layers",
    # "CP-qppo-ac-simple_reuploading_with_input_scaleing-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(97-params)",
    # "CP-qppo-ac-simple_reuploading_with_input_scaleing-exp_sced-ht100000-start-qlr1.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(97-params)",
    "CP-random-baseline-(0-params)",
    "CP-ppo-ac-NN(6,6)-(lr=1.0e-3)-(86-params)",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 500000
alpha = 0.05

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)



plot_dir = plots_dir + "/FL-actor-Hyperparameter-Test-0-NN(3)-lr"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-ppo-ac-NN(3)-(actor-lr=1.0e-3)-(67-params)",
    "FL-ppo-ac-NN(3)-(actor-lr=1.0e-2)-(67-params)",
    "FL-ppo-ac-NN(3)-(actor-lr=5.0e-2)-(67-params)",
    "FL-ppo-ac-NN(3)-(actor-lr=1.0e-1)-(67-params)",
    "random-baseline",
]
seeds = [10, 20, 30, 40, 50]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.1

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)


plot_dir = plots_dir + "/FL-actor-Hyperparameter-Test-0-NN(4)-lr"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-ppo-ac-NN(4)-(actor-lr=1.0e-3)-(88-params)",
    "FL-ppo-ac-NN(4)-(actor-lr=1.0e-2)-(88-params)",
    "FL-ppo-ac-NN(4)-(actor-lr=5.0e-2)-(88-params)",
    "FL-ppo-ac-NN(4)-(actor-lr=1.0e-1)-(88-params)",
    "random-baseline",
]
seeds = [10, 20, 30, 40, 50]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.1

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)


plot_dir = plots_dir + "/FL-actor-Hyperparameter-Test-0-NN(5)-lr"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-ppo-ac-NN(5)-(actor-lr=1.0e-3)-(109-params)",
    "FL-ppo-ac-NN(5)-(actor-lr=1.0e-2)-(109-params)",
    "FL-ppo-ac-NN(5)-(actor-lr=5.0e-2)-(109-params)",
    "FL-ppo-ac-NN(5)-(actor-lr=1.0e-1)-(109-params)",
    "random-baseline",
]
seeds = [10, 20, 30, 40, 50]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.1

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)


plot_dir = plots_dir + "/FL-actor-Hyperparameter-Test-0-NN(4,4)-lr"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-ppo-ac-NN(4,4)-(actor-lr=1.0e-3)-(108-params)",
    "FL-ppo-ac-NN(4,4)-(actor-lr=1.0e-2)-(108-params)",
    "FL-ppo-ac-NN(4,4)-(actor-lr=2.0e-2)-(108-params)",
    "FL-ppo-ac-NN(4,4)-(actor-lr=5.0e-2)-(108-params)",
    "random-baseline",
]
seeds = [10, 20, 30, 40, 50]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.1

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)


plot_dir = plots_dir + "/FL-actor-Hyperparameter-Test-0-NN(64,64)-lr"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    # "FL-ppo-ac-NN(64,64)-(actor-lr=5.0e-5)-(5508-params)",
    "FL-ppo-ac-NN(64,64)-(actor-lr=2.5e-4)-(5508-params)",
    "FL-ppo-ac-NN(64,64)-(actor-lr=1.0e-3)-(5508-params)",
    "FL-ppo-ac-NN(64,64)-(actor-lr=2.5e-3)-(5508-params)",
    "FL-ppo-ac-NN(64,64)-(actor-lr=5.0e-3)-(5508-params)",
    "FL-ppo-ac-NN(64,64)-(actor-lr=1.0e-2)-(5508-params)",
    "random-baseline",
]
seeds = [10, 20, 30, 40, 50]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.1

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)


plot_dir = plots_dir + "/FL-actor-Hyperparameter-Test-0b-best-NN"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-ppo-ac-NN(3)-(actor-lr=5.0e-2)-(67-params)",
    "FL-ppo-ac-NN(4)-(actor-lr=5.0e-2)-(88-params)",
    "FL-ppo-ac-NN(5)-(actor-lr=5.0e-2)-(109-params)",
    "FL-ppo-ac-NN(4,4)-(actor-lr=2.0e-2)-(108-params)",
    "FL-ppo-ac-NN(64,64)-(actor-lr=2.5e-3)-(5508-params)",
    "random-baseline",
    "ppo_default_lr100.0e-4",
]
seeds = [10, 20, 30, 40, 50]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.1

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)


plot_dir = plots_dir + "/FL-actor-Hyperparameter-Test-3-init"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-ac-simple_reuploading-exp_sced-ht25000-10->0.1e-3-output_scaleing-1e-3-(73-params)-randominit",
    "FL-qppo-ac-simple_reuploading-exp_sced-ht25000-10->0.1e-3-output_scaleing-1e-3-(73-params)-gaussinit",
    "FL-qppo-ac-simple_reuploading-exp_sced-ht25000-10->0.1e-3-output_scaleing-1e-3-(73-params)-allsmallinit",
    "FL-qppo-ac-simple_reuploading-exp_sced-ht25000-10->0.1e-3-output_scaleing-1e-3-(73-params)-allmidinit",
    "FL-qppo-ac-simple_reuploading-exp_sced-ht25000-10->0.1e-3-output_scaleing-1e-3-(73-params)-allbiginit",
    "random-baseline",
    "FL-ppo-ac-NN(3)-(actor-lr=5.0e-2)-(67-params)",
    "FL-ppo-ac-NN(4)-(actor-lr=5.0e-2)-(88-params)",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.05

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)

name1="FL-qppo-ac-simple_reuploading-exp_sced-ht25000-10->0.1e-3-output_scaleing-5e-3-(73-params)"
name2="FL-qppo-ac-simple-exp_sced-ht25000-10->0.1e-3-output_scaleing-5e-3-(73-params)"
name3="FL-qppo-ac-simple_reuploading-exp_sced-ht25000-10->0.1e-3-(73-params)"
name4="FL-qppo-ac-simple_reuploading-qlr-2.5e-3-output_scaleing-5e-3-(73-params)"

#############################################################

plot_dir = plots_dir + "/CP-actor-Hyperparameter-Test-0-NN(5,5)-lr"
gym_id = "CartPole-v1"
exp_names = [
    "CP-ppo-ac-NN(5,5)-(actor-lr=5.0e-5)-(67-params)",
    "CP-ppo-ac-NN(5,5)-(actor-lr=1.0e-4)-(67-params)",
    "CP-ppo-ac-NN(5,5)-(actor-lr=2.5e-4)-(67-params)",
    "CP-ppo-ac-NN(5,5)-(actor-lr=1.0e-3)-(67-params)",
    "CP-ppo-ac-NN(5,5)-(actor-lr=1.0e-2)-(67-params)",
    "CP-random-baseline-(0-params)",
]
seeds = [10, 20, 30, 40, 50]
batchsize = 4 * 128
max_steps = 500000
alpha = 0.1

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)


plot_dir = plots_dir + "/CP-actor-Hyperparameter-Test-0-NN(6,5)-lr"
gym_id = "CartPole-v1"
exp_names = [
    "CP-ppo-ac-NN(6,5)-(actor-lr=5.0e-5)-(77-params)",
    "CP-ppo-ac-NN(6,5)-(actor-lr=1.0e-4)-(77-params)",
    "CP-ppo-ac-NN(6,5)-(actor-lr=2.5e-4)-(77-params)",
    "CP-ppo-ac-NN(6,5)-(actor-lr=1.0e-3)-(77-params)",
    "CP-ppo-ac-NN(6,5)-(actor-lr=1.0e-2)-(77-params)",
    "CP-random-baseline-(0-params)",
]
seeds = [10, 20, 30, 40, 50]
batchsize = 4 * 128
max_steps = 500000
alpha = 0.1

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)


plot_dir = plots_dir + "/CP-actor-Hyperparameter-Test-0-NN(6,6)-lr"
gym_id = "CartPole-v1"
exp_names = [
    "CP-ppo-ac-NN(6,6)-(actor-lr=5.0e-5)-(86-params)",
    "CP-ppo-ac-NN(6,6)-(actor-lr=1.0e-4)-(86-params)",
    "CP-ppo-ac-NN(6,6)-(actor-lr=2.5e-4)-(86-params)",
    "CP-ppo-ac-NN(6,6)-(actor-lr=1.0e-3)-(86-params)",
    "CP-ppo-ac-NN(6,6)-(actor-lr=1.0e-2)-(86-params)",
    "CP-random-baseline-(0-params)",
]
seeds = [10, 20, 30, 40, 50]
batchsize = 4 * 128
max_steps = 500000
alpha = 0.1

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)


plot_dir = plots_dir + "/CP-actor-Hyperparameter-Test-0-NN(7,7)-lr"
gym_id = "CartPole-v1"
exp_names = [
    "CP-ppo-ac-NN(7,7)-(actor-lr=5.0e-5)-(107-params)",
    "CP-ppo-ac-NN(7,7)-(actor-lr=1.0e-4)-(107-params)",
    "CP-ppo-ac-NN(7,7)-(actor-lr=2.5e-4)-(107-params)",
    "CP-ppo-ac-NN(7,7)-(actor-lr=1.0e-3)-(107-params)",
    "CP-ppo-ac-NN(7,7)-(actor-lr=1.0e-2)-(107-params)",
    "CP-random-baseline-(0-params)",
]
seeds = [10, 20, 30, 40, 50]
batchsize = 4 * 128
max_steps = 500000
alpha = 0.1

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)


plot_dir = plots_dir + "/CP-actor-Hyperparameter-Test-0-NN(64,64)-lr"
gym_id = "CartPole-v1"
exp_names = [
    "CP-ppo-ac-NN(64,64)-(actor-lr=5.0e-5)-(4610-params)",
    "CP-ppo-ac-NN(64,64)-(actor-lr=1.0e-4)-(4610-params)",
    "CP-ppo-ac-NN(64,64)-(actor-lr=2.5e-4)-(4610-params)",
    "CP-ppo-ac-NN(64,64)-(actor-lr=1.0e-3)-(4610-params)",
    "CP-ppo-ac-NN(64,64)-(actor-lr=5.0e-3)-(4610-params)",
    "CP-random-baseline-(0-params)",
]
seeds = [10, 20, 30, 40, 50]
batchsize = 4 * 128
max_steps = 500000
alpha = 0.1

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)





plot_dir = plots_dir + "/CP-actor-Test-2-Ansatz-Comparison"
gym_id = "CartPole-v1"
exp_names = [
    "CP-qppo-ac-simple_reuploading-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-(73-params)-allsmallinit",
    "CP-qppo-ac-simple_reuploading----------------exp_sced-ht_80000-qlr(25-1)e-4-OutScale(1e-3)-allsmallinit-(73-params)",
    "CP-qppo-ac-simple_reuploading_sharedInpScale-exp_sced-ht_80000-qlr(25-1)e-4-OutScale(1e-3)-allsmallinit-(77-params)",
    "CP-qppo-ac-simple_reuploading_Input_Scaleing-exp_sced-ht_80000-qlr(25-1)e-4-OutScale(1e-3)-allsmallinit-(65-params)-4-layers",
    "CP-qppo-ac-simple_reuploading_Input_Scaleing-exp_sced-ht_80000-qlr(25-1)e-4-OutScale(1e-3)-allsmallinit-(81-params)-5-layers",
    "CP-qppo-ac-simple_reuploading_Input_Scaleing-exp_sced-ht_80000-qlr(10-0.5)e-4-OutScale(1e-3)-allsmallinit-(97-params)",
    "CP-random-baseline-(0-params)",
    "CP-ppo-ac-NN(5,5)-(actor-lr=1.0e-4)-(67-params)",
    "CP-ppo-ac-NN(6,5)-(actor-lr=1.0e-4)-(77-params)",
    "CP-ppo-ac-NN(6,6)-(actor-lr=1.0e-4)-(86-params)",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.05
labels=["QPPO(6-Layer)-KeinInpScale-Out(1e-4)-(73-Param)", "QPPO(6-Layer)-KeinInpScale-(73-Param)", "QPPO(6-Layer)-GeteiltesInpScale-(77-Param)", "QPPO(4-Layer)-InpScale-(65-Param)", "QPPO(5-Layer)-InpScale-(81-Param)", "QPPO(6-Layer)-InpScale-(96-Param)", "Zufällige Aktionsauswahl", "PPO(5,5)-(67-Param)", "PPO(6,5)-(77-Param)", "PPO(6,6)-(86-Param)"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)
"""

plot_dir = plots_dir + "/CP-actor-Test-2b-Ansatz-Comparison"
gym_id = "CartPole-v1"
exp_names = [
    "CP-qppo-ac-simple_reuploading-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-(73-params)-allsmallinit",
    "CP-qppo-ac-simple_reuploading----------------exp_sced-ht_80000-qlr(25-1)e-4-OutScale(5e-4)-allsmallinit-(73-params)",
    "CP-qppo-ac-simple_reuploading_sharedInpScale-exp_sced-ht_80000-qlr(25-1)e-4-OutScale(5e-4)-allsmallinit-(77-params)",
    "CP-qppo-ac-simple_reuploading_Input_Scaleing-exp_sced-ht_80000-qlr(25-1)e-4-OutScale(5e-4)-allsmallinit-(65-params)-4-layers",
    "CP-qppo-ac-simple_reuploading_Input_Scaleing-exp_sced-ht_80000-qlr(25-1)e-4-OutScale(5e-4)-allsmallinit-(81-params)-5-layers",
    "CP-qppo-ac-simple_reuploading_Input_Scaleing-exp_sced-ht_80000-qlr(10-0.5)e-4-OutScale(5e-4)-allsmallinit-(97-params)",
    "CP-random-baseline-(0-params)",
    "CP-ppo-ac-NN(5,5)-(actor-lr=1.0e-4)-(67-params)",
    "CP-ppo-ac-NN(6,5)-(actor-lr=1.0e-4)-(77-params)",
    "CP-ppo-ac-NN(6,6)-(actor-lr=1.0e-4)-(86-params)",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.05
labels=["QPPO(6-Layer)-ManuellesIRS-Out(1e-4)-(73-Param)", "QPPO(6-Layer)-ManuellesIRS-(73-Param)", "QPPO(6-Layer)-GeteiltesInpScale-(77-Param)", "QPPO(4-Layer)-InpScale-(65-Param)", "QPPO(5-Layer)-InpScale-(81-Param)", "QPPO(6-Layer)-InpScale-(96-Param)", "Zufällige Aktionsauswahl", "PPO(5,5)-(67-Param)", "PPO(6,5)-(77-Param)", "PPO(6,6)-(86-Param)"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)



plot_dir = plots_dir + "/CP-actor-Test-2c-Ansatz-Comparison"
gym_id = "CartPole-v1"
exp_names = [
    "CP-qppo-ac-simple_reuploading-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-(73-params)-allsmallinit",
    "CP-qppo-ac-simple_reuploading----------------exp_sced-ht100000-qlr(25-1)e-4-OutScale(1e-3)-allsmallinit-(73-params)-circularLE",
    "CP-qppo-ac-simple_reuploading_sharedInpScale-exp_sced-ht100000-qlr(25-1)e-4-OutScale(1e-3)-allsmallinit-(77-params)-circularLE",
    "CP-qppo-ac-simple_reuploading_Input_Scaleing-exp_sced-ht_80000-qlr(25-1)e-4-OutScale(1e-3)-allsmallinit-(65-params)-circularLE-4layers",
    "CP-qppo-ac-simple_reuploading_Input_Scaleing-exp_sced-ht_80000-qlr(25-1)e-4-OutScale(1e-3)-allsmallinit-(81-params)-circularLE-5layers",
    "CP-qppo-ac-simple_reuploading_Input_Scaleing-exp_sced-ht_80000-qlr(10-0.5)e-4-OutScale(1e-3)-allsmallinit-(97-params)-circularLE",
    "CP-random-baseline-(0-params)",
    "CP-ppo-ac-NN(5,5)-(actor-lr=1.0e-4)-(67-params)",
    "CP-ppo-ac-NN(6,5)-(actor-lr=1.0e-4)-(77-params)",
    "CP-ppo-ac-NN(6,6)-(actor-lr=1.0e-4)-(86-params)",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.05
labels=["QPPO(6-Layer)-ManuellesIRS-Out(1e-4)-(73-Param)", "QPPO(6-Layer)-ManuellesIRS-(73-Param)", "QPPO(6-Layer)-GeteiltesInpScale-(77-Param)", "QPPO(4-Layer)-InpScale-(65-Param)", "QPPO(5-Layer)-InpScale-(81-Param)", "QPPO(6-Layer)-InpScale-(96-Param)", "Zufällige Aktionsauswahl", "PPO(5,5)-(67-Param)", "PPO(6,5)-(77-Param)", "PPO(6,6)-(86-Param)"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)