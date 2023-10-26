from src.plot import plot_test_avg_final
from src.plot_grads import plot_gradient_avg, plot_insider_info
from src.plot_old import plot_training_results

results_dir = "qppo-slurm/results"
plots_dir = "plots"
test_plots_dir = "test-plots"

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
alpha = 0.05

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)


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
alpha = 0.05

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)


plot_dir = plots_dir + "/FL-actor-Hyperparameter-Test-1c-exp-sced"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-simple_reuploading-qlr2.5e-3-lr2.5e-4-clipcoef0.20-6varlayers-(72-params)",
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
alpha = 0.05

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)


#
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
alpha = 0.05

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)


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
alpha = 0.05

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)


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
alpha = 0.05

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)


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
alpha = 0.05

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)


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
alpha = 0.05

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)
"""


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
alpha = 0.05

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)


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
alpha = 0.05

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)


# Cartpole hyperparam tests
"""
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
alpha = 0.05

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)


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
alpha = 0.05

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)


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
alpha = 0.05

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)


plot_dir = plots_dir + "/CP-actor-Hyperparameter-Test-00-insider_info"
gym_id = "CartPole-v1"
exp_names = ["CP-ppo-ac-NN(6,6)-(lr=2.5e-3)-(86-params)-record-insider-info"]
seeds = [10, 20, 30, 40, 50]
batchsize = 4 * 128 * 8
max_steps = 150000
alpha = 0.05

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)
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
alpha = 0.05

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)




plot_dir = plots_dir + "/CP-actor-Hyperparameter-Test-1c-qlr-no_output_scale2"
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
alpha = 0.05

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)
"""

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
alpha = 0.05

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)


"""
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
"""

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
alpha = 0.05

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


# input scaleing

plot_dir = plots_dir + "/CP-actor-Test-1a-input-scaleing"
gym_id = "CartPole-v1"
exp_names = [
    "CP-qppo-ac-simple_reuploading-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-(73-params)-allsmallinit",
    "CP-qppo-ac-simple_reuploading-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(97-params)-8-layers",
    "CP-qppo-ac-simple_reuploading-arctan-rescale-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(73-params)",
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


plot_dir = plots_dir + "/CP-actor-Test-1b-input-rescaleings"
gym_id = "CartPole-v1"
exp_names = [
    "CP-qppo-ac-simple_reuploading-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-(73-params)-allsmallinit",
    "CP-qppo-ac-simple_reuploading-insider-input-rescale-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(73-params)",  # 10% clip
    # "CP-qppo-ac-simple_reuploading-insider-input-rescale-qlr0.5e-3-output_scaleing-1param-1e-4-(73-params)",
    "CP-qppo-ac-simple_reuploading-insider-input-rescale-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)",  # 10%clip
    # "CP-qppo-ac-simple_reuploading-insider-input-rescale-qlr10.e-3-output_scaleing-1param-1e-4-(73-params)",
    # "CP-qppo-ac-simple_reuploading-insider-input-rescale2-qlr0.5e-3-output_scaleing-1param-1e-4-allsmallinit-(73-params)",
    "CP-qppo-ac-simple_reuploading-insider-input-rescale2-qlr2.5e-3-output_scaleing-1param-1e-4-allsmallinit-(73-params)",  # 25%clip
    # "CP-qppo-ac-simple_reuploading-insider-input-rescale2-qlr10.e-3-output_scaleing-1param-1e-4-allsmallinit-(73-params)",
    "CP-qppo-ac-simple_reuploading-arctan-rescale-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(73-params)",  # 2*arctan(2x)
    "CP-qppo-ac-simple_reuploading-tanh-x2-rescale-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(73-params)",  # tanh(2x)
    "CP-random-baseline-(0-params)",
    "CP-ppo-ac-NN(6,6)-(lr=1.0e-3)-(86-params)",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.05

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)