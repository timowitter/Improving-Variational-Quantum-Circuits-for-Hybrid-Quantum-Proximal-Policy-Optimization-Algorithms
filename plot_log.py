from src.plot import plot_test_avg_final
from src.plot_grads import plot_gradient_avg
from src.plot_old import plot_training_results

results_dir = "qppo-slurm/results"
plots_dir = "plots"



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
batchsize = 4 * 128
max_steps = 150000
alpha = 0.3
labels=["Lr=0.5e-3", "Lr=2.5e-3", "Lr=5.0e-3", "Lr=10.0e-3", "Zufällige Aktionsauswahl"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)

