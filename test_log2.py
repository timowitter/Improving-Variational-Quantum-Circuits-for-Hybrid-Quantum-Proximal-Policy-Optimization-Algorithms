from src.plot import plot_test_avg_final
from src.plot_grads import plot_gradient_avg, plot_insider_info
from src.plot_old import plot_training_results

results_dir = "qppo-slurm/results"
plots_dir = "final-plots"
test_plots_dir = "test-plots"
"""
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
labels=["QPPO-Lr(0.5e-3)", "QPPO-Lr(2.5e-3)", "QPPO-Lr(5.0e-3)", "QPPO-Lr(10.0e-3)", "Zufällige Aktionsauswahl"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)


# We are makeing the average over all seeds of the average episode reward collected in an update epoch of 512 Timesteps of one seed in the Frozen Lake environment, 
# plot it vs the timesteps and smooth it with the exponentially weighted moveing average (alpha = 0.3). 
#We tested an VQC with 6 encodeing and 6 variational layers (72 parameters) with lernrates 
# (qlrs) of 0.5, 2.5, 5.0 and 10.0 (* e-3) for the actors and in all cases a NN with two hidden layers with 64 Nodes (5313 parameters) and an lernrate of 2.4 e-4 for 
# the critic. No output scaleing and no softmax are used for the circuit output.
# The qlr of 2.5e-3 works best, the higher lernrates have a faster start, but fail to converge later. The smallest lr is very slow, and seems to converge prematurely.




plot_dir = plots_dir + "/FL-actor-Hyperparameter-Test-1b-lr"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-simple_reuploading-qlr2.5e-3-lr0.5e-4-clipcoef0.20-6varlayers-(72-params)",
    #"FL-qppo-simple_reuploading-qlr2.5e-3-lr2.5e-4-clipcoef0.02-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-qlr2.5e-3-lr2.5e-4-clipcoef0.20-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-qlr2.5e-3-lr5.0e-4-clipcoef0.20-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-qlr2.5e-3-lr10.e-4-clipcoef0.20-6varlayers-(72-params)",
    "random-baseline",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.3
labels=["QPPO-Critic-Lr(0.5e-4)", "QPPO-Critic-Lr(2.5e-4)", "QPPO-Critic-Lr(5.0e-4)", "QPPO-Critic-Lr(10.0e-4)", "Zufällige Aktionsauswahl"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)

# We are makeing the average over all seeds of the average episode reward collected in an update epoch of 512 Timesteps of one seed in the Frozen Lake environment, 
# plot it vs the timesteps and smooth it with the exponentially weighted moveing average (alpha = 0.3). 
# We tested an VQC with 6 encodeing and 6 variational layers (72 parameters) 
# with an lernrate of 2.5e-3 for the actor and a NN with two hidden layers with 64 Nodes (5313 parameters) and with lernrates (lrs) of 0.5, 2.5, 5.0 and 10.0 (* e-4) 
# for the critic. No output scaleing and no softmax are used for the circuit output. 
# The default value of 2.5e-4 and the slightly higher value of 5.0e-4 get the best results. After this test we decide to choose the default lr of 2.5e-4
# and not to do any further tests on the critic lr, since it seems to work fine with the dafault lr and the number of tests we can run is limited.



plot_dir = plots_dir + "/FL-actor-Hyperparameter-Test-1c-exp-sced"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-simple_reuploading-exp_sced_fixed-ht15000-start-qlr10.e-3-end-qlr0.1e-3-lr2.5e-4-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-exp_sced_fixed-ht25000-start-qlr10.e-3-end-qlr0.1e-3-lr2.5e-4-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-exp_sced_fixed-ht40000-start-qlr10.e-3-end-qlr0.1e-3-lr2.5e-4-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-exp_sced_fixed-ht10000-start-qlr10.e-3-end-qlr1.0e-3-lr2.5e-4-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-exp_sced_fixed-ht20000-start-qlr10.e-3-end-qlr1.0e-3-lr2.5e-4-6varlayers-(72-params)",
    "FL-qppo-simple_reuploading-exp_sced_fixed-ht35000-start-qlr10.e-3-end-qlr1.0e-3-lr2.5e-4-6varlayers-(72-params)",
    "random-baseline",
    "FL-qppo-simple_reuploading-qlr2.5e-3-lr2.5e-4-clipcoef0.20-6varlayers-(72-params)",
    "FL-ppo-ac-NN(3)-(actor-lr=5.0e-2)-(67-params)",
    "FL-ppo-ac-NN(4)-(actor-lr=5.0e-2)-(88-params)",
]
seeds = [10, 20, 30, 40, 50]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.3
labels=["QPPO-Lr((10->0.1)*e-3, HWZ=15000)-(72-Param)", "QPPO-Lr((10->0.1)*e-3, HWZ=25000)-(72-Param)", "QPPO-Lr((10->0.1)*e-3, HWZ=40000)-(72-Param)", "QPPO-Lr((10->1.0)*e-3, HWZ=10000)-(72-Param)", "QPPO-Lr((10->1.0)*e-3, HWZ=20000)-(72-Param)", "QPPO-Lr((10->1.0)*e-3, HWZ=35000)-(72-Param)", "Zufällige Aktionsauswahl", "QPPO-Lr(2.5e-3)-(72-Param)", "PPO(3)-Lr(5.0e-2)-(67-Param)", "PPO(4)-Lr(5.0e-2)-(88-Param)"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)


# We are makeing the average over all seeds of the average episode reward collected in an update epoch of 512 Timesteps of one seed in the Frozen Lake environment, 
# plot it vs the timesteps and smooth it with the exponentially weighted moveing average (alpha = 0.3). 
# We tested QPPO with a VQC with 6 encodeing and 6 variational layers 
# (72 parameters) and with a fixed lernrate of 2.5e-3, an exponentially declineing lernrate from 10.0 to 0.1 (*e-3) or 10.0 to 1.0 (*e-3) with halftimes of 
# 15000, 25000, 40000 or 10000, 20000, 35000 for the actor respectively. We compared it to 2 PPOs with a NN with one hidden Layer with 3 (67 parameters) or 
# 4 (88 parameters) Nodes (and a lr of 5e-2) for the actor. All algotithms used a NN with two hidden layers and 64 Nodes (5313 parameters) and a lernrate of 2.5 * e-4 for the critic. 
# No output scaleing and no softmax are used for the circuit output.
# The falloff from 10 to 0.1 e-3 with a halftime of 25000 or 40000 achieves the best results. The smallest haftimes seem to fail in 1 of 3 seeds for an unknown 
# reason. The smaller fallof intervals of 10 to 1 e-3 work similar to the bigger ones in the early phase of leaning, but score significantly worse later.



plot_dir = plots_dir + "/FL-actor-Hyperparameter-Test-3-init"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-ac-simple_reuploading-exp_sced-ht25000-10->0.1e-3-output_scaleing-1e-3-(73-params)-randominit",
    "FL-qppo-ac-simple_reuploading-exp_sced-ht25000-10->0.1e-3-output_scaleing-1e-3-(73-params)-allsmallinit",
    "FL-qppo-ac-simple_reuploading-exp_sced-ht25000-10->0.1e-3-output_scaleing-1e-3-(73-params)-allmidinit",
    "FL-qppo-ac-simple_reuploading-exp_sced-ht25000-10->0.1e-3-output_scaleing-1e-3-(73-params)-allbiginit",
    "FL-qppo-ac-simple_reuploading-exp_sced-ht25000-10->0.1e-3-output_scaleing-1e-3-(73-params)-gaussinit",
    "random-baseline",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.3
labels=["QPPO-Standard-Init", "QPPO-Kleine-Init", "QPPO-Mittlere-Init", "QPPO-Große-Init", "QPPO-Gauß-Init", "Zufällige Aktionsauswahl"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)

# We are makeing the average over all seeds of the average episode reward collected in an update epoch of 512 Timesteps of one seed in the Frozen Lake environment, plot it vs the 
# timesteps and smooth it with the exponentially weighted moveing average (alpha = 0.3). 
# We used QPPO with a VQC with 6 encodeing and 6 variational layers and output scaleing (lr = 1e-3) (73 parameters) and an exponentially declineing lernrate from 10.0 to 0.1 (*e-3)
# with a halftime of 25000 timesteps for the actor. 
# The VQC was tested with different initialisations that are the arctanh of values drawn from a purely random distribution in the Intervals [-1,1], [-0.11, -0.01] || [0.01, 0.11], 
# [-0.75, -0.25] || [0.25, 0.75], [-0.99, -0.59] || [0.59, 0.99] or drawn from the gauss distribution with a mean of 0 and a Standard deviation of 1.
# All algotithms used a NN with two hidden layers and 64 Nodes (5313 parameters) and a lernrate of 2.5 * e-4 for the critic. 
#



plot_dir = plots_dir + "/FL-actor-Test-1-output-scaleing"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-1param-1e-3-(73-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-1param-1e-2-(73-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-1param-5e-2-(73-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-4params-1e-3-(76-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-4params-1e-2-(76-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-4params-5e-2-(76-params)",
    #"FL-qppo-ac-simple_reuploading-exp_sced-hybrid_output-20params-(92-params)",
    "random-baseline",
    "FL-qppo-simple_reuploading-exp_sced_fixed-ht25000-start-qlr10.e-3-end-qlr0.1e-3-lr2.5e-4-6varlayers-(72-params)",
    "FL-ppo-ac-NN(3)-(actor-lr=5.0e-2)-(67-params)",
    "FL-ppo-ac-NN(4)-(actor-lr=5.0e-2)-(88-params)",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.3
labels=["QPPO-OutScale(1e-3)-(73-Param)", "QPPO-OutScale(1e-2)-(73-Param)", "QPPO-OutScale(5e-2)-(73-Param)", "QPPO-4ParamOutScale(1e-3)-(76-Param)", "QPPO-4ParamOutScale(1e-2)-(76-Param)", "QPPO-4ParamOutScale(5e-2)-(76-Param)", "Zufällige Aktionsauswahl", "QPPO-keinOutScale-(72-Param)", "PPO(3)-Lr(5.0e-2)-(67-Param)", "PPO(4)-Lr(5.0e-2)-(88-Param)"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)


# We are makeing the average over all seeds of the average episode reward collected in an update epoch of 512 Timesteps of one seed in the Frozen Lake environment, plot it vs the 
# timesteps and smooth it with the exponentially weighted moveing average (alpha = 0.3). 
# We used QPPO with a VQC with 6 encodeing and 6 variational layers (73 parameters), an exponentially declineing lernrate from 10.0 to 0.1 (*e-3) with a halftimes of 25000 timesteps
# and test it with two types of output scaleing, with 1 and 4 Parameters, with lerning rates of 1e-3, 1e-2, 5e-2 for the actor.
# We compare it to an VQC without output scaleing and 2 PPOs with a NN with one hidden Layer with 3 (67 parameters) or 
# 4 (88 parameters) Nodes (and a lr of 5e-2) for the actor respectively.
# All Algorithms used a NN with two hidden layers and 64 Nodes (5313 parameters) and a lernrate of 2.5e-4 for the critic. 



plot_dir = plots_dir + "/FL-actor-Test-1a-output-scaleing-1-param"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-1param-1e-4-(73-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-1param-1e-3-(73-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-1param-1e-2-(73-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-1param-5e-2-(73-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-1param-1e-1-(73-params)",
    "random-baseline",
    "FL-qppo-simple_reuploading-exp_sced_fixed-ht25000-start-qlr10.e-3-end-qlr0.1e-3-lr2.5e-4-6varlayers-(72-params)",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.3
labels=["QPPO-OutScale(1e-4)-(73-Param)", "QPPO-OutScale(1e-3)-(73-Param)", "QPPO-OutScale(1e-2)-(73-Param)", "QPPO-OutScale(5e-2)-(73-Param)", "QPPO-OutScale(1e-1)-(73-Param)", "Zufällige Aktionsauswahl", "QPPO-keinOutScale-(72-Param)"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)

# We are makeing the average over all seeds of the average episode reward collected in an update epoch of 512 Timesteps of one seed in the Frozen Lake environment, plot it vs the 
# timesteps and smooth it with the exponentially weighted moveing average (alpha = 0.3). 
# We used QPPO with a VQC with 6 encodeing and 6 variational layers (73 parameters), an exponentially declineing lernrate from (10.0 to 0.1)e-3 with a halftimes of 25000 timesteps
# and test it with 1 parameter output scaleing, with lerning rates of 1e-4, 1e-3, 1e-2, 5e-2, 1e-1 for the actor.
# We compare it to an VQC without output scaleing.
# All QPPOs used a NN with two hidden layers and 64 Nodes (5313 parameters) and a lernrate of 2.5e-4 for the critic. 




plot_dir = plots_dir + "/FL-actor-Test-1a-output-scaleing-4-params"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-4params-1e-4-(76-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-4params-1e-3-(76-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-4params-1e-2-(76-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-4params-5e-2-(76-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-4params-1e-1-(76-params)",
    "random-baseline",
    "FL-qppo-simple_reuploading-exp_sced_fixed-ht25000-start-qlr10.e-3-end-qlr0.1e-3-lr2.5e-4-6varlayers-(72-params)",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.3
labels=["QPPO-4ParamOutScale(1e-4)-(76-Param)", "QPPO-4ParamOutScale(1e-3)-(76-Param)", "QPPO-4ParamOutScale(1e-2)-(76-Param)", "QPPO-4ParamOutScale(5e-2)-(76-Param)", "QPPO-4ParamOutScale(1e-1)-(76-Param)", "Zufällige Aktionsauswahl", "QPPO-keinOutScale-(72-Param)"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)

# We are makeing the average over all seeds of the average episode reward collected in an update epoch of 512 Timesteps of one seed in the Frozen Lake environment, plot it vs the 
# timesteps and smooth it with the exponentially weighted moveing average (alpha = 0.3). 
# We used QPPO with a VQC with 6 encodeing and 6 variational layers (76 parameters), an exponentially declineing lernrate from (10.0 to 0.1)e-3 with a halftimes of 25000 timesteps
# and test it with 4 parameter output scaleing, with lerning rates of 1e-4, 1e-3, 1e-2, 5e-2, 1e-1 for the actor.
# We compare it to an VQC without output scaleing.
# All QPPOs used a NN with two hidden layers and 64 Nodes (5313 parameters) and a lernrate of 2.5e-4 for the critic. 


plot_dir = plots_dir + "/FL-actor-Test-1c-output-scaleing-in-alternate-Frozen-Lake-Environment"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0-alt"
exp_names = [
    "FL-alt-qppo-ac-simple_reuploading-exp_sced-output_scaleing-1param-1e-2-(73-params)",
    "FL-alt-qppo-ac-simple_reuploading-exp_sced-output_scaleing-4params-1e-2-(76-params)",
    "FL-alt-qppo-ac-simple_reuploading-exp_sced-output_scaleing-4params-5e-2-(76-params)",
    "FL-alt-qppo-ac-NN(4,4)-(lr=1.0e-2)-(108-params)",
]
seeds = [10, 20, 30, 40, 50]
batchsize = 4 * 128
max_steps = 100000
alpha = 0.3
labels=["QPPO-OutScale(1e-2)-(73-Param)", "QPPO-4ParamOutScale(1e-2)-(76-Param)", "QPPO-4ParamOutScale(5e-2)-(76-Param)", "PPO(4,4)-Lr(1.0e-2)-(108-Param)"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)
# Tests in alternate-Frozen-Lake-Environment discontinued
"""

plot_dir = plots_dir + "/FL-actor-Test-2-reuploading"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-ac-simple_reuploading-exp_sced-ht25000-10->0.1e-3-output_scaleing-5e-3-(73-params)",
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-1param-1e-3-(73-params)",
    "FL-qppo-ac-simple-exp_sced-ht25000-10->0.1e-3-output_scaleing-5e-3-(73-params)",
    "FL-qppo-ac-simple-exp_sced-output_scaleing-1param-1e-3-(73-params)",
    "random-baseline",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.3
labels=["QPPO-Data-Reuploading-OutScale(5e-3)-(73-Param)", "QPPO-Data-Reuploading-OutScale(1e-3)-(73-Param)", "QPPO-ohne-Reuploading-OutScale(5e-3)-(73-Param)", "QPPO-ohne-Reuploading-OutScale(1e-3)-(73-Param)", "Zufällige Aktionsauswahl"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)

"""

plot_dir = plots_dir + "/FL-actor-Test-2-alternate-circuit-architecture"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-ac-simple_reuploading-exp_sced-output_scaleing-1param-1e-2-(73-params)",
    "FL-qppo-ac-Hgog_reuploading-exp_sced-output_scaleing-1param-1e-2-(73-params)",
    "FL-qppo-ac-Jerbi-reuploading-no-input-scaleing-exp_sced-output_scaleing-1param-1e-2-9var_8enc_layers(73-params)",
    "random-baseline",
    "FL-ppo-ac-NN(3)-(actor-lr=5.0e-2)-(67-params)",
    "FL-ppo-ac-NN(4)-(actor-lr=5.0e-2)-(88-params)",
]
seeds = [10, 20, 30, 40, 50]
batchsize = 4 * 128
max_steps = 50000
alpha = 0.3
labels=["QPPO-Standard-(73-Param)", "QPPO-Hgog+Reuploading-(73-Param)", "QPPO-Jerbi-(73-Param)", "Zufällige Aktionsauswahl", "PPO(3)-(67-Param)", "PPO(4)-(88-Param)"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)
"""


plot_dir = plots_dir + "/FL-actor-Test-3-Ansatz-comparison"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = [
    "FL-qppo-ac-simple_reuploading-exp_sced-ht25000-10->0.1e-3-output_scaleing-5e-3-(73-params)",
    #"FL-qppo-ac-simple-exp_sced-ht25000-10->0.1e-3-output_scaleing-5e-3-(73-params)",
    "FL-qppo-ac-simple-exp_sced-output_scaleing-1param-1e-3-(73-params)",               #lower OutScale lr (1e-3) since it fails for 5e-3
    "FL-qppo-ac-simple_reuploading-exp_sced-ht25000-10->0.1e-3-(73-params)",
    "FL-qppo-ac-simple_reuploading-qlr-2.5e-3-output_scaleing-5e-3-(73-params)",
    "random-baseline",
    "FL-ppo-ac-NN(3)-(actor-lr=5.0e-2)-(67-params)",
    "FL-ppo-ac-NN(4)-(actor-lr=5.0e-2)-(88-params)",
]
seeds = [10, 20, 30]
batchsize = 4 * 128
max_steps = 150000
alpha = 0.3
labels=["QPPO-Reuploading+OutScale+Scheduling-(73-Param)", "QPPO-ohne-Reuploading-(73-Param)", "QPPO-ohne-OutScale-(72-Param)", "QPPO-ohne-ExpLrScheduling-(73-Param)", "Zufällige Aktionsauswahl", "PPO(3)-(67-Param)", "PPO(4)-(88-Param)"]

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps, labels)

"""
"""

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
batchsize = 4 * 128
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
batchsize = 4 * 128
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
batchsize = 4 * 128
max_steps = 150000
alpha = 0.05

plot_test_avg_final(results_dir, plot_dir, gym_id, exp_names, seeds, alpha, max_steps)


plot_dir = plots_dir + "/CP-actor-Hyperparameter-Test-00-insider_info"
gym_id = "CartPole-v1"
exp_names = ["CP-ppo-ac-NN(6,6)-(lr=2.5e-3)-(86-params)-record-insider-info"]
seeds = [10, 20, 30, 40, 50]
batchsize = 4 * 128
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
"""