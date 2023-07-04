from src.plot import plot_test_avg

results_dir = "qppo-slurm/results"
plots_dir = "final-plots"

#Test 0 - ppo default:
"""
name="ppo_default_lr2.5e-4"
start_seed=10
seed_step=10
end_seed=50
envs=("Deterministic-ShortestPath-4x4-FrozenLake-v0")
circuits=("no_q_circuit")

for env in ${envs[@]}; do
    for circuit in ${circuits[@]}; do
        for seed in $(seq $start_seed $seed_step $end_seed); do
            sbatch --job-name="run-$env-$name-$seed" jobs/job.sh --exp-name $name --circuit $circuit --seed $seed --gym-id $env --num-steps 128 --total-timesteps 300000 --learning-rate 2.5e-4 --qlearning-rate 0 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer-nodes 4 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False
        done
    done
done
"""
#Test 0 plotting:
"""
plot_dir = plots_dir + "/Test_0"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = ["ppo_default_lr2.5e-4"]
seeds = [10, 20, 30, 40, 50]
stepsize = 4*128
max_steps = 300000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, stepsize, max_steps)
"""



#Hyperparameter Test 1: ppo with different learning rates
"""
name="ppo_default_lr3.0e-4"
start_seed=10
seed_step=10
end_seed=50
envs=("Deterministic-ShortestPath-4x4-FrozenLake-v0") #"CartPole-v1"
circuits=("no_q_circuit")               #"simple" / "Hgog" / "Jerbi-no-reuploading-no-input-scaleing" /" Jerbi-reuploading" / "Jerbi-reuploading-no-input-scaleing"

for env in ${envs[@]}; do
    for circuit in ${circuits[@]}; do
        for seed in $(seq $start_seed $seed_step $end_seed); do
            sbatch --job-name="run-$env-$name-$seed" jobs/job.sh --exp-name $name --circuit $circuit --seed $seed --gym-id $env --num-steps 128 --total-timesteps 300000 --learning-rate 3.0e-4 --qlearning-rate 0 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer-nodes 4 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False
        done
    done
done

name="ppo_default_lr2.0e-4"
start_seed=10
seed_step=10
end_seed=50
envs=("Deterministic-ShortestPath-4x4-FrozenLake-v0") #"CartPole-v1"
circuits=("no_q_circuit")               #"simple" / "Hgog" / "Jerbi-no-reuploading-no-input-scaleing" /" Jerbi-reuploading" / "Jerbi-reuploading-no-input-scaleing"

for env in ${envs[@]}; do
    for circuit in ${circuits[@]}; do
        for seed in $(seq $start_seed $seed_step $end_seed); do
            sbatch --job-name="run-$env-$name-$seed" jobs/job.sh --exp-name $name --circuit $circuit --seed $seed --gym-id $env --num-steps 128 --total-timesteps 300000 --learning-rate 2.0e-4 --qlearning-rate 0 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer-nodes 4 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False
        done
    done
done
"""
plot_dir = plots_dir + "/Hyperparameter-test_1"
gym_id = "Deterministic-ShortestPath-4x4-FrozenLake-v0"
exp_names = ["ppo_default_lr2.0e-4", "ppo_default_lr2.5e-4", "ppo_default_lr3.0e-4"]
seeds = [10, 20, 30, 40, 50]
stepsize = 4*128
max_steps = 300000

plot_test_avg(results_dir, plot_dir, gym_id, exp_names, seeds, stepsize, max_steps)

