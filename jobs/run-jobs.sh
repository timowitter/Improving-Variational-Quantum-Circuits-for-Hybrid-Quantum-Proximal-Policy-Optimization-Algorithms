#!/bin/bash

# Specify hyperparameters
# circuits: "no_q_circuit" / "simple" / "simple_reuploading" / "simple_reuploading_with_input_scaleing" /
#          "Hgog" / "Hgog_reuploading" / "Hgog_reuploading_with_input_scaleing" /
#          "Jerbi-no-reuploading-no-input-scaleing" / "Jerbi-reuploading-no-input-scaleing" / "Jerbi-reuploading"



name1b="FL-ppo-ac-NN(3)-(actor-lr=1.0e-2)-(67-params)-Det-Tests"
name2b="FL-ppo-ac-NN(4)-(actor-lr=1.0e-2)-(88-params)-Det-Tests"

start_seed=10
seed_step=10
end_seed=50
envs=("Deterministic-ShortestPath-4x4-FrozenLake-v0") 
timesteps=150000

circuits=("classic_NN")

for env in ${envs[@]}; do
    for circuit in ${circuits[@]}; do
        for seed in $(seq $start_seed $seed_step $end_seed); do
            sbatch --job-name="run-$env-$name1b-$seed" jobs/job.sh  --exp-name $name1b  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4  --classic-actor-learning-rate 1.0e-2 --qlearning-rate 0 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer1-nodes 3 --actor-hidden-layer2-nodes 0 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False --deterministic-tests-for-plotting True
            sbatch --job-name="run-$env-$name2b-$seed" jobs/job.sh  --exp-name $name2b  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4  --classic-actor-learning-rate 1.0e-2 --qlearning-rate 0 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer1-nodes 4 --actor-hidden-layer2-nodes 0 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False --deterministic-tests-for-plotting True
        done
    done
done


name1a="CP-ppo-ac-NN(5,5)-(actor-lr=1.0e-4)-(67-params)-Det-Tests"
name2a="CP-ppo-ac-NN(6,5)-(actor-lr=1.0e-4)-(77-params)-Det-Tests"

start_seed=10
seed_step=10
end_seed=50
envs=("CartPole-v1") 
timesteps=500000

circuits=("classic_NN")

for env in ${envs[@]}; do
    for circuit in ${circuits[@]}; do
        for seed in $(seq $start_seed $seed_step $end_seed); do
            sbatch --job-name="run-$env-$name1a-$seed" jobs/job.sh  --exp-name $name1a  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4  --classic-actor-learning-rate 1.0e-4 --qlearning-rate 0 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer1-nodes 5 --actor-hidden-layer2-nodes 5 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False --deterministic-tests-for-plotting True
            sbatch --job-name="run-$env-$name2a-$seed" jobs/job.sh  --exp-name $name2a  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4  --classic-actor-learning-rate 1.0e-4 --qlearning-rate 0 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer1-nodes 6 --actor-hidden-layer2-nodes 5 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False --deterministic-tests-for-plotting True
        done
    done
done
