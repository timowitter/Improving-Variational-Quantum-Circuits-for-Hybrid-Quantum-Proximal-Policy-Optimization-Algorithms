#!/bin/bash

# Specify hyperparameters
# circuits: "no_q_circuit" / "simple" / "simple_reuploading" / "simple_reuploading_with_input_scaleing" /
#          "Hgog" / "Hgog_reuploading" / "Hgog_reuploading_with_input_scaleing" /
#          "Jerbi-no-reuploading-no-input-scaleing" / "Jerbi-reuploading-no-input-scaleing" / "Jerbi-reuploading"

name5f="FL-ppo-ac-NN(64,64)-(actor-lr=2.5e-3)-(5508-params)"

start_seed=10
seed_step=10
end_seed=50
envs=("Deterministic-ShortestPath-4x4-FrozenLake-v0") 
timesteps=150000

circuits=("classic_NN")

for env in ${envs[@]}; do
    for circuit in ${circuits[@]}; do
        for seed in $(seq $start_seed $seed_step $end_seed); do
            sbatch --job-name="run-$env-$name5f-$seed" jobs/job.sh  --exp-name $name5f  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4  --classic-actor-learning-rate 2.5e-3 --qlearning-rate 0 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer1-nodes 64 --actor-hidden-layer2-nodes 64 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False
        done
    done
done





