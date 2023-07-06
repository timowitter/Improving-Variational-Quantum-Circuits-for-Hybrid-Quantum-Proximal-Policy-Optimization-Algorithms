#!/bin/bash

# Specify hyperparameters
start_seed=10
seed_step=10
end_seed=50
envs=("Deterministic-ShortestPath-4x4-FrozenLake-v0")
circuits=("no_q_circuit")

name7="ppo_default_lr6.0e-4"
name8="ppo_default_lr8.0e-4"
name9="ppo_default_lr10.0e-4"
name10="ppo_default_lr15.0e-4"

for env in ${envs[@]}; do
    for circuit in ${circuits[@]}; do
        for seed in $(seq $start_seed $seed_step $end_seed); do
            sbatch --job-name="run-$env-$name7-$seed" jobs/job.sh --exp-name $name7 --circuit $circuit --seed $seed --gym-id $env --num-steps 128 --total-timesteps 300000 --learning-rate 6.0e-4 --qlearning-rate 0 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer-nodes 4 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False
            sbatch --job-name="run-$env-$name8-$seed" jobs/job.sh --exp-name $name8 --circuit $circuit --seed $seed --gym-id $env --num-steps 128 --total-timesteps 300000 --learning-rate 8.0e-4 --qlearning-rate 0 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer-nodes 4 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False
            sbatch --job-name="run-$env-$name9-$seed" jobs/job.sh --exp-name $name9 --circuit $circuit --seed $seed --gym-id $env --num-steps 128 --total-timesteps 300000 --learning-rate 10.0e-4 --qlearning-rate 0 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer-nodes 4 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False
            sbatch --job-name="run-$env-$name10-$seed" jobs/job.sh --exp-name $name10 --circuit $circuit --seed $seed --gym-id $env --num-steps 128 --total-timesteps 300000 --learning-rate 15.0e-4 --qlearning-rate 0 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer-nodes 4 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False
        done
    done
done


#default settings ppo:  'python main.py --exp-name ppo_default  --circuit "simple" --seed 1 --gym-id Deterministic-ShortestPath-4x4-FrozenLake-v0 --num-steps 128 --total-timesteps 300000 --learning-rate 2.5e-4 --qlearning-rate 0e-3 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer-nodes 4 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False')

#default settings qppo: 'python main.py --exp-name qppo_default --circuit "simple" --seed 1 --gym-id Deterministic-ShortestPath-4x4-FrozenLake-v0 --num-steps 128 --total-timesteps 400000 --warmup-timesteps 50000 --warmup-learning-rate-bonus 0 --learning-rate 2.5e-4 --warmup-qlearning-rate-bonus 4.5e-3 --qlearning-rate 0.5e-3 --n-qubits 4 --n-var-layers 2 --n-enc-layers 1 --hybrid False --anneal-lr True --load-chkpt False')
