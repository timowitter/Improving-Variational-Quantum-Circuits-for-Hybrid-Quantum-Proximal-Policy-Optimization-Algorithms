#!/bin/bash

# Specify hyperparameters
#circuits: "no_q_circuit" / "simple" / "Hgog" / "Jerbi-no-reuploading-no-input-scaleing" /" Jerbi-reuploading" / "Jerbi-reuploading-no-input-scaleing"


name1="qppo-simple-qlr0.5e-3-6varlayers-outscale-(76-params)"
name2="qppo-simple-qlr0.5e-3-6varlayers-outscale-expLRscedule-(76-params)"

start_seed=10
seed_step=10
end_seed=30
envs=("Deterministic-ShortestPath-4x4-FrozenLake-v0") 
circuits=("simple")
timesteps=500000

for env in ${envs[@]}; do
    for circuit in ${circuits[@]}; do
        for seed in $(seq $start_seed $seed_step $end_seed); do
            sbatch --job-name="run-$env-$name1-$seed" jobs/job.sh --exp-name $name1 --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 0.5e-3 --n-qubits 4 --n-var-layers 6 --n-enc-layers 1 --exp-qlr-sceduling False --output-scaleing True --load-chkpt False
            sbatch --job-name="run-$env-$name2-$seed" jobs/job.sh --exp-name $name2 --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 0.5e-3 --n-qubits 4 --n-var-layers 6 --n-enc-layers 1 --exp-qlr-sceduling True --output-scaleing True --load-chkpt False
        done
    done
done



#default settings ppo:  'python main.py --exp-name ppo_default  --circuit "simple" --seed 1 --gym-id Deterministic-ShortestPath-4x4-FrozenLake-v0 --num-steps 128 --total-timesteps 300000 --learning-rate 2.5e-4 --qlearning-rate 0e-3 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer-nodes 4 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False')

#default settings qppo: 'python main.py --exp-name qppo_default --circuit "simple" --seed 1 --gym-id Deterministic-ShortestPath-4x4-FrozenLake-v0 --num-steps 128 --total-timesteps 400000 --warmup-timesteps 50000 --warmup-learning-rate-bonus 0 --learning-rate 2.5e-4 --warmup-qlearning-rate 5e-3 --qlearning-rate 0.5e-3 --n-qubits 4 --n-var-layers 2 --n-enc-layers 1 --hybrid False --anneal-lr True --load-chkpt False')



