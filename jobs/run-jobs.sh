#!/bin/bash

# Specify hyperparameters
#circuits: "no_q_circuit" / "simple" / "Hgog" / "Jerbi-no-reuploading-no-input-scaleing" /" Jerbi-reuploading" / "Jerbi-reuploading-no-input-scaleing"

name0="random-baseline"

start_seed=10
seed_step=10
end_seed=50
envs=("Deterministic-ShortestPath-4x4-FrozenLake-v0") 
circuits=("random-baseline")

for env in ${envs[@]}; do
    for circuit in ${circuits[@]}; do
        for seed in $(seq $start_seed $seed_step $end_seed); do
            sbatch --job-name="run-$env-$name10-$seed" jobs/job.sh --exp-name $name10 --circuit $circuit --seed $seed --gym-id $env --total-timesteps 1000000 --learning-rate 0 --qlearning-rate 0 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer-nodes 4 --critic-hidden-layer-nodes 4 --quantum-actor False --load-chkpt False --random-baseline True
        done
    done
done


name10="qppo-Jerbi_1enc_layer_no_input_scaleing-lrANDoutpSclx5.0_sceduling-qlr0.25e-3"
name11="qppo-Jerbi_1enc_layer_no_input_scaleing-lrANDoutpSclx5.0_sceduling-qlr0.5e-3"
name12="qppo-Jerbi_1enc_layer_no_input_scaleing-lrANDoutpSclx5.0_sceduling-qlr1.0e-3"
name13="qppo-Jerbi_2enc_layer_no_input_scaleing-lrANDoutpSclx3.0_sceduling-qlr0.1e-3"
name14="qppo-Jerbi_2enc_layer_no_input_scaleing-lrANDoutpSclx3.0_sceduling-qlr0.5e-3"
name15="qppo-Jerbi_2enc_layer_no_input_scaleing-lrANDoutpSclx3.0_sceduling-qlr2.5e-3"

start_seed=10
seed_step=10
end_seed=20
envs=("Deterministic-ShortestPath-4x4-FrozenLake-v0") 

circuits=("Jerbi-reuploading-no-input-scaleing")

for env in ${envs[@]}; do
    for circuit in ${circuits[@]}; do
        for seed in $(seq $start_seed $seed_step $end_seed); do
            sbatch --job-name="run-$env-$name13-$seed" jobs/job.sh --exp-name $name13 --circuit $circuit --seed $seed --gym-id $env --num-steps 128 --total-timesteps 500000 --warmup-timesteps 50000 --warmup-learning-rate-bonus 0 --learning-rate 2.5e-4 --warmup-qlearning-rate-bonus 4.9e-3 --qlearning-rate 0.1e-3 --n-qubits 4 --n-var-layers 3 --n-enc-layers 2 --hybrid False --anneal-lr True --load-chkpt False --sceduled-output-scaleing True --sced-out-scale-fac 3.0
            sbatch --job-name="run-$env-$name14-$seed" jobs/job.sh --exp-name $name14 --circuit $circuit --seed $seed --gym-id $env --num-steps 128 --total-timesteps 500000 --warmup-timesteps 50000 --warmup-learning-rate-bonus 0 --learning-rate 2.5e-4 --warmup-qlearning-rate-bonus 4.5e-3 --qlearning-rate 0.5e-3 --n-qubits 4 --n-var-layers 3 --n-enc-layers 2 --hybrid False --anneal-lr True --load-chkpt False --sceduled-output-scaleing True --sced-out-scale-fac 3.0
            sbatch --job-name="run-$env-$name15-$seed" jobs/job.sh --exp-name $name15 --circuit $circuit --seed $seed --gym-id $env --num-steps 128 --total-timesteps 500000 --warmup-timesteps 50000 --warmup-learning-rate-bonus 0 --learning-rate 2.5e-4 --warmup-qlearning-rate-bonus 2.5e-3 --qlearning-rate 2.5e-3 --n-qubits 4 --n-var-layers 3 --n-enc-layers 2 --hybrid False --anneal-lr True --load-chkpt False --sceduled-output-scaleing True --sced-out-scale-fac 3.0
        done
    done
done

circuits=("Jerbi-no-reuploading-no-input-scaleing")

for env in ${envs[@]}; do
    for circuit in ${circuits[@]}; do
        for seed in $(seq $start_seed $seed_step $end_seed); do
            sbatch --job-name="run-$env-$name10-$seed" jobs/job.sh --exp-name $name10 --circuit $circuit --seed $seed --gym-id $env --num-steps 128 --total-timesteps 500000 --warmup-timesteps 50000 --warmup-learning-rate-bonus 0 --learning-rate 2.5e-4 --warmup-qlearning-rate-bonus 4.75e-3 --qlearning-rate 0.25e-3 --n-qubits 4 --n-var-layers 2 --n-enc-layers 1 --hybrid False --anneal-lr True --load-chkpt False --sceduled-output-scaleing True --sced-out-scale-fac 5.0
            sbatch --job-name="run-$env-$name11-$seed" jobs/job.sh --exp-name $name11 --circuit $circuit --seed $seed --gym-id $env --num-steps 128 --total-timesteps 500000 --warmup-timesteps 50000 --warmup-learning-rate-bonus 0 --learning-rate 2.5e-4 --warmup-qlearning-rate-bonus 4.5e-3  --qlearning-rate 0.5e-3 --n-qubits 4 --n-var-layers 2 --n-enc-layers 1 --hybrid False --anneal-lr True --load-chkpt False --sceduled-output-scaleing True --sced-out-scale-fac 5.0
            sbatch --job-name="run-$env-$name12-$seed" jobs/job.sh --exp-name $name12 --circuit $circuit --seed $seed --gym-id $env --num-steps 128 --total-timesteps 500000 --warmup-timesteps 50000 --warmup-learning-rate-bonus 0 --learning-rate 2.5e-4 --warmup-qlearning-rate-bonus 4.0e-3  --qlearning-rate 1.0e-3 --n-qubits 4 --n-var-layers 2 --n-enc-layers 1 --hybrid False --anneal-lr True --load-chkpt False --sceduled-output-scaleing True --sced-out-scale-fac 5.0
        done
    done
done







#default settings ppo:  'python main.py --exp-name ppo_default  --circuit "simple" --seed 1 --gym-id Deterministic-ShortestPath-4x4-FrozenLake-v0 --num-steps 128 --total-timesteps 300000 --learning-rate 2.5e-4 --qlearning-rate 0e-3 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer-nodes 4 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False')

#default settings qppo: 'python main.py --exp-name qppo_default --circuit "simple" --seed 1 --gym-id Deterministic-ShortestPath-4x4-FrozenLake-v0 --num-steps 128 --total-timesteps 400000 --warmup-timesteps 50000 --warmup-learning-rate-bonus 0 --learning-rate 2.5e-4 --warmup-qlearning-rate-bonus 4.5e-3 --qlearning-rate 0.5e-3 --n-qubits 4 --n-var-layers 2 --n-enc-layers 1 --hybrid False --anneal-lr True --load-chkpt False')
