#!/bin/bash

# Specify hyperparameters

name1="qppo-Jerbi_1enc_layer_no_input_scaleing-lr_sceduling_AND_trainable_outpScl-qlr0.1e-3"
name2="qppo-Jerbi_1enc_layer_no_input_scaleing-lr_sceduling_AND_trainable_outpScl-qlr0.5e-3"
name3="qppo-Jerbi_1enc_layer_no_input_scaleing-lr_sceduling_AND_trainable_outpScl-qlr2.5e-3"
name4="qppo-Jerbi_1enc_layer_no_input_scaleing-lrANDoutpSclx1.0_sceduling-qlr0.1e-3"
name5="qppo-Jerbi_1enc_layer_no_input_scaleing-lrANDoutpSclx1.0_sceduling-qlr0.5e-3"
name6="qppo-Jerbi_1enc_layer_no_input_scaleing-lrANDoutpSclx1.0_sceduling-qlr2.5e-3"
name7="qppo-Jerbi_1enc_layer_no_input_scaleing-lrANDoutpSclx3.0_sceduling-qlr0.1e-3"
name8="qppo-Jerbi_1enc_layer_no_input_scaleing-lrANDoutpSclx3.0_sceduling-qlr0.5e-3"
name9="qppo-Jerbi_1enc_layer_no_input_scaleing-lrANDoutpSclx3.0_sceduling-qlr2.5e-3"

start_seed=10
seed_step=10
end_seed=20
envs=("Deterministic-ShortestPath-4x4-FrozenLake-v0") 
circuits=("Jerbi-no-reuploading-no-input-scaleing")

for env in ${envs[@]}; do
    for circuit in ${circuits[@]}; do
        for seed in $(seq $start_seed $seed_step $end_seed); do
            sbatch --job-name="run-$env-$name1-$seed" jobs/job.sh --exp-name $name1 --circuit $circuit --seed $seed --gym-id $env --num-steps 128 --total-timesteps 500000 --warmup-timesteps 50000 --warmup-learning-rate-bonus 0 --learning-rate 2.5e-4 --warmup-qlearning-rate-bonus 4.9e-3 --qlearning-rate 0.1e-3 --n-qubits 4 --n-var-layers 2 --n-enc-layers 1 --hybrid False --anneal-lr True --load-chkpt False --sceduled-output-scaleing False
            sbatch --job-name="run-$env-$name2-$seed" jobs/job.sh --exp-name $name2 --circuit $circuit --seed $seed --gym-id $env --num-steps 128 --total-timesteps 500000 --warmup-timesteps 50000 --warmup-learning-rate-bonus 0 --learning-rate 2.5e-4 --warmup-qlearning-rate-bonus 4.5e-3 --qlearning-rate 0.5e-3 --n-qubits 4 --n-var-layers 2 --n-enc-layers 1 --hybrid False --anneal-lr True --load-chkpt False --sceduled-output-scaleing False
            sbatch --job-name="run-$env-$name3-$seed" jobs/job.sh --exp-name $name3 --circuit $circuit --seed $seed --gym-id $env --num-steps 128 --total-timesteps 500000 --warmup-timesteps 50000 --warmup-learning-rate-bonus 0 --learning-rate 2.5e-4 --warmup-qlearning-rate-bonus 2.5e-3 --qlearning-rate 2.5e-3 --n-qubits 4 --n-var-layers 2 --n-enc-layers 1 --hybrid False --anneal-lr True --load-chkpt False --sceduled-output-scaleing False
            sbatch --job-name="run-$env-$name4-$seed" jobs/job.sh --exp-name $name4 --circuit $circuit --seed $seed --gym-id $env --num-steps 128 --total-timesteps 500000 --warmup-timesteps 50000 --warmup-learning-rate-bonus 0 --learning-rate 2.5e-4 --warmup-qlearning-rate-bonus 4.9e-3 --qlearning-rate 0.1e-3 --n-qubits 4 --n-var-layers 2 --n-enc-layers 1 --hybrid False --anneal-lr True --load-chkpt False --sceduled-output-scaleing True --sced-out-scale-fac 1.0
            sbatch --job-name="run-$env-$name5-$seed" jobs/job.sh --exp-name $name5 --circuit $circuit --seed $seed --gym-id $env --num-steps 128 --total-timesteps 500000 --warmup-timesteps 50000 --warmup-learning-rate-bonus 0 --learning-rate 2.5e-4 --warmup-qlearning-rate-bonus 4.5e-3 --qlearning-rate 0.5e-3 --n-qubits 4 --n-var-layers 2 --n-enc-layers 1 --hybrid False --anneal-lr True --load-chkpt False --sceduled-output-scaleing True --sced-out-scale-fac 1.0
            sbatch --job-name="run-$env-$name6-$seed" jobs/job.sh --exp-name $name6 --circuit $circuit --seed $seed --gym-id $env --num-steps 128 --total-timesteps 500000 --warmup-timesteps 50000 --warmup-learning-rate-bonus 0 --learning-rate 2.5e-4 --warmup-qlearning-rate-bonus 2.5e-3 --qlearning-rate 2.5e-3 --n-qubits 4 --n-var-layers 2 --n-enc-layers 1 --hybrid False --anneal-lr True --load-chkpt False --sceduled-output-scaleing True --sced-out-scale-fac 1.0
            sbatch --job-name="run-$env-$name7-$seed" jobs/job.sh --exp-name $name7 --circuit $circuit --seed $seed --gym-id $env --num-steps 128 --total-timesteps 500000 --warmup-timesteps 50000 --warmup-learning-rate-bonus 0 --learning-rate 2.5e-4 --warmup-qlearning-rate-bonus 4.9e-3 --qlearning-rate 0.1e-3 --n-qubits 4 --n-var-layers 2 --n-enc-layers 1 --hybrid False --anneal-lr True --load-chkpt False --sceduled-output-scaleing True --sced-out-scale-fac 3.0
            sbatch --job-name="run-$env-$name8-$seed" jobs/job.sh --exp-name $name8 --circuit $circuit --seed $seed --gym-id $env --num-steps 128 --total-timesteps 500000 --warmup-timesteps 50000 --warmup-learning-rate-bonus 0 --learning-rate 2.5e-4 --warmup-qlearning-rate-bonus 4.5e-3 --qlearning-rate 0.5e-3 --n-qubits 4 --n-var-layers 2 --n-enc-layers 1 --hybrid False --anneal-lr True --load-chkpt False --sceduled-output-scaleing True --sced-out-scale-fac 3.0
            sbatch --job-name="run-$env-$name9-$seed" jobs/job.sh --exp-name $name9 --circuit $circuit --seed $seed --gym-id $env --num-steps 128 --total-timesteps 500000 --warmup-timesteps 50000 --warmup-learning-rate-bonus 0 --learning-rate 2.5e-4 --warmup-qlearning-rate-bonus 2.5e-3 --qlearning-rate 2.5e-3 --n-qubits 4 --n-var-layers 2 --n-enc-layers 1 --hybrid False --anneal-lr True --load-chkpt False --sceduled-output-scaleing True --sced-out-scale-fac 3.0
        done
    done
done

#default settings ppo:  'python main.py --exp-name ppo_default  --circuit "simple" --seed 1 --gym-id Deterministic-ShortestPath-4x4-FrozenLake-v0 --num-steps 128 --total-timesteps 300000 --learning-rate 2.5e-4 --qlearning-rate 0e-3 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer-nodes 4 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False')

#default settings qppo: 'python main.py --exp-name qppo_default --circuit "simple" --seed 1 --gym-id Deterministic-ShortestPath-4x4-FrozenLake-v0 --num-steps 128 --total-timesteps 400000 --warmup-timesteps 50000 --warmup-learning-rate-bonus 0 --learning-rate 2.5e-4 --warmup-qlearning-rate-bonus 4.5e-3 --qlearning-rate 0.5e-3 --n-qubits 4 --n-var-layers 2 --n-enc-layers 1 --hybrid False --anneal-lr True --load-chkpt False')
