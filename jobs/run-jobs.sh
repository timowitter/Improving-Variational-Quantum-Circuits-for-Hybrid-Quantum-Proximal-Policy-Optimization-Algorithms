#!/bin/bash

# Specify hyperparameters
# circuits: "no_q_circuit" / "simple" / "simple_reuploading" / "simple_reuploading_with_input_scaleing" /
#          "Hgog" / "Hgog_reuploading" / "Hgog_reuploading_with_input_scaleing" /
#          "Jerbi-no-reuploading-no-input-scaleing" / "Jerbi-reuploading-no-input-scaleing" / "Jerbi-reuploading"

#name1="CP-qppo-ac-simple_reuploading-exp_sced-output_scaleing-1param-1e-2-(73-params)"
# sbatch --job-name="run-$env-$name1-$seed" jobs/job.sh  --exp-name $name1  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 0.1e-3 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --load-chkpt False --exp-qlr-scheduling True  --exp-scheduling-halftime 25000 --exp-scheduling-qlearning-rate 10e-3 --output-scaleing True --output-scaleing-learning-rate 1e-2 --shared-output-scaleing-param True

name2="CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)-gaussinit"
name3="CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)-allsmallinit"

start_seed=10
seed_step=10
end_seed=30

envs=("CartPole-v1")
timesteps=500000

circuits=("simple_reuploading")
for env in ${envs[@]}; do
    for circuit in ${circuits[@]}; do
        for seed in $(seq $start_seed $seed_step $end_seed); do
            sbatch --job-name="run-$env-$name2-$seed" jobs/job.sh  --exp-name $name2  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 2.5e-3 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --load-chkpt True --exp-qlr-scheduling False --output-scaleing True --output-scaleing-learning-rate 1e-4 --shared-output-scaleing-param True --param-init gauss_distribution --record-grads True
            sbatch --job-name="run-$env-$name3-$seed" jobs/job.sh  --exp-name $name3  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 2.5e-3 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --load-chkpt True --exp-qlr-scheduling False --output-scaleing True --output-scaleing-learning-rate 1e-4 --shared-output-scaleing-param True --param-init allsmall           --record-grads True
        done
    done
done

name6="CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)-unclipped_gaussinit"
name7="CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)-rescaled_gaussinit"
name8="CP-qppo-ac-simple_reuploading-qlr2.5e-3-output_scaleing-1param-1e-4-(73-params)-allverysmall"

start_seed=10
seed_step=10
end_seed=30

envs=("CartPole-v1")
timesteps=150000

circuits=("simple_reuploading")
for env in ${envs[@]}; do
    for circuit in ${circuits[@]}; do
        for seed in $(seq $start_seed $seed_step $end_seed); do
            sbatch --job-name="run-$env-$name6-$seed" jobs/job.sh  --exp-name $name6  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 2.5e-3 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --load-chkpt False --exp-qlr-scheduling False --output-scaleing True --output-scaleing-learning-rate 1e-4 --shared-output-scaleing-param True --param-init unclipped_gauss_distribution --record-grads True
            sbatch --job-name="run-$env-$name7-$seed" jobs/job.sh  --exp-name $name7  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 2.5e-3 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --load-chkpt False --exp-qlr-scheduling False --output-scaleing True --output-scaleing-learning-rate 1e-4 --shared-output-scaleing-param True --param-init rescaled_gauss_distribution  --record-grads True
            sbatch --job-name="run-$env-$name8-$seed" jobs/job.sh  --exp-name $name8  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 2.5e-3 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --load-chkpt False --exp-qlr-scheduling False --output-scaleing True --output-scaleing-learning-rate 1e-4 --shared-output-scaleing-param True --param-init allverysmall                 --record-grads True
        done
    done
done

#--weight-remapping none / clipped / pos_clipped / tanh / double_tanh / pos_tanh
#default settings ppo:  'python main.py --exp-name ppo_default  --circuit "simple" --seed 1 --gym-id Deterministic-ShortestPath-4x4-FrozenLake-v0 --num-steps 128 --total-timesteps 300000 --learning-rate 2.5e-4 --qlearning-rate 0e-3 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer-nodes 4 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False')

#default settings qppo: 'python main.py --exp-name qppo_default --circuit "simple" --seed 1 --gym-id Deterministic-ShortestPath-4x4-FrozenLake-v0 --num-steps 128 --total-timesteps 400000 --warmup-timesteps 50000 --warmup-learning-rate-bonus 0 --learning-rate 2.5e-4 --warmup-qlearning-rate 5e-3 --qlearning-rate 0.5e-3 --n-qubits 4 --n-var-layers 2 --n-enc-layers 1 --hybrid False --anneal-lr True --load-chkpt False')



