#!/bin/bash

# Specify hyperparameters
# circuits: "no_q_circuit" / "simple" / "simple_reuploading" / "simple_reuploading_with_input_scaleing" /
#          "Hgog" / "Hgog_reuploading" / "Hgog_reuploading_with_input_scaleing" /
#          "Jerbi-no-reuploading-no-input-scaleing" / "Jerbi-reuploading-no-input-scaleing" / "Jerbi-reuploading"



name8="CP-qppo-ac-simple_reuploading-arctan-x1-rescale-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(73-params)"

start_seed=30
seed_step=10
end_seed=30

envs=("CartPole-v1") 
timesteps=150000

circuits=("simple_reuploading")
for env in ${envs[@]}; do
    for circuit in ${circuits[@]}; do
        for seed in $(seq $start_seed $seed_step $end_seed); do
            sbatch --job-name="run-$env-$name8-$seed" jobs/job.sh  --exp-name $name8  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 1.0e-4 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --load-chkpt False --exp-qlr-scheduling True --exp-scheduling-halftime 100000 --exp-scheduling-qlearning-rate 2.5e-3 --output-scaleing True --output-scaleing-learning-rate 1e-4 --shared-output-scaleing-param True --param-init allsmall --insider-input-rescale True
        done
    done
done


name3b="CP-qppo-ac-simple_reuploading_with_input_scaleing-exp_sced-ht100000-start-qlr1.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(97-params)"

start_seed=10
seed_step=10
end_seed=10

envs=("CartPole-v1") 
timesteps=150000

circuits=("simple_reuploading_with_input_scaleing")
for env in ${envs[@]}; do
    for circuit in ${circuits[@]}; do
        for seed in $(seq $start_seed $seed_step $end_seed); do
            sbatch --job-name="run-$env-$name3b-$seed" jobs/job.sh  --exp-name $name3b  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 1.0e-4 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --load-chkpt False --exp-qlr-scheduling True --exp-scheduling-halftime 100000 --exp-scheduling-qlearning-rate 1.5e-3 --output-scaleing True --output-scaleing-learning-rate 1e-4 --shared-output-scaleing-param True --param-init allsmall
        done
    done
done



name5e="FL-ppo-ac-NN(64,64)-(actor-lr=5.0e-3)-(5508-params)"

start_seed=10
seed_step=10
end_seed=50
envs=("Deterministic-ShortestPath-4x4-FrozenLake-v0") 
timesteps=150000

circuits=("classic_NN")

for env in ${envs[@]}; do
    for circuit in ${circuits[@]}; do
        for seed in $(seq $start_seed $seed_step $end_seed); do
            sbatch --job-name="run-$env-$name5e-$seed" jobs/job.sh  --exp-name $name5e  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4  --classic-actor-learning-rate 5.0e-3 --qlearning-rate 0 --n-qubits 0 --n-var-layers 0 --n-enc-layers 0 --actor-hidden-layer1-nodes 64 --actor-hidden-layer2-nodes 64 --critic-hidden-layer-nodes 64 --quantum-actor False --load-chkpt False
        done
    done
done

