#!/bin/bash

# Specify hyperparameters
# circuits: "no_q_circuit" / "simple" / "simple_reuploading" / "simple_reuploading_with_input_scaleing" /
#          "Hgog" / "Hgog_reuploading" / "Hgog_reuploading_with_input_scaleing" /
#          "Jerbi-no-reuploading-no-input-scaleing" / "Jerbi-reuploading-no-input-scaleing" / "Jerbi-reuploading"

name4="CP-qppo-ac-simple_reuploading_with_input_scaleing-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(81-params)-5-layers"

start_seed=10
seed_step=10
end_seed=10

envs=("CartPole-v1") 
timesteps=150000

circuits=("simple_reuploading_with_input_scaleing")
for env in ${envs[@]}; do
    for circuit in ${circuits[@]}; do
        for seed in $(seq $start_seed $seed_step $end_seed); do
            sbatch --job-name="run-$env-$name4-$seed" jobs/job.sh  --exp-name $name4  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 1.0e-4 --n-qubits 4 --n-var-layers 5 --n-enc-layers 5 --load-chkpt False --exp-qlr-scheduling True --exp-scheduling-halftime 100000 --exp-scheduling-qlearning-rate 2.5e-3 --output-scaleing True --output-scaleing-learning-rate 1e-4 --shared-output-scaleing-param True --param-init allsmall
        done
    done
done


name1="CP-qppo-ac-simple_reuploading-tanh-x2-rescale-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(73-params)"

start_seed=10
seed_step=10
end_seed=30

envs=("CartPole-v1") 
timesteps=150000

circuits=("simple_reuploading")
for env in ${envs[@]}; do
    for circuit in ${circuits[@]}; do
        for seed in $(seq $start_seed $seed_step $end_seed); do
            sbatch --job-name="run-$env-$name1-$seed" jobs/job.sh  --exp-name $name1  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 1.0e-4 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --load-chkpt False --exp-qlr-scheduling True --exp-scheduling-halftime 100000 --exp-scheduling-qlearning-rate 2.5e-3 --output-scaleing True --output-scaleing-learning-rate 1e-4 --shared-output-scaleing-param True --param-init allsmall --insider-input-rescale True
        done
    done
done


name2="CP-qppo-ac-simple_reuploading_with_shared_input_scaleing-exp_sced-ht100000-start-qlr2.5e-3-end-qlr1e-4-output_scaleing-1param-1e-4-allsmallinit-(77-params)"

start_seed=10
seed_step=10
end_seed=30

envs=("CartPole-v1") 
timesteps=500000    #restarted after 150000

circuits=("simple_reuploading_with_shared_input_scaleing")
for env in ${envs[@]}; do
    for circuit in ${circuits[@]}; do
        for seed in $(seq $start_seed $seed_step $end_seed); do
            sbatch --job-name="run-$env-$name2-$seed" jobs/job.sh  --exp-name $name2  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 1.0e-4 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --load-chkpt True --exp-qlr-scheduling True --exp-scheduling-halftime 100000 --exp-scheduling-qlearning-rate 2.5e-3 --output-scaleing True --output-scaleing-learning-rate 1e-4 --shared-output-scaleing-param True --param-init allsmall
        done
    done
done


