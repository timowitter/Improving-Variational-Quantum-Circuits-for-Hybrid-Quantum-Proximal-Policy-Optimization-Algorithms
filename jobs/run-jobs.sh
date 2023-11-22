#!/bin/bash

# Specify hyperparameters
# circuits: "no_q_circuit" / "simple" / "simple_reuploading" / "simple_reuploading_with_input_scaleing" /
#          "Hgog" / "Hgog_reuploading" / "Hgog_reuploading_with_input_scaleing" /
#          "Jerbi-no-reuploading-no-input-scaleing" / "Jerbi-reuploading-no-input-scaleing" / "Jerbi-reuploading"



name1="CP-qppo-ac-simple_reuploading----------------exp_sced-ht100000-qlr(25-1)e-4-OutScale(2e-4)-allsmallinit-(73-params)"
name2="CP-qppo-ac-simple_reuploading_sharedInpScale-exp_sced-ht100000-qlr(25-1)e-4-OutScale(2e-4)-allsmallinit-(77-params)"
name3="CP-qppo-ac-simple_reuploading_Input_Scaleing-exp_sced-ht100000-qlr(25-1)e-4-OutScale(2e-4)-allsmallinit-(65-params)-4-layers"
name4="CP-qppo-ac-simple_reuploading_Input_Scaleing-exp_sced-ht100000-qlr(25-1)e-4-OutScale(2e-4)-allsmallinit-(81-params)-5-layers"
#name5="CP-qppo-ac-Jerbi_reuploading-----------------exp_sced-ht100000-qlr(25-1)e-4-OutScale(2e-4)-allsmallinit-(73-params)-4-enc-layers"


start_seed=10
seed_step=10
end_seed=10

envs=("CartPole-v1") 
timesteps=500000

circuits=("simple_reuploading_with_input_scaleing")
for env in ${envs[@]}; do
    for circuit in ${circuits[@]}; do
        for seed in $(seq $start_seed $seed_step $end_seed); do
            sbatch --job-name="run-$env-$name3-$seed" jobs/job.sh  --exp-name $name3  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 1.0e-4 --n-qubits 4 --n-var-layers 4 --n-enc-layers 4 --load-chkpt True --exp-qlr-scheduling True --exp-scheduling-halftime 100000 --exp-scheduling-qlearning-rate 2.5e-3 --output-scaleing True --output-scaleing-learning-rate 2e-4 --shared-output-scaleing-param True --param-init allsmall
        done
    done
done

start_seed=30
seed_step=10
end_seed=30

envs=("CartPole-v1") 
timesteps=150000

circuits=("simple_reuploading_with_input_scaleing")
for env in ${envs[@]}; do
    for circuit in ${circuits[@]}; do
        for seed in $(seq $start_seed $seed_step $end_seed); do
            sbatch --job-name="run-$env-$name4-$seed" jobs/job.sh  --exp-name $name4  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 1.0e-4 --n-qubits 4 --n-var-layers 5 --n-enc-layers 5 --load-chkpt True --exp-qlr-scheduling True --exp-scheduling-halftime 100000 --exp-scheduling-qlearning-rate 2.5e-3 --output-scaleing True --output-scaleing-learning-rate 2e-4 --shared-output-scaleing-param True --param-init allsmall
        done
    done
done

