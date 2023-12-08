#!/bin/bash

# Specify hyperparameters
# circuits: "no_q_circuit" / "simple" / "simple_reuploading" / "simple_reuploading_with_input_scaleing" /
#          "Hgog" / "Hgog_reuploading" / "Hgog_reuploading_with_input_scaleing" /
#          "Jerbi-no-reuploading-no-input-scaleing" / "Jerbi-reuploading-no-input-scaleing" / "Jerbi-reuploading"




name1="CP-qppo-ac-simple_reuploading----------------exp_sced-ht100000-qlr(25-1)e-4-OutScale(2e-4)-allsmallinit-(73-params)-Det-Tests"
name5="CP-qppo-ac-simple_reuploading----------------exp_sced-ht100000-qlr(25-1)e-4----------------allsmallinit-(72-params)-Det-Tests"
name2="CP-qppo-ac-simple_reuploading_sharedInpScale-exp_sced-ht100000-qlr(25-1)e-4-OutScale(2e-4)-allsmallinit-(77-params)-Det-Tests"
name3="CP-qppo-ac-simple_reuploading_Input_Scaleing-exp_sced-ht100000-qlr(25-1)e-4-OutScale(2e-4)-allsmallinit-(65-params)-4-layers-Det-Tests"
name4="CP-qppo-ac-simple_reuploading_Input_Scaleing-exp_sced-ht100000-qlr(25-1)e-4-OutScale(2e-4)-allsmallinit-(81-params)-5-layers-Det-Tests"


start_seed=10
seed_step=10
end_seed=30

envs=("CartPole-v1") 
timesteps=500000        #restarted after 150000

circuits=("simple_reuploading")
for env in ${envs[@]}; do
    for circuit in ${circuits[@]}; do
        for seed in $(seq $start_seed $seed_step $end_seed); do
            sbatch --job-name="run-$env-$name1-$seed" jobs/job.sh  --exp-name $name1  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 1.0e-4 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --load-chkpt True --exp-qlr-scheduling True --exp-scheduling-halftime 100000 --exp-scheduling-qlearning-rate 2.5e-3 --output-scaleing True --output-scaleing-learning-rate 2e-4 --shared-output-scaleing-param True --param-init allsmall --deterministic-tests-for-plotting True
        done
    done
done

circuits=("simple_reuploading")
for env in ${envs[@]}; do
    for circuit in ${circuits[@]}; do
        for seed in $(seq $start_seed $seed_step $end_seed); do
            sbatch --job-name="run-$env-$name5-$seed" jobs/job.sh  --exp-name $name5  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 1.0e-4 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --load-chkpt True --exp-qlr-scheduling True --exp-scheduling-halftime 100000 --exp-scheduling-qlearning-rate 2.5e-3 --output-scaleing False --param-init allsmall --deterministic-tests-for-plotting True
        done
    done
done

