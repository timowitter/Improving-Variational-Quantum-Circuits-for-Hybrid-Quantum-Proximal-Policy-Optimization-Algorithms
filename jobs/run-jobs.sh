#!/bin/bash

# Specify hyperparameters
# circuits: "no_q_circuit" / "simple" / "simple_reuploading" / "simple_reuploading_with_input_scaleing" /
#          "Hgog" / "Hgog_reuploading" / "Hgog_reuploading_with_input_scaleing" /
#          "Jerbi-no-reuploading-no-input-scaleing" / "Jerbi-reuploading-no-input-scaleing" / "Jerbi-reuploading"


name1="CP-qppo-ac-simple_reuploading-qlr0.5e-3-output_scaleing-1param-1e-4-(73-params)-allsmallinit"
name2="CP-qppo-ac-simple_reuploading-qlr10.e-3-output_scaleing-1param-1e-4-(73-params)-allsmallinit"
name3="CP-qppo-ac-simple_reuploading-exp_sced-ht_75000-start-qlr1e-2-end-qlr1e-4-output_scaleing-1param-1e-4-(73-params)-allsmallinit"
name4="CP-qppo-ac-simple_reuploading-exp_sced-ht100000-start-qlr1e-2-end-qlr1e-4-output_scaleing-1param-1e-4-(73-params)-allsmallinit"
name5="CP-qppo-ac-simple_reuploading-exp_sced-ht150000-start-qlr1e-2-end-qlr1e-4-output_scaleing-1param-1e-4-(73-params)-allsmallinit"


start_seed=10
seed_step=10
end_seed=30

envs=("CartPole-v1") 
timesteps=500000

circuits=("simple_reuploading")
for env in ${envs[@]}; do
    for circuit in ${circuits[@]}; do
        for seed in $(seq $start_seed $seed_step $end_seed); do
            sbatch --job-name="run-$env-$name1-$seed" jobs/job.sh  --exp-name $name1  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 5.0e-4 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --load-chkpt True --exp-qlr-scheduling False --output-scaleing True --output-scaleing-learning-rate 1e-4 --shared-output-scaleing-param True --param-init allsmall
            sbatch --job-name="run-$env-$name2-$seed" jobs/job.sh  --exp-name $name2  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 1.0e-2 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --load-chkpt True --exp-qlr-scheduling False --output-scaleing True --output-scaleing-learning-rate 1e-4 --shared-output-scaleing-param True --param-init allsmall
            sbatch --job-name="run-$env-$name3-$seed" jobs/job.sh  --exp-name $name3  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 1.0e-4 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --load-chkpt True --exp-qlr-scheduling True --exp-scheduling-halftime  75000 --exp-scheduling-qlearning-rate 1e-2 --output-scaleing True --output-scaleing-learning-rate 1e-4 --shared-output-scaleing-param True --param-init allsmall
            sbatch --job-name="run-$env-$name4-$seed" jobs/job.sh  --exp-name $name4  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 1.0e-4 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --load-chkpt True --exp-qlr-scheduling True --exp-scheduling-halftime 100000 --exp-scheduling-qlearning-rate 1e-2 --output-scaleing True --output-scaleing-learning-rate 1e-4 --shared-output-scaleing-param True --param-init allsmall
            sbatch --job-name="run-$env-$name5-$seed" jobs/job.sh  --exp-name $name5  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 1.0e-4 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --load-chkpt True --exp-qlr-scheduling True --exp-scheduling-halftime 150000 --exp-scheduling-qlearning-rate 1e-2 --output-scaleing True --output-scaleing-learning-rate 1e-4 --shared-output-scaleing-param True --param-init allsmall
        done
    done
done



