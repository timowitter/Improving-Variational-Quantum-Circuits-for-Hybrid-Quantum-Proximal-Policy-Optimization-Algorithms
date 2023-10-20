#!/bin/bash

# Specify hyperparameters
# circuits: "no_q_circuit" / "simple" / "simple_reuploading" / "simple_reuploading_with_input_scaleing" /
#          "Hgog" / "Hgog_reuploading" / "Hgog_reuploading_with_input_scaleing" /
#          "Jerbi-no-reuploading-no-input-scaleing" / "Jerbi-reuploading-no-input-scaleing" / "Jerbi-reuploading"
name1a="CP-qppo-ac-simple_reuploading-insider-input-rescale2-qlr0.5e-3-output_scaleing-1param-1e-4-allsmallinit-(73-params)"
name1b="CP-qppo-ac-simple_reuploading-insider-input-rescale2-qlr2.5e-3-output_scaleing-1param-1e-4-allsmallinit-(73-params)"
name1c="CP-qppo-ac-simple_reuploading-insider-input-rescale2-qlr10.e-3-output_scaleing-1param-1e-4-allsmallinit-(73-params)"

name5a="CP-qppo-ac-simple_reuploading-insider-input-rescale2-qlr0.5e-3-output_scaleing-1param-1e-4-(97-params)-allsmallinit-8-layers"
name5b="CP-qppo-ac-simple_reuploading-insider-input-rescale2-qlr2.5e-3-output_scaleing-1param-1e-4-(97-params)-allsmallinit-8-layers"
name5c="CP-qppo-ac-simple_reuploading-insider-input-rescale2-qlr10.e-3-output_scaleing-1param-1e-4-(97-params)-allsmallinit-8-layers"

name2a="CP-qppo-ac-simple_reuploading_with_shared_input_scaleing-qlr0.5e-3-output_scaleing-1param-1e-4-allsmallinit-(77-params)"
name2b="CP-qppo-ac-simple_reuploading_with_shared_input_scaleing-qlr2.5e-3-output_scaleing-1param-1e-4-allsmallinit-(77-params)"
name2c="CP-qppo-ac-simple_reuploading_with_shared_input_scaleing-qlr10.e-3-output_scaleing-1param-1e-4-allsmallinit-(77-params)"

name4a="CP-qppo-ac-simple_reuploading_with_input_scaleing-qlr0.5e-3-output_scaleing-1param-1e-4-allsmallinit-(81-params)-5-layers"
name4b="CP-qppo-ac-simple_reuploading_with_input_scaleing-qlr2.5e-3-output_scaleing-1param-1e-4-allsmallinit-(81-params)-5-layers"
name4c="CP-qppo-ac-simple_reuploading_with_input_scaleing-qlr10.e-3-output_scaleing-1param-1e-4-allsmallinit-(81-params)-5-layers"

name3a="CP-qppo-ac-simple_reuploading_with_input_scaleing-qlr0.5e-3-output_scaleing-1param-1e-4-allsmallinit-(97-params)"
name3b="CP-qppo-ac-simple_reuploading_with_input_scaleing-qlr2.5e-3-output_scaleing-1param-1e-4-allsmallinit-(97-params)"
name3c="CP-qppo-ac-simple_reuploading_with_input_scaleing-qlr10.e-3-output_scaleing-1param-1e-4-allsmallinit-(97-params)"

start_seed=10
seed_step=10
end_seed=30

envs=("CartPole-v1") 
timesteps=150000    #restarted after 150000

circuits=("simple_reuploading")
for env in ${envs[@]}; do
    for circuit in ${circuits[@]}; do
        for seed in $(seq $start_seed $seed_step $end_seed); do
            sbatch --job-name="run-$env-$name1a-$seed" jobs/job.sh  --exp-name $name1a  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 0.5e-3 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --load-chkpt False --exp-qlr-scheduling False --output-scaleing True --output-scaleing-learning-rate 1e-4 --shared-output-scaleing-param True --param-init allsmall --insider-input-rescale True
            sbatch --job-name="run-$env-$name1b-$seed" jobs/job.sh  --exp-name $name1b  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 2.5e-3 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --load-chkpt False --exp-qlr-scheduling False --output-scaleing True --output-scaleing-learning-rate 1e-4 --shared-output-scaleing-param True --param-init allsmall --insider-input-rescale True
            sbatch --job-name="run-$env-$name1c-$seed" jobs/job.sh  --exp-name $name1c  --circuit $circuit --seed $seed --gym-id $env --total-timesteps $timesteps --learning-rate 2.5e-4 --qlearning-rate 1.0e-2 --n-qubits 4 --n-var-layers 6 --n-enc-layers 6 --load-chkpt False --exp-qlr-scheduling False --output-scaleing True --output-scaleing-learning-rate 1e-4 --shared-output-scaleing-param True --param-init allsmall --insider-input-rescale True
        done
    done
done
